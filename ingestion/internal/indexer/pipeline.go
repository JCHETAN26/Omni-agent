// Package indexer turns document records into searchable chunks in both stores.
//
// Delivery guarantees (with the Kafka loop in worker.go):
//   - At-least-once: offsets are committed only after every record in the poll
//     is either written to both stores or parked in the DLQ.
//   - Idempotent: chunk IDs are UUIDv5(doc_id, position), used as the
//     Elasticsearch _id and the Qdrant point ID, so a replay overwrites.
//   - Outages (connection errors, 429, 5xx) are retried with backoff until they
//     clear; the partition stalls (lag grows) rather than dropping or DLQ'ing
//     healthy documents. Only input that can never succeed goes to the DLQ.
package indexer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/chunker"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/sink"
)

// Record is the slice of a Kafka record the pipeline needs.
type Record struct {
	Partition int32
	Offset    int64
	Key       []byte
	Value     []byte
}

// Outcome is what happened to one record.
type Outcome struct {
	Record Record
	Doc    *event.DocumentEvent // nil if undecodable
	Chunks int
	Err    error     // non-nil: permanent failure, route to the DLQ
	DoneAt time.Time // when the document's writes returned (searchable)
}

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Sparse interface {
	Bulk(ctx context.Context, chunks []chunker.Chunk) ([]sink.ItemError, error)
}

type Dense interface {
	Upsert(ctx context.Context, chunks []chunker.Chunk, vectors [][]float32) error
}

// StageTimes accumulates time spent per stage (for metrics).
type StageTimes struct {
	Chunk, Embed, Sparse, Dense time.Duration
}

type Pipeline struct {
	Embedder     Embedder // nil with Dense: BM25-only mode (no embeddings, no Qdrant)
	Sparse       Sparse
	Dense        Dense
	ChunkSize    int
	ChunkOverlap int
	Log          *slog.Logger
	// Backoff bounds for retryable failures.
	MinBackoff, MaxBackoff time.Duration
	// OnRetry is called for every retried outage (metrics hook; may be nil).
	OnRetry func(stage string, err error)
}

type pending struct {
	idx    int // index into outcomes
	doc    *event.DocumentEvent
	chunks []chunker.Chunk
}

// Process indexes a batch of records from one partition (in offset order) and
// returns one outcome per record. It returns an error only when ctx ends.
func (p *Pipeline) Process(ctx context.Context, recs []Record, times *StageTimes) ([]Outcome, error) {
	outcomes := make([]Outcome, len(recs))
	latest := map[string]int{} // doc key -> index of its newest version in this batch
	for i, r := range recs {
		outcomes[i].Record = r
		var d event.DocumentEvent
		if err := json.Unmarshal(r.Value, &d); err != nil {
			outcomes[i].Err = fmt.Errorf("undecodable record: %w", err)
			continue
		}
		outcomes[i].Doc = &d
		if err := d.Validate(); err != nil {
			outcomes[i].Err = err
			continue
		}
		latest[string(d.Key())] = i
	}
	t0 := time.Now()
	var work []pending
	for i := range outcomes {
		o := &outcomes[i]
		if o.Err != nil || latest[string(o.Doc.Key())] != i {
			continue // invalid, or superseded by a newer version later in this batch
		}
		chunks, err := chunker.Split(chunker.Document{DocID: o.Doc.DocID, Title: o.Doc.Title, Text: o.Doc.Text},
			p.ChunkSize, p.ChunkOverlap)
		if err != nil || len(chunks) == 0 {
			o.Err = errors.Join(errors.New("document produced no chunks"), err)
			continue
		}
		work = append(work, pending{i, o.Doc, chunks})
	}
	times.Chunk += time.Since(t0)

	if err := p.write(ctx, work, outcomes, times); err != nil {
		return nil, err
	}
	// Superseded versions are searchable as soon as their newest version is.
	for i := range outcomes {
		o := &outcomes[i]
		if o.Err == nil && o.Doc != nil {
			if j := latest[string(o.Doc.Key())]; j != i {
				o.Chunks, o.Err = outcomes[j].Chunks, outcomes[j].Err
			}
		}
	}
	return outcomes, nil
}

// write indexes all pending docs together; if the batch fails permanently it
// falls back to one document at a time so only the bad documents fail.
func (p *Pipeline) write(ctx context.Context, work []pending, outcomes []Outcome, times *StageTimes) error {
	if len(work) == 0 {
		return nil
	}
	failed, err := p.writeWithRetry(ctx, work, times)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if len(work) == 1 {
			outcomes[work[0].idx].Err = err
			return nil
		}
		p.Log.Warn("batch failed permanently; isolating documents", "docs", len(work), "err", err)
		for _, w := range work {
			if err := p.write(ctx, []pending{w}, outcomes, times); err != nil {
				return err
			}
		}
		return nil
	}
	for _, w := range work {
		o := &outcomes[w.idx]
		if e, bad := failed[w.doc.DocID]; bad {
			o.Err = e
		} else {
			o.Chunks = len(w.chunks)
		}
	}
	return nil
}

// writeWithRetry retries outages until they clear (or ctx ends). It returns
// per-document permanent item errors, or a permanent batch error.
func (p *Pipeline) writeWithRetry(ctx context.Context, work []pending, times *StageTimes) (map[string]error, error) {
	backoff := p.MinBackoff
	for attempt := 1; ; attempt++ {
		failed, stage, err := p.writeOnce(ctx, work, times)
		if err == nil || !sink.IsRetryable(err) {
			return failed, err
		}
		if p.OnRetry != nil {
			p.OnRetry(stage, err)
		}
		p.Log.Warn("store outage; retrying", "stage", stage, "attempt", attempt, "backoff", backoff, "err", err)
		jitter := time.Duration(rand.Int64N(int64(backoff)/2 + 1))
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff + jitter):
		}
		backoff = min(backoff*2, p.MaxBackoff)
	}
}

func (p *Pipeline) writeOnce(ctx context.Context, work []pending, times *StageTimes) (map[string]error, string, error) {
	var chunks []chunker.Chunk
	for _, w := range work {
		chunks = append(chunks, w.chunks...)
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	var vectors [][]float32
	if p.Dense != nil {
		t0 := time.Now()
		v, err := p.Embedder.Embed(ctx, texts)
		times.Embed += time.Since(t0)
		if err != nil {
			return nil, "embed", err
		}
		vectors = v
	}

	var (
		wg              sync.WaitGroup
		items           []sink.ItemError
		sparseErr       error
		denseErr        error
		sparseD, denseD time.Duration
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		t := time.Now()
		items, sparseErr = p.Sparse.Bulk(ctx, chunks)
		sparseD = time.Since(t)
	}()
	go func() {
		defer wg.Done()
		if p.Dense == nil {
			return
		}
		t := time.Now()
		denseErr = p.Dense.Upsert(ctx, chunks, vectors)
		denseD = time.Since(t)
	}()
	wg.Wait()
	times.Sparse += sparseD
	times.Dense += denseD
	// Retry the whole batch on any outage: both writes are idempotent upserts.
	for _, e := range []struct {
		stage string
		err   error
	}{{"elasticsearch", sparseErr}, {"qdrant", denseErr}} {
		if e.err != nil && sink.IsRetryable(e.err) {
			return nil, e.stage, e.err
		}
	}
	for _, it := range items {
		if it.Err.Retryable {
			return nil, "elasticsearch", it.Err
		}
	}
	if sparseErr != nil {
		return nil, "elasticsearch", sparseErr
	}
	if denseErr != nil {
		return nil, "qdrant", denseErr
	}
	if len(items) == 0 {
		return nil, "", nil
	}
	byChunk := map[string]string{}
	for _, c := range chunks {
		byChunk[c.ChunkID] = c.DocID
	}
	failed := map[string]error{}
	for _, it := range items {
		failed[byChunk[it.ChunkID]] = it.Err
	}
	return failed, "", nil
}

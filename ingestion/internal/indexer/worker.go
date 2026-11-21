package indexer

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/metrics"
)

// Worker is one member of the indexing consumer group.
type Worker struct {
	Client     *kgo.Client // consumer group on event.TopicDocuments, autocommit off, BlockRebalanceOnPoll
	Pipeline   *Pipeline
	Name       string
	MaxPoll    int // records per poll
	BatchDocs  int // documents per pipeline batch (per partition)
	DLQTopic   string
	IndexTopic string
	Log        *slog.Logger
	now        func() time.Time
}

// Run polls until ctx ends. Each poll is fully indexed (or DLQ'd), its DLQ and
// indexed events are acknowledged by Kafka, and only then are its offsets
// committed. A crash anywhere before the commit replays the poll.
func (w *Worker) Run(ctx context.Context) error {
	if w.now == nil {
		w.now = time.Now
	}
	for {
		fetches := w.Client.PollRecords(ctx, w.MaxPoll)
		if fetches.IsClientClosed() || ctx.Err() != nil {
			return ctx.Err()
		}
		fetches.EachError(func(topic string, p int32, err error) {
			w.Log.Warn("fetch error", "topic", topic, "partition", p, "err", err)
		})
		recs := fetches.Records()
		if len(recs) == 0 {
			w.Client.AllowRebalance()
			continue
		}
		if err := w.handle(ctx, recs); err != nil {
			return err // ctx ended mid-batch: exit without committing; the batch replays
		}
		w.Client.AllowRebalance()
	}
}

func (w *Worker) handle(ctx context.Context, recs []*kgo.Record) error {
	start := w.now()
	byPartition := map[int32][]*kgo.Record{}
	for _, r := range recs {
		byPartition[r.Partition] = append(byPartition[r.Partition], r)
	}
	var (
		mu       sync.Mutex
		wg       sync.WaitGroup
		outcomes []Outcome
		firstErr error
		times    StageTimes
	)
	for _, prs := range byPartition {
		wg.Add(1)
		go func(prs []*kgo.Record) { // partitions in parallel, each in offset order
			defer wg.Done()
			for s := 0; s < len(prs); s += w.BatchDocs {
				batch := make([]Record, 0, w.BatchDocs)
				for _, r := range prs[s:min(s+w.BatchDocs, len(prs))] {
					batch = append(batch, Record{r.Partition, r.Offset, r.Key, r.Value})
				}
				var t StageTimes
				out, err := w.Pipeline.Process(ctx, batch, &t)
				mu.Lock()
				times.Chunk += t.Chunk
				times.Embed += t.Embed
				times.Sparse += t.Sparse
				times.Dense += t.Dense
				if err != nil && firstErr == nil {
					firstErr = err
				}
				outcomes = append(outcomes, out...)
				mu.Unlock()
				if err != nil {
					return
				}
			}
		}(prs)
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	metrics.StageSeconds.WithLabelValues("chunk").Observe(times.Chunk.Seconds())
	metrics.StageSeconds.WithLabelValues("embed").Observe(times.Embed.Seconds())
	metrics.StageSeconds.WithLabelValues("elasticsearch").Observe(times.Sparse.Seconds())
	metrics.StageSeconds.WithLabelValues("qdrant").Observe(times.Dense.Seconds())

	indexedAt := w.now()
	var out []*kgo.Record
	for _, o := range outcomes {
		if o.Err != nil {
			out = append(out, w.dlqRecord(o))
			metrics.Documents.WithLabelValues("dlq").Inc()
			continue
		}
		ev := event.IndexedEvent{
			Tenant: o.Doc.Tenant, DocID: o.Doc.DocID, Chunks: o.Chunks,
			Partition: o.Record.Partition, Offset: o.Record.Offset,
			AcceptedAt: o.Doc.AcceptedAt, IndexedAt: indexedAt.UnixMicro(), Worker: w.Name,
		}
		v, _ := json.Marshal(ev)
		out = append(out, &kgo.Record{Topic: w.IndexTopic, Key: o.Record.Key, Value: v})
		metrics.Documents.WithLabelValues("indexed").Inc()
		if o.Doc.AcceptedAt > 0 {
			metrics.EndToEndSeconds.Observe(float64(ev.IndexedAt-o.Doc.AcceptedAt) / 1e6)
		}
	}
	if err := w.produceAll(ctx, out); err != nil {
		return err
	}
	// Commit the highest offset per partition in this poll.
	for {
		err := w.Client.CommitRecords(ctx, recs...)
		if err == nil {
			break
		}
		metrics.CommitErrors.Inc()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// A rebalance moved our partitions: the new owner replays from the last
		// commit, which is safe because every write is idempotent.
		w.Log.Warn("offset commit failed; records will be replayed by the partition owner", "err", err)
		break
	}
	metrics.BatchSeconds.Observe(w.now().Sub(start).Seconds())
	w.Log.Debug("committed", "records", len(recs), "partitions", len(byPartition))
	return nil
}

func (w *Worker) dlqRecord(o Outcome) *kgo.Record {
	w.Log.Warn("routing to DLQ", "partition", o.Record.Partition, "offset", o.Record.Offset, "err", o.Err)
	return &kgo.Record{
		Topic: w.DLQTopic,
		Key:   o.Record.Key,
		Value: o.Record.Value,
		Headers: []kgo.RecordHeader{
			{Key: "error", Value: []byte(o.Err.Error())},
			{Key: "source.topic", Value: []byte(event.TopicDocuments)},
			{Key: "source.partition", Value: []byte(strconv.Itoa(int(o.Record.Partition)))},
			{Key: "source.offset", Value: []byte(strconv.FormatInt(o.Record.Offset, 10))},
			{Key: "worker", Value: []byte(w.Name)},
		},
	}
}

// produceAll writes records with acks=all and waits for every ack, retrying
// through broker outages; nothing is committed until this returns nil.
func (w *Worker) produceAll(ctx context.Context, recs []*kgo.Record) error {
	backoff := 200 * time.Millisecond
	for len(recs) > 0 {
		results := w.Client.ProduceSync(ctx, recs...)
		var retry []*kgo.Record
		for _, r := range results {
			if r.Err != nil {
				retry = append(retry, r.Record)
			}
		}
		if len(retry) == 0 {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		w.Log.Warn("produce failed; retrying", "records", len(retry), "err", errors.Join(results.FirstErr()))
		for _, r := range retry {
			r.Partition, r.Offset, r.LeaderEpoch = 0, 0, 0
		}
		recs = retry
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 10*time.Second)
	}
	return nil
}

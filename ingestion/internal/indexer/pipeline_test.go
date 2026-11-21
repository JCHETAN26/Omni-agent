package indexer

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/chunker"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/sink"
)

type fakeEmbed struct{ calls int }

func (f *fakeEmbed) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls++
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1}
	}
	return out, nil
}

// fakeStore is both stores: a map keyed by chunk ID (an idempotent upsert),
// with injectable outages and per-document permanent rejections.
type fakeStore struct {
	mu        sync.Mutex
	es, qd    map[string]chunker.Chunk
	outages   int    // fail this many calls with a retryable error
	rejectDoc string // ES rejects this doc's chunks permanently
}

func newStore() *fakeStore {
	return &fakeStore{es: map[string]chunker.Chunk{}, qd: map[string]chunker.Chunk{}}
}

func (s *fakeStore) Bulk(_ context.Context, cs []chunker.Chunk) ([]sink.ItemError, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.outages > 0 {
		s.outages--
		return nil, &sink.Error{Store: "elasticsearch", Status: 503, Retryable: true, Err: errors.New("down")}
	}
	var items []sink.ItemError
	for _, c := range cs {
		if c.DocID == s.rejectDoc {
			items = append(items, sink.ItemError{ChunkID: c.ChunkID, Err: &sink.Error{Status: 400, Err: errors.New("mapper_parsing_exception")}})
			continue
		}
		s.es[c.ChunkID] = c
	}
	return items, nil
}

func (s *fakeStore) Upsert(_ context.Context, cs []chunker.Chunk, _ [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range cs {
		s.qd[c.ChunkID] = c
	}
	return nil
}

func rec(off int64, d event.DocumentEvent) Record {
	v, _ := json.Marshal(d)
	return Record{Partition: 0, Offset: off, Key: d.Key(), Value: v}
}

func doc(id, text string) event.DocumentEvent {
	return event.DocumentEvent{Tenant: "t", DocID: id, Title: id, Text: text, AcceptedAt: 1}
}

func pipeline(s *fakeStore) *Pipeline {
	return &Pipeline{
		Embedder: &fakeEmbed{}, Sparse: s, Dense: s, ChunkSize: 100, ChunkOverlap: 20,
		Log:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		MinBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond,
	}
}

func TestReplayIsIdempotent(t *testing.T) {
	s := newStore()
	p := pipeline(s)
	recs := []Record{rec(0, doc("a", strings.Repeat("alpha beta gamma. ", 30))), rec(1, doc("b", "short"))}
	for range 3 { // the same records delivered three times (crash before commit)
		out, err := p.Process(context.Background(), recs, &StageTimes{})
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range out {
			if o.Err != nil {
				t.Fatal(o.Err)
			}
		}
	}
	want, _ := chunker.Split(chunker.Document{DocID: "a", Title: "a", Text: strings.Repeat("alpha beta gamma. ", 30)}, 100, 20)
	if len(s.es) != len(want)+1 || len(s.qd) != len(want)+1 {
		t.Fatalf("es=%d qd=%d chunks after 3 replays, want %d", len(s.es), len(s.qd), len(want)+1)
	}
}

func TestOutagesAreRetriedNotDropped(t *testing.T) {
	s := newStore()
	s.outages = 3
	p := pipeline(s)
	retries := 0
	p.OnRetry = func(string, error) { retries++ }
	out, err := p.Process(context.Background(), []Record{rec(0, doc("a", "text"))}, &StageTimes{})
	if err != nil || out[0].Err != nil || out[0].Chunks != 1 {
		t.Fatalf("out=%+v err=%v", out, err)
	}
	if retries != 3 || len(s.es) != 1 {
		t.Fatalf("retries=%d es=%d", retries, len(s.es))
	}
}

func TestOutageUntilShutdownReturnsWithoutOutcome(t *testing.T) {
	s := newStore()
	s.outages = 1 << 30
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := pipeline(s).Process(ctx, []Record{rec(0, doc("a", "text"))}, &StageTimes{}); err == nil {
		t.Fatal("expected ctx error: nothing may be committed while the store is down")
	}
}

func TestPermanentFailuresGoToDLQ(t *testing.T) {
	s := newStore()
	s.rejectDoc = "bad"
	recs := []Record{
		{Partition: 0, Offset: 0, Value: []byte("{not json")},
		rec(1, doc("", "no id")),
		rec(2, doc("bad", "rejected by elasticsearch")),
		rec(3, doc("good", "fine")),
	}
	out, err := pipeline(s).Process(context.Background(), recs, &StageTimes{})
	if err != nil {
		t.Fatal(err)
	}
	for i, wantErr := range []bool{true, true, true, false} {
		if (out[i].Err != nil) != wantErr {
			t.Errorf("record %d: err=%v, want error=%v", i, out[i].Err, wantErr)
		}
	}
	if _, ok := s.es[chunker.ChunkID("good", 0)]; !ok {
		t.Error("the good document must still be indexed")
	}
}

func TestNewestVersionInBatchWins(t *testing.T) {
	s := newStore()
	recs := []Record{rec(0, doc("a", "old text")), rec(1, doc("a", "new text"))}
	out, err := pipeline(s).Process(context.Background(), recs, &StageTimes{})
	if err != nil || out[0].Err != nil || out[1].Err != nil {
		t.Fatal(err, out)
	}
	if got := s.es[chunker.ChunkID("a", 0)].Text; got != "new text" {
		t.Fatalf("indexed %q", got)
	}
}

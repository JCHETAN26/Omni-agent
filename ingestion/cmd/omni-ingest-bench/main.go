// omni-ingest-bench drives the ingestion pipeline end to end and verifies it.
//
//  1. Loads N documents (BEIR corpus.jsonl), doc_ids prefixed with a run ID.
//  2. Uploads them to omni-ingest (optionally rate-limited), retrying any
//     upload that is not acknowledged (e.g. while the broker is down).
//  3. Reads omni.documents.indexed for exact per-document latency
//     (accepted by omni-ingest -> searchable in Elasticsearch and Qdrant).
//  4. Waits for the consumer group's lag to reach 0.
//  5. Verifies, per document, that Elasticsearch and Qdrant hold exactly the
//     chunks the chunker produces (no loss, no duplicates), that nothing from
//     this run reached the DLQ, and spot-checks search (ES, Qdrant, /search).
//
// Prints a JSON report between REPORT_BEGIN / REPORT_END; exits 1 if any
// verification fails.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/chunker"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/embed"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/kafkax"
)

type config struct {
	ingestURL, esURL, esIndex, qdrantURL, collection, tritonURL, backendURL string
	group, corpus, runID, tenant, output                                    string
	docs, batch, concurrency, chunkSize, overlap, sample                    int
	rate                                                                    float64
	timeout                                                                 time.Duration
}

type doc struct {
	ID       string `json:"doc_id"`
	Title    string `json:"title"`
	Text     string `json:"text"`
	expected int
	first    string // first chunk text (search spot-check)
}

var log = kafkax.Logger("omni-ingest-bench")

func main() {
	var c config
	flag.StringVar(&c.ingestURL, "ingest-url", "http://omniagent-ingest:8090", "omni-ingest base URL")
	flag.StringVar(&c.esURL, "es-url", "http://omniagent-elasticsearch:9200", "")
	flag.StringVar(&c.esIndex, "es-index", "omniagent_chunks", "")
	flag.StringVar(&c.qdrantURL, "qdrant-url", "http://omniagent-qdrant:6333", "")
	flag.StringVar(&c.collection, "collection", "omniagent_chunks", "")
	flag.StringVar(&c.tritonURL, "triton-url", "http://omniagent-triton:8000", "")
	flag.StringVar(&c.backendURL, "backend-url", "http://omniagent-backend:8000", "gateway for /search spot checks ('' to skip)")
	flag.StringVar(&c.group, "group", "omni-indexers", "indexer consumer group")
	flag.StringVar(&c.corpus, "corpus", "/data/corpus.jsonl", "BEIR corpus.jsonl")
	flag.StringVar(&c.runID, "run-id", fmt.Sprintf("run%d", time.Now().Unix()), "doc_id prefix for this run")
	flag.StringVar(&c.tenant, "tenant", "bench", "")
	flag.StringVar(&c.output, "output", "", "also write the report here")
	flag.IntVar(&c.docs, "docs", 10000, "documents to ingest")
	flag.IntVar(&c.batch, "batch", 100, "documents per upload")
	flag.IntVar(&c.concurrency, "concurrency", 8, "concurrent uploads")
	flag.Float64Var(&c.rate, "rate", 0, "target docs/s (0 = as fast as possible)")
	flag.IntVar(&c.chunkSize, "chunk-size", 800, "")
	flag.IntVar(&c.overlap, "chunk-overlap", 120, "")
	flag.IntVar(&c.sample, "sample", 50, "documents to spot-check with searches")
	flag.DurationVar(&c.timeout, "timeout", 30*time.Minute, "")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	report, ok := run(ctx, c)
	out, _ := json.MarshalIndent(report, "", "  ")
	fmt.Printf("REPORT_BEGIN\n%s\nREPORT_END\n", out)
	if c.output != "" {
		_ = os.WriteFile(c.output, out, 0o644)
	}
	if !ok {
		os.Exit(1)
	}
}

func loadCorpus(c config) ([]doc, int, error) {
	f, err := os.Open(c.corpus)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 8<<20)
	var docs []doc
	chunks := 0
	for sc.Scan() && len(docs) < c.docs {
		var raw struct {
			ID    string `json:"_id"`
			Title string `json:"title"`
			Text  string `json:"text"`
		}
		if err := json.Unmarshal(sc.Bytes(), &raw); err != nil {
			return nil, 0, err
		}
		d := doc{ID: c.runID + "-" + raw.ID, Title: raw.Title, Text: raw.Text}
		cs, err := chunker.Split(chunker.Document{DocID: d.ID, Title: d.Title, Text: d.Text}, c.chunkSize, c.overlap)
		if err != nil || len(cs) == 0 {
			continue // empty text: not a valid document
		}
		d.expected, d.first = len(cs), cs[0].Text
		chunks += len(cs)
		docs = append(docs, d)
	}
	if len(docs) < c.docs {
		return nil, 0, fmt.Errorf("corpus has only %d usable documents", len(docs))
	}
	return docs, chunks, sc.Err()
}

type seen struct {
	acceptedUs, indexedUs int64
	worker                string
	partition             int32
}

// collector reads indexed events for this run, keeping the first per document.
type collector struct {
	mu      sync.Mutex
	first   map[string]seen
	events  int
	runPref string
}

func (col *collector) consume(ctx context.Context, cl *kgo.Client) {
	for {
		fs := cl.PollFetches(ctx)
		if ctx.Err() != nil || fs.IsClientClosed() {
			return
		}
		fs.EachRecord(func(r *kgo.Record) {
			var ev event.IndexedEvent
			if json.Unmarshal(r.Value, &ev) != nil || !strings.HasPrefix(ev.DocID, col.runPref) {
				return
			}
			col.mu.Lock()
			col.events++
			if _, ok := col.first[ev.DocID]; !ok {
				col.first[ev.DocID] = seen{ev.AcceptedAt, ev.IndexedAt, ev.Worker, ev.Partition}
			}
			col.mu.Unlock()
		})
	}
}

func (col *collector) count() int {
	col.mu.Lock()
	defer col.mu.Unlock()
	return len(col.first)
}

// consumeFrom starts a group-less consumer at the topic's current end.
func consumeFrom(ctx context.Context, adm *kadm.Client, topic string, atStart bool) (*kgo.Client, error) {
	var offs kadm.ListedOffsets
	var err error
	if atStart {
		offs, err = adm.ListStartOffsets(ctx, topic)
	} else {
		offs, err = adm.ListEndOffsets(ctx, topic)
	}
	if err != nil {
		return nil, err
	}
	start := map[string]map[int32]kgo.Offset{topic: {}}
	offs.Each(func(o kadm.ListedOffset) { start[topic][o.Partition] = kgo.NewOffset().At(o.Offset) })
	return kgo.NewClient(kgo.SeedBrokers(kafkax.Brokers()...), kgo.ConsumePartitions(start))
}

type sendStats struct {
	uploads, retries atomic.Int64
	firstErr         atomic.Value
}

func send(ctx context.Context, c config, docs []doc, st *sendStats) error {
	batches := make(chan []doc)
	go func() {
		defer close(batches)
		var tick <-chan time.Time
		if c.rate > 0 {
			t := time.NewTicker(time.Duration(float64(c.batch) / c.rate * float64(time.Second)))
			defer t.Stop()
			tick = t.C
		}
		for i := 0; i < len(docs); i += c.batch {
			if tick != nil {
				select {
				case <-tick:
				case <-ctx.Done():
					return
				}
			}
			select {
			case batches <- docs[i:min(i+c.batch, len(docs))]:
			case <-ctx.Done():
				return
			}
		}
	}()
	client := &http.Client{Timeout: 60 * time.Second}
	var wg sync.WaitGroup
	errs := make(chan error, c.concurrency)
	for range c.concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for b := range batches {
				if err := upload(ctx, client, c, b, st); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	return <-errs
}

func upload(ctx context.Context, client *http.Client, c config, b []doc, st *sendStats) error {
	body, _ := json.Marshal(map[string]any{"documents": b})
	backoff := 250 * time.Millisecond
	for attempt := 0; ; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.ingestURL+"/v1/documents", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Tenant", c.tenant)
		resp, err := client.Do(req)
		if err == nil {
			raw, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusAccepted {
				st.uploads.Add(1)
				return nil
			}
			if resp.StatusCode == http.StatusBadRequest {
				return fmt.Errorf("upload rejected: %s", raw)
			}
			err = fmt.Errorf("HTTP %d: %s", resp.StatusCode, bytes.TrimSpace(raw))
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		st.retries.Add(1)
		st.firstErr.CompareAndSwap(nil, err.Error())
		if attempt%10 == 0 {
			log.Warn("upload failed; retrying", "attempt", attempt+1, "err", err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 5*time.Second)
	}
}

func pct(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q*float64(len(sorted)-1) + 0.5)
	return float64(int(sorted[i]*10+0.5)) / 10
}

func run(ctx context.Context, c config) (map[string]any, bool) {
	report := map[string]any{"run_id": c.runID, "config": map[string]any{
		"docs": c.docs, "batch": c.batch, "concurrency": c.concurrency, "rate": c.rate,
		"corpus": c.corpus, "chunk_size": c.chunkSize, "chunk_overlap": c.overlap,
	}}
	fail := func(err error) (map[string]any, bool) {
		report["error"] = err.Error()
		return report, false
	}
	docs, expectedChunks, err := loadCorpus(c)
	if err != nil {
		return fail(err)
	}
	report["expected_chunks"] = expectedChunks
	admClient, err := kgo.NewClient(kgo.SeedBrokers(kafkax.Brokers()...))
	if err != nil {
		return fail(err)
	}
	defer admClient.Close()
	adm := kadm.NewClient(admClient)
	evClient, err := consumeFrom(ctx, adm, event.TopicIndexed, false)
	if err != nil {
		return fail(fmt.Errorf("indexed topic: %w", err))
	}
	defer evClient.Close()
	col := &collector{first: map[string]seen{}, runPref: c.runID + "-"}
	go col.consume(ctx, evClient)

	log.Info("sending", "docs", len(docs), "chunks", expectedChunks, "rate", c.rate)
	var st sendStats
	sendStart := time.Now()
	sendErr := make(chan error, 1)
	go func() { sendErr <- send(ctx, c, docs, &st) }()

	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
	var sendDone time.Duration
	sending := true
	for col.count() < len(docs) {
		select {
		case err := <-sendErr:
			if err != nil {
				return fail(fmt.Errorf("send: %w", err))
			}
			sendDone, sending = time.Since(sendStart), false
			sendErr = nil
		case <-tick.C:
			log.Info("progress", "uploads", st.uploads.Load(), "upload_retries", st.retries.Load(),
				"indexed", col.count(), "of", len(docs), "sending", sending)
		case <-ctx.Done():
			return fail(fmt.Errorf("timed out with %d/%d documents indexed", col.count(), len(docs)))
		}
	}
	if sending {
		if err := <-sendErr; err != nil {
			return fail(fmt.Errorf("send: %w", err))
		}
		sendDone = time.Since(sendStart)
	}
	report["throughput"], report["latency_ms"], report["sharding"] = summarize(col, len(docs))
	report["upload"] = map[string]any{
		"uploads": st.uploads.Load(), "retried_uploads": st.retries.Load(),
		"first_error": st.firstErr.Load(), "send_s": round(sendDone.Seconds()),
	}

	lag, err := waitForZeroLag(ctx, adm, c.group)
	report["final_consumer_lag"] = lag
	if err != nil {
		return fail(err)
	}
	ok := true
	verification, vok := verify(ctx, c, adm, docs, expectedChunks)
	report["verification"] = verification
	ok = ok && vok
	col.mu.Lock()
	report["indexed_events"] = map[string]any{"total": col.events, "distinct_docs": len(col.first),
		"replayed_or_duplicate_events": col.events - len(col.first)}
	col.mu.Unlock()
	report["search_spot_check"] = spotCheck(ctx, c, docs)
	report["passed"] = ok
	return report, ok
}

func round(v float64) float64 { return float64(int(v*100+0.5)) / 100 }

func summarize(col *collector, n int) (map[string]any, map[string]any, map[string]any) {
	col.mu.Lock()
	defer col.mu.Unlock()
	lat := make([]float64, 0, len(col.first))
	idx := make([]int64, 0, len(col.first))
	var firstAccepted int64 = 1 << 62
	workers, partitions := map[string]int{}, map[int32]int{}
	for _, s := range col.first {
		lat = append(lat, float64(s.indexedUs-s.acceptedUs)/1000)
		idx = append(idx, s.indexedUs)
		firstAccepted = min(firstAccepted, s.acceptedUs)
		workers[s.worker]++
		partitions[s.partition]++
	}
	sort.Float64s(lat)
	slices.Sort(idx)
	span := float64(idx[len(idx)-1]-firstAccepted) / 1e6
	// Steady state: the middle 80% of completions, excluding ramp-up and drain.
	lo, hi := idx[len(idx)/10], idx[len(idx)*9/10]
	steady := 0.0
	if hi > lo {
		steady = float64(len(idx)*9/10-len(idx)/10) / (float64(hi-lo) / 1e6)
	}
	return map[string]any{
		"docs": n, "wall_s": round(span), "docs_per_s": round(float64(n) / span),
		"steady_docs_per_s": round(steady),
	}, map[string]any{
		"p50": pct(lat, 0.50), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99), "max": pct(lat, 1),
	}, map[string]any{"docs_per_worker": workers, "partitions_used": len(partitions)}
}

func waitForZeroLag(ctx context.Context, adm *kadm.Client, group string) (int64, error) {
	var last int64 = -1
	for {
		lags, err := adm.Lag(ctx, group)
		if err == nil {
			if g, ok := lags[group]; ok && g.Error() == nil {
				last = g.Lag.Total()
				if last == 0 {
					return 0, nil
				}
			}
		}
		select {
		case <-ctx.Done():
			return last, fmt.Errorf("consumer lag still %d", last)
		case <-time.After(2 * time.Second):
		}
	}
}

func httpJSON(ctx context.Context, method, url string, body any, out any) error {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return fmt.Errorf("%s %s: HTTP %d: %.300s", method, url, resp.StatusCode, raw)
	}
	if out != nil {
		return json.Unmarshal(raw, out)
	}
	return nil
}

func esCounts(ctx context.Context, c config) (map[string]int, int, error) {
	if err := httpJSON(ctx, http.MethodPost, c.esURL+"/"+c.esIndex+"/_refresh", nil, nil); err != nil {
		return nil, 0, err
	}
	counts := map[string]int{}
	var after any
	total := 0
	for {
		comp := map[string]any{"size": 5000, "sources": []any{map[string]any{"doc": map[string]any{"terms": map[string]any{"field": "doc_id"}}}}}
		if after != nil {
			comp["after"] = after
		}
		var resp struct {
			Hits struct {
				Total struct{ Value int } `json:"total"`
			} `json:"hits"`
			Aggregations struct {
				D struct {
					AfterKey any `json:"after_key"`
					Buckets  []struct {
						Key      struct{ Doc string } `json:"key"`
						DocCount int                  `json:"doc_count"`
					} `json:"buckets"`
				} `json:"d"`
			} `json:"aggregations"`
		}
		err := httpJSON(ctx, http.MethodPost, c.esURL+"/"+c.esIndex+"/_search", map[string]any{
			"size": 0, "track_total_hits": true,
			"query": map[string]any{"prefix": map[string]any{"doc_id": c.runID + "-"}},
			"aggs":  map[string]any{"d": map[string]any{"composite": comp}},
		}, &resp)
		if err != nil {
			return nil, 0, err
		}
		total = resp.Hits.Total.Value
		for _, b := range resp.Aggregations.D.Buckets {
			counts[b.Key.Doc] = b.DocCount
		}
		if len(resp.Aggregations.D.Buckets) == 0 || resp.Aggregations.D.AfterKey == nil {
			return counts, total, nil
		}
		after = resp.Aggregations.D.AfterKey
	}
}

func qdrantCounts(ctx context.Context, c config) (map[string]int, int, error) {
	counts := map[string]int{}
	var offset any
	for {
		body := map[string]any{"limit": 5000, "with_payload": []string{"doc_id"}, "with_vector": false}
		if offset != nil {
			body["offset"] = offset
		}
		var resp struct {
			Result struct {
				Points []struct {
					ID      any `json:"id"`
					Payload struct {
						DocID string `json:"doc_id"`
					} `json:"payload"`
				} `json:"points"`
				Next any `json:"next_page_offset"`
			} `json:"result"`
		}
		if err := httpJSON(ctx, http.MethodPost, c.qdrantURL+"/collections/"+c.collection+"/points/scroll", body, &resp); err != nil {
			return nil, 0, err
		}
		for _, p := range resp.Result.Points {
			if strings.HasPrefix(p.Payload.DocID, c.runID+"-") {
				counts[p.Payload.DocID]++
			}
		}
		if resp.Result.Next == nil {
			total := 0
			for _, n := range counts {
				total += n
			}
			return counts, total, nil
		}
		offset = resp.Result.Next
	}
}

func dlqCount(ctx context.Context, adm *kadm.Client, runID string) (int, error) {
	ends, err := adm.ListEndOffsets(ctx, event.TopicDLQ)
	if err != nil {
		return 0, err
	}
	remaining := map[int32]int64{}
	ends.Each(func(o kadm.ListedOffset) {
		if o.Offset > 0 {
			remaining[o.Partition] = o.Offset
		}
	})
	if len(remaining) == 0 {
		return 0, nil
	}
	cl, err := consumeFrom(ctx, adm, event.TopicDLQ, true)
	if err != nil {
		return 0, err
	}
	defer cl.Close()
	n := 0
	for len(remaining) > 0 {
		fs := cl.PollFetches(ctx)
		if ctx.Err() != nil {
			return n, ctx.Err()
		}
		fs.EachRecord(func(r *kgo.Record) {
			if strings.Contains(string(r.Key), "/"+runID+"-") {
				n++
			}
			if r.Offset+1 >= remaining[r.Partition] {
				delete(remaining, r.Partition)
			}
		})
	}
	return n, nil
}

func verify(ctx context.Context, c config, adm *kadm.Client, docs []doc, expected int) (map[string]any, bool) {
	out := map[string]any{"documents": len(docs), "expected_chunks": expected}
	ok := true
	check := func(name string, counts map[string]int, total int, err error) {
		if err != nil {
			out[name] = map[string]any{"error": err.Error()}
			ok = false
			return
		}
		missing, short, extra := 0, 0, 0
		for _, d := range docs {
			switch got := counts[d.ID]; {
			case got == 0:
				missing++
			case got < d.expected:
				short++
			case got > d.expected:
				extra++
			}
		}
		res := map[string]any{
			"chunks": total, "documents_present": len(counts),
			"documents_missing": missing, "documents_with_missing_chunks": short,
			"documents_with_extra_chunks": extra,
			"exact":                       missing == 0 && short == 0 && extra == 0 && total == expected,
		}
		out[name] = res
		ok = ok && res["exact"].(bool)
	}
	esC, esT, err := esCounts(ctx, c)
	check("elasticsearch", esC, esT, err)
	qC, qT, err := qdrantCounts(ctx, c)
	check("qdrant", qC, qT, err)
	n, err := dlqCount(ctx, adm, c.runID)
	out["dlq_records"] = n
	if err != nil {
		out["dlq_error"] = err.Error()
		ok = false
	}
	ok = ok && n == 0
	out["zero_loss_zero_duplicates"] = ok
	return out, ok
}

// spotCheck searches for a sample of documents by a phrase from their first
// chunk: BM25 (Elasticsearch), dense (Qdrant) and the gateway's hybrid /search.
func spotCheck(ctx context.Context, c config, docs []doc) map[string]any {
	rng := rand.New(rand.NewPCG(7, 7))
	n := min(c.sample, len(docs))
	emb, err := embed.New(c.tritonURL, "minilm_embedder", 64)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	hits := map[string]int{"elasticsearch_top10": 0, "qdrant_top10": 0, "gateway_search_top10": 0}
	errs := map[string]string{}
	for _, i := range rng.Perm(len(docs))[:n] {
		d := docs[i]
		words := strings.Fields(d.first)
		query := strings.Join(words[:min(12, len(words))], " ")
		var es struct {
			Hits struct {
				Hits []struct {
					Source struct {
						DocID string `json:"doc_id"`
					} `json:"_source"`
				} `json:"hits"`
			} `json:"hits"`
		}
		if err := httpJSON(ctx, http.MethodPost, c.esURL+"/"+c.esIndex+"/_search", map[string]any{
			"size": 10, "_source": []string{"doc_id"}, "query": map[string]any{"match": map[string]any{"text": query}},
		}, &es); err != nil {
			errs["elasticsearch"] = err.Error()
		} else if slices.ContainsFunc(es.Hits.Hits, func(h struct {
			Source struct {
				DocID string `json:"doc_id"`
			} `json:"_source"`
		}) bool {
			return h.Source.DocID == d.ID
		}) {
			hits["elasticsearch_top10"]++
		}
		if vecs, err := emb.Embed(ctx, []string{query}); err != nil {
			errs["qdrant"] = err.Error()
		} else {
			var q struct {
				Result []struct {
					Payload struct {
						DocID string `json:"doc_id"`
					} `json:"payload"`
				} `json:"result"`
			}
			if err := httpJSON(ctx, http.MethodPost, c.qdrantURL+"/collections/"+c.collection+"/points/search",
				map[string]any{"vector": vecs[0], "limit": 10, "with_payload": []string{"doc_id"}}, &q); err != nil {
				errs["qdrant"] = err.Error()
			} else {
				for _, r := range q.Result {
					if r.Payload.DocID == d.ID {
						hits["qdrant_top10"]++
						break
					}
				}
			}
		}
		if c.backendURL != "" {
			var g struct {
				Hits []struct {
					Chunk struct {
						DocID string `json:"doc_id"`
					} `json:"chunk"`
				} `json:"hits"`
			}
			if err := httpJSON(ctx, http.MethodPost, c.backendURL+"/search",
				map[string]any{"query": query, "top_k": 10}, &g); err != nil {
				errs["gateway"] = err.Error()
			} else {
				for _, h := range g.Hits {
					if h.Chunk.DocID == d.ID {
						hits["gateway_search_top10"]++
						break
					}
				}
			}
		}
	}
	out := map[string]any{"sampled_docs": n, "query": "first 12 words of the document's first chunk"}
	for k, v := range hits {
		out[k] = v
	}
	if len(errs) > 0 {
		out["errors"] = errs
	}
	return out
}

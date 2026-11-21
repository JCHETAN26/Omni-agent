// omni-ingest accepts document uploads and publishes them to Kafka.
//
//	POST /v1/documents   {"documents":[{"doc_id","title","text","metadata"}...]}
//	                     or Content-Type: application/x-ndjson, one document per line
//	Header X-Tenant (default "default") scopes doc_ids; the record key is
//	"<tenant>/<doc_id>", so a document always lands on the same partition.
//
// 202 means every record of the upload was acknowledged by Kafka with acks=all
// (durable); indexing happens asynchronously in omni-indexer. 503 means some
// records may not be durable: retry the whole upload (re-sending is safe,
// indexing is idempotent per doc_id). 400 means the upload was rejected as a
// whole; nothing was published.
//
// Configuration: KAFKA_BROKERS, HTTP_ADDR (:8090), MAX_UPLOAD_BYTES (64 MiB),
// MAX_DOCS_PER_UPLOAD (5000), PRODUCE_TIMEOUT_S (20), KAFKA_REPLICATION (1).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/kafkax"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/metrics"
)

type server struct {
	cl         *kgo.Client
	log        *slog.Logger
	maxBytes   int64
	maxDocs    int
	timeout    time.Duration
	now        func() time.Time
	produceAll func(ctx context.Context, recs ...*kgo.Record) kgo.ProduceResults
}

type upload struct {
	Documents []event.DocumentEvent `json:"documents"`
}

type problem struct {
	Index int    `json:"index"`
	DocID string `json:"doc_id,omitempty"`
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *server) decode(r *http.Request) ([]event.DocumentEvent, error) {
	body := http.MaxBytesReader(nil, r.Body, s.maxBytes)
	ct, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if ct == "application/x-ndjson" {
		var docs []event.DocumentEvent
		sc := bufio.NewScanner(body)
		sc.Buffer(make([]byte, 64*1024), event.MaxTextBytes*2)
		for line := 1; sc.Scan(); line++ {
			if len(sc.Bytes()) == 0 {
				continue
			}
			var d event.DocumentEvent
			if err := json.Unmarshal(sc.Bytes(), &d); err != nil {
				return nil, fmt.Errorf("line %d: %w", line, err)
			}
			docs = append(docs, d)
			if len(docs) > s.maxDocs {
				return nil, fmt.Errorf("more than %d documents in one upload", s.maxDocs)
			}
		}
		return docs, sc.Err()
	}
	var u upload
	dec := json.NewDecoder(body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&u); err != nil {
		return nil, err
	}
	if len(u.Documents) > s.maxDocs {
		return nil, fmt.Errorf("%d documents in one upload (max %d)", len(u.Documents), s.maxDocs)
	}
	return u.Documents, nil
}

func (s *server) handleUpload(w http.ResponseWriter, r *http.Request) {
	docs, err := s.decode(r)
	if err != nil {
		var tooBig *http.MaxBytesError
		status := http.StatusBadRequest
		if errors.As(err, &tooBig) {
			status = http.StatusRequestEntityTooLarge
		}
		writeJSON(w, status, map[string]any{"error": err.Error()})
		return
	}
	if len(docs) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "no documents"})
		return
	}
	tenant := r.Header.Get("X-Tenant")
	if tenant == "" {
		tenant = "default"
	}
	accepted := s.now().UnixMicro()
	var problems []problem
	recs := make([]*kgo.Record, len(docs))
	for i := range docs {
		d := &docs[i]
		d.Tenant, d.AcceptedAt = tenant, accepted
		if err := d.Validate(); err != nil {
			problems = append(problems, problem{i, d.DocID, err.Error()})
			continue
		}
		v, _ := json.Marshal(d)
		recs[i] = &kgo.Record{Topic: event.TopicDocuments, Key: d.Key(), Value: v}
	}
	if len(problems) > 0 {
		metrics.Accepted.WithLabelValues("rejected").Add(float64(len(docs)))
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid documents", "documents": problems})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)
	defer cancel()
	t0 := time.Now()
	results := s.produceAll(ctx, recs...)
	metrics.ProduceSeconds.Observe(time.Since(t0).Seconds())
	if err := results.FirstErr(); err != nil {
		metrics.Accepted.WithLabelValues("failed").Add(float64(len(docs)))
		s.log.Warn("produce failed", "docs", len(docs), "err", err)
		w.Header().Set("Retry-After", "1")
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": "not all documents were acknowledged by Kafka; retry the upload", "detail": err.Error(),
		})
		return
	}
	partitions := map[int32]int{}
	for _, res := range results {
		partitions[res.Record.Partition]++
	}
	metrics.Accepted.WithLabelValues("accepted").Add(float64(len(docs)))
	writeJSON(w, http.StatusAccepted, map[string]any{
		"accepted": len(docs), "tenant": tenant, "accepted_at_us": accepted, "partitions": partitions,
	})
}

func main() {
	log := kafkax.Logger("omni-ingest")
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	timeout := time.Duration(kafkax.EnvInt("PRODUCE_TIMEOUT_S", 20)) * time.Second
	cl, err := kgo.NewClient(append(kafkax.ProducerOpts(),
		// Bound how long an upload waits on a down broker; the client retries.
		kgo.RecordDeliveryTimeout(timeout),
		kgo.ClientID("omni-ingest"),
	)...)
	if err != nil {
		log.Error("kafka client", "err", err)
		os.Exit(1)
	}
	if err := kafkax.EnsureTopics(ctx, cl, kafkax.Topics(), int16(kafkax.EnvInt("KAFKA_REPLICATION", 1)), log); err != nil {
		os.Exit(0) // ctx ended during startup
	}
	s := &server{
		cl: cl, log: log,
		maxBytes: int64(kafkax.EnvInt("MAX_UPLOAD_BYTES", 64<<20)),
		maxDocs:  kafkax.EnvInt("MAX_DOCS_PER_UPLOAD", 5000),
		timeout:  timeout, now: time.Now, produceAll: cl.ProduceSync,
	}
	srv := kafkax.Serve(kafkax.Env("HTTP_ADDR", ":8090"), func(ctx context.Context) error { return cl.Ping(ctx) },
		map[string]http.Handler{"POST /v1/documents": http.HandlerFunc(s.handleUpload)})
	log.Info("ingest started", "addr", srv.Addr)
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdown) // finish in-flight uploads
	cl.Close()
	log.Info("ingest stopped")
}

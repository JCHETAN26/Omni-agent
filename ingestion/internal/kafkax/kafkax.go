// Package kafkax holds shared Kafka setup: client options, topic creation,
// environment config, and the small HTTP server every binary exposes.
package kafkax

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kerr"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
)

// Env reads a string setting.
func Env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// EnvInt reads an integer setting (exits on a malformed value).
func EnvInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		slog.Error("invalid integer setting", "key", key, "value", v)
		os.Exit(2)
	}
	return n
}

// Brokers parses KAFKA_BROKERS (comma separated).
func Brokers() []string { return strings.Split(Env("KAFKA_BROKERS", "localhost:9092"), ",") }

// ProducerOpts: idempotent producer (franz-go's default), acks from all ISRs,
// unlimited retries (the default), small linger for batching, lz4.
func ProducerOpts() []kgo.Opt {
	return []kgo.Opt{
		kgo.SeedBrokers(Brokers()...),
		kgo.RequiredAcks(kgo.AllISRAcks()),
		kgo.ProducerLinger(5 * time.Millisecond),
		kgo.ProducerBatchCompression(kgo.Lz4Compression(), kgo.NoCompression()),
	}
}

// Topic is a topic to create on startup.
type Topic struct {
	Name       string
	Partitions int32
}

// Topics returns the pipeline's topics with partition counts from the environment.
func Topics() []Topic {
	return []Topic{
		{event.TopicDocuments, int32(EnvInt("KAFKA_DOCUMENT_PARTITIONS", 12))},
		{event.TopicIndexed, int32(EnvInt("KAFKA_EVENT_PARTITIONS", 3))},
		{event.TopicDLQ, int32(EnvInt("KAFKA_EVENT_PARTITIONS", 3))},
	}
}

// EnsureTopics creates missing topics (idempotent; safe from every replica),
// retrying until the broker answers or ctx ends.
func EnsureTopics(ctx context.Context, cl *kgo.Client, topics []Topic, replication int16, log *slog.Logger) error {
	adm := kadm.NewClient(cl)
	for {
		err := ensure(ctx, adm, topics, replication)
		if err == nil {
			return nil
		}
		log.Warn("creating topics failed; retrying", "err", err)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
}

func ensure(ctx context.Context, adm *kadm.Client, topics []Topic, replication int16) error {
	var errs []error
	for _, t := range topics {
		resp, err := adm.CreateTopic(ctx, t.Partitions, replication,
			map[string]*string{"min.insync.replicas": kadm.StringPtr("1")}, t.Name)
		if err == nil {
			err = resp.Err
		}
		if err != nil && !errors.Is(err, kerr.TopicAlreadyExists) {
			errs = append(errs, fmt.Errorf("%s: %w", t.Name, err))
		}
	}
	return errors.Join(errs...)
}

// Serve exposes /healthz, /readyz and /metrics (plus extra handlers) on addr.
func Serve(addr string, ready func(context.Context) error, extra map[string]http.Handler) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := ready(ctx); err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		w.Write([]byte("ok\n"))
	})
	mux.Handle("GET /metrics", promhttp.Handler())
	for pattern, h := range extra {
		mux.Handle(pattern, h)
	}
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("http server failed", "err", err)
			os.Exit(1)
		}
	}()
	return srv
}

// Logger is a JSON logger at LOG_LEVEL (default info).
func Logger(service string) *slog.Logger {
	var level slog.Level
	_ = level.UnmarshalText([]byte(Env("LOG_LEVEL", "info")))
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})).With("service", service)
}

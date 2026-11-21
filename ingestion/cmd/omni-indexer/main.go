// omni-indexer is a sharded indexing worker: one member of the Kafka consumer
// group on omni.documents. Each worker owns a subset of partitions; it chunks
// documents, embeds them with Triton, writes Elasticsearch + Qdrant, then
// commits offsets. Run more replicas to shard further (up to the partition count).
//
// Configuration (environment):
//
//	KAFKA_BROKERS            broker list (default localhost:9092)
//	KAFKA_GROUP              consumer group (default omni-indexers)
//	ELASTICSEARCH_URL        ELASTICSEARCH_INDEX (omniagent_chunks), ELASTICSEARCH_API_KEY
//	QDRANT_URL               QDRANT_COLLECTION (omniagent_chunks), QDRANT_API_KEY, QDRANT_QUANTIZATION
//	TRITON_HTTP_URL          TRITON_EMBED_MODEL (minilm_embedder), EMBED_BATCH (64)
//	CHUNK_SIZE / CHUNK_OVERLAP   800 / 120 (must match the gateway)
//	MAX_POLL_RECORDS (256), BATCH_DOCS (32), HTTP_ADDR (:9090), WORKER_NAME (hostname)
package main

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/JCHETAN26/Omni-agent/ingestion/internal/embed"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/event"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/indexer"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/kafkax"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/metrics"
	"github.com/JCHETAN26/Omni-agent/ingestion/internal/sink"
)

func main() {
	log := kafkax.Logger("omni-indexer")
	host, _ := os.Hostname()
	name := kafkax.Env("WORKER_NAME", host)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	es := sink.NewElasticsearch(kafkax.Env("ELASTICSEARCH_URL", "http://localhost:9200"),
		os.Getenv("ELASTICSEARCH_API_KEY"), kafkax.Env("ELASTICSEARCH_INDEX", "omniagent_chunks"))
	qd := sink.NewQdrant(kafkax.Env("QDRANT_URL", "http://localhost:6333"),
		os.Getenv("QDRANT_API_KEY"), kafkax.Env("QDRANT_COLLECTION", "omniagent_chunks"))
	qd.Quantization = os.Getenv("QDRANT_QUANTIZATION")
	emb, err := embed.New(kafkax.Env("TRITON_HTTP_URL", "http://localhost:8000"),
		kafkax.Env("TRITON_EMBED_MODEL", "minilm_embedder"), kafkax.EnvInt("EMBED_BATCH", 64))
	if err != nil {
		log.Error("tokenizer", "err", err)
		os.Exit(1)
	}

	cl, err := kgo.NewClient(append(kafkax.ProducerOpts(),
		kgo.ConsumerGroup(kafkax.Env("KAFKA_GROUP", "omni-indexers")),
		kgo.ConsumeTopics(event.TopicDocuments),
		kgo.DisableAutoCommit(),
		kgo.BlockRebalanceOnPoll(),
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
		// Fail over a SIGKILLed worker's partitions within ~10 s.
		kgo.SessionTimeout(10*time.Second),
		kgo.HeartbeatInterval(2*time.Second),
		kgo.RebalanceTimeout(2*time.Minute),
		kgo.FetchMaxWait(500*time.Millisecond),
		kgo.ClientID(name),
	)...)
	if err != nil {
		log.Error("kafka client", "err", err)
		os.Exit(1)
	}

	ready := func(ctx context.Context) error { return cl.Ping(ctx) }
	srv := kafkax.Serve(kafkax.Env("HTTP_ADDR", ":9090"), ready, nil)

	// Setup retries until dependencies come up (they may start after us).
	for _, step := range []struct {
		name string
		fn   func(context.Context) error
	}{
		{"topics", func(ctx context.Context) error {
			return kafkax.EnsureTopics(ctx, cl, kafkax.Topics(), int16(kafkax.EnvInt("KAFKA_REPLICATION", 1)), log)
		}},
		{"elasticsearch index", es.EnsureIndex},
		{"qdrant collection", func(ctx context.Context) error { return qd.EnsureCollection(ctx, embed.Dim) }},
	} {
		for {
			err := step.fn(ctx)
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				os.Exit(0)
			}
			log.Warn("setup step failed; retrying", "step", step.name, "err", err)
			time.Sleep(2 * time.Second)
		}
	}

	w := &indexer.Worker{
		Client: cl,
		Pipeline: &indexer.Pipeline{
			Embedder: emb, Sparse: es, Dense: qd,
			ChunkSize: kafkax.EnvInt("CHUNK_SIZE", 800), ChunkOverlap: kafkax.EnvInt("CHUNK_OVERLAP", 120),
			Log: log, MinBackoff: 200 * time.Millisecond, MaxBackoff: 30 * time.Second,
			OnRetry: func(stage string, _ error) { metrics.Retries.WithLabelValues(stage).Inc() },
		},
		Name:       name,
		MaxPoll:    kafkax.EnvInt("MAX_POLL_RECORDS", 256),
		BatchDocs:  kafkax.EnvInt("BATCH_DOCS", 32),
		DLQTopic:   event.TopicDLQ,
		IndexTopic: event.TopicIndexed,
		Log:        log,
	}
	log.Info("indexer started", "worker", name)
	err = w.Run(ctx)
	// Graceful stop (SIGTERM): an interrupted batch was not committed and will
	// be replayed by the next owner. Leave the group so it rebalances at once.
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdown)
	cl.Close()
	if err != nil && !errors.Is(err, context.Canceled) {
		log.Error("indexer stopped", "err", err)
		os.Exit(1)
	}
	log.Info("indexer stopped")
}

// Package metrics holds the Prometheus metrics of the ingestion services.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	Accepted = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_ingest_documents_total",
		Help: "Documents received by omni-ingest, by result (accepted, rejected, failed).",
	}, []string{"result"})
	ProduceSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "omni_ingest_produce_seconds",
		Help:    "Time for an upload's records to be acknowledged by Kafka (acks=all).",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 16),
	})
	Documents = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_indexer_documents_total",
		Help: "Documents processed by omni-indexer, by outcome (indexed, dlq).",
	}, []string{"outcome"})
	EndToEndSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "omni_indexer_end_to_end_seconds",
		Help:    "Accepted by omni-ingest to searchable in Elasticsearch and Qdrant.",
		Buckets: prometheus.ExponentialBuckets(0.01, 2, 16),
	})
	BatchSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "omni_indexer_poll_seconds",
		Help:    "Time to index and commit one poll.",
		Buckets: prometheus.ExponentialBuckets(0.01, 2, 14),
	})
	StageSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "omni_indexer_stage_seconds",
		Help:    "Summed time per pipeline stage within one poll.",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 18),
	}, []string{"stage"})
	Retries = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_indexer_retries_total",
		Help: "Retried downstream outages, by stage.",
	}, []string{"stage"})
	CommitErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "omni_indexer_commit_errors_total",
		Help: "Failed offset commits (records are replayed idempotently).",
	})
)

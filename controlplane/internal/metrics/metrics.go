// Package metrics holds the control plane's Prometheus metrics.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	Requests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_requests_total",
		Help: "Router RPCs by method, gRPC code and degradation mode.",
	}, []string{"method", "code", "degradation"})
	Latency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "omni_cp_request_seconds",
		Help:    "Router RPC latency on this node, by method (Chat: until the stream ends).",
		Buckets: prometheus.ExponentialBuckets(0.002, 2, 16),
	}, []string{"method"})
	Inflight = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "omni_cp_inflight", Help: "Router RPCs in progress on this node.",
	})
	Attempts = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_upstream_attempts_total",
		Help: "Gateway attempts by result (ok, retryable_error, client_error).",
	}, []string{"result"})
	Failovers = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_failovers_total",
		Help: "Calls moved to another gateway after an upstream failure, by method.",
	}, []string{"method"})
	Transitions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_upstream_transitions_total",
		Help: "Gateway endpoint state transitions, by new state.",
	}, []string{"to"})
	Healthy = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "omni_cp_upstreams_healthy", Help: "Healthy gateway endpoints (this node's view).",
	})
	Degraded = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_degraded_responses_total",
		Help: "Answers served in a degraded mode, by mode (bm25_only, stale_cache, passages_only).",
	}, []string{"mode"})
	ChatRestarts = promauto.NewCounter(prometheus.CounterOpts{
		Name: "omni_cp_chat_restarts_total",
		Help: "Chat streams restarted on another gateway after a mid-stream upstream failure.",
	})
	PeersAlive = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "omni_cp_peers_alive", Help: "Peer control-plane nodes answering heartbeats.",
	})
	PeerEvents = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "omni_cp_peer_events_total", Help: "Peer membership events (dead, recovered).",
	}, []string{"event"})
	PeerDetectSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "omni_cp_peer_failure_detection_seconds",
		Help:    "Time from a peer's last answered heartbeat to declaring it dead.",
		Buckets: prometheus.LinearBuckets(0.25, 0.25, 16),
	})
	PeerEjections = promauto.NewCounter(prometheus.CounterOpts{
		Name: "omni_cp_peer_propagated_ejections_total",
		Help: "Gateways taken out of rotation because a peer reported them failed.",
	})
)

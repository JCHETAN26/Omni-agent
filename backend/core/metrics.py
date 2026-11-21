"""Prometheus metrics, exposed at GET /metrics.

Names follow Prometheus conventions (`_total` counters, `_seconds` units). The
Grafana dashboard in deploy/kubernetes/helm/omniagent/dashboards/ is built on
these series.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

WS_SESSIONS = Gauge("omniagent_ws_sessions_active", "Open WebSocket sessions", ["endpoint"])
WS_FRAMES = Counter("omniagent_ws_frames_total", "WebSocket frames", ["direction"])
WS_RATE_LIMITED = Counter("omniagent_ws_rate_limited_total", "Frames rejected by the token bucket")
WS_BUFFER_DROPPED = Counter(
    "omniagent_ws_buffer_dropped_total", "Items evicted from full session ring buffers"
)
TOKENS_STREAMED = Counter("omniagent_tokens_streamed_total", "Answer tokens streamed to clients")
AGENT_TURNS = Counter("omniagent_agent_turns_total", "Completed agent turns", ["tool", "outcome"])
AGENT_RETRIES = Histogram(
    "omniagent_agent_retries", "Self-correction rewrites per agent turn", buckets=(0, 1, 2, 3, 5)
)
AGENT_DISCARDED = Counter(
    "omniagent_agent_generations_discarded_total", "Answers discarded by the grounding check"
)
AGENT_FALLBACKS = Counter(
    "omniagent_agent_reasoner_fallbacks_total", "LLM reasoning steps that fell back", ["node"]
)
TURN_SECONDS = Histogram(
    "omniagent_turn_seconds", "End-to-end answer latency", ["endpoint"], buckets=LATENCY_BUCKETS
)
TTFT_SECONDS = Histogram(
    "omniagent_time_to_first_token_seconds",
    "Time to first answer token",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
RERANK_SKIPPED = Counter(
    "omniagent_rerank_skipped_total", "Rerank fallbacks to RRF order", ["reason"]
)
LLM_CACHE = Counter(
    "omniagent_llm_cache_total",
    "LLM calls by cache outcome (hit, miss, coalesced)",
    ["kind", "outcome"],
)
LLM_TOKENS = Counter("omniagent_llm_tokens_total", "LLM tokens reported by the provider", ["type"])
LTR_SKIPPED = Counter("omniagent_ltr_skipped_total", "Learning-to-rank stages skipped", ["reason"])
TRITON_SECONDS = Histogram(
    "omniagent_triton_request_seconds",
    "Triton gRPC inference latency",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
TRITON_ERRORS = Counter("omniagent_triton_errors_total", "Failed Triton inference calls", ["model"])
LOOP_LAG = Gauge(
    "omniagent_event_loop_lag_max_seconds", "Max event-loop lag over the recent window"
)

FRAMES = Counter(
    "omniagent_frames_total",
    "Live frames handled on /ws/frames",
    ["outcome"],  # classified | unavailable | invalid
)
FRAME_SECONDS = Histogram(
    "omniagent_frame_seconds",
    "Server time per frame on /ws/frames, by stage",
    ["stage"],  # preprocess | infer | total
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)


# --- collaborative incident rooms ----------------------------------------------
ROOM_SESSIONS = Gauge("omniagent_room_sessions", "Room WebSocket sessions on this replica")
ROOM_EVENTS = Counter(
    "omniagent_room_events_total", "Room events published by this replica", ["type"]
)
ROOM_SLOW_CONSUMERS = Counter(
    "omniagent_room_slow_consumers_total",
    "Room sessions closed for a resync because their send queue overflowed",
)

"""Structured logging and OpenTelemetry tracing.

Spans are always created (cheap), so trace context propagates across WebSocket
frames and gRPC calls even when no exporter is configured. Export goes to OTLP
when OTEL_EXPORTER_OTLP_ENDPOINT is set, and/or to the console.

Propagation uses W3C Trace Context (`traceparent`/`tracestate`):
* WebSocket: a client may put `traceparent` in any JSON frame. Server events
  for that request carry the resulting `trace_id`.
* gRPC: injected as request metadata on every Triton call. Triton reads it
  when its OpenTelemetry trace mode is enabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from backend.core.config import Settings

_configured = False


def setup_tracing(settings: Settings) -> TracerProvider:
    """Install the global tracer provider once per process (idempotent)."""
    global _configured
    current = trace.get_tracer_provider()
    if _configured and isinstance(current, TracerProvider):
        return current
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    if settings.otel_console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    return provider


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("omniagent")


def inject_carrier() -> dict[str, str]:
    """W3C trace headers for the current context."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_context(carrier: dict[str, Any] | None) -> otel_context.Context | None:
    if not carrier or "traceparent" not in carrier:
        return None
    return propagate.extract({k: str(v) for k, v in carrier.items()})


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


class LoopLagMonitor:
    """Samples event-loop scheduling delay: how late a short timer fires.

    Anything that blocks the loop (sync I/O, CPU-heavy code) shows up directly
    as lag, so this is the check that inference never runs on the loop.
    """

    def __init__(self, interval_s: float = 0.01, window: int = 3000):
        self.interval_s = interval_s
        self.samples: deque[float] = deque(maxlen=window)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await asyncio.sleep(self.interval_s)
            self.samples.append(max(0.0, (loop.time() - t0 - self.interval_s) * 1000))

    def snapshot(self) -> dict[str, float]:
        if not self.samples:
            return {"samples": 0}
        ordered = sorted(self.samples)
        return {
            "samples": len(ordered),
            "p50_ms": round(ordered[len(ordered) // 2], 2),
            "p99_ms": round(ordered[int(len(ordered) * 0.99) - 1], 2),
            "max_ms": round(ordered[-1], 2),
        }

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with trace/span ids from the active span."""

    _RESERVED = set(vars(logging.makeLogRecord({})))

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            out["trace_id"] = format(span_ctx.trace_id, "032x")
            out["span_id"] = format(span_ctx.span_id, "016x")
        for key, value in vars(record).items():
            if key not in self._RESERVED and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())

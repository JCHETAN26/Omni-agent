"""Pace Uvicorn's synchronous HTTP-to-WebSocket handoffs per worker."""

from __future__ import annotations

from asyncio import AbstractEventLoop
from collections import deque
from time import perf_counter

from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

_UPGRADE_SLICE_S = 0.001
_UPGRADE_YIELD_S = 0.001


class _PendingUpgrades:
    def __init__(self, loop: AbstractEventLoop) -> None:
        self.loop = loop
        self.pending: deque[PacedHttpToolsProtocol] = deque()
        self.scheduled = False

    def submit(self, protocol: PacedHttpToolsProtocol) -> None:
        self.pending.append(protocol)
        if not self.scheduled:
            self.scheduled = True
            self.loop.call_soon(self._drain)

    def _drain(self) -> None:
        deadline = perf_counter() + _UPGRADE_SLICE_S
        try:
            while self.pending:
                self.pending.popleft()._complete_websocket_upgrade()
                if perf_counter() >= deadline:
                    break
        finally:
            if self.pending:
                self.loop.call_later(_UPGRADE_YIELD_S, self._drain)
            else:
                self.scheduled = False


class PacedHttpToolsProtocol(HttpToolsProtocol):
    """Yield between batches of handshakes while leaving validation to Uvicorn."""

    def handle_websocket_upgrade(self) -> None:
        upgrades = getattr(self.server_state, "_omniagent_pending_upgrades", None)
        if upgrades is None:
            upgrades = _PendingUpgrades(self.loop)
            self.server_state._omniagent_pending_upgrades = upgrades
        # The HTTP parser has stopped at the upgrade boundary. Hold incoming
        # bytes until the original Uvicorn callback installs the WebSocket parser.
        self.transport.pause_reading()
        upgrades.submit(self)

    def _complete_websocket_upgrade(self) -> None:
        if self.transport.is_closing():
            return
        try:
            super().handle_websocket_upgrade()
        except Exception:
            self.transport.close()
            raise
        if not self.transport.is_closing():
            self.transport.resume_reading()

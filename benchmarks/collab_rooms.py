"""Collaborative-room synchronization load test (/ws/rooms).

    python -m benchmarks.collab_rooms --url http://127.0.0.1:8000 \
        --sessions 1000 --per-room 20 --rate 0.5 --duration 60 --procs 4 --output out.json

N WebSocket sessions join N/per_room rooms (created over REST). After all are
connected, every session sends chat messages at --rate msgs/s (jittered),
each carrying its send time (`client_ts`, wall clock ms) and a unique
client_msg_id. Every session records, for every message from a peer:

  delivery latency   receive time - client_ts   (one sample per receiving peer)
  room sync latency  the slowest peer's delivery, per message (all peers have it)

plus throughput, connections dropped or refused, and sequence integrity: every
sequenced event a session receives must have seq = previous + 1 (no gap, no
duplicate, no reordering). Sessions are split across --procs client processes
so the client is not the bottleneck; all share the machine clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import platform
import random
import time
import urllib.request
import uuid

from websockets.asyncio.client import connect


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 2)


def create_rooms(url: str, n: int) -> list[str]:
    rooms = []
    for i in range(n):
        req = urllib.request.Request(
            f"{url}/rooms",
            json.dumps({"title": f"bench {i}"}).encode(),
            {"Content-Type": "application/json"},
        )
        rooms.append(json.load(urllib.request.urlopen(req, timeout=10))["room_id"])
    return rooms


class Session:
    def __init__(self, idx: int, room: str):
        self.idx, self.room = idx, room
        self.name = f"u{idx}"
        self.sid: str | None = None
        self.last_seq: int | None = None
        self.gaps = self.dupes = self.reorders = 0
        self.received = 0
        self.closed: int | None = None
        self.close_reason = ""
        self.error = ""


async def run_sessions(
    url: str,
    assignments: list[tuple[int, str]],
    rate: float,
    duration: float,
    start_at: float,
    connect_conc: int,
) -> dict:
    ws_url = url.replace("http", "ws", 1)
    deliveries: list[float] = []
    timeline: list[tuple[int, float, str]] = []  # (receive second, latency, room)
    per_msg: dict[
        str, list[float]
    ] = {}  # client_msg_id -> delivery latencies (this process's receivers)
    sent: dict[
        str, tuple[str, float]
    ] = {}  # client_msg_id -> (room, client_ts) for this process's senders
    sessions = [Session(i, r) for i, r in assignments]
    sem = asyncio.Semaphore(connect_conc)
    connected = 0
    refused = 0
    stop_at = start_at + duration

    async def one(s: Session) -> None:
        nonlocal connected, refused
        try:
            async with sem:
                ws = await connect(
                    f"{ws_url}/ws/rooms/{s.room}?name={s.name}",
                    max_size=2**20,
                    ping_interval=None,
                    open_timeout=30,
                )
        except Exception:  # noqa: BLE001
            refused += 1
            return
        connected += 1
        try:

            async def reader() -> None:
                async for raw in ws:
                    now = time.time() * 1000
                    ev = json.loads(raw)
                    t = ev.get("type")
                    if t == "room.snapshot":
                        s.sid = ev["session"]
                        continue
                    seq = ev.get("seq")
                    if seq is not None:
                        if s.last_seq is not None:
                            if seq == s.last_seq:
                                s.dupes += 1
                            elif seq < s.last_seq:
                                s.reorders += 1
                            elif seq > s.last_seq + 1:
                                s.gaps += 1
                        s.last_seq = max(seq, s.last_seq or 0)
                    if t == "message" and ev.get("session") != s.sid and "client_ts" in ev:
                        s.received += 1
                        if ev["client_ts"] >= start_at * 1000:
                            lat = now - ev["client_ts"]
                            deliveries.append(lat)
                            timeline.append((int(now // 1000), lat, s.room))
                            per_msg.setdefault(ev["client_msg_id"], []).append(lat)

            rtask = asyncio.create_task(reader())
            # Wait for the common start, then send at `rate` with jitter.
            await asyncio.sleep(max(0.0, start_at - time.time()))
            await asyncio.sleep(random.random() / max(rate, 1e-9))
            last_ping = time.time()
            while time.time() < stop_at:
                cid = uuid.uuid4().hex[:16]
                ts = time.time() * 1000
                sent[cid] = (s.room, ts)
                await ws.send(
                    json.dumps(
                        {
                            "type": "message",
                            "text": f"status from {s.name}",
                            "client_msg_id": cid,
                            "client_ts": ts,
                        }
                    )
                )
                if time.time() - last_ping > 10:
                    await ws.send(json.dumps({"type": "ping"}))
                    last_ping = time.time()
                await asyncio.sleep(random.expovariate(rate))
            await asyncio.sleep(3)  # drain deliveries still in flight
            if rtask.done() and rtask.exception():
                s.error = repr(rtask.exception())
            rtask.cancel()
        except Exception as exc:  # noqa: BLE001
            s.error = repr(exc)[:200]
        finally:
            s.closed, s.close_reason = ws.close_code, ws.close_reason or ""
            await ws.close()

    await asyncio.gather(*(one(s) for s in sessions))
    return {
        "connected": connected,
        "refused": refused,
        "deliveries": deliveries,
        "per_msg": per_msg,
        "sent": {k: v[1] for k, v in sent.items()},
        "sent_room": {k: v[0] for k, v in sent.items()},
        "gaps": sum(s.gaps for s in sessions),
        "dupes": sum(s.dupes for s in sessions),
        "reorders": sum(s.reorders for s in sessions),
        "abnormal_closes": sum(1 for s in sessions if s.closed not in (None, 1000, 1005)),
        "close_details": [
            {
                "session": s.name,
                "room": s.room,
                "code": s.closed,
                "reason": s.close_reason,
                "error": s.error,
            }
            for s in sessions
            if s.closed not in (None, 1000, 1005) or s.error
        ],
        "timeline": timeline,
    }


def _proc(args) -> dict:
    import gc

    # The load generator's own GC pauses would show up as "sync latency":
    # keep the collector out of the measurement (each process is short-lived).
    gc.disable()
    url, assignments, rate, duration, start_at, conc = args
    return asyncio.run(run_sessions(url, assignments, rate, duration, start_at, conc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--sessions", type=int, default=1000)
    p.add_argument("--per-room", type=int, default=20)
    p.add_argument("--rate", type=float, default=0.5, help="messages per second per session")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--connect-concurrency", type=int, default=50, help="per process")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    a = p.parse_args()

    n_rooms = max(1, a.sessions // a.per_room)
    rooms = create_rooms(a.url, n_rooms)
    # Whole rooms per process keeps the client work balanced.
    assignments: list[list[tuple[int, str]]] = [[] for _ in range(a.procs)]
    for i in range(a.sessions):
        room_idx = i % n_rooms
        assignments[room_idx % a.procs].append((i, rooms[room_idx]))
    room_size = {r: sum(1 for i in range(a.sessions) if rooms[i % n_rooms] == r) for r in rooms}
    # Everyone connects first; sending starts together once all are in.
    connect_budget = 10 + a.sessions / 100
    start_at = time.time() + connect_budget
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        parts = pool.map(
            _proc,
            [
                (a.url, asg, a.rate, a.duration, start_at, a.connect_concurrency)
                for asg in assignments
            ],
        )
    wall = time.time() - t0

    deliveries = [x for pr in parts for x in pr["deliveries"]]
    # Where do slow deliveries come from: a moment (stall), a client process, a room?
    by_sec: dict[int, list[float]] = {}
    for pr in parts:
        for sec, lat, _room in pr["timeline"]:
            by_sec.setdefault(sec, []).append(lat)
    per_second = [
        {"t": sec, "n": len(v), "p99": pct(v, 0.99), "max": pct(v, 1)}
        for sec, v in sorted(by_sec.items())
    ]
    stall_seconds = [x for x in per_second if (x["p99"] or 0) > 50]
    per_proc = [
        {"proc": i, "deliveries": len(pr["deliveries"]), "p99": pct(pr["deliveries"], 0.99)}
        for i, pr in enumerate(parts)
    ]
    per_msg: dict[str, list[float]] = {}
    for pr in parts:
        for k, v in pr["per_msg"].items():
            per_msg.setdefault(k, []).extend(v)
    sent = {k: v for pr in parts for k, v in pr["sent"].items()}
    sent_room = {k: v for pr in parts for k, v in pr["sent_room"].items()}
    measured = {k: ts for k, ts in sent.items() if ts >= start_at * 1000}
    complete, incomplete, sync = 0, 0, []
    for cid in measured:
        peers = room_size[sent_room[cid]] - 1
        got = per_msg.get(cid, [])
        if len(got) >= peers:
            complete += 1
            sync.append(max(got))
        else:
            incomplete += 1
    connected = sum(pr["connected"] for pr in parts)
    report = {
        "label": a.label,
        "host": platform.platform(),
        "url": a.url,
        "sessions": a.sessions,
        "connected": connected,
        "refused": sum(pr["refused"] for pr in parts),
        "abnormal_closes": sum(pr["abnormal_closes"] for pr in parts),
        "close_details": [d for pr in parts for d in pr["close_details"]][:50],
        "rooms": n_rooms,
        "participants_per_room": a.per_room,
        "rate_per_session": a.rate,
        "duration_s": a.duration,
        "client_procs": a.procs,
        "messages_sent": len(measured),
        "messages_per_s": round(len(measured) / a.duration, 1),
        "deliveries": len(deliveries),
        "deliveries_per_s": round(len(deliveries) / a.duration, 1),
        "delivery_latency_ms": {
            q: pct(deliveries, v)
            for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
        },
        "room_sync_latency_ms": {
            q: pct(sync, v) for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
        },
        "messages_reaching_all_peers": complete,
        "messages_missing_a_peer": incomplete,
        "sequence": {
            "gaps": sum(pr["gaps"] for pr in parts),
            "duplicates": sum(pr["dupes"] for pr in parts),
            "reorders": sum(pr["reorders"] for pr in parts),
        },
        "wall_s": round(wall, 1),
        "seconds_with_p99_over_50ms": stall_seconds,
        "per_client_process": per_proc,
        "per_second": per_second,
    }
    print(json.dumps(report, indent=1))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()

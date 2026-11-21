"""Summarize deploy/kind/grpc-control-plane-bench.sh output (prints JSON).

python3 deploy/kind/grpc_cp_analyze.py <out_dir>
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def calls(report: dict) -> list[tuple[int, float, str, str, str]]:
    """(start_unix_ms, latency_ms, code, degradation, node) for every call."""
    t0 = report["start_unix_ms"]
    return [(t0 + r[0], r[1], r[2], r[3], r[4]) for r in report["calls_detail"]["rows"]]


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[int(q * (len(v) - 1) + 0.5)], 2)


def overhead(out: Path) -> dict:
    res: dict = {"runs": {}}
    by_mode: dict[str, list[dict]] = {"direct": [], "grpc": []}
    for p in sorted(out.glob("overhead-*.json")):
        r = load(p)
        by_mode[r["mode"]].append(r)
        res["runs"][p.stem] = {
            "calls_per_s": r["calls_per_s"],
            "failed": r["failed"],
            "latency_ms": r["latency_ms"],
        }
    for mode, runs in by_mode.items():
        lat = [c[1] for r in runs for c in calls(r) if c[2] == "OK"]
        res[mode] = {
            "calls": len(lat),
            "calls_per_s_mean": round(statistics.mean(r["calls_per_s"] for r in runs), 1),
            "p50": pct(lat, 0.5),
            "p95": pct(lat, 0.95),
            "p99": pct(lat, 0.99),
        }
    if by_mode["direct"] and by_mode["grpc"]:
        res["added_ms"] = {
            q: round(res["grpc"][q] - res["direct"][q], 2) for q in ("p50", "p95", "p99")
        }
        res["throughput_ratio_grpc_over_direct"] = round(
            res["grpc"]["calls_per_s_mean"] / res["direct"]["calls_per_s_mean"], 3
        )
    return res


def failover(out: Path) -> dict:
    r = load(out / "failover.json")
    cs = calls(r)
    kills = [
        int(line.split()[0]) for line in (out / "kills.txt").read_text().split("\n") if line.strip()
    ]
    downs = []
    for line in (out / "controlplane-membership.log").read_text().splitlines():
        if '"peer down"' in line:
            m = re.search(r'"detected_unix_ms":(\d+)', line)
            n = re.search(r'"peer":"([^"]+)"', line)
            downs.append((int(m.group(1)), n.group(1) if n else "?"))
    ok_lat = [c[1] for c in cs if c[2] == "OK"]
    base = [c[1] for c in cs if c[2] == "OK" and c[0] < kills[0] - 1000]
    per_kill = []
    for i, k in enumerate(kills):
        end = kills[i + 1] if i + 1 < len(kills) else 10**15
        inflight = [c for c in cs if c[0] <= k < c[0] + c[1]]
        window = [c for c in cs if k <= c[0] < end]
        after5 = [c for c in cs if k + 5000 <= c[0] < end]
        dets = sorted(d - k for d, _ in downs if 0 <= d - k < 10_000)
        per_kill.append(
            {
                "kill_unix_ms": k,
                "calls_in_flight_at_kill": len(inflight),
                "in_flight_failed": sum(c[2] != "OK" for c in inflight),
                "in_flight_max_latency_ms": round(max((c[1] for c in inflight), default=0), 1),
                "calls_until_next_kill": len(window),
                "errors_until_next_kill": sum(c[2] != "OK" for c in window),
                "errors_from_5s_after_kill": sum(c[2] != "OK" for c in after5),
                "p99_first_1s_after_kill_ms": pct(
                    [c[1] for c in window if c[0] < k + 1000 and c[2] == "OK"], 0.99
                ),
                "peer_detection_ms": dets,  # one entry per surviving peer that logged "peer down"
            }
        )
    all_det = [d for pk in per_kill for d in pk["peer_detection_ms"]]
    return {
        "calls": len(cs),
        "failed": r["failed"],
        "by_code": r["by_code"],
        "error_rate": r["error_rate"],
        "calls_per_s": r["calls_per_s"],
        "latency_ms": r["latency_ms"],
        "baseline_before_first_kill_ms": {"p50": pct(base, 0.5), "p99": pct(base, 0.99)},
        "served_by": r["served_by"],
        "kills": per_kill,
        "peer_detection_ms": {
            "samples": all_det,
            "median": statistics.median(all_det) if all_det else None,
            "max": max(all_det, default=None),
        },
        "restarts": (out / "restarts.txt").read_text().split(),
        "ok_latency_ms": {"p50": pct(ok_lat, 0.5), "p99": pct(ok_lat, 0.99)},
    }


def degradation(out: Path) -> dict:
    r = load(out / "degradation.json")
    cs = calls(r)
    ev = dict(
        line.split()
        for line in (out / "degradation-events.txt").read_text().splitlines()
        if line.strip()
    )
    down, up = int(ev["triton_down"]), int(ev["triton_up"])
    ready = int(ev.get("triton_ready", 0))
    deg = [c for c in cs if c[3]]
    first_deg = min((c[0] for c in deg if c[0] >= down), default=None)
    last_deg = max((c[0] + c[1] for c in deg), default=None)
    phase = lambda a, b: [c for c in cs if a <= c[0] < b]  # noqa: E731

    def summ(cc):
        return {
            "calls": len(cc),
            "errors": sum(c[2] != "OK" for c in cc),
            "degraded": sum(bool(c[3]) for c in cc),
            "p50_ms": pct([c[1] for c in cc if c[2] == "OK"], 0.5),
            "p99_ms": pct([c[1] for c in cc if c[2] == "OK"], 0.99),
        }

    return {
        "calls": len(cs),
        "failed": r["failed"],
        "by_code": r["by_code"],
        "degraded_total": r["degraded"],
        "before_triton_down": summ(phase(0, down)),
        "triton_down_to_scaled_up": summ(phase(down, up)),
        "after_scale_up": summ(phase(up, 10**15)),
        "first_degraded_after_down_ms": None if first_deg is None else first_deg - down,
        "last_degraded_end_after_scale_up_ms": None if last_deg is None else last_deg - up,
        "triton_ready_after_scale_up_ms": ready - up if ready else None,
        "modes": sorted({c[3] for c in deg}),
    }


def main() -> None:
    out = Path(sys.argv[1])
    summary = {}
    for name, fn in (("overhead", overhead), ("failover", failover), ("degradation", degradation)):
        try:
            summary[name] = fn(out)
        except FileNotFoundError as exc:
            summary[name] = {"missing": str(exc)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Collaborative-room sync benchmark on this machine: a private redis-server and
# a multi-worker FastAPI (uvicorn) stack, then benchmarks/collab_rooms.py at
# several session counts. Nothing leaves the machine (heuristic reasoner,
# hashing embedder, mock inference).
#   scripts/collab_rooms_bench.sh <out_dir> [workers=4] [rate=0.5] [duration=60] [levels="100 500 1000 1200"] [repeats=3]
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1
workers=${2:-4}
rate=${3:-0.5}
duration=${4:-60}
levels=${5:-"100 500 1000 1200"}
repeats=${6:-3}
redis_port=6391
api_port=8191
mkdir -p "$out"
ulimit -n 16384 2>/dev/null || ulimit -n "$(ulimit -Hn)"
note() { echo "$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))') $*" | tee -a "$out/bench.log"; }
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

redis-server --port "$redis_port" --save "" --appendonly no --loglevel warning > "$out/redis.log" 2>&1 &
pids+=($!)
until redis-cli -p "$redis_port" ping >/dev/null 2>&1; do sleep 0.2; done

OPENAI_API_KEY="" LLM_PROVIDER=heuristic EMBEDDING_PROVIDER=hashing INFERENCE_BACKEND=mock \
  REDIS_URL="redis://127.0.0.1:$redis_port/0" LOG_LEVEL=WARNING WS_MAX_SESSIONS=5000 \
  .venv/bin/uvicorn backend.main:server_app --factory --host 127.0.0.1 --port "$api_port" \
  --workers "$workers" --log-level warning --no-access-log > "$out/uvicorn.log" 2>&1 &
pids+=($!)
until curl -fsS "http://127.0.0.1:$api_port/livez" >/dev/null 2>&1; do sleep 0.3; done
note "stack: redis $(redis-server --version | awk '{print $3}'), uvicorn x$workers, fd limit $(ulimit -n), $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(sysctl -n hw.ncpu) cpus"

for rep in $(seq 1 "$repeats"); do
  for n in $levels; do
    f="$out/sessions-$n-run$rep.json"
    note "=== run $rep: $n sessions, 20 per room, $rate msgs/s each, ${duration}s"
    .venv/bin/python -m benchmarks.collab_rooms --url "http://127.0.0.1:$api_port" --sessions "$n" \
      --per-room 20 --rate "$rate" --duration "$duration" --procs 4 --label "sessions-$n-run$rep" \
      --output "$f" > /dev/null
    note "$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['connected'],'connected,', r['messages_per_s'],'msg/s in,', r['deliveries_per_s'],'deliveries/s, sync p50/p95/p99', [r['room_sync_latency_ms'][k] for k in ('p50','p95','p99')], 'delivery', [r['delivery_latency_ms'][k] for k in ('p50','p95','p99')], 'seq', r['sequence'], 'missing', r['messages_missing_a_peer'], 'abnormal', r['abnormal_closes'], r['close_details'][:3])" "$f")"
    redis-cli -p "$redis_port" flushall >/dev/null
  done
done
note "done"

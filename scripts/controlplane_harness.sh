#!/usr/bin/env bash
# Local multi-node harness for the gRPC control plane (no containers):
# 2 FastAPI gateways (in-memory stores, hashing embedder, mock inference,
# heuristic reasoner: nothing leaves the machine) + 3 omni-controlplane nodes.
#   scripts/controlplane_harness.sh <out_dir> <bin_dir with omni-controlplane, cp-loadgen>
#
# 1. Routing overhead: closed loop, 8 workers, 30 s, unique queries; direct to
#    the gateways (round-robin) vs through the control plane; 2 rounds. 1b: the
#    same with 1 worker (no queueing, so balancing cannot matter: hop cost only).
# 2. Node failover: closed loop through the control plane for 90 s; one node is
#    SIGKILLed at +20/+45/+70 s and restarted 3 s later (as a kubelet would).
# 3. Gateway failover: 60 s; one gateway SIGKILLed at +20 s, restarted at +40 s.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1
bin=$2
mkdir -p "$out/logs"
log="$out/bench.log"
: > "$log"
note() { echo "$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"))') $*" | tee -a "$log"; }
now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

gw_ports=(8101 8102)
cp_ports=(9101 9102 9103)
gateways=$(printf '127.0.0.1:%s,' "${gw_ports[@]}"); gateways=${gateways%,}
peers=$(printf '127.0.0.1:%s,' "${cp_ports[@]}"); peers=${peers%,}

gw_pid=()  # indexed like gw_ports (bash 3.2: no associative arrays)
cp_pid=()  # indexed like cp_ports
start_gateway() {
  local idx=$1 port=${gw_ports[$1]}
  OPENAI_API_KEY="" LLM_PROVIDER=heuristic EMBEDDING_PROVIDER=hashing INFERENCE_BACKEND=mock \
    VISION_BACKEND=mock RATE_LIMIT_CAPACITY=1000000 RATE_LIMIT_REFILL_PER_S=1000000 LOG_LEVEL=WARNING \
    .venv/bin/uvicorn backend.main:server_app --factory --host 127.0.0.1 --port "$port" --log-level warning \
    >> "$out/logs/gateway-$port.log" 2>&1 &
  gw_pid[$idx]=$!
  pids+=($!)
  until curl -fsS "http://127.0.0.1:$port/readyz" >/dev/null 2>&1; do sleep 0.2; done
}
start_node() {
  local i=$1 port=${cp_ports[$1]}
  GRPC_HOST=127.0.0.1 GRPC_PORT=$port HTTP_ADDR="127.0.0.1:$((port + 90))" GATEWAY_ADDRS=$gateways PEER_ADDRS=$peers POD_IP=127.0.0.1 \
    NODE_NAME="n$((i + 1))" HEARTBEAT_MS=500 DEAD_AFTER=3 DRAIN_SECONDS=1 \
    "$bin/omni-controlplane" >> "$out/logs/node-n$((i + 1)).log" 2>&1 &
  cp_pid[$i]=$!
  pids+=($!)
}
seed() {
  .venv/bin/python - "$@" <<'PY'
import json, sys, urllib.request
from backend.ingestion.loader import load_markdown_dir
from scripts.gen_ingestion_golden import _random_docs
docs = [d.model_dump() for d in load_markdown_dir("samples") + _random_docs(500, seed=3)]
for port in sys.argv[1:]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/ingest", json.dumps({"documents": docs}).encode(),
                                 {"Content-Type": "application/json"})
    print(port, json.load(urllib.request.urlopen(req, timeout=300))["chunks"], "chunks")
PY
}
loadgen() { # loadgen <report.json> [flags]
  local report=$1
  shift
  "$bin/cp-loadgen" --addrs "$peers" --unique "$@" 2>>"$out/logs/loadgen.log" |
    sed -n '/^REPORT_BEGIN$/,/^REPORT_END$/p' | sed '1d;$d' > "$report"
}

note "=== start 2 gateways, seed, start 3 control-plane nodes"
start_gateway 0
start_gateway 1
seed "${gw_ports[@]}" | tee -a "$log"
for i in 0 1 2; do start_node "$i"; done
sleep 3
note "nodes: $(for p in "${cp_ports[@]}"; do curl -fsS "http://127.0.0.1:$((p + 90))/healthz" >/dev/null && echo -n "$p:up "; done)"

note "=== 1. routing overhead (8 workers, 30 s, unique queries)"
for round in 1 2; do
  loadgen "$out/overhead-direct-$round.json" --mode direct --direct-url "http://127.0.0.1:8101,http://127.0.0.1:8102" \
    --workers 8 --duration 30s --label "overhead-direct-$round"
  loadgen "$out/overhead-grpc-$round.json" --mode grpc --workers 8 --duration 30s --label "overhead-grpc-$round"
  note "round $round done"
done
note "=== 1b. per-hop overhead without concurrency (1 worker, 30 s)"
for round in 1 2; do
  loadgen "$out/overhead1-direct-$round.json" --mode direct --direct-url "http://127.0.0.1:8101,http://127.0.0.1:8102" \
    --workers 1 --duration 30s --label "overhead1-direct-$round"
  loadgen "$out/overhead1-grpc-$round.json" --mode grpc --workers 1 --duration 30s --label "overhead1-grpc-$round"
done

note "=== 2. node failover: SIGKILL at +20/+45/+70 s, restart after 3 s"
: > "$out/kills.txt"
loadgen "$out/failover.json" --mode grpc --workers 8 --duration 90s --label failover &
lg=$!
sleep 1
t0=$(now_ms)
for k in 0 1 2; do
  at=(20 45 70)
  sleep $(( ${at[$k]} - ($(now_ms) - t0) / 1000 ))
  victim=${cp_pid[$k]}
  ts=$(now_ms)
  kill -9 "$victim"
  echo "$ts $victim n$((k + 1))" >> "$out/kills.txt"
  note "SIGKILL n$((k + 1)) (pid $victim) at $ts"
  sleep 3
  start_node "$k"
  note "restarted n$((k + 1))"
done
wait "$lg"
cat "$out"/logs/node-*.log | grep -E '"peer (down|recovered|registered)"' > "$out/controlplane-membership.log" || true
echo "local harness: nodes restarted by the script 3 s after each kill" > "$out/restarts.txt"

note "=== 3. gateway failover: SIGKILL gateway 8101 at +20 s, restart at +40 s"
loadgen "$out/gateway-failover.json" --mode grpc --workers 8 --duration 60s --label gateway-failover &
lg=$!
sleep 21
gk=$(now_ms)
kill -9 "${gw_pid[0]}"
note "SIGKILL gateway 8101 at $gk"
sleep 20
gr=$(now_ms)
start_gateway 0
seed 8101 >/dev/null
gready=$(now_ms)
note "gateway 8101 restarted at $gr, ready+seeded at $gready"
wait "$lg"
echo "gateway_kill $gk" > "$out/gateway-events.txt"
echo "gateway_restart $gr" >> "$out/gateway-events.txt"
echo "gateway_ready $gready" >> "$out/gateway-events.txt"
cat "$out"/logs/node-*.log | grep '"gateway state"' | grep '127.0.0.1:8101' > "$out/gateway-transitions.log" || true

.venv/bin/python deploy/kind/grpc_cp_analyze.py "$out" > "$out/summary.json"
note "done"

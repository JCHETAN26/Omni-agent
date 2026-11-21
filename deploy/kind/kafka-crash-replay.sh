#!/usr/bin/env bash
# Crash-and-replay fault injection for the Go/Kafka ingestion pipeline on kind.
#   deploy/kind/kafka-crash-replay.sh <out_dir> [docs=20000] [rate=docs/s]
#
# While omni-ingest-bench streams a known corpus at a fixed rate, this SIGKILLs
# (kill -9, no graceful shutdown) processes from inside the kind node:
#   1. every omni-indexer worker
#   2. the Kafka broker (JVM)
#   3. two of the indexer workers
#   4. the broker and every indexer at once
# Kubernetes restarts the containers. The bench then waits for every document's
# indexed event and consumer lag 0, and verifies that Elasticsearch and Qdrant
# hold exactly the expected chunks for all N documents and the DLQ is empty.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1
docs=${2:-20000}
rate=${3:-100}
node=omniagent-control-plane
ns=omniagent
mkdir -p "$out"
log="$out/faults.log"
: > "$log"
note() { echo "$(date -u +%FT%T.%3NZ) $*" | tee -a "$log"; }
kill9() { # kill9 <pgrep args...>: SIGKILL matching processes in the node
  local pids
  pids=$(docker exec "$node" pgrep "$@" | tr '\n' ' ')
  # The node image has no kill binary: use the shell builtin.
  [ -n "$pids" ] && docker exec "$node" sh -c "kill -9 $pids"
  echo "$pids"
}
restarts() {
  kubectl -n "$ns" get pods -l 'app.kubernetes.io/component in (indexer,kafka,ingest)' \
    -o jsonpath='{range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount} {end}'
}

note "restart counts before: $(restarts)"
run_id="crash$(date +%s)"
deploy/kind/kafka-bench.sh "$out/crash-replay.json" --docs "$docs" --rate "$rate" \
  --run-id "$run_id" --timeout 45m > "$out/bench-driver.log" 2>&1 &
bench=$!
until kubectl -n "$ns" logs -l app.kubernetes.io/component=ingest-bench --tail=50 2>/dev/null | grep -q '"msg":"progress"'; do
  sleep 2
done
note "bench $run_id streaming $docs docs at $rate docs/s"

sleep 20
note "FAULT 1: SIGKILL all indexers: pids $(kill9 -x omni-indexer)"
sleep 30
note "FAULT 2: SIGKILL Kafka broker: pids $(kill9 -f kafka.Kafka)"
sleep 40
two=$(docker exec "$node" pgrep -x omni-indexer | head -2 | tr '\n' ' ')
docker exec "$node" sh -c "kill -9 $two"
note "FAULT 3: SIGKILL two indexers: pids $two"
sleep 30
note "FAULT 4: SIGKILL broker and all indexers: pids $(kill9 -f kafka.Kafka) $(kill9 -x omni-indexer)"

wait "$bench" && status=0 || status=$?
note "bench exited $status"
note "restart counts after: $(restarts)"
cat "$out/bench-driver.log" | tail -3
exit "$status"

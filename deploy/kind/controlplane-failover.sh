#!/usr/bin/env bash
# Live failover test of the gRPC control plane on kind.
#   deploy/kind/controlplane-failover.sh <out_dir> [rate=10] [duration=150s]
# With the operator running a 3-node control plane in front of 2 gateways,
# cp-loadgen sends Search calls at a fixed rate while this injects:
#   +30 s  SIGKILL one control-plane node (no drain, connections reset)
#   +60 s  delete a control-plane pod (SIGTERM: NOT_SERVING, drain, stop)
#   +90 s  SIGKILL one gateway (FastAPI) process
#   +110 s SIGKILL two control-plane nodes at once
# Every call carries a 3 s deadline; any non-OK result counts as a failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/kind/lib-cp.sh
out=$1
rate=${2:-10}
duration=${3:-150s}
mkdir -p "$out"
log="$out/faults.log"
: > "$log"

kubectl apply -f deploy/kind/omniagentcluster.yaml >/dev/null
kubectl -n "$ns" patch oac omniagent --type=merge -p '{"spec":{"autoscaling":{"mode":"off"}}}' >/dev/null
kubectl -n "$ns" scale deploy/omniagent-backend --replicas=2 >/dev/null
kubectl -n "$ns" wait --for=condition=Ready oac/omniagent --timeout=180s >/dev/null
kubectl -n "$ns" rollout status deploy/omniagent-backend --timeout=300s >/dev/null
note "control plane: $(kubectl -n "$ns" get pods -l app.kubernetes.io/component=controlplane --no-headers | awk '{print $1}' | tr '\n' ' ')"

job="cp-failover-$(date +%s)"
loadgen_start "$job" --rate "$rate" --duration "$duration" --label failover
note "load started: $rate calls/s for $duration"
sleep 30
note "FAULT 1: SIGKILL one control-plane node: pid $(p=$(pids -f /omni-controlplane | awk '{print $1}'); kill9 "$p"; echo "$p")"
sleep 30
victim=$(kubectl -n "$ns" get pods -l app.kubernetes.io/component=controlplane -o name | tail -1)
kubectl -n "$ns" delete "$victim" --wait=false >/dev/null
note "FAULT 2: graceful delete (SIGTERM drain) of $victim"
sleep 30
note "FAULT 3: SIGKILL one gateway: pid $(p=$(pids -f 'uvicorn' | awk '{print $1}'); kill9 "$p"; echo "$p")"
sleep 20
note "FAULT 4: SIGKILL two control-plane nodes: pids $(p=$(pids -f /omni-controlplane | awk '{print $1, $2}'); kill9 "$p"; echo "$p")"

loadgen_wait "$job" "$out/failover.json"
note "load finished"
note "restarts: $(kubectl -n "$ns" get pods -l 'app.kubernetes.io/component in (controlplane,backend)' -o jsonpath='{range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount} {end}')"
kubectl -n "$ns" logs -l app.kubernetes.io/component=controlplane --tail=-1 --prefix 2>/dev/null |
  grep '"gateway state"' > "$out/gateway-transitions.log" || true
python3 - "$out" <<'PY'
import json, re, sys, datetime as dt
out = sys.argv[1]
r = json.load(open(f"{out}/failover.json"))
faults = [(int(dt.datetime.fromisoformat(l.split()[0].replace("Z", "+00:00")).timestamp()), l.split(" ", 1)[1].strip())
          for l in open(f"{out}/faults.log") if " FAULT " in l]
tl = r["timeline"]
def window(a, b):
    rows = [x for x in tl if a <= x["t"] < b]
    return {"ok": sum(x["ok"] for x in rows), "failed": sum(x["failed"] for x in rows),
            "degraded": sum(x["degraded"] for x in rows)}
summary = {"sent": r["sent"], "failed": r["failed"], "by_code": r["by_code"], "degraded": r["degraded"],
           "latency_ms": r["latency_ms"], "served_by_node": r["served_by_node"],
           "served_by_gateway": r["served_by_gateway"], "faults": []}
for i, (t, name) in enumerate(faults):
    end = faults[i + 1][0] if i + 1 < len(faults) else tl[-1]["t"] + 1
    w = window(t, end)
    p99 = max((x["p99_ms"] for x in tl if t <= x["t"] < min(t + 10, end)), default=0)
    summary["faults"].append({"fault": name, "at_unix": t, **w, "worst_1s_p99_ms_in_next_10s": p99})
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
PY

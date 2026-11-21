#!/usr/bin/env bash
# Operator benchmark on kind (deploy/kind/phase4-up.sh first).
#   deploy/kind/operator-bench.sh <out_dir> [trials=5] [profile=60s@2,180s@12,150s@0]
#
# 1. Provisioning: time from `kubectl apply` until all 3 control-plane nodes
#    are Ready, (a) applying the operator's own rendered objects by hand
#    ("before": no operator) and (b) applying an OmniAgentCluster and letting
#    the operator reconcile ("after"). Then self-healing: delete the
#    control-plane Deployment and time the operator's recovery.
# 2. Autoscaling: the same step load through the control plane with the gateway
#    scaled by (a) a CPU HPA ("hpa" mode) and (b) the operator's in-flight
#    autoscaler ("operator" mode); gateway replicas are recorded every second.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/kind/lib-cp.sh
out=$1
trials=${2:-5}
profile=${3:-60s@2,180s@12,150s@0}
mkdir -p "$out"
log="$out/operator-bench.log"
: > "$log"
elapsed() { awk -v a="$1" -v b="$(date +%s%N)" 'BEGIN { printf "%.2f", (b - a) / 1e9 }'; }
cp_ready() { kubectl -n "$ns" get deploy omniagent-controlplane -o jsonpath='{.status.readyReplicas}' 2>/dev/null; }
wait_ready() { # seconds from $1 (epoch ns) until 3 control-plane nodes are Ready
  until [ "$(cp_ready)" = "3" ]; do sleep 0.2; done
  elapsed "$1"
}
wait_gone() {
  while kubectl -n "$ns" get deploy omniagent-controlplane >/dev/null 2>&1 ||
    [ -n "$(kubectl -n "$ns" get pods -l app.kubernetes.io/component=controlplane -o name)" ]; do sleep 1; done
}

# Objects the operator would create (its golden render), retargeted to kind.
manual=$(mktemp)
python3 - "$manual" <<'PY'
import sys, yaml
docs = [d for d in yaml.safe_load_all(open("operator/internal/controller/testdata/rendered.yaml")) if d]
out = []
for d in docs:
    if d["kind"] == "HorizontalPodAutoscaler":
        continue
    if d["kind"] == "Deployment":
        c = d["spec"]["template"]["spec"]["containers"][0]
        c["image"], c["imagePullPolicy"] = "omniagent-controlplane:dev", "Never"
    out.append(d)
yaml.safe_dump_all(out, open(sys.argv[1], "w"))
PY

note "=== provisioning: manual apply of the operator's objects (no operator)"
kubectl -n "$ns" delete oac omniagent --ignore-not-found --wait=true >/dev/null
wait_gone
for i in $(seq 1 "$trials"); do
  t0=$(date +%s%N)
  kubectl apply -f "$manual" >/dev/null
  note "manual trial $i: ready in $(wait_ready "$t0") s"
  kubectl delete -f "$manual" --wait=true >/dev/null
  wait_gone
done

note "=== provisioning: OmniAgentCluster reconciled by the operator"
for i in $(seq 1 "$trials"); do
  t0=$(date +%s%N)
  kubectl apply -f deploy/kind/omniagentcluster.yaml >/dev/null
  note "operator trial $i: ready in $(wait_ready "$t0") s"
  kubectl -n "$ns" delete oac omniagent --wait=true >/dev/null
  wait_gone
done

note "=== self-healing: delete the control-plane Deployment under the operator"
kubectl apply -f deploy/kind/omniagentcluster.yaml >/dev/null
wait_ready "$(date +%s%N)" >/dev/null
for i in $(seq 1 "$trials"); do
  uid=$(kubectl -n "$ns" get deploy omniagent-controlplane -o jsonpath='{.metadata.uid}')
  t0=$(date +%s%N)
  kubectl -n "$ns" delete deploy omniagent-controlplane --wait=false >/dev/null
  # Recreated = a Deployment with a different UID exists again.
  until now=$(kubectl -n "$ns" get deploy omniagent-controlplane -o jsonpath='{.metadata.uid}' 2>/dev/null) &&
    [ -n "$now" ] && [ "$now" != "$uid" ]; do sleep 0.1; done
  recreated=$(elapsed "$t0")
  note "self-heal trial $i: Deployment recreated in $recreated s, 3 nodes Ready in $(wait_ready "$t0") s"
  sleep 5
done

for mode in hpa operator; do
  note "=== autoscaling: $mode mode, profile $profile"
  kubectl -n "$ns" patch oac omniagent --type=merge -p "{\"spec\":{\"autoscaling\":{\"mode\":\"$mode\"}}}" >/dev/null
  kubectl -n "$ns" scale deploy/omniagent-backend --replicas=1 >/dev/null
  sleep 45 # settle at the minimum (and let metrics-server see idle CPU)
  note "$mode: gateway at $(kubectl -n "$ns" get deploy omniagent-backend -o jsonpath='{.spec.replicas}') replica(s) before load"
  csv="$out/replicas-$mode.csv"
  echo "t,spec,ready" > "$csv"
  (while true; do
    echo "$(date +%s),$(kubectl -n "$ns" get deploy omniagent-backend -o jsonpath='{.spec.replicas},{.status.readyReplicas}')" >> "$csv"
    sleep 1
  done) &
  watcher=$!
  job="cp-scale-$mode-$(date +%s)"
  loadgen_start "$job" --profile "$profile" --label "autoscale-$mode"
  loadgen_wait "$job" "$out/autoscale-$mode.json"
  sleep 30
  kill "$watcher"
  note "$mode: done; gateway at $(kubectl -n "$ns" get deploy omniagent-backend -o jsonpath='{.spec.replicas}') replica(s)"
  kubectl -n "$ns" get events --field-selector involvedObject.name=omniagent-backend,reason=SuccessfulRescale \
    -o custom-columns=TIME:.lastTimestamp,MSG:.message --no-headers 2>/dev/null >> "$out/hpa-events.log" || true
done
kubectl -n "$ns" patch oac omniagent --type=merge -p '{"spec":{"autoscaling":{"mode":"off"}}}' >/dev/null
kubectl -n "$ns" scale deploy/omniagent-backend --replicas=2 >/dev/null

python3 - "$out" <<'PY'
import csv, json, re, statistics, sys
out = sys.argv[1]
lines = open(f"{out}/operator-bench.log").read()
def times(prefix):
    return [float(x) for x in re.findall(rf"{prefix} trial \d+: .*?ready in ([0-9.]+) s", lines)]
summary = {"provisioning_s": {}, "autoscaling": {}}
for k, p in (("manual_apply", "manual"), ("operator_reconcile", "operator"), ("self_heal_ready", "self-heal")):
    v = times(p)
    if v:
        summary["provisioning_s"][k] = {"trials": v, "median": statistics.median(v), "min": min(v), "max": max(v)}
rec = [float(x) for x in re.findall(r"recreated in ([0-9.]+) s", lines)]
if rec:
    summary["provisioning_s"]["self_heal_recreated"] = {"trials": rec, "median": statistics.median(rec)}
for mode in ("hpa", "operator"):
    try:
        r = json.load(open(f"{out}/autoscale-{mode}.json"))
        rows = [(int(a), int(b or 0), int(c or 0)) for a, b, c in
                (x.split(",") for x in open(f"{out}/replicas-{mode}.csv").read().split()[1:]) if a]
    except (FileNotFoundError, ValueError):
        continue
    step, off = r["stage_starts_unix"][1], r["stage_starts_unix"][2]
    first = lambda cond, after: next((t - after for t, s, rd in rows if t >= after and cond(s, rd)), None)
    peak = max((s for t, s, rd in rows if step <= t < off), default=None)
    tl = r["timeline"]
    def p95(a, b):
        v = sorted(x["p99_ms"] for x in tl if a <= x["t"] < b)
        return v[int(0.95 * (len(v) - 1))] if v else None
    summary["autoscaling"][mode] = {
        "scale_up_decision_s": first(lambda s, rd: s > 1, step),
        "first_new_replica_ready_s": first(lambda s, rd: rd > 1, step),
        "peak_replicas": peak,
        "scale_down_to_1_after_load_off_s": first(lambda s, rd: s == 1, off),
        "calls": r["sent"], "failed": r["failed"], "by_code": r["by_code"],
        "latency_ms_overall": r["latency_ms"],
        "p95_of_1s_p99_ms": {"first_60s_of_step": p95(step, step + 60), "rest_of_step": p95(step + 60, off)},
    }
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
PY

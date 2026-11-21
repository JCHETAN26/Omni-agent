#!/usr/bin/env bash
# Phase 4A on the kind cluster (deploy/kind/up.sh first): deploy the gRPC
# control plane with Helm and measure it.
#   deploy/kind/grpc-control-plane-bench.sh <out_dir>
#
# 1. Routing overhead: closed-loop Search (8 workers, 60 s), straight to the
#    gateways' POST /search vs through the control plane, alternated twice.
# 2. Failover: closed-loop Search through the control plane (8 workers, 120 s);
#    one control-plane replica is SIGKILLed at +30 s, +60 s and +90 s.
#    Measured: errors, calls in flight at each kill and their latency, peer
#    detection time (kill -> peers log "peer down"), restart to Ready.
# 3. Degradation: closed loop (4 workers, 210 s); Triton is scaled to 0 at
#    +30 s and back to 1 at +90 s. Measured: errors, BM25-degraded answers, and
#    when degradation starts and ends.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1
ns=omniagent
node=omniagent-control-plane
mkdir -p "$out"
log="$out/bench.log"
: > "$log"
note() { echo "$(date -u +%FT%T.%3NZ) $*" | tee -a "$log"; }
now_ms() { date +%s%3N; }

loadgen() { # loadgen <name> <report.json> [cp-loadgen flags...]: run as a Job, wait, extract
  local name=$1 report=$2
  shift 2
  local args
  args=$(printf '"%s", ' "$@")
  kubectl apply -n "$ns" -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: { name: $name }
spec:
  backoffLimit: 0
  template:
    metadata:
      labels: { app.kubernetes.io/component: cp-loadgen, omniagent.io/controlplane-client: "true" }
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext: { runAsNonRoot: true, runAsUser: 65532, seccompProfile: { type: RuntimeDefault } }
      containers:
        - name: loadgen
          image: omniagent-controlplane:dev
          imagePullPolicy: Never
          command: ["/cp-loadgen"]
          args: [${args%, }]
          resources: { requests: { cpu: 100m, memory: 64Mi }, limits: { cpu: "1", memory: 512Mi } }
          securityContext: { allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: { drop: [ALL] } }
YAML
  echo "$name"
}
collect() { # collect <name> <report.json>
  kubectl -n "$ns" wait --for=condition=complete "job/$1" --timeout=30m >/dev/null
  kubectl -n "$ns" logs "job/$1" | sed -n '/^REPORT_BEGIN$/,/^REPORT_END$/p' | sed '1d;$d' > "$2"
  kubectl -n "$ns" delete job "$1" >/dev/null
}
started() { until kubectl -n "$ns" logs "job/$1" 2>/dev/null | grep -q '"msg":"progress"\|REPORT_BEGIN'; do sleep 0.5; done; }
cp_pids() { docker exec "$node" pgrep -f /omni-controlplane | tr '\n' ' '; }

note "=== deploy (Phase 4B operator objects removed; idle Phase 3 workloads off to fit 4 vCPU)"
kubectl -n "$ns" delete omniagentclusters.platform.omniagent.io --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete ns operator-system --ignore-not-found >/dev/null 2>&1 || true
kubectl -n "$ns" delete deploy,svc,pdb,networkpolicy -l app.kubernetes.io/managed-by=omniagent-operator --ignore-not-found >/dev/null
docker build -q -f deploy/docker/controlplane.Dockerfile -t omniagent-controlplane:dev . >/dev/null
docker build -q -f deploy/docker/backend.Dockerfile -t omniagent-backend:dev . >/dev/null
kind load docker-image --name omniagent omniagent-controlplane:dev omniagent-backend:dev >/dev/null
helm upgrade omniagent deploy/kubernetes/helm/omniagent -n "$ns" -f deploy/kind/values-kind.yaml \
  --set kafka.enabled=false --set ingestion.enabled=false --set frontend.enabled=false \
  --wait --timeout 15m >/dev/null
kubectl -n "$ns" rollout restart deploy/omniagent-backend >/dev/null
kubectl -n "$ns" rollout status deploy/omniagent-backend --timeout=10m >/dev/null
kubectl -n "$ns" rollout status deploy/omniagent-controlplane --timeout=5m >/dev/null
note "control plane: $(kubectl -n "$ns" get pods -l app.kubernetes.io/component=controlplane -o wide --no-headers | awk '{print $1"@"$6}' | tr '\n' ' ')"

note "=== 1. routing overhead (closed loop, 8 workers, 60 s each)"
for round in 1 2; do
  for mode in direct grpc; do
    j=$(loadgen "cp-overhead-$mode-$round-$(date +%s)" x --mode "$mode" --workers 8 --duration 60s \
      --target dns:///omniagent-controlplane:9000 --direct-url http://omniagent-backend:8000 --label "overhead-$mode-$round")
    collect "$j" "$out/overhead-$mode-$round.json"
    note "overhead $mode round $round: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['calls_per_s'], 'calls/s', r['latency_ms'], 'failed', r['failed'])" "$out/overhead-$mode-$round.json")"
  done
done

note "=== 2. failover: SIGKILL one control-plane replica at +30/+60/+90 s"
j=$(loadgen "cp-failover-$(date +%s)" x --mode grpc --workers 8 --duration 120s --label failover)
started "$j"
t0=$(now_ms)
: > "$out/kills.txt"
for at in 30 60 90; do
  sleep $(( at - ($(now_ms) - t0) / 1000 ))
  pid=$(cp_pids | awk '{print $1}')
  k=$(now_ms)
  docker exec "$node" sh -c "kill -9 $pid"
  echo "$k $pid" >> "$out/kills.txt"
  note "SIGKILL control-plane pid $pid at $k"
done
collect "$j" "$out/failover.json"
kubectl -n "$ns" logs -l app.kubernetes.io/component=controlplane --prefix --tail=-1 --since=10m 2>/dev/null |
  grep -E '"peer (down|recovered|registered)"|"control plane node started"' > "$out/controlplane-membership.log" || true
kubectl -n "$ns" get pods -l app.kubernetes.io/component=controlplane \
  -o jsonpath='{range .items[*]}{.metadata.name}{" restarts="}{.status.containerStatuses[0].restartCount}{"\n"}{end}' > "$out/restarts.txt"

note "=== 3. degradation: Triton down at +30 s, back at +90 s"
j=$(loadgen "cp-degrade-$(date +%s)" x --mode grpc --workers 4 --duration 210s --label degradation)
started "$j"
t0=$(now_ms)
sleep 30
echo "triton_down $(now_ms)" > "$out/degradation-events.txt"
kubectl -n "$ns" scale deploy/omniagent-triton --replicas=0 >/dev/null
note "Triton scaled to 0"
sleep $(( 90 - ($(now_ms) - t0) / 1000 ))
echo "triton_up $(now_ms)" >> "$out/degradation-events.txt"
kubectl -n "$ns" scale deploy/omniagent-triton --replicas=1 >/dev/null
note "Triton scaled back to 1"
collect "$j" "$out/degradation.json"
kubectl -n "$ns" rollout status deploy/omniagent-triton --timeout=10m >/dev/null
echo "triton_ready $(now_ms)" >> "$out/degradation-events.txt"

python3 deploy/kind/grpc_cp_analyze.py "$out" | tee "$out/summary.json"
note "done"

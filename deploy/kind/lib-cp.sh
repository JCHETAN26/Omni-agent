# Shared helpers for the Phase 4 kind scripts (source this file).
ns=omniagent
node=omniagent-control-plane
note() { echo "$(date -u +%FT%T.%3NZ) $*" | tee -a "${log:-/dev/null}"; }

# loadgen_start <job-name> [cp-loadgen flags...]: run cp-loadgen as a Job carrying
# the client label the control plane's NetworkPolicy admits.
loadgen_start() {
  local name=$1
  shift
  local args
  args=$(printf '"%s", ' "$@")
  kubectl apply -n "$ns" -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: $name
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app.kubernetes.io/component: cp-loadgen
        omniagent.io/controlplane-client: "true"
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
        seccompProfile: { type: RuntimeDefault }
      containers:
        - name: loadgen
          image: omniagent-controlplane:dev
          imagePullPolicy: Never
          command: ["/cp-loadgen"]
          args: [--target, "dns:///omniagent-controlplane:9000", ${args%, }]
          resources:
            requests: { cpu: 100m, memory: 64Mi }
            limits: { cpu: "1", memory: 256Mi }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: [ALL] }
YAML
  until kubectl -n "$ns" logs "job/$name" 2>/dev/null | grep -q '"msg":"stage"'; do sleep 1; done
}

# loadgen_wait <job-name> <report.json>: wait for the Job and extract its report.
loadgen_wait() {
  local name=$1 out=$2
  kubectl -n "$ns" wait --for=condition=complete "job/$name" --timeout=30m >/dev/null
  kubectl -n "$ns" logs "job/$name" > "${out%.json}.log"
  sed -n '/^REPORT_BEGIN$/,/^REPORT_END$/p' "${out%.json}.log" | sed '1d;$d' > "$out"
}

# pids <pgrep args...>: matching process IDs inside the kind node.
pids() { docker exec "$node" pgrep "$@" | tr '\n' ' '; }
# kill9 <pid...>: SIGKILL inside the node (its image has no kill binary).
kill9() { docker exec "$node" sh -c "kill -9 $*"; }

#!/usr/bin/env bash
# Run omni-ingest-bench as a Job inside the kind cluster and save its report.
#   deploy/kind/kafka-bench.sh <report.json> [omni-ingest-bench flags...]
#   e.g. deploy/kind/kafka-bench.sh out/throughput.json --docs 20000 --concurrency 8
# Corpus: BEIR FiQA-2018 (57,638 financial Q&A posts), downloaded once to .cache/.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1
shift
ns=omniagent
corpus=.cache/fiqa/corpus.jsonl
if [ ! -s "$corpus" ]; then
  mkdir -p .cache/fiqa
  curl -fsSL -o .cache/fiqa.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
  unzip -p .cache/fiqa.zip fiqa/corpus.jsonl > "$corpus"
fi
if [ -z "${SKIP_BUILD:-}" ]; then
  docker build -q -f deploy/docker/ingestion.Dockerfile --target bench -t omniagent-ingest-bench:dev . >/dev/null
  kind load docker-image --name omniagent omniagent-ingest-bench:dev >/dev/null
fi

name="ingest-bench-$(date +%s)"
args=$(printf '"%s", ' "$@")
kubectl apply -n "$ns" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $name
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app.kubernetes.io/name: omniagent
        app.kubernetes.io/instance: omniagent
        app.kubernetes.io/component: ingest-bench
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
        seccompProfile: { type: RuntimeDefault }
      containers:
        - name: bench
          image: omniagent-ingest-bench:dev
          imagePullPolicy: Never
          args: [${args%, }]
          env:
            - { name: KAFKA_BROKERS, value: "omniagent-kafka:9092" }
          resources:
            requests: { cpu: 100m, memory: 256Mi }
            limits: { cpu: "1", memory: 1Gi }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: [ALL] }
EOF
echo "job/$name started"
until kubectl -n "$ns" get pod -l job-name="$name" -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -qE 'Running|Succeeded|Failed'; do sleep 2; done
kubectl -n "$ns" logs -f "job/$name" | grep --line-buffered -E '"msg":"(progress|sending|upload failed)' | cut -c1-220 || true
for _ in $(seq 1 60); do
  s=$(kubectl -n "$ns" get job "$name" -o jsonpath='{.status.succeeded}{.status.failed}')
  [ -n "$s" ] && break
  sleep 2
done
mkdir -p "$(dirname "$out")"
kubectl -n "$ns" logs "job/$name" > "${out%.json}.log"
sed -n '/^REPORT_BEGIN$/,/^REPORT_END$/p' "${out%.json}.log" | sed '1d;$d' > "$out"
echo "report: $out"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r.get(k) for k in ('throughput','latency_ms','final_consumer_lag','passed')}))" "$out"

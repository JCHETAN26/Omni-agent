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
  # BEIR's Hugging Face mirror (Parquet, checksum-verified); the original TU
  # Darmstadt host is often unreachable. Needs pyarrow (pip install pyarrow).
  curl -fsSL -o .cache/fiqa/corpus.parquet \
    https://huggingface.co/datasets/BeIR/fiqa/resolve/main/corpus/corpus-00000-of-00001.parquet
  echo "b36bb07dab3bfad488190818828e749a52faf5ba391c1d51dac8b06064576f99  .cache/fiqa/corpus.parquet" | sha256sum -c -
  python3 - "$corpus" <<'PY'
import json, sys
import pyarrow.parquet as pq
with open(sys.argv[1], "w") as f:
    for r in pq.read_table(".cache/fiqa/corpus.parquet").to_pylist():
        doc = {"_id": str(r["_id"]), "title": r.get("title") or "", "text": r["text"]}
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
PY
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

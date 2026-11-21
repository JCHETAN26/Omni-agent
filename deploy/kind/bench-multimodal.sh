#!/usr/bin/env bash
# Dynamic batching OFF vs ON for live frames on the kind cluster.
#   deploy/kind/bench-multimodal.sh [out_dir] [rounds]
# Each run points the backend at one Triton model (same ONNX graph, same
# instance count; only dynamic batching differs), waits for the rollout, and drives
# closed-loop /ws/frames sessions through the NodePort (localhost:8080).
# Rounds alternate the order so neither config always runs first.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:-benchmarks/results/$(date -u +%F)/multimodal}
rounds=${2:-2}
sessions=${SESSIONS:-"1 2 4 8 16 32"}
duration=${DURATION:-30}
mkdir -p "$out"

kubectl -n omniagent port-forward svc/omniagent-triton 18002:8002 >/tmp/pf-metrics.log 2>&1 &
pf=$!
trap 'kill $pf 2>/dev/null || true' EXIT
sleep 3

use_model() {
  helm upgrade omniagent deploy/kubernetes/helm/omniagent -n omniagent --reuse-values \
    --set config.tritonVisionModel="$1" --wait --timeout 10m >/dev/null
  # The chart's checksum/config annotation rolls the backend on a config change.
  kubectl -n omniagent rollout status deploy/omniagent-backend --timeout 10m >/dev/null
  for i in $(seq 1 60); do curl -fsS http://127.0.0.1:8080/readyz >/dev/null 2>&1 && break; sleep 2; done
  kubectl -n omniagent get configmap omniagent-config -o jsonpath='{.data.TRITON_VISION_MODEL}'; echo
}

for r in $(seq 1 "$rounds"); do
  if (( r % 2 )); then order="vit_b16_nobatch vit_b16"; else order="vit_b16 vit_b16_nobatch"; fi
  for model in $order; do
    label=$([ "$model" = vit_b16 ] && echo batching-on || echo batching-off)
    echo "=== $(date -u +%T) round $r: $label ($(use_model "$model"))"
    python -m benchmarks.multimodal --url ws://127.0.0.1:8080/ws/frames \
      --sessions $sessions --duration "$duration" --warmup 5 \
      --triton-metrics http://127.0.0.1:18002/metrics --model "$model" \
      --label "$label" --output "$out/$label-r$r.json"
  done
done
use_model vit_b16 >/dev/null   # leave the default (batching on) deployed
echo MULTIMODAL_COMPLETE

#!/usr/bin/env bash
# Build every image, create the kind cluster, and deploy OmniAgent with real Triton
# and the Go/Kafka ingestion pipeline.
#   TRITON_TAG=<yy.mm>-py3 deploy/kind/up.sh
set -euo pipefail
: "${TRITON_TAG:?set TRITON_TAG, e.g. 25.01-py3}"
cd "$(dirname "$0")/../.."
repo=model_server/model_repository
step() { echo "=== $(date -u +%T) $*"; }

step "export ONNX models"
[ -f $repo/minilm_embedder/1/model.onnx ] || python -m model_server.export_onnx
[ -f $repo/vit_b16/1/model.onnx ] || python -m model_server.export_vit

step "build images"
docker build -q -f deploy/docker/backend.Dockerfile -t omniagent-backend:dev .
docker build -q -t omniagent-frontend:dev frontend
docker build -q -f deploy/docker/triton.Dockerfile --build-arg TRITON_TAG="$TRITON_TAG" -t omniagent-triton:dev .
docker build -q -f deploy/docker/ingestion.Dockerfile -t omniagent-ingestion:dev .
docker build -q -f deploy/docker/controlplane.Dockerfile -t omniagent-controlplane:dev .
docker build -q -f deploy/docker/operator.Dockerfile -t omniagent-operator:dev .

step "kind cluster"
kind get clusters | grep -qx omniagent || kind create cluster --config deploy/kind/kind-config.yaml --wait 120s
kind load docker-image --name omniagent omniagent-backend:dev omniagent-frontend:dev omniagent-triton:dev omniagent-ingestion:dev \
  omniagent-controlplane:dev omniagent-operator:dev

step "helm install"
helm upgrade --install omniagent deploy/kubernetes/helm/omniagent \
  --namespace omniagent --create-namespace \
  -f deploy/kind/values-kind.yaml --wait --timeout 20m
kubectl apply -n omniagent -f deploy/kind/backend-nodeport.yaml

step "smoke test"
for i in $(seq 1 60); do curl -fsS http://127.0.0.1:8080/readyz >/dev/null 2>&1 && break; sleep 2; done
curl -fsS http://127.0.0.1:8080/health
echo
kubectl get pods -n omniagent -o wide

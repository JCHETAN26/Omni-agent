#!/usr/bin/env bash
# Phase 4B on the existing kind cluster (deploy/kind/up.sh first): build and
# load the gRPC control plane and the operator, hand the control plane to the
# operator (the chart's own is switched off), install metrics-server (needed
# only for the operator's "hpa" mode), and install the operator. Idle Phase 3
# workloads are switched off to fit the 4-vCPU node.
set -euo pipefail
cd "$(dirname "$0")/../.."
step() { echo "=== $(date -u +%T) $*"; }

step "build + load images"
docker build -q -f deploy/docker/controlplane.Dockerfile -t omniagent-controlplane:dev . >/dev/null
docker build -q -f deploy/docker/operator.Dockerfile -t omniagent-operator:dev . >/dev/null
kind load docker-image --name omniagent omniagent-controlplane:dev omniagent-operator:dev

step "chart: control plane managed by the operator"
helm upgrade omniagent deploy/kubernetes/helm/omniagent -n omniagent -f deploy/kind/values-kind.yaml \
  --set controlPlane.enabled=false --set controlPlane.operatorManaged=true \
  --set kafka.enabled=false --set ingestion.enabled=false --set frontend.enabled=false \
  --wait --timeout 15m >/dev/null

step "metrics-server"
# Pinned release; kind kubelets serve self-signed certificates.
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.1/components.yaml >/dev/null
kubectl -n kube-system get deploy metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q kubelet-insecure-tls ||
  kubectl -n kube-system patch deploy metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl -n kube-system rollout status deploy/metrics-server --timeout=180s

step "operator (CRD, RBAC, manager)"
sed 's#ghcr.io/jchetan26/omniagent-operator:0.4.0#omniagent-operator:dev#' deploy/kubernetes/manifests/operator.yaml |
  kubectl apply -f - >/dev/null
kubectl -n operator-system rollout restart deploy/operator-controller-manager >/dev/null
kubectl -n operator-system rollout status deploy/operator-controller-manager --timeout=180s
kubectl get crd omniagentclusters.platform.omniagent.io

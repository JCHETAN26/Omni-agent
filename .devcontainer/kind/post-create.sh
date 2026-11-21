#!/usr/bin/env bash
# One-time setup for the kind + Triton Codespace.
set -euxo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends libopencv-dev cmake acl
# kind: pinned release, verified against its published SHA-256.
KIND_VERSION=v0.27.0
curl -fsSLo /tmp/kind "https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-amd64"
curl -fsSLo /tmp/kind.sha256 "https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-amd64.sha256sum"
(cd /tmp && echo "$(cut -d' ' -f1 kind.sha256)  kind" | sha256sum -c -)
sudo install -m 0755 /tmp/kind /usr/local/bin/kind
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt -r model_server/requirements-export.txt
[ -f .env ] || cp .env.example .env

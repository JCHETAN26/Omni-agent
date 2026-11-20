#!/usr/bin/env bash
# Move Docker's data root to the large ephemeral /tmp disk on every start
# (/tmp is wiped when the Codespace stops, so images are pulled again after that).
set -euo pipefail
want=/tmp/docker-data
if ! grep -qs "\"data-root\": \"$want\"" /etc/docker/daemon.json; then
  sudo mkdir -p /etc/docker "$want"
  echo "{\"data-root\": \"$want\"}" | sudo tee /etc/docker/daemon.json >/dev/null
fi
if [ "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" != "$want" ]; then
  sudo pkill dockerd || true
  for i in $(seq 1 20); do pgrep dockerd >/dev/null || break; sleep 1; done
  sudo sh -c 'nohup dockerd > /tmp/dockerd.log 2>&1 &'
  for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
fi
docker info --format 'docker root: {{.DockerRootDir}}'

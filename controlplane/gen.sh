#!/usr/bin/env bash
# Regenerate the Go stubs from proto/ (protoc 36.x, protoc-gen-go v1.36.12,
# protoc-gen-go-grpc v1.6.0):
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.0
set -euo pipefail
cd "$(dirname "$0")"
m=github.com/JCHETAN26/Omni-agent/controlplane/gen
protoc -I proto --go_out=gen --go_opt=module=$m --go-grpc_out=gen --go-grpc_opt=module=$m \
  proto/omni/control/v1/control.proto

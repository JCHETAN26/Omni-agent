# syntax=docker/dockerfile:1.7
# gRPC control plane node (omni-controlplane) and its load generator
# (cp-loadgen), static binaries on distroless (non-root, no shell).
#   docker build -f deploy/docker/controlplane.Dockerfile -t omniagent-controlplane:dev .
FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY controlplane/go.mod controlplane/go.sum ./
RUN go mod download
COPY controlplane/ ./
RUN CGO_ENABLED=0 go test ./... \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ ./cmd/...

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/omni-controlplane /out/cp-loadgen /
USER nonroot:nonroot
EXPOSE 9000 9090
ENTRYPOINT ["/omni-controlplane"]

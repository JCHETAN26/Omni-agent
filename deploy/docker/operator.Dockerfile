# syntax=docker/dockerfile:1.7
# omniagent-operator (controller-runtime manager). Built from the repo root:
# the operator imports the control plane's generated gRPC client (../controlplane).
#   docker build -f deploy/docker/operator.Dockerfile -t omniagent-operator:dev .
FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY controlplane/ controlplane/
COPY operator/go.mod operator/go.sum operator/
RUN cd operator && go mod download
COPY operator/ operator/
RUN cd operator && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/manager ./cmd

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/manager /manager
USER 65532:65532
ENTRYPOINT ["/manager"]

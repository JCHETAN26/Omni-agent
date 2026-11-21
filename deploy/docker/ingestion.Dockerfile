# syntax=docker/dockerfile:1.7
# Go ingestion pipeline: omni-ingest, omni-indexer and omni-ingest-bench as
# static binaries on distroless (non-root, no shell).
#   docker build -f deploy/docker/ingestion.Dockerfile -t omniagent-ingestion:dev .
#   docker build -f deploy/docker/ingestion.Dockerfile --target bench -t omniagent-ingest-bench:dev .
#     (bench needs .cache/fiqa/corpus.jsonl: deploy/kind/kafka-bench.sh downloads it)
FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY ingestion/go.mod ingestion/go.sum ./
RUN go mod download
COPY ingestion/ ./
RUN CGO_ENABLED=0 go test ./... \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ ./cmd/...

# Benchmark/verification image (deploy/kind/kafka-bench.sh).
FROM gcr.io/distroless/static-debian12:nonroot AS bench
COPY --from=build /out/omni-ingest-bench /
COPY .cache/fiqa/corpus.jsonl /data/corpus.jsonl
USER nonroot:nonroot
ENTRYPOINT ["/omni-ingest-bench"]

# Default target: the services.
FROM gcr.io/distroless/static-debian12:nonroot AS runtime
COPY --from=build /out/omni-ingest /out/omni-indexer /
USER nonroot:nonroot
EXPOSE 8090 9090
ENTRYPOINT ["/omni-indexer"]

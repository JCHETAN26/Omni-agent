# syntax=docker/dockerfile:1.7
# Triton Inference Server with OmniAgent's model repository baked in (/models):
# vit_b16 (dynamic batching), vit_b16_nobatch, minilm_embedder, cross_encoder_reranker.
# Export the ONNX files first:
#   python -m model_server.export_onnx && python -m model_server.export_vit
#   docker build -f deploy/docker/triton.Dockerfile --build-arg TRITON_TAG=<yy.mm>-py3 -t omniagent-triton:dev .
ARG TRITON_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}
COPY --chown=1000:1000 model_server/model_repository /models
# NGC images include the non-root "triton-server" user (UID 1000).
USER 1000
HEALTHCHECK --interval=15s --timeout=3s CMD ["curl", "-fsS", "http://127.0.0.1:8000/v2/health/ready"]
CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]

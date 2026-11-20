# syntax=docker/dockerfile:1.7
# OmniAgent API (FastAPI + LangGraph). Build from the repo root:
#   docker build -f deploy/docker/backend.Dockerfile -t omniagent-backend:dev .
# Build the C++ OpenCV preprocessing extension (native/vision) as a wheel.
FROM python:3.12-slim-bookworm AS vision-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config libopencv-dev \
 && rm -rf /var/lib/apt/lists/*
COPY native/vision /src/native/vision
RUN pip wheel --no-cache-dir /src/native/vision -w /wheels

FROM python:3.12-slim-bookworm

# Runtime OpenCV libraries for the extension (core, imgproc, imgcodecs only).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles several GB of CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY search/requirements.txt search/requirements.txt
COPY model_server/requirements.txt model_server/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY --from=vision-build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels \
 && python -c "import omni_vision; print('omni_vision', omni_vision.opencv_version)"

# Bake the embedding model and tokenizers into the image so pods start without
# internet egress (HF_HUB_OFFLINE below).
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
S('sentence-transformers/all-MiniLM-L6-v2'); \
from transformers import AutoTokenizer as T; \
T.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && chmod -R a+rX /opt/hf-cache

COPY backend backend
COPY search search
COPY model_server/__init__.py model_server/inference_client.py model_server/vision_client.py model_server/
COPY samples samples

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TMPDIR=/tmp

EXPOSE 8000
CMD ["uvicorn", "backend.main:server_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--http", "backend.uvicorn_protocol:PacedHttpToolsProtocol", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--ws-max-size", "1048576", \
     "--timeout-graceful-shutdown", "20"]

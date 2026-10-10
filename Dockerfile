# Heart Disease Risk API - production image
#   docker build -t heart-risk-api:latest .
#   docker run -p 8000:8000 heart-risk-api:latest
# The model artefact (models/model.joblib + model_metadata.json) must exist,
# i.e. run `make train` (or the CI "train" job) before building.

ARG BASE_IMAGE=python:3.11-slim

# ---------- stage 1: build an isolated virtualenv with only serving deps
FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && python -c "import cardiorisk.serving.app"

# ---------- stage 2: slim runtime, non-root
FROM ${BASE_IMAGE} AS runtime
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI + scikit-learn)" \
      org.opencontainers.image.authors="Shankaraguru Ramdurg (2025AE05523)" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    LOG_LEVEL=INFO \
    PORT=8000

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY models/model.joblib models/model_metadata.json /app/models/

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python3", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["sh", "-c", "exec uvicorn cardiorisk.serving.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]

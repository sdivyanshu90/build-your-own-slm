# =============================================================================
# Multi-stage build for the BYO-SLM inference API.
#   * builder  — installs CPU PyTorch + the package into an isolated venv.
#   * runtime  — minimal, non-root image carrying only the venv + source.
# Build:  docker build -t byo-slm:latest .
# For a CUDA image, swap the base images and the torch index URL.
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build tooling required by some wheels; removed by virtue of stage discard.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual environment we will copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU PyTorch first (large, rarely-changing layer -> cached).
RUN pip install --upgrade pip \
    && pip install "torch>=2.1,<3.0" --index-url https://download.pytorch.org/whl/cpu

# Install the application. Copy only what the build needs for good caching.
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SLM_ENV=production \
    SLM_LOG_JSON=true \
    SLM_API_HOST=0.0.0.0 \
    SLM_API_PORT=8000 \
    SLM_MODEL_DIR=/app/checkpoints/tiny

# Run as an unprivileged user.
RUN groupadd --system --gid 1001 slm \
    && useradd --system --uid 1001 --gid slm --create-home slm

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Model artifacts are mounted at runtime (kept out of the image).
RUN mkdir -p /app/checkpoints && chown -R slm:slm /app

USER slm
EXPOSE 8000

# Liveness check used by orchestrators and `docker compose`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

# Production server: a factory app behind uvicorn. Scale via replicas, not
# workers, when serving on a single GPU.
CMD ["uvicorn", "slm.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

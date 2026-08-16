# Phase 7 — Production Dockerfile (multi-stage)
# Stage 1: Build
FROM python:3.12-slim AS builder

SHELL ["/bin/bash", "-cu"]

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m minigpt

# Set working directory
WORKDIR /opt/minigpt_llm

# Copy requirements and install deps (layer caching)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
COPY minigpt_llm /opt/minigpt_llm/minigpt_llm
COPY serving /opt/minigpt_llm/serving
COPY training /opt/minigpt_llm/training
COPY inference /opt/minigpt_llm/inference
COPY configs /opt/minigpt_llm/configs
COPY scripts /opt/minigpt_llm/scripts
COPY pyproject.toml /opt/minigpt_llm/pyproject.toml
COPY README.md LICENSE /opt/minigpt_llm/

# Create mount points for external artifacts
RUN mkdir -p /opt/minigpt_llm/checkpoints /opt/minigpt_llm/tokenizer /opt/minigpt_llm/data

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime (minimal)
FROM python:3.12-slim AS runtime

SHELL ["/bin/bash", "-cu"]

# Create non-root user
RUN useradd -m minigpt || true

# Install minimal runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/minigpt_llm

# Copy only essential artifacts from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/minigpt_llm/minigpt_llm /opt/minigpt_llm/minigpt_llm
COPY --from=builder /opt/minigpt_llm/serving /opt/minigpt_llm/serving
COPY --from=builder /opt/minigpt_llm/inference /opt/minigpt_llm/inference
COPY --from=builder /opt/minigpt_llm/configs /opt/minigpt_llm/configs
RUN mkdir -p /opt/minigpt_llm/checkpoints /opt/minigpt_llm/tokenizer /opt/minigpt_llm/data && \
    chown -R minigpt:minigpt /opt/minigpt_llm

# Set as minigpt user
USER minigpt

# Expose FastAPI port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint
CMD ["uvicorn", "serving.server:app", "--host", "0.0.0.0", "--port", "8080"]

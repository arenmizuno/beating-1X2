# Multi-stage build for the beating-1X2 serving container.
#
# Stage 1 builds wheels so the runtime image carries no compiler toolchain.
# Only requirements.txt is installed -- not requirements-ops.txt -- so Prefect,
# DVC, Evidently and Streamlit stay out of the served API image.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim AS runtime

# libgomp is LightGBM's OpenMP runtime. It is absent from -slim, and without it
# `import lightgbm` fails at container start rather than at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY params.yaml ./
COPY data/mappings/ ./data/mappings/

# Run as a non-root user. mlruns/ and data/ arrive as mounts at runtime.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/mlruns /app/data /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Hits the real readiness signal: /health reports "degraded" rather than failing
# when no champion is registered, so the check asserts on the payload.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"status": *"ok"' || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

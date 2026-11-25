# syntax=docker/dockerfile:1

# =============================================================================
# Artery Stenosis Detection — Backend Dockerfile (Optimized)
# =============================================================================
# Build:   docker build -t artery-stenosis-backend .
# Run:     docker run -p 8000:8000 --env-file .env artery-stenosis-backend
# =============================================================================

# -----------------------------
# Stage 1: Build dependencies
# -----------------------------
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level dependencies required by OpenCV, PyTorch, and psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install runtime-only Python dependencies (CPU-only PyTorch, no dev/test deps)
# Installed in separate layers for caching efficiency:
# 1. torch (CPU-only) - largest dependency, changes infrequently
# 2. Everything else together - faster resolution, fewer layers
COPY requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --no-deps torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic[email] pydantic-settings \
    sqlalchemy psycopg[binary] pwdlib[argon2] pyjwt[crypto] \
    opencv-python-headless numpy python-multipart filelock fsspec jinja2 markupsafe mpmath networkx

# -----------------------------
# Stage 2: Application runtime
# -----------------------------
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install OS-level dependencies (same as deps stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source code
COPY backend/ /app/backend/

# Copy alembic.ini to app root for Alembic to find it
COPY backend/alembic.ini /app/alembic.ini

# Copy ML model checkpoints (required for inference)
COPY AIML_testing/outputs/models/ /app/AIML_testing/outputs/models/

# Create upload directory (mounted volume at runtime for persistence)
RUN mkdir -p /app/uploads/images

# Create non-root user for security
RUN groupadd -r arteryuser && useradd -r -g arteryuser arteryuser
RUN chown -R arteryuser:arteryuser /app/uploads /app/AIML_testing
USER arteryuser

# Expose FastAPI port
EXPOSE 8000

# Health check using Python (curl not available in python:3.12-slim)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Start the application
CMD ["uvicorn", "app.extended_main:app", "--host", "0.0.0.0", "--port", "8000"]

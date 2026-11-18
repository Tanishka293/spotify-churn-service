# -----------------------------------------------------------------------------
# Stage 1: Base image with security hardening and non-root user
# -----------------------------------------------------------------------------
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install security patches & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and group (UID 1000)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY --chown=appuser:appgroup . .

# Ensure data and artifact directories are writable by appuser
RUN mkdir -p /app/data /app/artifacts && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose microservice port
EXPOSE 8000

# Docker healthcheck using the liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production server entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

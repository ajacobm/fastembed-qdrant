FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .
COPY src/ src/

# Create virtual environment and install dependencies
RUN uv venv --seed && \
    uv pip install -r requirements.txt

# Add missing dependencies for health checking
RUN uv pip install psutil GPUtil

# Generate protobuf files
RUN uv run python -m grpc_tools.protoc \
    --proto_path=src/proto \
    --python_out=src/proto \
    --grpc_python_out=src \
    src/proto/embed.proto

# Create necessary directories
RUN mkdir -p .cache

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CACHE_DIR=/app/.cache
ENV PYTHONPATH=/app/src

# Default environment variables
ENV GRPC_PORT=50051
ENV HTTP_PORT=50052
ENV DEFAULT_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV USE_CUDA=false
ENV DEFAULT_CHUNK_SIZE=512
ENV DEFAULT_CHUNK_OVERLAP=50
ENV START_MODE=both

# Enhanced health check configuration with container boundary awareness
ENV GRPC_STARTUP_TIMEOUT=300
ENV GRPC_HEALTH_CHECK_INTERVAL=5
ENV HTTP_STARTUP_TIMEOUT=60

# Health check system configuration
ENV HEALTH_ENABLE=true
ENV HEALTH_CHECK_INTERVAL=30
ENV HEALTH_STARTUP_GRACE=120
ENV HEALTH_CONTAINER_MONITORING=true
ENV HEALTH_MEMORY_THRESHOLD=85.0
ENV HEALTH_CPU_THRESHOLD=90.0
ENV HEALTH_DISK_THRESHOLD=90.0

# Observability configuration
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
ENV LOG_OUTPUT=console

# Expose both ports
EXPOSE 50051 50052

# Copy startup scripts and environment loader
COPY start_unified_improved.sh /app/start_unified_improved.sh
COPY load_env.sh /app/load_env.sh
RUN chmod +x /app/start_unified_improved.sh /app/load_env.sh

# Docker health check with enhanced container boundary awareness
# Uses a layered approach: basic connectivity -> detailed health -> full diagnostic
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:${HTTP_PORT:-50052}/health || \
        curl -f http://localhost:${HTTP_PORT:-50052}/health/detailed || \
        curl -f http://localhost:${HTTP_PORT:-50052}/liveness || exit 1

# Health check script for manual use and Kubernetes
COPY docker-health-check.sh /app/docker-health-check.sh
RUN chmod +x /app/docker-health-check.sh

# Use the improved unified startup script
CMD ["/app/start_unified_improved.sh"]
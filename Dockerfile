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

# Expose both ports
EXPOSE 50051 50052

# Copy startup scripts and environment loader
COPY start_unified_improved.sh /app/start_unified_improved.sh
COPY load_env.sh /app/load_env.sh
RUN chmod +x /app/start_unified_improved.sh /app/load_env.sh

# Health check configuration
ENV GRPC_STARTUP_TIMEOUT=300
ENV GRPC_HEALTH_CHECK_INTERVAL=5
ENV HTTP_STARTUP_TIMEOUT=60

# Health check for Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${HTTP_PORT:-50052}/health || exit 1

# Use the improved unified startup script
CMD ["/app/start_unified_improved.sh"]

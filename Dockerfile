FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && add-apt-repository ppa:graphics-drivers/ppa \
    && apt-get update \
    && apt-get install -y \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
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
    --python_out=src \
    --grpc_python_out=src \
    src/proto/embed.proto

# Create necessary directories
RUN mkdir -p .cache

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CACHE_DIR=/app/.cache
# Set Python path to include src directory
ENV PYTHONPATH=/app/src

# Default environment variables
ENV GRPC_PORT=50051
ENV HTTP_PORT=50052
ENV DEFAULT_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV USE_CUDA=true
ENV DEFAULT_CHUNK_SIZE=512
ENV DEFAULT_CHUNK_OVERLAP=50

# Expose ports
EXPOSE 50051 50052

# Default command runs the enhanced gRPC server
# Use: docker run -e "START_HTTP=true" to run HTTP server instead
CMD ["sh", "-c", "if [ \"$START_HTTP\" = \"true\" ]; then uv run python src/http_server.py; else uv run python src/enhanced_server.py; fi"]
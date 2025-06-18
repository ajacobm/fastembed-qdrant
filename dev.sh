#!/bin/bash

# FastEmbed-Qdrant Development Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
UV_PATH="${HOME}/.local/bin/uv"

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  FastEmbed-Qdrant Development Helper${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [command] [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  setup           - Install uv and sync dependencies"
    echo "  protobuf        - Regenerate protobuf files"
    echo "  test-grpc       - Run gRPC client example"
    echo "  test-http       - Test HTTP endpoints"
    echo "  test-observability - Test observability features"
    echo "  demo-observability - Run observability demo"
    echo "  run-grpc        - Run gRPC server locally"
    echo "  run-http        - Run HTTP server locally"
    echo "  run-http-obs    - Run HTTP server with observability"
    echo "  run-enhanced    - Run enhanced server with observability"
    echo "  docker-build    - Build Docker image"
    echo "  docker-up       - Start services with Docker Compose"
    echo "  docker-down     - Stop Docker Compose services"
    echo "  docker-logs     - Show Docker logs"
    echo "  qdrant-status   - Check Qdrant connection"
    echo "  clean           - Clean up cache and temporary files"
    echo "  help            - Show this help message"
    echo ""
}

ensure_uv() {
    if [ ! -f "$UV_PATH" ]; then
        echo -e "${YELLOW}Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:$PATH"
    fi
}

setup_project() {
    print_header
    echo -e "${GREEN}Setting up FastEmbed-Qdrant development environment...${NC}"
    
    ensure_uv
    
    echo -e "${YELLOW}Syncing dependencies with uv...${NC}"
    export PATH="${HOME}/.local/bin:$PATH"
    uv sync
    
    echo -e "${YELLOW}Generating protobuf files...${NC}"
    generate_protobuf
    
    echo -e "${GREEN}Setup complete!${NC}"
}

generate_protobuf() {
    echo -e "${YELLOW}Generating protobuf files...${NC}"
    
    export PATH="${HOME}/.local/bin:$PATH"
    uv run python -m grpc_tools.protoc \
        --proto_path=src/proto \
        --python_out=src \
        --grpc_python_out=src \
        src/proto/embed.proto
    
    echo -e "${GREEN}Protobuf files generated successfully${NC}"
}

run_grpc_server() {
    echo -e "${GREEN}Starting FastEmbed gRPC server...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    # Set default environment variables
    export GRPC_PORT=50051
    export DEFAULT_MODEL="BAAI/bge-base-en-v1.5"
    export USE_CUDA=true
    
    echo -e "${YELLOW}Server configuration:${NC}"
    echo "  gRPC Port: $GRPC_PORT"
    echo "  Default Model: $DEFAULT_MODEL"
    echo "  Use CUDA: $USE_CUDA"
    echo ""
    
    uv run python src/enhanced_server.py
}

run_http_server_observability() {
    echo -e "${GREEN}Starting FastEmbed HTTP server with observability...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    # Set default environment variables for observability
    export HTTP_PORT=8000
    export GRPC_PORT=50051
    export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
    export LOG_FORMAT=${LOG_FORMAT:-"json"}
    export LOG_OUTPUT=${LOG_OUTPUT:-"console"}
    
    echo -e "${YELLOW}HTTP Server with Observability configuration:${NC}"
    echo "  HTTP Port: $HTTP_PORT"
    echo "  gRPC Port: $GRPC_PORT"
    echo "  API Docs: http://localhost:$HTTP_PORT/docs"
    echo "  Log Level: $LOG_LEVEL"
    echo "  Log Format: $LOG_FORMAT"
    echo "  Log Output: $LOG_OUTPUT"
    echo ""
    
    uv run python src/http_server_with_observability.py
}

run_enhanced_server() {
    echo -e "${GREEN}Starting FastEmbed Enhanced Server with Observability...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    # Set default environment variables for observability
    export GRPC_PORT=50051
    export DEFAULT_MODEL="BAAI/bge-base-en-v1.5"
    export USE_CUDA=true
    export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
    export LOG_FORMAT=${LOG_FORMAT:-"json"}
    export LOG_OUTPUT=${LOG_OUTPUT:-"console"}
    
    echo -e "${YELLOW}Enhanced Server configuration:${NC}"
    echo "  gRPC Port: $GRPC_PORT"
    echo "  Default Model: $DEFAULT_MODEL"
    echo "  Use CUDA: $USE_CUDA"
    echo "  Log Level: $LOG_LEVEL"
    echo "  Log Format: $LOG_FORMAT"
    echo "  Log Output: $LOG_OUTPUT"
    echo ""
    
    uv run python src/enhanced_server_with_observability.py
}

run_http_server() {
    echo -e "${GREEN}Starting FastEmbed HTTP server...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    # Set default environment variables
    export HTTP_PORT=8000
    export GRPC_PORT=50051
    
    echo -e "${YELLOW}Server configuration:${NC}"
    echo "  HTTP Port: $HTTP_PORT"
    echo "  gRPC Port: $GRPC_PORT"
    echo "  API Docs: http://localhost:$HTTP_PORT/docs"
    echo ""
    
    uv run python src/http_server.py
}

test_grpc() {
    echo -e "${GREEN}Running gRPC client example...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    echo -e "${YELLOW}Make sure the gRPC server is running on localhost:50051${NC}"
    sleep 2
    
    uv run python src/client_example.py
}

test_http() {
    echo -e "${GREEN}Testing HTTP endpoints...${NC}"
    
    BASE_URL="http://localhost:8000"
    
    echo -e "${YELLOW}Testing health endpoint...${NC}"
    curl -s "$BASE_URL/health" | jq . || echo "Health check failed"
    echo ""
    
    echo -e "${YELLOW}Testing status endpoint...${NC}"
    curl -s "$BASE_URL/status" | jq . || echo "Status check failed"
    echo ""
    
    echo -e "${YELLOW}Testing models endpoint...${NC}"
    curl -s "$BASE_URL/models" | jq '. | length' || echo "Models check failed"
    echo ""
    
    echo -e "${YELLOW}Testing embeddings endpoint...${NC}"
    curl -s -X POST "$BASE_URL/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"texts":["Hello world","Test embedding"],"model_name":"BAAI/bge-base-en-v1.5"}' | \
        jq '.embeddings | length' || echo "Embeddings test failed"
    echo ""
    
    echo -e "${GREEN}HTTP tests completed${NC}"
}

test_observability() {
    echo -e "${GREEN}Testing observability features...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    echo -e "${YELLOW}Running basic observability tests...${NC}"
    
    uv run python -c "
import sys
from pathlib import Path

# Add src to path
src_path = Path('.') / 'src'
sys.path.insert(0, str(src_path))

from observability import setup_logging, get_logger, RequestContext, LogConfig
import time

print('🧪 Testing observability system...')

# Test JSON logging
setup_logging(level='INFO', format_type='json', output_type='console')
logger = get_logger('test')
logger.info('Observability test started', component='dev_script')

# Test request context
with RequestContext('test-123', method='TestMethod', model='test-model') as ctx:
    logger.info('Request processing test', operation='basic_test')
    time.sleep(0.1)

logger.info('Observability test completed successfully')
print('✅ Observability system working correctly!')
"
    
    echo -e "${GREEN}Observability tests completed${NC}"
}

demo_observability() {
    echo -e "${GREEN}Running observability demo...${NC}"
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    export PATH="${HOME}/.local/bin:$PATH"
    
    uv run python demo_observability_phase1.py
}

docker_build() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t fastembed-qdrant:latest .
    echo -e "${GREEN}Docker image built successfully${NC}"
}

docker_up() {
    echo -e "${GREEN}Starting services with Docker Compose...${NC}"
    
    if [ "$1" = "http" ]; then
        docker-compose --profile http up -d
        echo -e "${YELLOW}Services started with HTTP API${NC}"
        echo -e "${YELLOW}gRPC: localhost:50051${NC}"
        echo -e "${YELLOW}HTTP: localhost:50052${NC}"
        echo -e "${YELLOW}Qdrant: localhost:6333${NC}"
    else
        docker-compose up -d
        echo -e "${YELLOW}Services started (gRPC only)${NC}"
        echo -e "${YELLOW}gRPC: localhost:50051${NC}"
        echo -e "${YELLOW}Qdrant: localhost:6333${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Use '$0 docker-logs' to see logs${NC}"
    echo -e "${YELLOW}Use '$0 qdrant-status' to check Qdrant${NC}"
}

docker_down() {
    echo -e "${GREEN}Stopping Docker Compose services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped${NC}"
}

docker_logs() {
    echo -e "${GREEN}Showing Docker Compose logs...${NC}"
    docker-compose logs -f
}

check_qdrant() {
    echo -e "${GREEN}Checking Qdrant status...${NC}"
    
    QDRANT_URL="http://localhost:6333"
    
    echo -e "${YELLOW}Cluster info:${NC}"
    curl -s "$QDRANT_URL/cluster" | jq . || echo "Qdrant not accessible"
    
    echo -e "${YELLOW}Collections:${NC}" 
    curl -s "$QDRANT_URL/collections" | jq . || echo "Cannot get collections"
    
    echo -e "${YELLOW}Telemetry:${NC}"
    curl -s "$QDRANT_URL/telemetry" | jq . || echo "Cannot get telemetry"
}

clean_project() {
    echo -e "${GREEN}Cleaning up project files...${NC}"
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean build artifacts
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ 2>/dev/null || true
    
    # Clean uv cache (optional)
    if [ "$1" = "all" ]; then
        export PATH="${HOME}/.local/bin:$PATH"
        uv cache clean 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Main command handling
case "${1:-help}" in
    setup)
        setup_project
        ;;
    protobuf)
        generate_protobuf
        ;;
    test-grpc)
        test_grpc
        ;;
    test-http)
        test_http
        ;;
    test-observability)
        test_observability
        ;;
    demo-observability)
        demo_observability
        ;;
    run-grpc)
        run_grpc_server
        ;;
    run-http)
        run_http_server
        ;;
    run-http-obs)
        run_http_server_observability
        ;;
    run-enhanced)
        run_enhanced_server
        ;;
    docker-build)
        docker_build
        ;;
    docker-up)
        docker_up "$2"
        ;;
    docker-down)
        docker_down
        ;;
    docker-logs)
        docker_logs
        ;;
    qdrant-status)
        check_qdrant
        ;;
    clean)
        clean_project "$2"
        ;;
    help|*)
        print_usage
        ;;
esac
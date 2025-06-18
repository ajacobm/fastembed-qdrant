#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    source ./load_env.sh
else
    echo "Warning: .env file not found, using defaults"
fi

echo "FastEmbed Unified Server Startup"
echo "================================"
echo "START_MODE: ${START_MODE:-both}"
echo "GRPC_PORT: ${GRPC_PORT:-50051}"
echo "HTTP_PORT: ${HTTP_PORT:-8000}"
echo "USE_CUDA: ${USE_CUDA:-false}"
echo "DEFAULT_MODEL: ${DEFAULT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Function to start gRPC server
start_grpc() {
    echo "Starting FastEmbed gRPC server on port ${GRPC_PORT:-50051}..."
    exec uv run python src/enhanced_server.py
}

# Function to start HTTP server
start_http() {
    echo "Starting FastEmbed HTTP API server on port ${HTTP_PORT:-50052}..."
    exec uv run python src/http_server.py --host 0.0.0.0 --port ${HTTP_PORT:-50052} --grpc-host localhost --grpc-port ${GRPC_PORT:-50051}
}

# Function to start both servers
start_both() {
    echo "Starting both gRPC and HTTP servers..."
    
    # Start gRPC server in background
    echo "Starting gRPC server..."
    uv run python src/enhanced_server.py &
    GRPC_PID=$!
    
    # Wait for gRPC server to be ready
    echo "Waiting for gRPC server to initialize..."
    sleep 10
    
    # Check if gRPC server is still running
    if ! kill -0 $GRPC_PID 2>/dev/null; then
        echo "ERROR: gRPC server failed to start"
        exit 1
    fi
    
    echo "Starting HTTP API server..."
    uv run python src/http_server.py --host 0.0.0.0 --port ${HTTP_PORT:-50052} --grpc-host localhost --grpc-port ${GRPC_PORT:-50051} &
    HTTP_PID=$!
    
    # Wait for HTTP server to be ready
    sleep 5
    
    # Check if HTTP server is still running
    if ! kill -0 $HTTP_PID 2>/dev/null; then
        echo "ERROR: HTTP server failed to start"
        kill $GRPC_PID 2>/dev/null
        exit 1
    fi
    
    echo ""
    echo "✅ Both servers are running:"
    echo "   - gRPC Server: localhost:${GRPC_PORT:-50051}"
    echo "   - HTTP API:    localhost:${HTTP_PORT:-50052}"
    echo ""
    echo "Health check URLs:"
    echo "   - HTTP: http://localhost:${HTTP_PORT:-50052}/health"
    echo "   - Status: http://localhost:${HTTP_PORT:-50052}/status"
    echo ""
    
    # Wait for any process to exit
    wait -n
    
    # If we get here, one of the processes exited
    echo "One of the servers exited, stopping all..."
    kill $GRPC_PID $HTTP_PID 2>/dev/null || true
    exit 1
}

# Main startup logic
case "${START_MODE:-both}" in
    "grpc")
        start_grpc
        ;;
    "http")
        start_http
        ;;
    "both")
        start_both
        ;;
    *)
        echo "ERROR: Invalid START_MODE: ${START_MODE}"
        echo "Valid options: grpc, http, both"
        exit 1
        ;;
esac
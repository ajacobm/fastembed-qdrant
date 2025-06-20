#!/bin/bash
set -e

# Load environment variables from .env file
echo "Current working directory: $(pwd)"
echo "Files in current directory:"
ls -la | head -10

if [ -f "/app/.env" ]; then
    echo "Loading environment variables from /app/.env..."
    source /app/load_env.sh
elif [ -f ".env" ]; then
    echo "Loading environment variables from ./.env..."
    source ./load_env.sh
else
    echo "⚠️  Warning: .env file not found in /app/.env or ./.env, using defaults"
    echo "Available files:"
    find /app -name "*.env" -o -name "*env*" | head -5
fi

echo "FastEmbed Unified Server Startup (Improved)"
echo "=========================================="
echo "START_MODE: ${START_MODE:-both}"
echo "GRPC_PORT: ${GRPC_PORT:-50051}"
echo "HTTP_PORT: ${HTTP_PORT:-50052}"
echo "USE_CUDA: ${USE_CUDA:-false}"
echo "DEFAULT_MODEL: ${DEFAULT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Configuration for health checking
GRPC_STARTUP_TIMEOUT=${GRPC_STARTUP_TIMEOUT:-300}  # 5 minutes for model loading
GRPC_HEALTH_CHECK_INTERVAL=${GRPC_HEALTH_CHECK_INTERVAL:-5}  # Check every 5 seconds
HTTP_STARTUP_TIMEOUT=${HTTP_STARTUP_TIMEOUT:-60}   # 1 minute for HTTP server

# Function to check if gRPC port is listening
check_grpc_port() {
    local port="${1:-50051}"
    if timeout 2 bash -c "</dev/tcp/localhost/${port}" 2>/dev/null; then
        echo "✓ gRPC port ${port} is listening"
        return 0
    else
        echo "✗ gRPC port ${port} not listening"
        return 1
    fi
}

# Function to check if gRPC service is responding (basic connectivity)
check_grpc_connectivity() {
    local port="${1:-50051}"
    PYTHONPATH=/app/src python3 -c "
import grpc
import sys

try:
    channel = grpc.insecure_channel('localhost:${port}')
    # Just try to create a connection - don't call any RPCs yet
    grpc.channel_ready_future(channel).result(timeout=3)
    print('✓ gRPC channel connectivity OK')
    channel.close()
    sys.exit(0)
except Exception as e:
    print(f'✗ gRPC connectivity failed: {e}')
    sys.exit(1)
" 2>/dev/null
}

# Function to check if gRPC server is fully healthy (model loaded)
check_grpc_health() {
    local port="${1:-50051}"
    # Try to connect to gRPC server using Python client
    PYTHONPATH=/app/src python3 -c "
import grpc
import sys
import os
sys.path.insert(0, '/app/src')

try:
    from embed_pb2_grpc import EmbeddingServiceStub
    from embed_pb2 import StatusRequest
    
    channel = grpc.insecure_channel('localhost:${port}')
    stub = EmbeddingServiceStub(channel)
    # Try to call GetStatus with a short timeout
    response = stub.GetStatus(StatusRequest(), timeout=5.0)
    print(f'✓ gRPC server fully healthy - Model: {response.current_model}')
    channel.close()
    sys.exit(0)
except ImportError as ie:
    print(f'✗ Import error (protobuf not ready): {ie}')
    sys.exit(1)
except grpc.RpcError as ge:
    if ge.code() == grpc.StatusCode.UNAVAILABLE:
        print(f'✗ gRPC server not available: {ge.details()}')
    elif ge.code() == grpc.StatusCode.UNIMPLEMENTED:
        print(f'✗ gRPC service not implemented yet: {ge.details()}')
    else:
        print(f'✗ gRPC error: {ge.code()} - {ge.details()}')
    sys.exit(1)
except Exception as e:
    print(f'✗ gRPC server not ready: {e}')
    sys.exit(1)
" 2>/dev/null
}

# Layered health checking function
check_grpc_layered() {
    local port="${1:-50051}"
    local stage="${2:-full}"
    
    case "$stage" in
        "port")
            check_grpc_port "$port"
            ;;
        "connectivity")
            check_grpc_connectivity "$port"
            ;;
        "full")
            check_grpc_health "$port"
            ;;
        *)
            echo "Invalid stage: $stage"
            return 1
            ;;
    esac
}

# Function to check if HTTP server is healthy
check_http_health() {
    local port="${1:-50052}"
    curl -f -s "http://localhost:${port}/health" > /dev/null 2>&1
}

# Function to wait for gRPC server to be ready with layered checking
wait_for_grpc() {
    local port="${1:-50051}"
    local timeout="${2:-300}"
    local interval="${3:-5}"
    
    echo "Waiting for gRPC server on port ${port} to become healthy..."
    echo "Timeout: ${timeout}s, Check interval: ${interval}s"
    echo "Using layered health checking: Port -> Connectivity -> Full Service"
    
    local elapsed=0
    local checks=0
    local port_ready=false
    local connectivity_ready=false
    local service_ready=false
    
    while [ $elapsed -lt $timeout ]; do
        checks=$((checks + 1))
        echo ""
        echo "=== Health check #${checks} (elapsed: ${elapsed}s) ==="
        
        # Stage 1: Check if port is listening
        if [ "$port_ready" = false ]; then
            echo "🔍 Stage 1: Checking if gRPC port is listening..."
            if check_grpc_layered "$port" "port"; then
                port_ready=true
                echo "✅ Stage 1 PASSED: gRPC port is listening"
            else
                echo "⏳ Stage 1: gRPC port not ready yet"
            fi
        fi
        
        # Stage 2: Check basic gRPC connectivity (only if port is ready)
        if [ "$port_ready" = true ] && [ "$connectivity_ready" = false ]; then
            echo "🔍 Stage 2: Checking gRPC connectivity..."
            if check_grpc_layered "$port" "connectivity"; then
                connectivity_ready=true
                echo "✅ Stage 2 PASSED: gRPC connectivity established"
            else
                echo "⏳ Stage 2: gRPC connectivity not ready"
            fi
        fi
        
        # Stage 3: Check full service health (only if connectivity is ready)
        if [ "$connectivity_ready" = true ] && [ "$service_ready" = false ]; then
            echo "🔍 Stage 3: Checking full gRPC service health..."
            if check_grpc_layered "$port" "full"; then
                service_ready=true
                echo "✅ Stage 3 PASSED: gRPC service fully healthy!"
                echo ""
                echo "🎉 All stages completed successfully!"
                return 0
            else
                echo "⏳ Stage 3: gRPC service not fully ready (model may still be loading)"
            fi
        fi
        
        # Show progress summary
        echo ""
        echo "Progress Summary:"
        echo "  Port Listening:    $([ "$port_ready" = true ] && echo "✅ READY" || echo "❌ NOT READY")"
        echo "  gRPC Connectivity: $([ "$connectivity_ready" = true ] && echo "✅ READY" || echo "❌ NOT READY")"
        echo "  Service Health:    $([ "$service_ready" = true ] && echo "✅ READY" || echo "❌ NOT READY")"
        
        echo "Waiting ${interval}s before next check..."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    
    echo ""
    echo "❌ Timeout waiting for gRPC server to become healthy after ${timeout}s"
    echo ""
    echo "Final Status:"
    echo "  Port Listening:    $([ "$port_ready" = true ] && echo "✅ READY" || echo "❌ FAILED")"
    echo "  gRPC Connectivity: $([ "$connectivity_ready" = true ] && echo "✅ READY" || echo "❌ FAILED")"
    echo "  Service Health:    $([ "$service_ready" = true ] && echo "✅ READY" || echo "❌ FAILED")"
    
    # Provide diagnostic suggestions
    if [ "$port_ready" = false ]; then
        echo ""
        echo "🔧 DIAGNOSTIC: gRPC server port not listening"
        echo "   - Server may not be starting properly"
        echo "   - Check server logs for startup errors"
        echo "   - Verify GRPC_PORT environment variable"
    elif [ "$connectivity_ready" = false ]; then
        echo ""
        echo "🔧 DIAGNOSTIC: gRPC port listening but no connectivity"
        echo "   - Server may be in startup phase"
        echo "   - Check if server process is running"
        echo "   - Network connectivity issues possible"
    elif [ "$service_ready" = false ]; then
        echo ""
        echo "🔧 DIAGNOSTIC: gRPC server responding but service not healthy"
        echo "   - Model loading may still be in progress"
        echo "   - Check for CUDA/model loading errors"
        echo "   - Consider increasing GRPC_STARTUP_TIMEOUT"
    fi
    
    return 1
}

# Function to wait for HTTP server to be ready
wait_for_http() {
    local port="${1:-50052}"
    local timeout="${2:-60}"
    local interval="${3:-2}"
    
    echo "Waiting for HTTP server on port ${port} to become healthy..."
    
    local elapsed=0
    local checks=0
    
    while [ $elapsed -lt $timeout ]; do
        checks=$((checks + 1))
        echo "HTTP health check #${checks} (elapsed: ${elapsed}s)..."
        
        if check_http_health "$port"; then
            echo "✅ HTTP server is healthy and ready!"
            return 0
        fi
        
        echo "HTTP server not ready yet, waiting ${interval}s..."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    
    echo "❌ Timeout waiting for HTTP server to become healthy after ${timeout}s"
    return 1
}

# Function to start gRPC server
start_grpc() {
    echo "Starting FastEmbed gRPC server on port ${GRPC_PORT:-50051}..."
    exec uv run python src/grpc_server.py
}

# Function to start HTTP server
start_http() {
    echo "Starting FastEmbed HTTP API server on port ${HTTP_PORT:-50052}..."
    exec uv run python src/http_server.py --host 0.0.0.0 --port ${HTTP_PORT:-50052} --grpc-host localhost --grpc-port ${GRPC_PORT:-50051}
}

# Function to start both servers with proper coordination
start_both() {
    echo "Starting both gRPC and HTTP servers with intelligent health checking..."
    
    # Start gRPC server in background
    echo "🚀 Starting gRPC server..."
    uv run python src/grpc_server.py &
    GRPC_PID=$!
    
    # Wait for gRPC server to be ready
    echo "⏳ Waiting for gRPC server to initialize and load models..."
    if ! wait_for_grpc "${GRPC_PORT:-50051}" "$GRPC_STARTUP_TIMEOUT" "$GRPC_HEALTH_CHECK_INTERVAL"; then
        echo "❌ Failed to start gRPC server within timeout"
        kill $GRPC_PID 2>/dev/null || true
        exit 1
    fi
    
    # Check if gRPC server is still running
    if ! kill -0 $GRPC_PID 2>/dev/null; then
        echo "❌ gRPC server process died"
        exit 1
    fi
    
    echo "🚀 Starting HTTP API server..."
    uv run python src/http_server.py --host 0.0.0.0 --port ${HTTP_PORT:-50052} --grpc-host localhost --grpc-port ${GRPC_PORT:-50051} &
    HTTP_PID=$!
    
    # Wait for HTTP server to be ready
    echo "⏳ Waiting for HTTP server to start..."
    if ! wait_for_http "${HTTP_PORT:-50052}" "$HTTP_STARTUP_TIMEOUT"; then
        echo "❌ Failed to start HTTP server within timeout"
        kill $GRPC_PID $HTTP_PID 2>/dev/null || true
        exit 1
    fi
    
    # Check if HTTP server is still running
    if ! kill -0 $HTTP_PID 2>/dev/null; then
        echo "❌ HTTP server process died"
        kill $GRPC_PID 2>/dev/null || true
        exit 1
    fi
    
    echo ""
    echo "🎉 Both servers are running and healthy:"
    echo "   - gRPC Server: localhost:${GRPC_PORT:-50051}"
    echo "   - HTTP API:    localhost:${HTTP_PORT:-50052}"
    echo ""
    echo "Health check URLs:"
    echo "   - HTTP: http://localhost:${HTTP_PORT:-50052}/health"
    echo "   - Status: http://localhost:${HTTP_PORT:-50052}/status"
    echo ""
    echo "🔄 Monitoring server health..."
    
    # Monitor both processes
    while true; do
        # Check if either process has died
        if ! kill -0 $GRPC_PID 2>/dev/null; then
            echo "❌ gRPC server process died, stopping all services..."
            kill $HTTP_PID 2>/dev/null || true
            exit 1
        fi
        
        if ! kill -0 $HTTP_PID 2>/dev/null; then
            echo "❌ HTTP server process died, stopping all services..."
            kill $GRPC_PID 2>/dev/null || true
            exit 1
        fi
        
        # Optional: Periodic health checks
        sleep 30
        echo "🔍 Periodic health check - both servers still running"
    done
}

# Cleanup function
cleanup() {
    echo "🛑 Shutting down servers..."
    if [ ! -z "$GRPC_PID" ]; then
        kill $GRPC_PID 2>/dev/null || true
    fi
    if [ ! -z "$HTTP_PID" ]; then
        kill $HTTP_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

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
        echo "❌ Invalid START_MODE: ${START_MODE}"
        echo "Valid options: grpc, http, both"
        exit 1
        ;;
esac

#!/bin/bash

# Comprehensive Container Startup Test Script
# This script tests the new layered health checking and diagnostics

set -e

echo "🧪 FastEmbed Container Startup Test Suite"
echo "=========================================="

# Configuration
CONTAINER_NAME="fastembed-v2"
TEST_TIMEOUT=400  # 6+ minutes for full model loading
CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "[$(date '+%H:%M:%S')] $1"
}

# Function to check if container exists
check_container_exists() {
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        return 0
    else
        return 1
    fi
}

# Function to check container status
get_container_status() {
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep "${CONTAINER_NAME}" | awk '{print $2, $3, $4, $5}' || echo "Not Running"
}

# Function to check container health
get_container_health() {
    docker inspect "${CONTAINER_NAME}" --format="{{.State.Health.Status}}" 2>/dev/null || echo "no health check"
}

# Function to test HTTP endpoints
test_http_endpoints() {
    local port=${1:-50052}
    
    echo "🌐 Testing HTTP endpoints on port ${port}..."
    
    # Test health endpoint
    if curl -f -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        log "${GREEN}✅ Health endpoint responding${NC}"
    else
        log "${RED}❌ Health endpoint not responding${NC}"
        return 1
    fi
    
    # Test status endpoint
    if curl -f -s "http://localhost:${port}/status" > /dev/null 2>&1; then
        log "${GREEN}✅ Status endpoint responding${NC}"
        # Get and display status info
        local status_info=$(curl -s "http://localhost:${port}/status" | jq -r '.current_model // "No model info"' 2>/dev/null || echo "Status retrieved")
        log "${BLUE}ℹ️  Current model: ${status_info}${NC}"
    else
        log "${RED}❌ Status endpoint not responding${NC}"
    fi
    
    return 0
}

# Function to test gRPC endpoint
test_grpc_endpoint() {
    local port=${1:-50051}
    
    echo "🔌 Testing gRPC endpoint on port ${port}..."
    
    # Test if port is listening
    if timeout 5 bash -c "</dev/tcp/localhost/${port}" 2>/dev/null; then
        log "${GREEN}✅ gRPC port ${port} is listening${NC}"
        return 0
    else
        log "${RED}❌ gRPC port ${port} not listening${NC}"
        return 1
    fi
}

# Function to monitor container logs in real-time
monitor_logs() {
    local duration=${1:-60}
    
    echo "📋 Monitoring container logs for ${duration} seconds..."
    echo "Press Ctrl+C to stop early"
    
    timeout "${duration}" docker logs -f "${CONTAINER_NAME}" 2>&1 | while read line; do
        # Highlight important log lines
        if echo "$line" | grep -q "✅\|✓"; then
            echo -e "${GREEN}${line}${NC}"
        elif echo "$line" | grep -q "❌\|✗\|ERROR\|Failed"; then
            echo -e "${RED}${line}${NC}"
        elif echo "$line" | grep -q "⏳\|Health check\|Stage"; then
            echo -e "${YELLOW}${line}${NC}"
        elif echo "$line" | grep -q "🔍\|Starting\|Loading"; then
            echo -e "${BLUE}${line}${NC}"
        else
            echo "$line"
        fi
    done || true
}

# Function to run full container test
run_container_test() {
    echo "🚀 Starting comprehensive container test..."
    
    # Step 1: Stop existing container
    if check_container_exists; then
        log "${YELLOW}Stopping existing container...${NC}"
        docker-compose down
        sleep 2
    fi
    
    # Step 2: Start container
    log "${BLUE}Starting container with docker-compose...${NC}"
    docker-compose up -d
    
    # Step 3: Monitor startup process
    log "${BLUE}Monitoring startup process...${NC}"
    
    local elapsed=0
    local startup_completed=false
    
    while [ $elapsed -lt $TEST_TIMEOUT ]; do
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        
        local status=$(get_container_status)
        local health=$(get_container_health)
        
        log "${BLUE}Status: ${status} | Health: ${health} | Elapsed: ${elapsed}s${NC}"
        
        # Check if container is healthy
        if [ "$health" = "healthy" ]; then
            log "${GREEN}🎉 Container is healthy!${NC}"
            startup_completed=true
            break
        elif [ "$health" = "unhealthy" ]; then
            log "${RED}❌ Container is unhealthy${NC}"
            break
        elif echo "$status" | grep -q "Exited"; then
            log "${RED}❌ Container exited unexpectedly${NC}"
            break
        fi
        
        # Show progress indicator
        if [ $((elapsed % 30)) -eq 0 ]; then
            log "${YELLOW}⏳ Still waiting for container to become healthy...${NC}"
        fi
    done
    
    # Step 4: Test endpoints if startup completed
    if [ "$startup_completed" = true ]; then
        echo ""
        log "${GREEN}🧪 Running endpoint tests...${NC}"
        sleep 5  # Give services a moment to fully stabilize
        
        test_http_endpoints 50052
        test_grpc_endpoint 50051
        
        log "${GREEN}✅ All tests completed successfully!${NC}"
        return 0
    else
        log "${RED}❌ Container startup failed or timed out${NC}"
        return 1
    fi
}

# Function to show container diagnostic info
show_diagnostics() {
    echo ""
    echo "🔧 Container Diagnostics"
    echo "======================="
    
    if check_container_exists; then
        echo "Container Status:"
        docker ps -a | grep "${CONTAINER_NAME}" || echo "Container not found"
        
        echo ""
        echo "Container Health:"
        docker inspect "${CONTAINER_NAME}" --format="{{.State.Health}}" 2>/dev/null || echo "No health info"
        
        echo ""
        echo "Recent Logs (last 50 lines):"
        docker logs "${CONTAINER_NAME}" --tail 50 2>&1
        
        echo ""
        echo "Resource Usage:"
        docker stats "${CONTAINER_NAME}" --no-stream 2>/dev/null || echo "Cannot get stats"
    else
        echo "Container does not exist"
    fi
}

# Main menu
case "${1:-test}" in
    "test"|"run")
        run_container_test
        ;;
    "logs")
        duration=${2:-120}
        monitor_logs "$duration"
        ;;
    "endpoints")
        test_http_endpoints 50052
        test_grpc_endpoint 50051
        ;;
    "diagnostics"|"diag")
        show_diagnostics
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  test        - Run full container startup test (default)"
        echo "  logs [sec]  - Monitor container logs for specified seconds (default: 120)"
        echo "  endpoints   - Test HTTP and gRPC endpoints"
        echo "  diagnostics - Show container diagnostic information"
        echo "  help        - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

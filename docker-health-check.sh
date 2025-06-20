#!/bin/bash
set -e

# Docker Health Check Script with Container Boundary Awareness
# This script provides layered health checking for different scenarios:
# 1. Docker HEALTHCHECK (basic connectivity)
# 2. Manual container health check (detailed)
# 3. Kubernetes probes (readiness/liveness)
# 4. External monitoring (comprehensive)

# Configuration
HTTP_PORT=${HTTP_PORT:-50052}
GRPC_PORT=${GRPC_PORT:-50051}
CHECK_LEVEL=${CHECK_LEVEL:-basic}  # basic, detailed, diagnostic
TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
VERBOSE=${VERBOSE:-false}

# Colors for output (if terminal supports it)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$VERBOSE" == "true" ]]; then
        case $level in
            "INFO")  echo -e "${BLUE}[$timestamp] INFO:${NC} $message" ;;
            "WARN")  echo -e "${YELLOW}[$timestamp] WARN:${NC} $message" ;;
            "ERROR") echo -e "${RED}[$timestamp] ERROR:${NC} $message" ;;
            "SUCCESS") echo -e "${GREEN}[$timestamp] SUCCESS:${NC} $message" ;;
            *) echo "[$timestamp] $level: $message" ;;
        esac
    fi
}

# Function to check if we're running inside a container
detect_container_environment() {
    if [[ -f "/.dockerenv" ]] || [[ -n "${KUBERNETES_SERVICE_HOST}" ]]; then
        echo "container"
    else
        echo "host"
    fi
}

# Function to check basic HTTP connectivity
check_http_basic() {
    local url="http://localhost:${HTTP_PORT}/health"
    log "INFO" "Checking basic HTTP health at $url"
    
    if curl -f -s --max-time "$TIMEOUT" "$url" >/dev/null 2>&1; then
        log "SUCCESS" "Basic HTTP health check passed"
        return 0
    else
        log "ERROR" "Basic HTTP health check failed"
        return 1
    fi
}

# Function to check detailed HTTP health
check_http_detailed() {
    local url="http://localhost:${HTTP_PORT}/health/detailed"
    log "INFO" "Checking detailed HTTP health at $url"
    
    local response=$(curl -f -s --max-time "$TIMEOUT" "$url" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        local status=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        log "SUCCESS" "Detailed HTTP health check passed - Status: $status"
        
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  Uptime: {data.get(\"uptime_seconds\", 0):.1f}s')
    print(f'  Components: {len(data.get(\"components\", {}))}')
    for comp, info in data.get('components', {}).items():
        print(f'    {comp}: {info.get(\"status\", \"unknown\")}')
    resource_usage = data.get('resource_usage', {})
    if resource_usage:
        print(f'  CPU: {resource_usage.get(\"cpu_percent\", 0):.1f}%')
        print(f'  Memory: {resource_usage.get(\"memory_percent\", 0):.1f}%')
        if 'gpu_memory_percent' in resource_usage:
            print(f'  GPU Memory: {resource_usage[\"gpu_memory_percent\"]:.1f}%')
except:
    print('  (Could not parse detailed response)')
" 2>/dev/null
        fi
        return 0
    else
        log "ERROR" "Detailed HTTP health check failed"
        return 1
    fi
}

# Function to check diagnostic health
check_http_diagnostic() {
    local url="http://localhost:${HTTP_PORT}/health/diagnostic"
    log "INFO" "Checking diagnostic HTTP health at $url"
    
    local response=$(curl -f -s --max-time "$TIMEOUT" "$url" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        log "SUCCESS" "Diagnostic HTTP health check passed"
        
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  Overall Status: {data.get(\"overall_status\", \"unknown\")}')
    print(f'  Message: {data.get(\"message\", \"No message\")}')
    print(f'  Version: {data.get(\"version\", \"unknown\")}')
    
    container_info = data.get('container_info', {})
    runtime_info = container_info.get('runtime', {})
    if runtime_info:
        print(f'  Container Runtime: {runtime_info.get(\"type\", \"unknown\")}')
        if runtime_info.get('container_id'):
            print(f'  Container ID: {runtime_info[\"container_id\"][:12]}...')
        if runtime_info.get('pod_name'):
            print(f'  Pod Name: {runtime_info[\"pod_name\"]}')
        if runtime_info.get('namespace'):
            print(f'  Namespace: {runtime_info[\"namespace\"]}')
    
    recommendations = data.get('recommendations', [])
    if recommendations:
        print('  Recommendations:')
        for rec in recommendations[:3]:
            print(f'    - {rec}')
except:
    print('  (Could not parse diagnostic response)')
" 2>/dev/null
        fi
        return 0
    else
        log "ERROR" "Diagnostic HTTP health check failed"
        return 1
    fi
}

# Function to check readiness probe
check_readiness() {
    local url="http://localhost:${HTTP_PORT}/readiness"
    log "INFO" "Checking readiness at $url"
    
    local response=$(curl -f -s --max-time "$TIMEOUT" "$url" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        local status=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        log "SUCCESS" "Readiness probe passed - Status: $status"
        return 0
    else
        log "ERROR" "Readiness probe failed"
        return 1
    fi
}

# Function to check liveness probe
check_liveness() {
    local url="http://localhost:${HTTP_PORT}/liveness"
    log "INFO" "Checking liveness at $url"
    
    local response=$(curl -f -s --max-time "$TIMEOUT" "$url" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        local status=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        log "SUCCESS" "Liveness probe passed - Status: $status"
        return 0
    else
        log "ERROR" "Liveness probe failed"
        return 1
    fi
}

# Function to check if HTTP port is listening
check_port_listening() {
    log "INFO" "Checking if HTTP port $HTTP_PORT is listening"
    
    if timeout 3 bash -c "</dev/tcp/localhost/${HTTP_PORT}" 2>/dev/null; then
        log "SUCCESS" "HTTP port $HTTP_PORT is listening"
        return 0
    else
        log "ERROR" "HTTP port $HTTP_PORT is not listening"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Docker Health Check Script with Container Boundary Awareness"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --level LEVEL     Health check level: basic, detailed, diagnostic (default: basic)"
    echo "  --port PORT       HTTP port to check (default: $HTTP_PORT)"
    echo "  --timeout SECONDS Timeout for each check (default: $TIMEOUT)"
    echo "  --verbose         Enable verbose output"
    echo "  --readiness       Check readiness probe (Kubernetes)"
    echo "  --liveness        Check liveness probe (Kubernetes)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Basic health check (Docker HEALTHCHECK compatible)"
    echo "  $0 --level detailed --verbose  # Detailed health check with verbose output"
    echo "  $0 --readiness             # Kubernetes readiness probe"
    echo "  $0 --liveness              # Kubernetes liveness probe"
    echo ""
    echo "Container Environment Detection:"
    local env=$(detect_container_environment)
    echo "  Current environment: $env"
    echo ""
    echo "Health Check Endpoints:"
    echo "  Basic:      http://localhost:${HTTP_PORT}/health"
    echo "  Detailed:   http://localhost:${HTTP_PORT}/health/detailed"
    echo "  Diagnostic: http://localhost:${HTTP_PORT}/health/diagnostic"
    echo "  Readiness:  http://localhost:${HTTP_PORT}/readiness"
    echo "  Liveness:   http://localhost:${HTTP_PORT}/liveness"
}

# Parse command line arguments
PROBE_TYPE="health"
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            CHECK_LEVEL="$2"
            shift 2
            ;;
        --port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --readiness)
            PROBE_TYPE="readiness"
            shift
            ;;
        --liveness)
            PROBE_TYPE="liveness"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main health check logic
main() {
    log "INFO" "Starting health check - Level: $CHECK_LEVEL, Probe: $PROBE_TYPE, Environment: $(detect_container_environment)"
    
    # First, check if port is listening
    if ! check_port_listening; then
        log "ERROR" "Port check failed - service may not be running"
        exit 1
    fi
    
    # Execute the appropriate check based on probe type
    case $PROBE_TYPE in
        "readiness")
            if check_readiness; then
                log "SUCCESS" "Readiness probe successful"
                exit 0
            else
                log "ERROR" "Readiness probe failed"
                exit 1
            fi
            ;;
        "liveness")
            if check_liveness; then
                log "SUCCESS" "Liveness probe successful"
                exit 0
            else
                log "ERROR" "Liveness probe failed"
                exit 1
            fi
            ;;
        "health")
            case $CHECK_LEVEL in
                "basic")
                    if check_http_basic; then
                        log "SUCCESS" "Basic health check successful"
                        exit 0
                    else
                        log "ERROR" "Basic health check failed"
                        exit 1
                    fi
                    ;;
                "detailed")
                    if check_http_detailed; then
                        log "SUCCESS" "Detailed health check successful"
                        exit 0
                    else
                        log "ERROR" "Detailed health check failed"
                        exit 1
                    fi
                    ;;
                "diagnostic")
                    if check_http_diagnostic; then
                        log "SUCCESS" "Diagnostic health check successful"
                        exit 0
                    else
                        log "ERROR" "Diagnostic health check failed"
                        exit 1
                    fi
                    ;;
                *)
                    log "ERROR" "Invalid check level: $CHECK_LEVEL"
                    show_usage
                    exit 1
                    ;;
            esac
            ;;
        *)
            log "ERROR" "Invalid probe type: $PROBE_TYPE"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
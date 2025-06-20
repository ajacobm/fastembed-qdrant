# FastEmbed Server Startup Fix - Summary

## Problem Identified

The original issue was a **race condition** where:
1. **gRPC server takes 111+ seconds** to load models (especially CUDA models)
2. **HTTP server starts after only 10 seconds** (hardcoded sleep)
3. **HTTP health checks fail** because gRPC isn't ready yet
4. **Container appears unhealthy** despite eventually becoming ready

## Root Cause Analysis

From the logs, we observed:
- Model loading takes significant time: `"duration_ms": 111064.89`
- HTTP server starts too early and fails health checks
- gRPC connection refused errors during startup period
- Eventually both servers work correctly once gRPC is fully loaded

## Comprehensive Solution

### 1. Improved Startup Script (`start_unified_improved.sh`)

**Key Features:**
- **Intelligent Health Checking**: Uses actual gRPC calls instead of just port checks
- **Configurable Timeouts**: 5-minute default for model loading
- **Progressive Health Monitoring**: Checks every 5 seconds with detailed logging
- **Process Monitoring**: Ensures both servers stay healthy
- **Proper Error Handling**: Cleanup and graceful shutdown

**Configuration Variables:**
```bash
GRPC_STARTUP_TIMEOUT=300     # 5 minutes for model loading
GRPC_HEALTH_CHECK_INTERVAL=5 # Check every 5 seconds
HTTP_STARTUP_TIMEOUT=60      # 1 minute for HTTP server
```

### 2. Docker Configuration Updates

**Dockerfile Changes:**
- Added `python3-pip` for health check scripts
- Longer health check start period: `start-period=120s`
- More retries: `retries=5`
- Environment variables for timeout configuration

**docker-compose.yml Changes:**
- Extended health check start period: `start_period: 300s` (5 minutes)
- Increased retries: `retries: 10`
- Better health check intervals

### 3. Health Check Functions

**gRPC Health Check:**
```python
# Actually calls GetStatus RPC to verify service is ready
response = stub.GetStatus(StatusRequest(), timeout=5.0)
```

**HTTP Health Check:**
```bash
curl -f -s "http://localhost:${port}/health"
```

## Usage Instructions

### Quick Start (Recommended)
```bash
# Stop existing container first
docker-compose down

# Rebuild and start with improved configuration
docker-compose up --build
```

### Alternative: Manual Testing
```bash
# Test locally first
./start_unified_improved.sh

# Or test specific modes
START_MODE=grpc ./start_unified_improved.sh
START_MODE=http ./start_unified_improved.sh
```

### Monitoring Startup Progress
```bash
# Watch container logs with timestamps
docker-compose logs -f --timestamps fastembed

# Check health status
docker-compose ps
curl http://localhost:50052/health
```

## Expected Behavior

### Startup Sequence:
1. **gRPC server starts** (background process)
2. **Health checks begin** every 5 seconds
3. **Model loading progress** shown in logs
4. **gRPC becomes ready** after model loading completes
5. **HTTP server starts** only after gRPC is confirmed healthy
6. **Both servers monitored** continuously

### Timing Expectations:
- **CUDA Model Loading**: 90-120 seconds (as observed)
- **Total Startup Time**: 2-3 minutes for first run
- **Subsequent Starts**: Faster due to cached models
- **Health Check Success**: Only after both servers are truly ready

## Troubleshooting

### If Container Still Shows Unhealthy:
```bash
# Check detailed logs
docker logs fastembed-v2 --tail 100

# Test health endpoints manually
curl -v http://localhost:50052/health
curl -v http://localhost:50052/status
```

### If gRPC Server Doesn't Start:
```bash
# Check CUDA availability
docker exec -it fastembed-v2 nvidia-smi

# Test gRPC connection manually
docker exec -it fastembed-v2 python3 -c "
import grpc
from proto.embed_pb2_grpc import EmbeddingServiceStub
from proto.embed_pb2 import StatusRequest
channel = grpc.insecure_channel('localhost:50051')
stub = EmbeddingServiceStub(channel)
print(stub.GetStatus(StatusRequest()))
"
```

### Performance Optimization:
```bash
# Use persistent model cache
mkdir -p ./cache
# Models will be cached after first download

# Monitor resource usage
docker stats fastembed-v2
```

## Configuration Options

### Environment Variables:
```bash
# Startup timeouts
GRPC_STARTUP_TIMEOUT=300
GRPC_HEALTH_CHECK_INTERVAL=5
HTTP_STARTUP_TIMEOUT=60

# Server configuration
START_MODE=both              # Options: grpc, http, both
DEFAULT_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_CUDA=true               # Enable CUDA acceleration
```

### Docker Compose Profiles:
```bash
# Unified service (default)
docker-compose up

# Separate services
docker-compose --profile separate up
```

## Benefits of This Fix

1. **Eliminates Race Condition**: HTTP waits for gRPC to be truly ready
2. **Proper Health Reporting**: Container shows healthy only when actually ready
3. **Better Observability**: Detailed startup progress logging
4. **Graceful Failure**: Clear error messages and cleanup
5. **Production Ready**: Configurable timeouts and monitoring
6. **Backward Compatible**: Existing environment variables still work

## Testing Verification

The fix addresses the specific error pattern observed:
```
"grpc_code": "UNAVAILABLE", 
"grpc_details": "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:50051: Failed to connect to remote host: connect: Connection refused (111)"
```

This error will no longer occur because HTTP server waits for gRPC to be ready before starting.

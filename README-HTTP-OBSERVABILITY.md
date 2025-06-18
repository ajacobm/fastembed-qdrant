# FastEmbed HTTP Server with Observability

The FastEmbed HTTP server provides a RESTful API interface to the gRPC FastEmbed service with comprehensive observability features including structured logging, request correlation, and performance monitoring.

## Features

### 🌐 HTTP API
- **RESTful endpoints** for all FastEmbed operations
- **OpenAPI documentation** available at `/docs`
- **CORS support** for cross-origin requests
- **File upload support** for document processing
- **Model management** endpoints

### 📊 Observability
- **Request correlation** with unique request IDs
- **Structured logging** in JSON or text format
- **Performance timing** for all operations
- **Error correlation** across request boundaries
- **Operation context tracking** for nested operations
- **gRPC operation monitoring** with error translation

### 🔧 Configuration
- **Environment-based** configuration
- **Multiple log formats** (JSON for production, text for development)
- **Flexible output options** (console, file, remote)
- **Log rotation** and retention policies

## Quick Start

### 1. Start the gRPC Server
```bash
./dev.sh run-grpc
```

### 2. Start HTTP Server with Observability
```bash
./dev.sh run-http-obs
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Get server status
curl http://localhost:8000/status

# Generate embeddings
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Test embedding"],
    "model_name": "BAAI/bge-base-en-v1.5"
  }'
```

### 4. View API Documentation
Open http://localhost:8000/docs in your browser for interactive API documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with version info |
| GET | `/health` | Health check endpoint |
| GET | `/status` | Server status and configuration |
| GET | `/models` | List available models |
| POST | `/embeddings` | Generate text embeddings |
| POST | `/process-file` | Process uploaded files |
| POST | `/update-model-config` | Update model configuration |

## Observability Features

### Request Correlation
Every HTTP request gets a unique correlation ID that appears in all related log entries:

```json
{
  "timestamp": "2025-06-18T05:50:00.000Z",
  "level": "info",
  "message": "HTTP request started",
  "request_id": "abc123-def456",
  "method": "POST",
  "path": "/embeddings",
  "client_ip": "192.168.1.100",
  "user_agent": "curl/8.0.1"
}
```

### Operation Context
Operations are tracked with nested contexts:

```json
{
  "timestamp": "2025-06-18T05:50:01.000Z", 
  "level": "info",
  "message": "Operation in progress",
  "request_id": "abc123-def456",
  "operation": "grpc_get_embeddings",
  "text_count": 2,
  "model_name": "BAAI/bge-base-en-v1.5",
  "duration": 0.045
}
```

### Performance Monitoring
All operations include timing information:

```json
{
  "timestamp": "2025-06-18T05:50:01.100Z",
  "level": "info", 
  "message": "HTTP request completed",
  "request_id": "abc123-def456",
  "status_code": 200,
  "duration": 0.156,
  "duration_ms": 156.0
}
```

### Error Correlation
Errors are tracked with full context and stack traces:

```json
{
  "timestamp": "2025-06-18T05:50:02.000Z",
  "level": "error",
  "message": "gRPC error getting embeddings", 
  "request_id": "abc123-def456",
  "grpc_code": "UNAVAILABLE",
  "grpc_details": "Connection refused",
  "text_count": 2,
  "model_name": "BAAI/bge-base-en-v1.5"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | 8000 | HTTP server port |
| `GRPC_PORT` | 50051 | gRPC server port |
| `LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | json | Log format (json, text) |
| `LOG_OUTPUT` | console | Output destination (console, file) |
| `LOG_FILE_PATH` | ./logs/fastembed.log | Log file path |
| `LOG_MAX_SIZE` | 100MB | Maximum log file size |
| `LOG_BACKUP_COUNT` | 5 | Number of log file backups |

### Example Configuration
```bash
# Development
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
export LOG_OUTPUT=console

# Production  
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export LOG_OUTPUT=file
export LOG_FILE_PATH=/var/log/fastembed/api.log
```

## Development Tools

### Run with Observability
```bash
# Start HTTP server with observability
./dev.sh run-http-obs

# Test observability components
./dev.sh test-observability

# Run observability demo
python demo_http_observability.py
```

### Log Format Examples

#### Text Format (Development)
```
2025-06-18T05:50:00.000Z [info] HTTP request started [fastembed.api] 
  request_id=abc123 method=POST path=/embeddings client_ip=192.168.1.100
```

#### JSON Format (Production)
```json
{
  "timestamp": "2025-06-18T05:50:00.000Z",
  "level": "info",
  "message": "HTTP request started",
  "logger": "fastembed.api",
  "request_id": "abc123",
  "method": "POST", 
  "path": "/embeddings",
  "client_ip": "192.168.1.100"
}
```

## Integration Examples

### Using with curl
```bash
# Generate embeddings with request tracking
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-custom-id" \
  -d '{
    "texts": ["Hello world"],
    "model_name": "BAAI/bge-base-en-v1.5"
  }'
```

### Using with Python
```python
import httpx
import asyncio

async def generate_embeddings():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/embeddings",
            json={
                "texts": ["Hello world", "Test embedding"],
                "model_name": "BAAI/bge-base-en-v1.5"
            },
            headers={"X-Request-ID": "python-client-001"}
        )
        
        # Request ID is returned in response headers
        request_id = response.headers.get("X-Request-ID")
        print(f"Request ID: {request_id}")
        
        return response.json()

# Run the async function
result = asyncio.run(generate_embeddings())
```

### File Upload Example
```python
import httpx

async def process_file():
    async with httpx.AsyncClient() as client:
        files = {"file": ("document.txt", "This is test content", "text/plain")}
        data = {
            "model_name": "BAAI/bge-base-en-v1.5",
            "store_in_qdrant": "true",
            "collection_name": "documents"
        }
        
        response = await client.post(
            "http://localhost:8000/process-file",
            files=files,
            data=data
        )
        
        return response.json()
```

## Monitoring and Alerting

### Health Check
```bash
# Simple health check
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "server_version": "2.0.0", 
  "timestamp": 1750226400.0
}
```

### Status Monitoring
```bash
# Detailed status
curl http://localhost:8000/status

# Response includes server health, model status, and configuration
{
  "server_version": "2.0.0",
  "current_model": "BAAI/bge-base-en-v1.5",
  "cuda_available": true,
  "qdrant_connected": true,
  "uptime_seconds": 3600
}
```

### Log Aggregation
For production deployments, configure log aggregation:

```yaml
# Docker Compose example
services:
  fastembed-http:
    image: fastembed:latest
    environment:
      - LOG_FORMAT=json
      - LOG_OUTPUT=console
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Performance Considerations

- **Request correlation** adds minimal overhead (~1ms per request)
- **Structured logging** is optimized for high-throughput scenarios
- **File output** with rotation prevents disk space issues
- **Async operations** maintain responsiveness under load

## Troubleshooting

### Common Issues

1. **gRPC Connection Refused**
   - Ensure gRPC server is running on port 50051
   - Check network connectivity between services

2. **High Log Volume**
   - Adjust LOG_LEVEL to WARNING or ERROR for production
   - Configure log rotation with appropriate limits

3. **Missing Correlation**
   - Verify request middleware is properly installed
   - Check that structured logging is configured

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
./dev.sh run-http-obs
```

## Next Steps

- **Phase 2**: Prometheus metrics integration
- **Health checks**: Advanced health monitoring
- **Tracing**: Distributed tracing with Jaeger
- **Dashboards**: Grafana dashboard templates
- **Alerts**: Alert rules for common issues
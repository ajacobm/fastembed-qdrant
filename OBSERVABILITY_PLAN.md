# FastEmbed Server Observability Implementation Plan

## Current State Analysis

The FastEmbed server currently has:
- ✅ Basic Python logging with timestamps and levels
- ✅ gRPC and HTTP endpoints (enhanced_server.py and http_server.py)
- ✅ Environment-based configuration system
- ✅ Health check endpoints
- ❌ **Missing**: Structured metrics collection
- ❌ **Missing**: Distributed tracing
- ❌ **Missing**: Performance monitoring
- ❌ **Missing**: Business metrics (requests/embeddings/failures)

---

## Phase 1: Enhanced Logging System

### 1.1 Structured Logging Implementation
**Goal**: Replace basic logging with structured JSON logs for better parsing and analysis.

**Components to Create**:
- `src/observability/logger.py` - Enhanced logger with structured format
- `src/observability/log_context.py` - Request context tracking
- `src/observability/log_config.py` - Environment-based log configuration

**Key Features**:
- JSON structured logging with consistent fields
- Request ID tracking through gRPC metadata
- Performance timing logs
- Error tracking with stack traces
- Configurable log levels per component
- Log correlation across gRPC/HTTP layers

### 1.2 Log Output Routing
**Options**:
1. **Console Output** (current) - JSON formatted for log aggregation
2. **File Output** - Rotated log files with configurable retention
3. **Remote Logging** - Direct integration with log aggregation services
4. **Multiple Sinks** - Console + File + Remote simultaneously

---

## Phase 2: Prometheus Metrics Integration

### 2.1 Metrics Collection Framework
**Components to Create**:
- `src/observability/metrics.py` - Prometheus metrics definitions
- `src/observability/middleware.py` - gRPC/HTTP interceptors for automatic metrics
- `src/observability/metrics_server.py` - Dedicated metrics endpoint

**Core Metrics Categories**:

#### 2.1.1 Request Metrics
```python
# Request counters
embedding_requests_total = Counter('embedding_requests_total', 
    ['method', 'model', 'status'])
file_processing_requests_total = Counter('file_processing_requests_total',
    ['method', 'status', 'store_destination'])

# Request duration
request_duration_seconds = Histogram('request_duration_seconds',
    ['method', 'model'], buckets=[0.1, 0.5, 1, 2, 5, 10, 30])

# Request size metrics
embedding_batch_size = Histogram('embedding_batch_size',
    ['model'], buckets=[1, 5, 10, 25, 50, 100, 500])
file_size_bytes = Histogram('file_size_bytes', 
    buckets=[1024, 10240, 102400, 1048576, 10485760])
```

#### 2.1.2 Business Logic Metrics
```python
# Embedding generation
embeddings_generated_total = Counter('embeddings_generated_total', ['model'])
chunks_processed_total = Counter('chunks_processed_total', ['model'])

# Model operations
model_load_duration_seconds = Histogram('model_load_duration_seconds', ['model'])
model_loads_total = Counter('model_loads_total', ['model', 'status'])

# Qdrant operations  
qdrant_operations_total = Counter('qdrant_operations_total', 
    ['operation', 'collection', 'status'])
qdrant_points_stored_total = Counter('qdrant_points_stored_total', ['collection'])
```

#### 2.1.3 System Resource Metrics
```python
# CUDA/GPU metrics
cuda_memory_usage_bytes = Gauge('cuda_memory_usage_bytes', ['device'])
cuda_utilization_percent = Gauge('cuda_utilization_percent', ['device'])

# Model memory usage
model_memory_usage_bytes = Gauge('model_memory_usage_bytes', ['model'])

# Cache metrics
cache_size_bytes = Gauge('cache_size_bytes')
cache_hit_rate = Gauge('cache_hit_rate', ['model'])
```

#### 2.1.4 Error and Health Metrics
```python
# Error tracking
errors_total = Counter('errors_total', ['component', 'error_type'])
grpc_errors_total = Counter('grpc_errors_total', ['method', 'code'])

# Health status
server_uptime_seconds = Counter('server_uptime_seconds')
qdrant_connection_status = Gauge('qdrant_connection_status')
model_load_status = Gauge('model_load_status', ['model'])
```

### 2.2 Metrics Collection Points

**Enhanced gRPC Server Integration**:
- Request/response interceptors for automatic timing
- Model loading/switching metrics
- File processing pipeline metrics
- Qdrant operation metrics

**HTTP Server Integration**:
- FastAPI middleware for HTTP metrics
- File upload size and processing metrics
- Health check metrics

---

## Phase 3: Implementation Architecture

### 3.1 Directory Structure
```
src/observability/
├── __init__.py
├── logger.py              # Enhanced structured logging
├── log_config.py         # Log configuration management
├── log_context.py        # Request context tracking
├── metrics.py            # Prometheus metrics definitions
├── middleware.py         # gRPC/HTTP interceptors
├── metrics_server.py     # Dedicated metrics endpoint
├── health_checks.py      # Enhanced health monitoring
└── exporters.py          # Optional: Custom metric exporters
```

### 3.2 Configuration Integration
**Environment Variables to Add**:
```bash
# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json  # json|text
LOG_OUTPUT=console  # console|file|remote
LOG_FILE_PATH=/app/logs/fastembed.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# Metrics Configuration  
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics
METRICS_SUBSYSTEM=fastembed

# Remote Logging (optional)
LOG_REMOTE_ENDPOINT=http://logstash:5000
LOG_REMOTE_FORMAT=json

# Tracing (future)
TRACING_ENABLED=false
JAEGER_ENDPOINT=http://jaeger:14268
```

### 3.3 Dependencies to Add
**Update pyproject.toml**:
```toml
dependencies = [
    # ... existing dependencies
    "prometheus-client>=0.19.0",
    "structlog>=23.2.0",
    "python-json-logger>=2.0.0",
    "contextvars>=2.4",  # For request context
]

[project.optional-dependencies]
monitoring = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0", 
    "opentelemetry-instrumentation-grpc>=0.41b0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
]
```

---

## Phase 4: Deployment Integration

### 4.1 Docker Configuration Updates
**Dockerfile additions**:
```dockerfile
# Expose metrics port
EXPOSE 9090

# Create logs directory
RUN mkdir -p /app/logs

# Add metrics endpoint environment
ENV METRICS_PORT=9090
ENV METRICS_ENABLED=true
```

### 4.2 Kubernetes Service Monitor
**For Kubernetes deployment**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fastembed-metrics
spec:
  selector:
    matchLabels:
      app: fastembed
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### 4.3 Log Aggregation Integration
**Options for log routing**:
1. **Kubernetes logs** → **Fluentd/Fluent Bit** → **ELK/EFK Stack**
2. **Docker logs** → **Promtail** → **Loki** → **Grafana**
3. **Direct HTTP** → **LogStash** → **Elasticsearch**

---

## Phase 5: Dashboard and Alerting

### 5.1 Grafana Dashboard Panels
1. **Request Overview**: Request rate, duration, error rate
2. **Model Performance**: Load times, embedding generation rate
3. **Resource Usage**: Memory, CUDA utilization, cache stats
4. **Qdrant Integration**: Connection status, points stored
5. **Error Analysis**: Error breakdown by type and component

### 5.2 Key Alert Rules
```yaml
- alert: HighErrorRate
  expr: rate(errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"

- alert: ModelLoadFailure  
  expr: model_load_status == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Model failed to load"

- alert: QdrantDisconnected
  expr: qdrant_connection_status == 0
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Qdrant connection lost"

- alert: HighMemoryUsage
  expr: cuda_memory_usage_bytes / cuda_memory_total_bytes > 0.9
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "CUDA memory usage high"
```

---

## Implementation Phases

### **Phase 1 (Week 1)**: Core Logging ⭐ **START HERE**
- [ ] Implement structured JSON logging
- [ ] Add request context tracking  
- [ ] Update existing log statements
- [ ] Create observability directory structure
- [ ] Test log output formats

**Priority**: HIGH - Foundation for all other observability

### **Phase 2 (Week 2)**: Basic Metrics
- [ ] Add Prometheus client library
- [ ] Implement core request/response metrics
- [ ] Create metrics endpoint
- [ ] Basic gRPC interceptors
- [ ] HTTP middleware for FastAPI

**Priority**: HIGH - Core monitoring capability

### **Phase 3 (Week 3)**: Advanced Metrics
- [ ] Add business logic metrics
- [ ] Implement resource monitoring
- [ ] CUDA/GPU metrics collection
- [ ] Qdrant operation metrics
- [ ] Model performance metrics

**Priority**: MEDIUM - Enhanced monitoring

### **Phase 4 (Week 4)**: Integration & Testing
- [ ] Docker/Kubernetes integration
- [ ] Dashboard creation
- [ ] Performance testing
- [ ] Alert rule configuration
- [ ] Documentation updates

**Priority**: MEDIUM - Production readiness

---

## Quick Start Commands

### Phase 1 Implementation
```bash
# Create observability structure
mkdir -p src/observability
touch src/observability/__init__.py

# Install dependencies
uv add prometheus-client structlog python-json-logger

# Create initial logging configuration
# (see Phase 1 implementation details below)
```

### Testing Plan
```bash
# Test structured logging
python src/enhanced_server.py --log-level debug --log-format json

# Test metrics endpoint
curl http://localhost:9090/metrics

# Load testing with metrics
hey -n 1000 -c 10 http://localhost:8080/health
```

---

## Phase 1 Implementation Details

### File: `src/observability/logger.py`
```python
"""Enhanced structured logging for FastEmbed server."""
import logging
import time
import uuid
from typing import Any, Dict, Optional
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

# Request context tracking
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

class ContextualJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter that includes request context."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add context fields
        log_record['request_id'] = request_id.get()
        log_record['user_id'] = user_id.get()
        log_record['timestamp'] = time.time()
        log_record['service'] = 'fastembed-server'
        
        # Add performance timing if available
        if hasattr(record, 'duration'):
            log_record['duration_ms'] = record.duration * 1000
            
        # Add model information if available
        if hasattr(record, 'model_name'):
            log_record['model_name'] = record.model_name

def setup_logging(
    level: str = 'INFO',
    format_type: str = 'json',
    output_type: str = 'console'
) -> logging.Logger:
    """Set up structured logging configuration."""
    
    logger = logging.getLogger('fastembed')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if format_type == 'json':
        formatter = ContextualJsonFormatter(
            '%(timestamp)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
        )
    
    if output_type == 'console':
        handler = logging.StreamHandler()
    else:  # file output
        handler = logging.FileHandler('/app/logs/fastembed.log')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f'fastembed.{name}')

def set_request_context(req_id: str, uid: Optional[str] = None) -> str:
    """Set request context for logging correlation."""
    if not req_id:
        req_id = str(uuid.uuid4())
    
    request_id.set(req_id)
    if uid:
        user_id.set(uid)
    
    return req_id

def clear_request_context() -> None:
    """Clear the current request context."""
    request_id.set(None)
    user_id.set(None)
```

---

## Next Steps

1. **Review the plan** and prioritize phases based on immediate needs
2. **Set up development environment** with observability dependencies
3. **Implement Phase 1** (Enhanced Logging) first as it provides immediate value
4. **Test integration** with existing FastEmbed server components
5. **Plan Phase 2** (Basic Metrics) implementation timeline

---

## Notes

- This plan is designed to be implemented incrementally without breaking existing functionality
- Each phase builds on the previous one
- Metrics and logging can be enabled/disabled via environment variables
- The implementation should be backward compatible
- Consider performance impact of metrics collection in production

---

## Resources

- [Prometheus Python Client Documentation](https://prometheus.github.io/client_python/)
- [Structlog Documentation](https://www.structlog.org/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/)
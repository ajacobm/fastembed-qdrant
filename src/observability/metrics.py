
from prometheus_client import Counter, Histogram, Gauge

# Request Metrics
embedding_requests_total = Counter('embedding_requests_total', 'Total embedding requests', ['method', 'model', 'status'])
file_processing_requests_total = Counter('file_processing_requests_total', 'Total file processing requests', ['method', 'status', 'store_destination'])
request_duration_seconds = Histogram('request_duration_seconds', 'Request duration in seconds', ['method', 'model'], buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
embedding_batch_size = Histogram('embedding_batch_size', 'Embedding batch size', ['model'], buckets=[1, 5, 10, 25, 50, 100, 500])
file_size_bytes = Histogram('file_size_bytes', 'File size in bytes', buckets=[1024, 10240, 102400, 1048576, 10485760])

# Business Logic Metrics
embeddings_generated_total = Counter('embeddings_generated_total', 'Total embeddings generated', ['model'])
chunks_processed_total = Counter('chunks_processed_total', 'Total chunks processed', ['model'])
model_load_duration_seconds = Histogram('model_load_duration_seconds', 'Model load duration in seconds', ['model'])
model_loads_total = Counter('model_loads_total', 'Total model loads', ['model', 'status'])
qdrant_operations_total = Counter('qdrant_operations_total', 'Total Qdrant operations', ['operation', 'collection', 'status'])
qdrant_points_stored_total = Counter('qdrant_points_stored_total', 'Total points stored in Qdrant', ['collection'])

# System Resource Metrics
cuda_memory_usage_bytes = Gauge('cuda_memory_usage_bytes', 'CUDA memory usage in bytes', ['device'])
cuda_utilization_percent = Gauge('cuda_utilization_percent', 'CUDA utilization in percent', ['device'])
model_memory_usage_bytes = Gauge('model_memory_usage_bytes', 'Model memory usage in bytes', ['model'])
cache_size_bytes = Gauge('cache_size_bytes', 'Cache size in bytes')
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate', ['model'])

# Error and Health Metrics
errors_total = Counter('errors_total', 'Total errors', ['component', 'error_type'])
grpc_errors_total = Counter('grpc_errors_total', 'Total gRPC errors', ['method', 'code'])
server_uptime_seconds = Counter('server_uptime_seconds', 'Server uptime in seconds')
qdrant_connection_status = Gauge('qdrant_connection_status', 'Qdrant connection status')
model_load_status = Gauge('model_load_status', 'Model load status', ['model'])

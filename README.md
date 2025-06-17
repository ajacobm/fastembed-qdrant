# FastEmbed-Qdrant Integration Server

A high-performance gRPC and HTTP server for text embeddings with **Qdrant integration**, **file streaming support**, and **CUDA acceleration**. Built on FastEmbed with additional features for production RAG workflows.

## 🚀 Features

- **File Streaming**: Process files of arbitrary length through streaming gRPC endpoints
- **Qdrant Integration**: Direct storage of embeddings in Qdrant vector database
- **Smart Chunking**: Model-aware text chunking with configurable overlap
- **Metadata Support**: Rich metadata support for documents and chunks
- **HTTP API**: FastAPI wrapper for easy integration with web applications
- **Environment Configuration**: Full configuration through environment variables
- **UV Package Management**: Modern Python packaging with uv
- **Enhanced Monitoring**: Status endpoints and health checks
- **CUDA Support**: GPU acceleration for faster embedding generation

## 📋 Supported Models

The server includes optimized configurations for 15+ FastEmbed models:

| Model | Dimensions | Max Length | Default Chunk Size | Size (GB) |
|-------|------------|------------|-------------------|-----------|
| BAAI/bge-small-en-v1.5 | 384 | 512 | 400 | 0.067 |
| BAAI/bge-base-en-v1.5 | 768 | 512 | 400 | 0.210 |
| BAAI/bge-large-en-v1.5 | 1024 | 512 | 400 | 1.200 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 256 | 200 | 0.090 |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 8192 | 2000 | 0.520 |

[View full model list](src/model_config.py)

## 🔧 Quick Start

### Using Docker (Recommended)

1. **Clone and start services:**
```bash
git clone <repository-url>
cd fastembed-qdrant

# Start gRPC server + Qdrant
docker-compose up -d

# Include HTTP API server  
docker-compose --profile http up -d
```

2. **Test the server:**
```bash
# Test gRPC
python src/client_example.py

# Test HTTP
curl http://localhost:8000/health
```

### Using UV (Development)

1. **Setup:**
```bash
./dev.sh setup
```

2. **Run servers:**
```bash
# gRPC server
./dev.sh run-grpc

# HTTP server  
./dev.sh run-http
```

## 🌐 Environment Configuration

Configure using environment variables:

### Basic Settings
```bash
GRPC_PORT=50051                    # gRPC server port
HTTP_PORT=8000                     # HTTP server port
DEFAULT_MODEL=BAAI/bge-base-en-v1.5 # Default embedding model
USE_CUDA=true                      # Enable CUDA acceleration
```

### Qdrant Integration
```bash
QDRANT_HOST=localhost              # Qdrant server host
QDRANT_PORT=6333                   # Qdrant HTTP port
QDRANT_COLLECTION=documents        # Default collection name
QDRANT_API_KEY=your-api-key        # Optional API key
```

### File Processing
```bash
DEFAULT_CHUNK_SIZE=512             # Default text chunk size
DEFAULT_CHUNK_OVERLAP=50           # Default chunk overlap
MAX_FILE_SIZE_MB=100               # Maximum file size limit
```

## 📡 API Usage

### HTTP API (Recommended)

#### Upload and Process Files
```bash
curl -X POST "http://localhost:8000/process-file" \\
  -F "file=@document.txt" \\
  -F "model_name=BAAI/bge-base-en-v1.5" \\
  -F "chunk_size=400" \\
  -F "store_in_qdrant=true" \\
  -F "collection_name=documents"
```

#### Get Embeddings for Text
```bash
curl -X POST "http://localhost:8000/embeddings" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Hello world", "FastEmbed embeddings"],
    "model_name": "BAAI/bge-base-en-v1.5"
  }'
```

#### Check Server Status
```bash
curl http://localhost:8000/status
```

### gRPC API

See [client_example.py](src/client_example.py) for comprehensive gRPC usage examples.

## 🔬 Qdrant Integration

The server integrates directly with Qdrant for vector storage and search:

### Features
- **Automatic Collection Management**: Creates collections with correct vector dimensions
- **Batch Processing**: Efficient batch insertion of embeddings  
- **Metadata Preservation**: Stores chunk metadata with embeddings
- **Search Capabilities**: Built-in similarity search functions

### Example: Vector Search
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Search for similar content
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    with_payload=True
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.payload['text']}")
    print(f"File: {result.payload['filename']}")
```

## 🧪 Development

Use the development helper script:

```bash
# Setup development environment
./dev.sh setup

# Generate protobuf files
./dev.sh protobuf

# Run tests
./dev.sh test-grpc
./dev.sh test-http

# Docker operations
./dev.sh docker-build
./dev.sh docker-up
./dev.sh docker-logs
```

## 📊 Performance

### Typical Performance (RTX 4090)
- **Text Embedding**: ~2000 texts/second (batch size 32)
- **File Processing**: ~1MB/second text processing + embedding
- **Qdrant Storage**: ~500 embeddings/second insertion

### Resource Requirements
- **Base Memory**: ~2GB (model + overhead)
- **CUDA Memory**: +2GB VRAM for model weights
- **Per 1K embeddings**: ~50MB additional memory

## 🚀 Production Deployment

### Docker Compose (Production)
```yaml
version: '3.8'
services:
  fastembed:
    image: your-registry/fastembed-qdrant:latest
    environment:
      - QDRANT_HOST=your-qdrant-host
      - QDRANT_API_KEY=your-api-key
      - DEFAULT_MODEL=BAAI/bge-large-en-v1.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

- [FastEmbed](https://github.com/qdrant/fastembed) - Core embedding library
- [Qdrant](https://github.com/qdrant/qdrant) - Vector database
- [UV](https://github.com/astral-sh/uv) - Python package manager
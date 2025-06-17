"""
Environment configuration for the FastEmbed server with Qdrant integration.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database connection."""
    host: str
    port: int
    collection_name: str
    api_key: Optional[str] = None
    use_ssl: bool = False
    timeout: int = 60
    
    @property
    def url(self) -> str:
        """Get the full Qdrant URL."""
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"


@dataclass
class ServerConfig:
    """Configuration for the FastEmbed server."""
    # Server settings
    grpc_port: int = 50051
    http_port: int = 8000
    max_workers: int = 10
    
    # FastEmbed settings
    default_model: str = "BAAI/bge-large-en-v1.5"
    use_cuda: bool = True
    max_model_length: int = 512
    embedding_threads: int = 8
    
    # File processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    
    # Cache directory
    cache_dir: str = "/app/.cache"
    
    # Qdrant configuration
    qdrant: Optional[QdrantConfig] = None


def load_config() -> ServerConfig:
    """Load server configuration from environment variables."""
    
    # Qdrant configuration
    qdrant_config = None
    qdrant_host = os.getenv("QDRANT_HOST")
    if qdrant_host:
        qdrant_config = QdrantConfig(
            host=qdrant_host,
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
            api_key=os.getenv("QDRANT_API_KEY"),
            use_ssl=os.getenv("QDRANT_USE_SSL", "false").lower() == "true",
            timeout=int(os.getenv("QDRANT_TIMEOUT", "60"))
        )
    
    config = ServerConfig(
        # Server settings
        grpc_port=int(os.getenv("GRPC_PORT", "50051")),
        http_port=int(os.getenv("HTTP_PORT", "8000")),
        max_workers=int(os.getenv("MAX_WORKERS", "10")),
        
        # FastEmbed settings
        default_model=os.getenv("DEFAULT_MODEL", "BAAI/bge-large-en-v1.5"),
        use_cuda=os.getenv("USE_CUDA", "true").lower() == "true",
        max_model_length=int(os.getenv("MAX_MODEL_LENGTH", "512")),
        embedding_threads=int(os.getenv("EMBEDDING_THREADS", "8")),
        
        # File processing settings
        chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50")),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
        
        cache_dir=os.getenv("CACHE_DIR", "/app/.cache"),
        
        # Qdrant
        qdrant=qdrant_config
    )
    
    return config


def get_env_variables() -> Dict[str, Any]:
    """Get all relevant environment variables for debugging."""
    env_vars = {}
    
    # Server configuration
    for key in ["GRPC_PORT", "HTTP_PORT", "MAX_WORKERS"]:
        env_vars[key] = os.getenv(key)
    
    # FastEmbed configuration
    for key in ["DEFAULT_MODEL", "USE_CUDA", "MAX_MODEL_LENGTH", "EMBEDDING_THREADS"]:
        env_vars[key] = os.getenv(key)
    
    # File processing
    for key in ["DEFAULT_CHUNK_SIZE", "DEFAULT_CHUNK_OVERLAP", "MAX_FILE_SIZE_MB"]:
        env_vars[key] = os.getenv(key)
    
    # Directories
    for key in ["CACHE_DIR"]:
        env_vars[key] = os.getenv(key)
    
    # Qdrant configuration
    for key in ["QDRANT_HOST", "QDRANT_PORT", "QDRANT_COLLECTION", "QDRANT_API_KEY", "QDRANT_USE_SSL", "QDRANT_TIMEOUT"]:
        value = os.getenv(key)
        if key == "QDRANT_API_KEY" and value:
            env_vars[key] = "*" * len(value)  # Mask API key for security
        else:
            env_vars[key] = value
    
    return env_vars
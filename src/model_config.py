"""
Model configuration and chunking strategies for FastEmbed models
Based on FastEmbed supported models documentation: https://qdrant.github.io/fastembed/examples/Supported_Models/
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific FastEmbed model."""
    model_name: str
    dimensions: int
    max_length: int
    default_chunk_size: int
    default_chunk_overlap: int
    size_gb: float
    license: str
    description: str


# FastEmbed model configurations based on the official documentation
FASTEMBED_MODELS = {
    # Small models (good for high-throughput scenarios)
    "BAAI/bge-small-en-v1.5": ModelConfig(
        model_name="BAAI/bge-small-en-v1.5",
        dimensions=384,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.067,
        license="mit",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "BAAI/bge-small-zh-v1.5": ModelConfig(
        model_name="BAAI/bge-small-zh-v1.5",
        dimensions=512,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.090,
        license="mit",
        description="Text embeddings, Unimodal (text), Chinese, 512 sequence length"
    ),
    "snowflake/snowflake-arctic-embed-xs": ModelConfig(
        model_name="snowflake/snowflake-arctic-embed-xs",
        dimensions=384,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.090,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        max_length=256,
        default_chunk_size=200,
        default_chunk_overlap=40,
        size_gb=0.090,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 256 sequence length"
    ),
    "jinaai/jina-embeddings-v2-small-en": ModelConfig(
        model_name="jinaai/jina-embeddings-v2-small-en",
        dimensions=512,
        max_length=8192,
        default_chunk_size=2000,
        default_chunk_overlap=200,
        size_gb=0.120,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 8192 sequence length"
    ),
    
    # Base models (balanced performance/size)
    "BAAI/bge-base-en-v1.5": ModelConfig(
        model_name="BAAI/bge-base-en-v1.5",
        dimensions=768,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.210,
        license="mit",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "snowflake/snowflake-arctic-embed-m": ModelConfig(
        model_name="snowflake/snowflake-arctic-embed-m",
        dimensions=768,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.430,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "nomic-ai/nomic-embed-text-v1.5": ModelConfig(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        dimensions=768,
        max_length=8192,
        default_chunk_size=2000,
        default_chunk_overlap=200,
        size_gb=0.520,
        license="apache-2.0",
        description="Text embeddings, Multimodal (text, image), English, 8192 sequence length"
    ),
    "jinaai/jina-embeddings-v2-base-en": ModelConfig(
        model_name="jinaai/jina-embeddings-v2-base-en",
        dimensions=768,
        max_length=8192,
        default_chunk_size=2000,
        default_chunk_overlap=200,
        size_gb=0.520,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 8192 sequence length"
    ),
    "snowflake/snowflake-arctic-embed-m-long": ModelConfig(
        model_name="snowflake/snowflake-arctic-embed-m-long",
        dimensions=768,
        max_length=2048,
        default_chunk_size=1600,
        default_chunk_overlap=150,
        size_gb=0.540,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 2048 sequence length"
    ),
    
    # Large models (best performance)
    "BAAI/bge-large-en-v1.5": ModelConfig(
        model_name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=1.200,
        license="mit",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "thenlper/gte-large": ModelConfig(
        model_name="thenlper/gte-large",
        dimensions=1024,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=1.200,
        license="mit",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "snowflake/snowflake-arctic-embed-l": ModelConfig(
        model_name="snowflake/snowflake-arctic-embed-l",
        dimensions=1024,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=1.020,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "mixedbread-ai/mxbai-embed-large-v1": ModelConfig(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        dimensions=1024,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=0.640,
        license="apache-2.0",
        description="Text embeddings, Unimodal (text), English, 512 sequence length"
    ),
    "intfloat/multilingual-e5-large": ModelConfig(
        model_name="intfloat/multilingual-e5-large",
        dimensions=1024,
        max_length=512,
        default_chunk_size=400,
        default_chunk_overlap=50,
        size_gb=2.240,
        license="mit",
        description="Text embeddings, Unimodal (text), Multilingual, 512 sequence length"
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get model configuration for a given model name."""
    return FASTEMBED_MODELS.get(model_name)


def get_optimal_chunk_size(model_name: str, custom_chunk_size: Optional[int] = None) -> Tuple[int, int]:
    """
    Get optimal chunk size and overlap for a model.
    
    Args:
        model_name: Name of the FastEmbed model
        custom_chunk_size: Optional custom chunk size
        
    Returns:
        Tuple of (chunk_size, chunk_overlap)
    """
    config = get_model_config(model_name)
    if not config:
        # Default fallback values
        chunk_size = custom_chunk_size or 400
        chunk_overlap = max(50, chunk_size // 8)  # Default to ~12.5% overlap
        return chunk_size, chunk_overlap
    
    if custom_chunk_size:
        # Ensure custom chunk size doesn't exceed model's max length
        chunk_size = min(custom_chunk_size, config.max_length - 50)  # Leave some buffer
        chunk_overlap = max(config.default_chunk_overlap, chunk_size // 8)
    else:
        chunk_size = config.default_chunk_size
        chunk_overlap = config.default_chunk_overlap
    
    return chunk_size, chunk_overlap


def list_supported_models() -> Dict[str, ModelConfig]:
    """Get all supported model configurations."""
    return FASTEMBED_MODELS.copy()


def get_model_dimensions(model_name: str) -> int:
    """Get the embedding dimensions for a model."""
    config = get_model_config(model_name)
    return config.dimensions if config else 768  # Default to 768 dimensions


def validate_model_name(model_name: str) -> bool:
    """Check if a model name is supported."""
    return model_name in FASTEMBED_MODELS
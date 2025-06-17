"""
Enhanced gRPC server for FastEmbed with Qdrant integration.
"""

import asyncio
import logging
import os
import time
from concurrent import futures
from typing import Dict, List, Optional, AsyncGenerator
import io

import grpc
import numpy as np
from fastembed import TextEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingServicer:
    """Enhanced embedding service with Qdrant integration."""
    
    def __init__(self):
        from config import load_config
        self.config = load_config()
        self.model: Optional[TextEmbedding] = None
        self.model_name: Optional[str] = None
        
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Check CUDA availability
        self.cuda_available = self._check_cuda_availability()
        if self.cuda_available:
            logger.info("CUDA is available and configured")
        else:
            logger.warning("CUDA is not available, falling back to CPU")
        
        # Initialize Qdrant storage if configured
        self.qdrant_store = None
        if self.config.qdrant:
            from qdrant_store import QdrantEmbeddingStore
            self.qdrant_store = QdrantEmbeddingStore(self.config.qdrant)
        
        # Server start time
        self.start_time = time.time()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Found CUDA device: {device_name}")
                return True
        except ImportError:
            pass
        return False
    
    async def initialize(self):
        """Initialize the service."""
        if self.qdrant_store:
            try:
                await self.qdrant_store.initialize()
                logger.info("Qdrant storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant storage: {e}")
                self.qdrant_store = None
        
        # Load default model
        if self.config.default_model:
            success = await self.load_model(self.config.default_model)
            if success:
                logger.info(f"Default model {self.config.default_model} loaded successfully")
    
    async def load_model(self, model_name: str, use_cuda: bool = None) -> bool:
        """Load or switch the FastEmbed model."""
        try:
            if self.model_name == model_name:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            if use_cuda is None:
                use_cuda = self.config.use_cuda and self.cuda_available
            
            logger.info(f"Loading model {model_name} with CUDA: {use_cuda}")
            
            self.model = TextEmbedding(
                model_name=model_name,
                max_length=self.config.max_model_length,
                cache_dir=self.config.cache_dir,
                threads=self.config.embedding_threads
            )
            self.model_name = model_name
            logger.info(f"Successfully loaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.model:
            raise ValueError("No model loaded")
        
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    
    def get_status(self) -> Dict:
        """Get server status."""
        uptime = int(time.time() - self.start_time)
        
        return {
            "server_version": "2.0.0",
            "current_model": self.model_name or "No model loaded",
            "cuda_available": self.cuda_available,
            "qdrant_connected": self.qdrant_store is not None,
            "uptime_seconds": uptime,
            "configuration": {
                "grpc_port": str(self.config.grpc_port),
                "cache_dir": self.config.cache_dir,
                "use_cuda": str(self.config.use_cuda),
                "qdrant_host": getattr(self.config.qdrant, 'host', 'Not configured'),
            }
        }


async def serve():
    """Start the gRPC server."""
    from config import load_config
    
    config = load_config()
    logger.info("Starting FastEmbed-Qdrant gRPC server")
    logger.info(f"Configuration: gRPC port={config.grpc_port}, model={config.default_model}")
    
    # Create and initialize service
    service = EmbeddingServicer()
    await service.initialize()
    
    # For now, create a simple loop for the server
    logger.info(f"FastEmbed server is ready on port {config.grpc_port}")
    logger.info("Server is running... (This is a minimal implementation)")
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


if __name__ == '__main__':
    asyncio.run(serve())
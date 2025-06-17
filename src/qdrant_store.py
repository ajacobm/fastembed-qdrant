"""
Qdrant client for storing embeddings with metadata.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionStatus,
    PointStruct, Filter, FieldCondition, MatchValue
)
from qdrant_client.http.exceptions import ResponseHandlingException

from config import QdrantConfig
from text_chunker import TextChunk

logger = logging.getLogger(__name__)


class QdrantEmbeddingStore:
    """Handles storing and retrieving embeddings in Qdrant."""
    
    def __init__(self, config: QdrantConfig):
        self.config = config
        self._client = None
        self._async_client = None
        
    async def initialize(self):
        """Initialize Qdrant clients."""
        try:
            # Initialize async client
            self._async_client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                https=self.config.use_ssl
            )
            
            # Initialize sync client for collection management
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                https=self.config.use_ssl
            )
            
            logger.info(f"Connected to Qdrant at {self.config.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    async def ensure_collection(
        self, 
        collection_name: str, 
        vector_size: int,
        distance_metric: Distance = Distance.COSINE
    ) -> bool:
        """
        Ensure a collection exists with the correct configuration.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            distance_metric: Distance metric to use
            
        Returns:
            True if collection is ready, False otherwise
        """
        try:
            # Check if collection exists
            collections = await self._async_client.get_collections()
            collection_exists = any(
                col.name == collection_name for col in collections.collections
            )
            
            if collection_exists:
                # Verify collection configuration
                collection_info = await self._async_client.get_collection(collection_name)
                if collection_info.config.params.vectors.size != vector_size:
                    logger.warning(
                        f"Collection {collection_name} exists but has wrong vector size: "
                        f"{collection_info.config.params.vectors.size} != {vector_size}"
                    )
                    return False
                
                logger.info(f"Collection {collection_name} already exists and is correctly configured")
                return True
            else:
                # Create collection
                await self._async_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_metric
                    )
                )
                logger.info(f"Created collection {collection_name} with vector size {vector_size}")
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            return False
    
    async def store_embeddings(
        self,
        collection_name: str,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        document_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Store embeddings in Qdrant with associated chunk metadata.
        
        Args:
            collection_name: Name of the collection
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: Optional document identifier
            
        Returns:
            Tuple of (success, list of point IDs)
        """
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
            return False, []
        
        try:
            points = []
            point_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare metadata
                payload = {
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'chunk_index': chunk.chunk_index,
                    'start_offset': chunk.start_offset,
                    'end_offset': chunk.end_offset,
                    'chunk_size': len(chunk.text),
                }
                
                # Add document ID if provided
                if document_id:
                    payload['document_id'] = document_id
                
                # Add custom metadata from the chunk
                for key, value in chunk.metadata.items():
                    if key not in payload and value is not None:
                        # Ensure the value is JSON serializable
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            payload[key] = value
                        else:
                            payload[key] = str(value)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Store points in batches to avoid overwhelming the server
            batch_size = 100
            stored_ids = []
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                result = await self._async_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                
                if result.status == "completed":
                    batch_ids = [point.id for point in batch]
                    stored_ids.extend(batch_ids)
                    logger.debug(f"Stored batch of {len(batch)} points in {collection_name}")
                else:
                    logger.error(f"Failed to store batch in {collection_name}: {result}")
                    return False, stored_ids
                
                # Add small delay between batches to avoid overwhelming the server
                if i + batch_size < len(points):
                    await asyncio.sleep(0.01)
            
            logger.info(f"Successfully stored {len(stored_ids)} embeddings in {collection_name}")
            return True, stored_ids
            
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {e}")
            return False, []
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and reachable."""
        try:
            collections = await self._async_client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def close(self):
        """Close Qdrant connections."""
        if self._async_client:
            await self._async_client.close()
        if self._client:
            self._client.close()
        logger.info("Closed Qdrant connections")
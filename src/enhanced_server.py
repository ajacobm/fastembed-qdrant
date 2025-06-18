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
from google.protobuf import empty_pb2

# Import generated protobuf code
from proto import embed_pb2
from proto import embed_pb2_grpc

# Import all message types from protobuf
from proto.embed_pb2 import (
    EmbeddingRequest, EmbeddingResponse, Embedding,
    LoadModelRequest, LoadModelResponse,
    FileStreamRequest, FileStreamResponse, FileMetadata, ProcessingOptions,
    StatusRequest, StatusResponse,
    ListModelsRequest, ListModelsResponse, ModelInfo,
    UpdateModelConfigRequest, UpdateModelConfigResponse
)

# Import our new modules
from config import load_config, get_env_variables, ServerConfig
from model_config import (
    get_model_config, get_optimal_chunk_size, list_supported_models,
    get_model_dimensions, validate_model_name
)
from text_chunker import TextChunker, TextChunk
from qdrant_store import QdrantEmbeddingStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedEmbeddingService(embed_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model: Optional[TextEmbedding] = None
        self.model_name: Optional[str] = None
        self.chunk_size: Optional[int] = None
        self.chunk_overlap: Optional[int] = None
        self.qdrant_collection: Optional[str] = None
        
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Configure CUDA options
        self.cuda_available = self._check_cuda_availability()
        if self.cuda_available:
            self.cuda_provider_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8GB
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }
            logger.info("CUDA is available and configured")
        else:
            logger.warning("CUDA is not available, falling back to CPU")

        # Initialize Qdrant storage if configured
        self.qdrant_store: Optional[QdrantEmbeddingStore] = None
        if self.config.qdrant:
            self.qdrant_store = QdrantEmbeddingStore(self.config.qdrant)

        # Server start time for uptime tracking
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

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                logger.info("ONNX Runtime CUDA provider is available")
                return True
        except ImportError:
            pass

        return False

    async def initialize(self):
        """Initialize the service including Qdrant connection."""
        if self.qdrant_store:
            try:
                await self.qdrant_store.initialize()
                logger.info("Qdrant storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant storage: {e}")
                self.qdrant_store = None

        # Load default model
        if self.config.default_model:
            success = await self.load_model(
                model_name=self.config.default_model,
                use_cuda=self.config.use_cuda,
                max_length=self.config.max_model_length,
                threads=self.config.embedding_threads
            )
            if success:
                logger.info(f"Default model {self.config.default_model} loaded successfully")
            else:
                logger.warning(f"Failed to load default model {self.config.default_model}")

    async def load_model(
        self,
        model_name: str,
        use_cuda: bool = True,
        max_length: int = 512,
        threads: int = 8
    ) -> bool:
        """Load or switch the FastEmbed model."""
        try:
            if self.model_name == model_name:
                logger.info(f"Model {model_name} already loaded")
                return True

            # Validate model name
            if not validate_model_name(model_name):
                logger.warning(f"Model {model_name} not in supported models list, but attempting to load")

            # Configure providers based on CUDA availability
            providers = []
            if use_cuda and self.cuda_available:
                providers.append(("CUDAExecutionProvider", self.cuda_provider_options))
            providers.append("CPUExecutionProvider")

            logger.info(f"Initializing model {model_name} with providers: {providers}")

            self.model = TextEmbedding(
                model_name=model_name,
                max_length=max_length,
                cache_dir=self.config.cache_dir,
                threads=threads,
                use_cuda=use_cuda and self.cuda_available,
                providers=providers,
            )
            self.model_name = model_name
            logger.info(f"Successfully loaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    async def LoadModel(
        self,
        request: LoadModelRequest,
        context: grpc.aio.ServicerContext
    ) -> LoadModelResponse:
        """gRPC method to load a model."""
        success = await self.load_model(
            model_name=request.model_name,
            use_cuda=request.use_cuda if request.HasField('use_cuda') else self.config.use_cuda,
            max_length=request.max_length if request.HasField('max_length') else self.config.max_model_length,
            threads=request.threads if request.HasField('threads') else self.config.embedding_threads
        )

        if success:
            model_info = f"Model: {self.model_name}, Dimensions: {get_model_dimensions(self.model_name)}"
            return LoadModelResponse(
                success=True,
                message=f"Successfully loaded model {request.model_name}",
                model_info=model_info
            )
        else:
            return LoadModelResponse(
                success=False,
                message=f"Failed to load model {request.model_name}"
            )

    async def GetEmbeddings(
        self,
        request: EmbeddingRequest,
        context: grpc.aio.ServicerContext
    ) -> EmbeddingResponse:
        """gRPC method to get embeddings for text."""
        if not self.model:
            if not request.model_name:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "No model loaded and no model name provided"
                )
            success = await self.load_model(request.model_name)
            if not success:
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Failed to load model {request.model_name}"
                )

        try:
            # Get embeddings
            embeddings = list(self.model.embed(request.texts))

            # Convert to response format
            embedding_responses = []
            for emb in embeddings:
                embedding_responses.append(
                    Embedding(
                        vector=emb.tolist(),
                        dimension=len(emb)
                    )
                )

            return EmbeddingResponse(embeddings=embedding_responses)
        except Exception as e:
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Error generating embeddings: {str(e)}"
            )

    async def ProcessFileStream(
        self,
        request_stream: AsyncGenerator[FileStreamRequest, None],
        context: grpc.aio.ServicerContext
    ) -> FileStreamResponse:
        """gRPC method to process streaming file data and store embeddings."""
        try:
            file_metadata = None
            processing_options = None
            model_name = None
            file_buffer = io.BytesIO()
            
            # Process the stream
            async for request in request_stream:
                if request.HasField('metadata'):
                    file_metadata = request.metadata
                    logger.info(f"Received file metadata: {file_metadata.filename}")
                
                elif request.HasField('chunk_data'):
                    file_buffer.write(request.chunk_data)
                
                elif request.HasField('options'):
                    processing_options = request.options
                
                if request.model_name:
                    model_name = request.model_name

            if not file_metadata:
                return FileStreamResponse(
                    success=False,
                    message="No file metadata received"
                )

            # Get the complete file content
            file_content = file_buffer.getvalue().decode('utf-8', errors='ignore')
            file_buffer.close()

            if not file_content.strip():
                return FileStreamResponse(
                    success=False,
                    message="No text content found in file"
                )

            # Use provided model or current model
            target_model = model_name or self.model_name
            if not target_model:
                return FileStreamResponse(
                    success=False,
                    message="No model specified and no default model loaded"
                )

            # Load model if needed
            if target_model != self.model_name:
                success = await self.load_model(target_model)
                if not success:
                    return FileStreamResponse(
                        success=False,
                        message=f"Failed to load model {target_model}"
                    )

            # Determine chunking parameters
            chunk_size = processing_options.chunk_size if processing_options and processing_options.chunk_size > 0 else None
            chunk_overlap = processing_options.chunk_overlap if processing_options and processing_options.chunk_overlap > 0 else None
            
            optimal_chunk_size, optimal_overlap = get_optimal_chunk_size(target_model, chunk_size)
            if chunk_overlap is None:
                chunk_overlap = optimal_overlap

            # Create text chunker
            chunker = TextChunker(chunk_size=optimal_chunk_size, chunk_overlap=chunk_overlap)

            # Prepare base metadata
            base_metadata = {
                'filename': file_metadata.filename,
                'content_type': file_metadata.content_type,
                'file_size': file_metadata.file_size,
                'model_name': target_model,
                'document_id': file_metadata.document_id or f"doc_{int(time.time())}"
            }

            # Add custom metadata
            if file_metadata.custom_metadata:
                base_metadata.update(file_metadata.custom_metadata)

            # Chunk the text
            chunks = chunker.chunk_text(file_content, base_metadata)
            logger.info(f"Created {len(chunks)} chunks from file {file_metadata.filename}")

            if not chunks:
                return FileStreamResponse(
                    success=False,
                    message="No valid chunks created from file content"
                )

            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = list(self.model.embed(chunk_texts))
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Store in Qdrant if configured and requested
            stored_ids = []
            points_stored = 0
            
            if (self.qdrant_store and 
                processing_options and 
                processing_options.store_in_qdrant):
                
                collection_name = processing_options.collection_name or self.config.qdrant.collection_name
                vector_size = get_model_dimensions(target_model)
                
                # Ensure collection exists
                collection_ready = await self.qdrant_store.ensure_collection(
                    collection_name, vector_size
                )
                
                if collection_ready:
                    success, stored_ids = await self.qdrant_store.store_embeddings(
                        collection_name=collection_name,
                        chunks=chunks,
                        embeddings=embeddings,
                        document_id=base_metadata['document_id']
                    )
                    
                    if success:
                        points_stored = len(stored_ids)
                        logger.info(f"Stored {points_stored} points in Qdrant collection {collection_name}")
                    else:
                        logger.error("Failed to store embeddings in Qdrant")
                else:
                    logger.error(f"Failed to ensure collection {collection_name} exists")

            return FileStreamResponse(
                success=True,
                message=f"Successfully processed file {file_metadata.filename}",
                chunks_processed=len(chunks),
                embeddings_created=len(embeddings),
                points_stored=points_stored,
                chunk_ids=stored_ids,
                document_id=base_metadata['document_id']
            )

        except Exception as e:
            logger.error(f"Error processing file stream: {e}")
            return FileStreamResponse(
                success=False,
                message=f"Error processing file: {str(e)}"
            )

    async def GetStatus(
        self,
        request: StatusRequest,
        context: grpc.aio.ServicerContext
    ) -> StatusResponse:
        """Get server status and configuration."""
        # Check Qdrant connection
        qdrant_connected = False
        if self.qdrant_store:
            qdrant_connected = await self.qdrant_store.health_check()

        # Get configuration info including current model configuration
        env_vars = get_env_variables()
        config_info = {
            "grpc_port": str(self.config.grpc_port),
            "cache_dir": self.config.cache_dir,
            "use_cuda": str(self.config.use_cuda),
            "qdrant_host": env_vars.get("QDRANT_HOST", "Not configured"),
            "qdrant_collection": env_vars.get("QDRANT_COLLECTION", "Not configured"),
        }
        
        # Add current model configuration if a model is loaded
        if self.model_name:
            model_config = get_model_config(self.model_name)
            if model_config:
                config_info.update({
                    "model_dimensions": str(model_config.dimensions),
                    "model_max_length": str(model_config.max_length),
                    "current_chunk_size": str(self.chunk_size or model_config.default_chunk_size),
                    "current_chunk_overlap": str(self.chunk_overlap or model_config.default_chunk_overlap),
                    "model_size_gb": str(model_config.size_gb),
                    "model_license": model_config.license,
                    "model_description": model_config.description,
                    "current_qdrant_collection": self.qdrant_collection or env_vars.get("QDRANT_COLLECTION", "Not configured")
                })

        uptime = int(time.time() - self.start_time)

        return StatusResponse(
            server_version="2.0.0",
            current_model=self.model_name or "No model loaded",
            cuda_available=self.cuda_available,
            qdrant_connected=qdrant_connected,
            configuration=config_info,
            uptime_seconds=uptime
        )

    async def UpdateModelConfig(
        self,
        request: UpdateModelConfigRequest,
        context: grpc.aio.ServicerContext
    ) -> UpdateModelConfigResponse:
        """Dynamically update model config, chunk size/overlap, and Qdrant collection."""
        model_name = request.model_name
        qdrant_collection = request.qdrant_collection if request.qdrant_collection else None
        from model_config import get_model_config
        config = get_model_config(model_name)
        if not config:
            return UpdateModelConfigResponse(
                success=False,
                message=f"Model '{model_name}' not found in config.",
                active_model=self.model_name or "",
                chunk_size=self.chunk_size or 0,
                chunk_overlap=self.chunk_overlap or 0,
                qdrant_collection=self.qdrant_collection or ""
            )
        # Switch model, chunk size, and overlap
        success = await self.load_model(model_name)
        if not success:
            return UpdateModelConfigResponse(
                success=False,
                message=f"Failed to load model {model_name}",
                active_model=self.model_name or "",
                chunk_size=self.chunk_size or 0,
                chunk_overlap=self.chunk_overlap or 0,
                qdrant_collection=self.qdrant_collection or ""
            )
        self.chunk_size = config.default_chunk_size
        self.chunk_overlap = config.default_chunk_overlap
        # Qdrant collection
        if qdrant_collection:
            self.qdrant_collection = qdrant_collection
        elif self.config.qdrant and self.config.qdrant.collection_name:
            self.qdrant_collection = self.config.qdrant.collection_name
        return UpdateModelConfigResponse(
            success=True,
            message=f"Model config updated. Model: {model_name}, Chunk size: {self.chunk_size}, Chunk overlap: {self.chunk_overlap}, Qdrant collection: {self.qdrant_collection}",
            active_model=model_name or "",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            qdrant_collection=self.qdrant_collection or ""
        )

    async def ListModels(
        self,
        request: ListModelsRequest,
        context: grpc.aio.ServicerContext
    ) -> ListModelsResponse:
        """List all supported models."""
        models = list_supported_models()
        model_info_list = []
        
        for model_config in models.values():
            model_info = ModelInfo(
                model_name=model_config.model_name,
                dimensions=model_config.dimensions,
                max_length=model_config.max_length,
                default_chunk_size=model_config.default_chunk_size,
                size_gb=model_config.size_gb,
                license=model_config.license,
                description=model_config.description
            )
            model_info_list.append(model_info)

        return ListModelsResponse(models=model_info_list)


async def serve():
    """Start the enhanced gRPC server."""
    # Load configuration
    config = load_config()
    logger.info("Starting Enhanced FastEmbed Server with Qdrant integration")
    logger.info(f"Configuration: {get_env_variables()}")

    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # Create and register service
    embedding_service = EnhancedEmbeddingService(config)
    embed_pb2_grpc.add_EmbeddingServiceServicer_to_server(embedding_service, server)

    # Initialize the service
    await embedding_service.initialize()

    # Listen on configured port
    listen_addr = f'[::]:{config.grpc_port}'
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")

    # Start server
    await server.start()
    logger.info("Enhanced FastEmbed Server is ready to accept connections")
    
    # Wait for termination
    await server.wait_for_termination()


if __name__ == '__main__':
    asyncio.run(serve())
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

# Import observability
from observability import (
    setup_logging, get_logger, RequestContext, operation_context,
    log_model_operation, log_qdrant_operation, log_file_processing,
    LogConfig, update_model_context
)

# Configure observability logging
log_config = LogConfig.from_env()
log_config.validate()

setup_logging(
    level=log_config.level,
    format_type=log_config.format,
    output_type=log_config.output,
    file_path=log_config.file_path,
    max_size=log_config.max_size,
    backup_count=log_config.backup_count
)

logger = get_logger('server')
model_logger = get_logger('model')
qdrant_logger = get_logger('qdrant')


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
            logger.info("CUDA is available and configured", cuda_device=True)
        else:
            logger.warning("CUDA is not available, falling back to CPU", cuda_device=False)

        # Initialize Qdrant storage if configured
        self.qdrant_store: Optional[QdrantEmbeddingStore] = None
        if self.config.qdrant:
            self.qdrant_store = QdrantEmbeddingStore(self.config.qdrant)

        # Server start time for uptime tracking
        self.start_time = time.time()
        logger.info("EnhancedEmbeddingService initialized", 
                   cache_dir=self.config.cache_dir,
                   cuda_available=self.cuda_available)

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        with operation_context("cuda_detection") as op:
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info("CUDA device detected", device_name=device_name, **op)
                    return True
            except ImportError:
                logger.debug("PyTorch not available for CUDA detection", **op)

            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    logger.info("ONNX Runtime CUDA provider available", providers=providers, **op)
                    return True
            except ImportError:
                logger.debug("ONNX Runtime not available for CUDA detection", **op)

            logger.info("No CUDA support detected", **op)
            return False

    async def initialize(self):
        """Initialize the service including Qdrant connection."""
        with operation_context("service_initialization") as op:
            if self.qdrant_store:
                try:
                    with operation_context("qdrant_initialization") as qdrant_op:
                        await self.qdrant_store.initialize()
                        logger.info("Qdrant storage initialized successfully", **qdrant_op)
                        log_qdrant_operation(
                            operation="initialize",
                            collection="system",
                            duration=qdrant_op.get('duration', 0)
                        )
                except Exception as e:
                    logger.error("Failed to initialize Qdrant storage", 
                               error=str(e), **op, exc_info=True)
                    log_qdrant_operation(
                        operation="initialize",
                        collection="system",
                        error=str(e)
                    )
                    self.qdrant_store = None

            # Load default model
            if self.config.default_model:
                with operation_context("default_model_loading", model=self.config.default_model) as model_op:
                    success = await self.load_model(
                        model_name=self.config.default_model,
                        use_cuda=self.config.use_cuda,
                        max_length=self.config.max_model_length,
                        threads=self.config.embedding_threads
                    )
                    if success:
                        logger.info("Default model loaded successfully", **model_op)
                    else:
                        logger.warning("Failed to load default model", **model_op)

    async def load_model(
        self,
        model_name: str,
        use_cuda: bool = True,
        max_length: int = 512,
        threads: int = 8
    ) -> bool:
        """Load or switch the FastEmbed model."""
        start_time = time.time()
        
        try:
            with operation_context("model_loading", model=model_name, use_cuda=use_cuda) as op:
                if self.model_name == model_name:
                    logger.info("Model already loaded", **op)
                    return True

                # Validate model name
                if not validate_model_name(model_name):
                    logger.warning("Model not in supported list, attempting to load", **op)

                # Configure providers based on CUDA availability
                providers = []
                if use_cuda and self.cuda_available:
                    providers.append(("CUDAExecutionProvider", self.cuda_provider_options))
                providers.append("CPUExecutionProvider")

                logger.info("Initializing model", 
                          providers=[p[0] if isinstance(p, tuple) else p for p in providers], 
                          **op)

                self.model = TextEmbedding(
                    model_name=model_name,
                    max_length=max_length,
                    cache_dir=self.config.cache_dir,
                    threads=threads,
                    use_cuda=use_cuda and self.cuda_available,
                    providers=providers,
                )
                
                self.model_name = model_name
                update_model_context(model_name)
                
                duration = time.time() - start_time
                logger.info("Model loaded successfully", 
                          duration=duration,
                          **op)
                
                log_model_operation(
                    operation="model_load",
                    model_name=model_name,
                    duration=duration,
                    use_cuda=use_cuda and self.cuda_available,
                    max_length=max_length,
                    threads=threads
                )
                
                return True
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Error loading model", 
                        model=model_name, 
                        error=str(e), 
                        duration=duration,
                        exc_info=True)
            
            log_model_operation(
                operation="model_load",
                model_name=model_name,
                duration=duration,
                error=str(e)
            )
            return False

    async def LoadModel(
        self,
        request: LoadModelRequest,
        context: grpc.aio.ServicerContext
    ) -> LoadModelResponse:
        """gRPC method to load a model."""
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        
        with RequestContext(request_id, method="LoadModel", model=request.model_name) as ctx:
            logger.info("LoadModel request received", 
                       model=request.model_name,
                       use_cuda=request.use_cuda if request.HasField('use_cuda') else self.config.use_cuda)
            
            success = await self.load_model(
                model_name=request.model_name,
                use_cuda=request.use_cuda if request.HasField('use_cuda') else self.config.use_cuda,
                max_length=request.max_length if request.HasField('max_length') else self.config.max_model_length,
                threads=request.threads if request.HasField('threads') else self.config.embedding_threads
            )

            if success:
                model_info = f"Model: {self.model_name}, Dimensions: {get_model_dimensions(self.model_name)}"
                logger.info("LoadModel request completed successfully", 
                           model=request.model_name,
                           model_info=model_info)
                
                return LoadModelResponse(
                    success=True,
                    message=f"Successfully loaded model {request.model_name}",
                    model_info=model_info
                )
            else:
                logger.error("LoadModel request failed", model=request.model_name)
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
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        target_model = request.model_name or self.model_name
        
        with RequestContext(request_id, method="GetEmbeddings", model=target_model) as ctx:
            logger.info("GetEmbeddings request received", 
                       batch_size=len(request.texts),
                       model=target_model)
            
            if not self.model:
                if not request.model_name:
                    logger.error("No model loaded and no model name provided")
                    await context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "No model loaded and no model name provided"
                    )
                
                success = await self.load_model(request.model_name)
                if not success:
                    logger.error("Failed to load model", model=request.model_name)
                    await context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"Failed to load model {request.model_name}"
                    )

            try:
                with operation_context("embedding_generation", 
                                     batch_size=len(request.texts),
                                     model=self.model_name) as op:
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

                    logger.info("Embeddings generated successfully", 
                              embeddings_count=len(embedding_responses),
                              dimension=len(embeddings[0]) if embeddings else 0,
                              **op)
                    
                    log_model_operation(
                        operation="embedding_generation",
                        model_name=self.model_name,
                        duration=op.get('duration', 0),
                        batch_size=len(request.texts),
                        embeddings_generated=len(embeddings)
                    )

                    return EmbeddingResponse(embeddings=embedding_responses)
                    
            except Exception as e:
                logger.error("Error generating embeddings", 
                           error=str(e), 
                           batch_size=len(request.texts),
                           exc_info=True)
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
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        
        with RequestContext(request_id, method="ProcessFileStream") as ctx:
            start_time = time.time()
            
            try:
                file_metadata = None
                processing_options = None
                model_name = None
                file_buffer = io.BytesIO()
                
                logger.info("ProcessFileStream request started")
                
                # Process the stream
                async for request in request_stream:
                    if request.HasField('metadata'):
                        file_metadata = request.metadata
                        logger.info("File metadata received", 
                                   filename=file_metadata.filename,
                                   content_type=file_metadata.content_type,
                                   file_size=file_metadata.file_size)
                    
                    elif request.HasField('chunk_data'):
                        file_buffer.write(request.chunk_data)
                    
                    elif request.HasField('options'):
                        processing_options = request.options
                    
                    if request.model_name:
                        model_name = request.model_name

                if not file_metadata:
                    logger.error("No file metadata received")
                    return FileStreamResponse(
                        success=False,
                        message="No file metadata received"
                    )

                # Get the complete file content
                file_content = file_buffer.getvalue().decode('utf-8', errors='ignore')
                file_buffer.close()

                if not file_content.strip():
                    logger.error("No text content found in file", filename=file_metadata.filename)
                    return FileStreamResponse(
                        success=False,
                        message="No text content found in file"
                    )

                # Use provided model or current model
                target_model = model_name or self.model_name
                if not target_model:
                    logger.error("No model specified and no default model loaded")
                    return FileStreamResponse(
                        success=False,
                        message="No model specified and no default model loaded"
                    )

                update_model_context(target_model)

                # Load model if needed
                if target_model != self.model_name:
                    with operation_context("model_switching", from_model=self.model_name, to_model=target_model):
                        success = await self.load_model(target_model)
                        if not success:
                            logger.error("Failed to load model", model=target_model)
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
                with operation_context("text_chunking", 
                                     chunk_size=optimal_chunk_size,
                                     chunk_overlap=chunk_overlap) as chunk_op:
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
                    logger.info("Text chunked successfully", 
                               chunks_count=len(chunks),
                               filename=file_metadata.filename,
                               **chunk_op)

                if not chunks:
                    logger.error("No valid chunks created from file content", 
                               filename=file_metadata.filename)
                    return FileStreamResponse(
                        success=False,
                        message="No valid chunks created from file content"
                    )

                # Generate embeddings
                with operation_context("embedding_generation", 
                                     chunks_count=len(chunks),
                                     model=target_model) as embed_op:
                    chunk_texts = [chunk.text for chunk in chunks]
                    embeddings = list(self.model.embed(chunk_texts))
                    logger.info("Embeddings generated successfully", 
                               embeddings_count=len(embeddings),
                               **embed_op)

                # Store in Qdrant if configured and requested
                stored_ids = []
                points_stored = 0
                
                if (self.qdrant_store and 
                    processing_options and 
                    processing_options.store_in_qdrant):
                    
                    collection_name = processing_options.collection_name or self.config.qdrant.collection_name
                    vector_size = get_model_dimensions(target_model)
                    
                    with operation_context("qdrant_storage", 
                                         collection=collection_name,
                                         points_count=len(chunks)) as qdrant_op:
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
                                logger.info("Embeddings stored in Qdrant successfully", 
                                           points_stored=points_stored,
                                           collection=collection_name,
                                           **qdrant_op)
                                
                                log_qdrant_operation(
                                    operation="store_embeddings",
                                    collection=collection_name,
                                    points_count=points_stored,
                                    duration=qdrant_op.get('duration', 0)
                                )
                            else:
                                logger.error("Failed to store embeddings in Qdrant", 
                                           collection=collection_name, **qdrant_op)
                        else:
                            logger.error("Failed to ensure collection exists", 
                                       collection=collection_name, **qdrant_op)

                total_duration = time.time() - start_time
                
                # Log file processing summary
                log_file_processing(
                    filename=file_metadata.filename,
                    size_bytes=file_metadata.file_size,
                    chunks_count=len(chunks),
                    duration=total_duration,
                    model=target_model,
                    stored_to_qdrant=points_stored > 0
                )

                logger.info("ProcessFileStream completed successfully",
                           filename=file_metadata.filename,
                           chunks_processed=len(chunks),
                           embeddings_created=len(embeddings),
                           points_stored=points_stored,
                           total_duration=total_duration)

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
                total_duration = time.time() - start_time
                logger.error("Error processing file stream", 
                           error=str(e), 
                           duration=total_duration,
                           exc_info=True)
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
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        
        with RequestContext(request_id, method="GetStatus") as ctx:
            logger.info("GetStatus request received")
            
            # Determine server state
            if self.model is None and self.config.default_model:
                server_state = f"Initializing (loading {self.config.default_model})"
                current_model = f"Loading {self.config.default_model}..."
            elif self.model is None:
                server_state = "Ready (no model loaded)"
                current_model = "No model loaded"
            else:
                server_state = "Ready"
                current_model = self.model_name or "Unknown model"
            
            # Check Qdrant connection (with timeout to avoid blocking)
            qdrant_connected = False
            try:
                if self.qdrant_store:
                    with operation_context("qdrant_health_check") as op:
                        # Quick health check with timeout
                        qdrant_connected = await asyncio.wait_for(
                            self.qdrant_store.health_check(), 
                            timeout=2.0
                        )
                        logger.info("Qdrant health check completed", 
                                   connected=qdrant_connected, **op)
            except asyncio.TimeoutError:
                logger.warning("Qdrant health check timed out")
                qdrant_connected = False
            except Exception as e:
                logger.warning("Qdrant health check failed", error=str(e))
                qdrant_connected = False

            # Get configuration info including current model configuration
            env_vars = get_env_variables()
            config_info = {
                "server_state": server_state,
                "grpc_port": str(self.config.grpc_port),
                "cache_dir": self.config.cache_dir,
                "use_cuda": str(self.config.use_cuda),
                "default_model": self.config.default_model or "Not configured",
                "qdrant_host": env_vars.get("QDRANT_HOST", "Not configured"),
                "qdrant_collection": env_vars.get("QDRANT_COLLECTION", "Not configured"),
                "log_level": env_vars.get("LOG_LEVEL", "INFO"),
                "log_format": env_vars.get("LOG_FORMAT", "json"),
            }
            
            # Add current model configuration if a model is loaded
            if self.model_name and self.model is not None:
                try:
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
                except Exception as e:
                    logger.warning("Could not get model config details", error=str(e))
                    config_info["model_config_error"] = str(e)

            uptime = int(time.time() - self.start_time)

            logger.info("GetStatus request completed", 
                       uptime_seconds=uptime,
                       server_state=server_state,
                       current_model=current_model)

            return StatusResponse(
                server_version="2.0.0-enhanced",
                current_model=current_model,
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
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        
        with RequestContext(request_id, method="UpdateModelConfig", model=request.model_name) as ctx:
            logger.info("UpdateModelConfig request received", 
                       model=request.model_name,
                       qdrant_collection=request.qdrant_collection)
            
            model_name = request.model_name
            qdrant_collection = request.qdrant_collection if request.qdrant_collection else None
            
            from model_config import get_model_config
            config = get_model_config(model_name)
            if not config:
                logger.error("Model not found in config", model=model_name)
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
                logger.error("Failed to load model", model=model_name)
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
            
            logger.info("UpdateModelConfig completed successfully",
                       model=model_name,
                       chunk_size=self.chunk_size,
                       chunk_overlap=self.chunk_overlap,
                       qdrant_collection=self.qdrant_collection)
            
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
        request_id = context.peer() + "-" + str(int(time.time() * 1000))
        
        with RequestContext(request_id, method="ListModels") as ctx:
            logger.info("ListModels request received")
            
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

            logger.info("ListModels request completed", models_count=len(model_info_list))
            return ListModelsResponse(models=model_info_list)


async def serve():
    """Start the enhanced gRPC server."""
    # Load configuration
    config = load_config()
    
    logger.info("Starting Enhanced FastEmbed Server with Observability", 
               version="2.0.0",
               grpc_port=config.grpc_port,
               use_cuda=config.use_cuda,
               default_model=config.default_model)
    
    env_vars = get_env_variables()
    logger.info("Server configuration loaded", 
               cache_dir=config.cache_dir,
               max_workers=config.max_workers,
               qdrant_configured=bool(config.qdrant),
               log_level=env_vars.get("LOG_LEVEL", "INFO"),
               log_format=env_vars.get("LOG_FORMAT", "json"))

    # Create gRPC server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # Create and register service
    embedding_service = EnhancedEmbeddingService(config)
    embed_pb2_grpc.add_EmbeddingServiceServicer_to_server(embedding_service, server)

    # Listen on configured port and start server BEFORE initialization
    listen_addr = f'[::]:{config.grpc_port}'
    server.add_insecure_port(listen_addr)
    logger.info("gRPC server configured", listen_addr=listen_addr)

    # Start server to accept connections immediately
    await server.start()
    logger.info("Enhanced FastEmbed Server started and accepting connections",
               status="started",
               listen_addr=listen_addr,
               note="Service initialization will happen in background")
    
    # Initialize the service in background (non-blocking)
    async def background_initialization():
        try:
            with operation_context("background_service_initialization") as op:
                logger.info("Starting background service initialization...")
                await embedding_service.initialize()
                logger.info("Background service initialization completed successfully", **op)
        except Exception as e:
            logger.error("Background service initialization failed", 
                        error=str(e), exc_info=True)
    
    # Start background initialization task
    initialization_task = asyncio.create_task(background_initialization())
    
    logger.info("Enhanced FastEmbed Server is ready to accept connections",
               status="ready",
               listen_addr=listen_addr,
               initialization_status="in_progress")
    
    # Wait for termination
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested", reason="keyboard_interrupt")
        # Cancel background task if still running
        if not initialization_task.done():
            initialization_task.cancel()
            try:
                await initialization_task
            except asyncio.CancelledError:
                logger.info("Background initialization cancelled")
    except Exception as e:
        logger.error("Server terminated unexpectedly", error=str(e), exc_info=True)
    finally:
        logger.info("Enhanced FastEmbed Server shutdown complete")


if __name__ == '__main__':
    asyncio.run(serve())

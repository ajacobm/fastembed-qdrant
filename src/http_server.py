"""
FastAPI HTTP wrapper for the FastEmbed gRPC server with full observability.
This provides HTTP endpoints with structured logging and request correlation.
"""

import asyncio
import io
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import grpc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import protobuf classes
from proto.embed_pb2 import (
    EmbeddingRequest as GrpcEmbeddingRequest, 
    FileStreamRequest, FileMetadata, ProcessingOptions,
    StatusRequest, ListModelsRequest, UpdateModelConfigRequest
)
from proto.embed_pb2_grpc import EmbeddingServiceStub

from prometheus_client import make_asgi_app
from observability.metrics import request_duration_seconds, embedding_requests_total, file_processing_requests_total
from config import load_config
from observability.logger import setup_logging, get_logger
from observability.log_config import LogConfig
from observability.log_context import RequestContext, operation_context, log_file_processing


# Set up logging early
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

logger = get_logger(__name__)


# Pydantic models for API
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None


class EmbeddingResponse(BaseModel):
    embeddings: List[Dict[str, Any]]
    model_name: str


class FileProcessRequest(BaseModel):
    model_name: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    store_in_qdrant: bool = True
    collection_name: Optional[str] = None
    document_id: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None


class FileProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    embeddings_created: int
    points_stored: int
    chunk_ids: List[str]
    document_id: Optional[str] = None


class ModelInfo(BaseModel):
    model_name: str
    dimensions: int
    max_length: int
    default_chunk_size: int
    size_gb: float
    license: str
    description: str


class CurrentModelConfig(BaseModel):
    model_name: str
    dimensions: int
    max_length: int
    chunk_size: int
    chunk_overlap: int
    size_gb: float
    license: str
    description: str
    qdrant_collection: Optional[str] = None


class ServerStatus(BaseModel):
    server_version: str
    current_model: str
    cuda_available: bool
    qdrant_connected: bool
    configuration: Dict[str, str]
    uptime_seconds: int
    current_model_config: Optional[CurrentModelConfig] = None


class UpdateModelConfigRequestBody(BaseModel):
    model_name: str
    qdrant_collection: Optional[str] = None


class HTTPEmbeddingService:
    """HTTP service that connects to the gRPC FastEmbed server with full observability."""
    
    def __init__(self, grpc_host: str = "localhost", grpc_port: int = 50051):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.channel = None
        self.stub = None
        self.logger = get_logger("http_client")
    
    async def connect(self):
        """Connect to the gRPC server."""
        with operation_context("grpc_connect", host=self.grpc_host, port=self.grpc_port):
            try:
                self.channel = grpc.aio.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
                self.stub = EmbeddingServiceStub(self.channel)
                self.logger.info("Connected to gRPC server", extra={
                    "grpc_host": self.grpc_host,
                    "grpc_port": self.grpc_port
                })
            except Exception as e:
                self.logger.error("Failed to connect to gRPC server", extra={
                    "grpc_host": self.grpc_host,
                    "grpc_port": self.grpc_port,
                    "error": str(e)
                })
                raise
    
    async def disconnect(self):
        """Disconnect from the gRPC server."""
        if self.channel:
            with operation_context("grpc_disconnect", host=self.grpc_host, port=self.grpc_port):
                try:
                    await self.channel.close()
                    self.logger.info("Disconnected from gRPC server")
                except Exception as e:
                    self.logger.error("Error disconnecting from gRPC server", extra={"error": str(e)})
    
    async def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get embeddings for a list of texts."""
        operation_data = {
            "text_count": len(texts), 
            "model_name": model_name or "current",
            "total_chars": sum(len(text) for text in texts)
        }
        
        with operation_context("grpc_get_embeddings", **operation_data):
            try:
                request = GrpcEmbeddingRequest(texts=texts, model_name=model_name or "")
                response = await self.stub.GetEmbeddings(request)
                
                embeddings = []
                for emb in response.embeddings:
                    embeddings.append({
                        "vector": list(emb.vector),
                        "dimension": emb.dimension
                    })
                
                result = {
                    "embeddings": embeddings,
                    "model_name": model_name or "current"
                }
                
                self.logger.info("Successfully generated embeddings", extra={
                    "embeddings_count": len(embeddings),
                    "model_name": result["model_name"]
                })
                
                return result
                
            except grpc.RpcError as e:
                self.logger.error("gRPC error getting embeddings", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details(),
                    **operation_data
                })
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error("Unexpected error getting embeddings", extra={
                    "error": str(e),
                    **operation_data
                })
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    async def process_file_stream(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        options: FileProcessRequest
    ) -> Dict[str, Any]:
        """Process a file through the streaming endpoint."""
        operation_data = {
            "filename": filename,
            "content_type": content_type,
            "file_size": len(file_content),
            "model_name": options.model_name or "current",
            "chunk_size": options.chunk_size,
            "store_in_qdrant": options.store_in_qdrant,
            "collection_name": options.collection_name
        }
        
        with operation_context("grpc_process_file_stream", **operation_data) as ctx:
            try:
                # Create async generator for the stream
                async def create_stream():
                    # Send metadata first
                    metadata = FileMetadata(
                        filename=filename,
                        content_type=content_type,
                        file_size=len(file_content),
                        custom_metadata=options.custom_metadata or {},
                        document_id=options.document_id or ""
                    )
                    
                    yield FileStreamRequest(
                        metadata=metadata,
                        model_name=options.model_name or ""
                    )
                    
                    # Send processing options
                    processing_opts = ProcessingOptions(
                        chunk_size=options.chunk_size or 0,
                        chunk_overlap=options.chunk_overlap or 0,
                        store_in_qdrant=options.store_in_qdrant,
                        collection_name=options.collection_name or ""
                    )
                    
                    yield FileStreamRequest(options=processing_opts)
                    
                    # Send file content in chunks
                    chunk_size = 8192  # 8KB chunks for streaming
                    for i in range(0, len(file_content), chunk_size):
                        chunk = file_content[i:i + chunk_size]
                        yield FileStreamRequest(chunk_data=chunk)
                
                # Call the streaming RPC
                response = await self.stub.ProcessFileStream(create_stream())
                
                result = {
                    "success": response.success,
                    "message": response.message,
                    "chunks_processed": response.chunks_processed,
                    "embeddings_created": response.embeddings_created,
                    "points_stored": response.points_stored,
                    "chunk_ids": list(response.chunk_ids),
                    "document_id": response.document_id or None
                }
                
                # Log file processing completion
                log_file_processing(
                    filename=filename,
                    size_bytes=len(file_content),
                    chunks_count=response.chunks_processed,
                    duration=ctx.get('duration', 0),  # Get duration from context
                    model=options.model_name or "current",
                    stored_to_qdrant=response.points_stored > 0
                )
                
                self.logger.info("Successfully processed file", extra={
                    "chunks_processed": response.chunks_processed,
                    "embeddings_created": response.embeddings_created,
                    "points_stored": response.points_stored,
                    **operation_data
                })
                
                return result
                
            except grpc.RpcError as e:
                self.logger.error("gRPC error processing file", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details(),
                    **operation_data
                })
                
                # Log file processing failure
                log_file_processing(
                    filename=filename,
                    size_bytes=len(file_content),
                    chunks_count=0,
                    duration=ctx.get('duration', 0),
                    model=options.model_name or "current",
                    stored_to_qdrant=False,
                    error=f"gRPC error: {e.details()}"
                )
                
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error("Unexpected error processing file", extra={
                    "error": str(e),
                    **operation_data
                })
                
                # Log file processing failure
                log_file_processing(
                    filename=filename,
                    size_bytes=len(file_content),
                    chunks_count=0,
                    duration=ctx.get('duration', 0),
                    model=options.model_name or "current",
                    stored_to_qdrant=False,
                    error=str(e)
                )
                
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    async def update_model_config(self, model_name: str, qdrant_collection: Optional[str]=None) -> Dict[str, Any]:
        """Update the active model/chunking/Qdrant collection via gRPC."""
        with operation_context("grpc_update_model_config", model_name=model_name, collection=qdrant_collection):
            try:
                request = UpdateModelConfigRequest(model_name=model_name)
                if qdrant_collection:
                    request.qdrant_collection = qdrant_collection
                response = await self.stub.UpdateModelConfig(request)
                
                result = {
                    "success": response.success,
                    "message": response.message,
                    "active_model": response.active_model,
                    "chunk_size": response.chunk_size,
                    "chunk_overlap": response.chunk_overlap,
                    "qdrant_collection": response.qdrant_collection,
                }
                
                self.logger.info("Model configuration updated", extra={
                    "model_name": model_name,
                    "success": response.success,
                    "active_model": response.active_model,
                    "qdrant_collection": response.qdrant_collection
                })
                
                return result
                
            except grpc.RpcError as e:
                self.logger.error("gRPC error updating model config", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details(),
                    "model_name": model_name
                })
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error("Unexpected error updating model config", extra={
                    "error": str(e),
                    "model_name": model_name
                })
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get server status with full model configuration."""
        with operation_context("grpc_get_status"):
            try:
                response = await self.stub.GetStatus(StatusRequest())
                
                status_data = {
                    "server_version": response.server_version,
                    "current_model": response.current_model,
                    "cuda_available": response.cuda_available,
                    "qdrant_connected": response.qdrant_connected,
                    "configuration": dict(response.configuration),
                    "uptime_seconds": response.uptime_seconds,
                    "current_model_config": None
                }
                
                # If a model is currently loaded, extract its configuration from the config dict
                config_dict = dict(response.configuration)
                if response.current_model and response.current_model != "No model loaded" and "model_dimensions" in config_dict:
                    status_data["current_model_config"] = {
                        "model_name": response.current_model,
                        "dimensions": int(config_dict.get("model_dimensions", "0")),
                        "max_length": int(config_dict.get("model_max_length", "0")),
                        "chunk_size": int(config_dict.get("current_chunk_size", "0")),
                        "chunk_overlap": int(config_dict.get("current_chunk_overlap", "0")),
                        "size_gb": float(config_dict.get("model_size_gb", "0.0")),
                        "license": config_dict.get("model_license", "unknown"),
                        "description": config_dict.get("model_description", "No description available"),
                        "qdrant_collection": config_dict.get("current_qdrant_collection", None) if config_dict.get("current_qdrant_collection") != "Not configured" else None
                    }
                
                self.logger.debug("Retrieved server status", extra={
                    "server_version": response.server_version,
                    "current_model": response.current_model,
                    "cuda_available": response.cuda_available,
                    "qdrant_connected": response.qdrant_connected,
                    "uptime_seconds": response.uptime_seconds
                })
                
                return status_data
                
            except grpc.RpcError as e:
                self.logger.error("gRPC error getting status", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details()
                })
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error("Unexpected error getting status", extra={"error": str(e)})
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List supported models."""
        with operation_context("grpc_list_models"):
            try:
                response = await self.stub.ListModels(ListModelsRequest())
                models = []
                for model in response.models:
                    models.append({
                        "model_name": model.model_name,
                        "dimensions": model.dimensions,
                        "max_length": model.max_length,
                        "default_chunk_size": model.default_chunk_size,
                        "size_gb": model.size_gb,
                        "license": model.license,
                        "description": model.description
                    })
                
                self.logger.debug("Retrieved model list", extra={"model_count": len(models)})
                return models
                
            except grpc.RpcError as e:
                self.logger.error("gRPC error listing models", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details()
                })
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error("Unexpected error listing models", extra={"error": str(e)})
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Global service instance
service: Optional[HTTPEmbeddingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager with observability."""
    global service
    startup_logger = get_logger("startup")
    
    try:
        startup_logger.info("Starting FastEmbed HTTP API server")
        config = load_config()
        
        # Initialize the gRPC client service
        service = HTTPEmbeddingService(grpc_port=config.grpc_port)
        await service.connect()
        
        startup_logger.info("FastEmbed HTTP API server started successfully", extra={
            "grpc_port": config.grpc_port
        })
        
        yield
        
    except Exception as e:
        startup_logger.error("Failed to start FastEmbed HTTP API server", extra={
            "error": str(e)
        })
        raise
    finally:
        shutdown_logger = get_logger("shutdown")
        try:
            if service:
                await service.disconnect()
            shutdown_logger.info("FastEmbed HTTP API server stopped")
        except Exception as e:
            shutdown_logger.error("Error during shutdown", extra={"error": str(e)})


# Create FastAPI app
app = FastAPI(
    title="FastEmbed HTTP API", 
    description="HTTP wrapper for FastEmbed gRPC server with Qdrant integration and full observability",
    version="2.0.0",
    lifespan=lifespan
)

# Mount the Prometheus metrics app to the /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request correlation and metrics
@app.middleware("http")
async def request_correlation_and_metrics_middleware(request: Request, call_next):
    """Add request correlation ID, timing, and Prometheus metrics to all HTTP requests."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Extract client info
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Create request context
    with RequestContext(
        request_id=request_id,
        method=f"HTTP {request.method} {request.url.path}",
        user_id=client_ip  # Using IP as user ID for now
    ):
        # Log request start
        api_logger = get_logger("api")
        api_logger.info("HTTP request started", extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent
        })
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful completion
            api_logger.info("HTTP request completed", extra={
                "status_code": response.status_code,
                "duration": duration,
                "duration_ms": round(duration * 1000, 2)
            })
            
            # Add correlation ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Update Prometheus metrics for successful requests
            request_duration_seconds.labels(method=request.method, model="unknown").observe(duration)
            if request.url.path == "/embeddings":
                embedding_requests_total.labels(method=request.method, model="unknown", status="success").inc()
            elif request.url.path == "/process-file":
                file_processing_requests_total.labels(method=request.method, status="success", store_destination="unknown").inc()
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Log error
            api_logger.error("HTTP request failed", extra={
                "duration": duration,
                "duration_ms": round(duration * 1000, 2),
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Update Prometheus metrics for failed requests
            request_duration_seconds.labels(method=request.method, model="unknown").observe(duration)
            if request.url.path == "/embeddings":
                embedding_requests_total.labels(method=request.method, model="unknown", status="error").inc()
            elif request.url.path == "/process-file":
                file_processing_requests_total.labels(method=request.method, status="error", store_destination="unknown").inc()
            
            # Re-raise to let FastAPI handle it
            raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FastEmbed HTTP API", "version": "2.0.0", "observability": "enabled"}


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text."""
    with operation_context("create_embeddings", text_count=len(request.texts), model_name=request.model_name):
        result = await service.get_embeddings(request.texts, request.model_name)
        return EmbeddingResponse(**result)


@app.post("/process-file", response_model=FileProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    store_in_qdrant: bool = Form(True),
    collection_name: Optional[str] = Form(None),
    document_id: Optional[str] = Form(None),
    custom_metadata: Optional[str] = Form(None)
):
    """Process an uploaded file and generate embeddings."""
    operation_data = {
        "filename": file.filename,
        "content_type": file.content_type,
        "model_name": model_name,
        "store_in_qdrant": store_in_qdrant
    }
    
    with operation_context("process_file", **operation_data):
        # Read file content
        content = await file.read()
        
        # Parse custom metadata if provided
        metadata = {}
        if custom_metadata:
            try:
                metadata = json.loads(custom_metadata)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in custom_metadata", extra={
                    "custom_metadata": custom_metadata,
                    "filename": file.filename
                })
                raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
        # Create options
        options = FileProcessRequest(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            store_in_qdrant=store_in_qdrant,
            collection_name=collection_name,
            document_id=document_id,
            custom_metadata=metadata
        )
        
        # Process the file
        result = await service.process_file_stream(
            file_content=content,
            filename=file.filename or "unknown",
            content_type=file.content_type or "text/plain",
            options=options
        )
        
        return FileProcessResponse(**result)


@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    with operation_context("get_status"):
        result = await service.get_status()
        return ServerStatus(**result)


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List supported models."""
    with operation_context("list_models"):
        result = await service.list_models()
        return [ModelInfo(**model) for model in result]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    with operation_context("health_check"):
        try:
            status = await service.get_status()
            return {
                "status": "healthy", 
                "server_version": status["server_version"],
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error("Health check failed", extra={"error": str(e)})
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/update-model-config")
async def update_model_config(request: UpdateModelConfigRequestBody):
    """
    Update the embedding model, chunk_size, chunk_overlap, or Qdrant collection.
    """
    with operation_context("update_model_config", model_name=request.model_name, collection=request.qdrant_collection):
        result = await service.update_model_config(
            model_name=request.model_name,
            qdrant_collection=request.qdrant_collection,
        )
        return result


# Exception handlers for better error logging
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with logging."""
    error_logger = get_logger("http_error")
    error_logger.warning("HTTP exception occurred", extra={
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": request.url.path,
        "method": request.method
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with logging."""
    error_logger = get_logger("error")
    error_logger.error("Unhandled exception occurred", extra={
        "error": str(exc),
        "error_type": type(exc).__name__,
        "path": request.url.path,
        "method": request.method
    }, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="FastEmbed HTTP API Server with Observability")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--grpc-host", default="localhost", help="gRPC server host")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set environment variables for the config
    os.environ.setdefault("GRPC_PORT", str(args.grpc_port))
    os.environ.setdefault("GRPC_HOST", args.grpc_host)
    os.environ.setdefault("LOG_LEVEL", args.log_level.upper())
    
    startup_logger = get_logger("main")
    startup_logger.info("Starting FastEmbed HTTP API server", extra={
        "host": args.host,
        "port": args.port,
        "grpc_host": args.grpc_host,
        "grpc_port": args.grpc_port,
        "log_level": args.log_level
    })
    
    uvicorn.run(
        "http_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        access_log=False  # We handle our own logging
    )

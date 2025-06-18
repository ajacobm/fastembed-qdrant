"""
FastAPI HTTP wrapper for the FastEmbed gRPC server.
This provides HTTP endpoints that can be wrapped by MCP tools.
"""

import asyncio
import io
import logging
import time
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import uuid

import grpc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi import status
from pydantic import BaseModel

# Import protobuf classes
from proto.embed_pb2 import (
    EmbeddingRequest as GrpcEmbeddingRequest, 
    FileStreamRequest, FileMetadata, ProcessingOptions,
    StatusRequest, ListModelsRequest, UpdateModelConfigRequest
)
from proto.embed_pb2_grpc import EmbeddingServiceStub

from config import load_config
from observability.logger import get_logger, RequestContext, log_operation_context
from observability.log_context import get_current_request_id

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


class HTTPEmbeddingService:
    """HTTP service that connects to the gRPC FastEmbed server."""
    
    def __init__(self, grpc_host: str = "localhost", grpc_port: int = 50051):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.channel = None
        self.stub = None
    
    async def connect(self):
        """Connect to the gRPC server."""
        with log_operation_context("grpc_connect", {"host": self.grpc_host, "port": self.grpc_port}):
            self.channel = grpc.aio.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
            self.stub = EmbeddingServiceStub(self.channel)
            logger.info(f"Connected to gRPC server at {self.grpc_host}:{self.grpc_port}")
    
    async def disconnect(self):
        """Disconnect from the gRPC server."""
        if self.channel:
            with log_operation_context("grpc_disconnect", {"host": self.grpc_host, "port": self.grpc_port}):
                await self.channel.close()
    
    async def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get embeddings for a list of texts."""
        operation_data = {
            "text_count": len(texts), 
            "model_name": model_name or "current",
            "total_chars": sum(len(text) for text in texts)
        }
        
        with log_operation_context("grpc_get_embeddings", operation_data):
            request = GrpcEmbeddingRequest(texts=texts, model_name=model_name or "")
            
            operation_data = {
            "filename": filename,
            "content_type": content_type,
            "file_size": len(file_content),
            "model_name": options.model_name or "current",
            "chunk_size": options.chunk_size,
            "store_in_qdrant": options.store_in_qdrant,
            "collection_name": options.collection_name
        }
        
        with log_operation_context("grpc_process_file_stream", operation_data):
            try:
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
                
                logger.info("Successfully generated embeddings", extra={
                    "embeddings_count": len(embeddings),
                    "model_name": result["model_name"]
                })
                
                return result
                
            except grpc.RpcError as e:
                logger.error(f"gRPC error getting embeddings: {e}", extra={
                    "grpc_code": e.code().name if e.code() else "UNKNOWN",
                    "grpc_details": e.details()
                })
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
    
    async def process_file_stream(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        options: FileProcessRequest
    ) -> Dict[str, Any]:
        """Process a file through the streaming endpoint."""
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
            
            return {
                "success": response.success,
                "message": response.message,
                "chunks_processed": response.chunks_processed,
                "embeddings_created": response.embeddings_created,
                "points_stored": response.points_stored,
                "chunk_ids": list(response.chunk_ids),
                "document_id": response.document_id or None
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error processing file: {e}")
            raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
    
    async def update_model_config(self, model_name: str, qdrant_collection: Optional[str]=None) -> Dict[str, Any]:
        """Update the active model/chunking/Qdrant collection via gRPC."""
        try:
            request = UpdateModelConfigRequest(model_name=model_name)
            if qdrant_collection:
                request.qdrant_collection = qdrant_collection
            response = await self.stub.UpdateModelConfig(request)
            return {
                "success": response.success,
                "message": response.message,
                "active_model": response.active_model,
                "chunk_size": response.chunk_size,
                "chunk_overlap": response.chunk_overlap,
                "qdrant_collection": response.qdrant_collection,
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error updating model config: {e}")
            raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get server status with full model configuration."""
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
            
            return status_data
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting status: {e}")
            raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List supported models."""
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
            return models
        except grpc.RpcError as e:
            logger.error(f"gRPC error listing models: {e}")
            raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")


# Global service instance
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global service
    config = load_config()
    service = HTTPEmbeddingService(grpc_port=config.grpc_port)
    await service.connect()
    yield
    await service.disconnect()


# Create FastAPI app
app = FastAPI(
    title="FastEmbed HTTP API", 
    description="HTTP wrapper for FastEmbed gRPC server with Qdrant integration",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FastEmbed HTTP API", "version": "2.0.0"}


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text."""
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
    # Note: custom_metadata as JSON string in form data
    custom_metadata: Optional[str] = Form(None)
):
    """Process an uploaded file and generate embeddings."""
    # Read file content
    content = await file.read()
    
    # Parse custom metadata if provided
    metadata = {}
    if custom_metadata:
        try:
            import json
            metadata = json.loads(custom_metadata)
        except json.JSONDecodeError:
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
    result = await service.get_status()
    return ServerStatus(**result)


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List supported models."""
    result = await service.list_models()
    return [ModelInfo(**model) for model in result]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        status = await service.get_status()
        return {"status": "healthy", "server_version": status["server_version"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

from fastapi import Body

class UpdateModelConfigRequestBody(BaseModel):
    model_name: str
    qdrant_collection: Optional[str] = None

@app.post("/update-model-config")
async def update_model_config(request: UpdateModelConfigRequestBody = Body(...)):
    """
    Update the embedding model, chunk_size, chunk_overlap, or Qdrant collection.
    Body params:
        model_name: str
        qdrant_collection: Optional[str]
    """
    result = await service.update_model_config(
        model_name=request.model_name,
        qdrant_collection=request.qdrant_collection,
    )
    return result


if __name__ == "__main__":
    import uvicorn
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="FastEmbed HTTP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--grpc-host", default="localhost", help="gRPC server host")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set environment variables for the config
    os.environ.setdefault("GRPC_PORT", str(args.grpc_port))
    os.environ.setdefault("GRPC_HOST", args.grpc_host)
    
    logger.info(f"Starting FastEmbed HTTP API server on {args.host}:{args.port}")
    logger.info(f"Connecting to gRPC server at {args.grpc_host}:{args.grpc_port}")
    
    uvicorn.run(
        "http_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload
    )

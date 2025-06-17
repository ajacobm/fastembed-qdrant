"""
FastAPI HTTP wrapper for the FastEmbed server.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str

class ServerStatus(BaseModel):
    server_version: str
    current_model: str
    cuda_available: bool
    qdrant_connected: bool
    configuration: Dict[str, str]
    uptime_seconds: int

# Create FastAPI app
app = FastAPI(
    title="FastEmbed-Qdrant HTTP API",
    description="HTTP API for FastEmbed server with Qdrant integration",
    version="2.0.0"
)

# Global service instance
service = None

@app.on_event("startup")
async def startup():
    """Initialize the service on startup."""
    global service
    from enhanced_server import EmbeddingServicer
    service = EmbeddingServicer()
    await service.initialize()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FastEmbed-Qdrant HTTP API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        status = service.get_status()
        return {"status": "healthy", "server_version": status["server_version"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    result = service.get_status()
    return ServerStatus(**result)

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text."""
    try:
        # Load model if specified and different from current
        if request.model_name and request.model_name != service.model_name:
            success = await service.load_model(request.model_name)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to load model {request.model_name}")
        
        embeddings = service.get_embeddings(request.texts)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=service.model_name or "default"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    store_in_qdrant: bool = Form(False)
):
    """Process an uploaded file and generate embeddings."""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8', errors='ignore')
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Load model if specified
        if model_name and model_name != service.model_name:
            success = await service.load_model(model_name)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
        
        # Simple chunking (for demonstration)
        from text_chunker import TextChunker
        chunk_size = chunk_size or service.config.chunk_size
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=service.config.chunk_overlap)
        
        chunks = chunker.chunk_text(text_content, {
            'filename': file.filename,
            'content_type': file.content_type
        })
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = service.get_embeddings(chunk_texts)
        
        # Store in Qdrant if requested and available
        points_stored = 0
        if store_in_qdrant and service.qdrant_store:
            try:
                from model_config import get_model_dimensions
                collection_name = service.config.qdrant.collection_name
                vector_size = get_model_dimensions(service.model_name)
                
                # Ensure collection exists
                collection_ready = await service.qdrant_store.ensure_collection(
                    collection_name, vector_size
                )
                
                if collection_ready:
                    success, stored_ids = await service.qdrant_store.store_embeddings(
                        collection_name=collection_name,
                        chunks=chunks,
                        embeddings=embeddings
                    )
                    if success:
                        points_stored = len(stored_ids)
            except Exception as e:
                logger.error(f"Failed to store in Qdrant: {e}")
        
        return {
            "success": True,
            "message": f"Successfully processed file {file.filename}",
            "chunks_processed": len(chunks),
            "embeddings_created": len(embeddings),
            "points_stored": points_stored
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    port = int(os.getenv("HTTP_PORT", "8000"))
    host = os.getenv("HTTP_HOST", "0.0.0.0")
    
    logger.info(f"Starting FastEmbed HTTP server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)
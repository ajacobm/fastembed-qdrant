"""
Example client for testing the enhanced FastEmbed server with file streaming.
"""

import asyncio
import grpc
from typing import AsyncGenerator
import io

# Import protobuf classes
from proto.embed_pb2 import (
    EmbeddingRequest, EmbeddingResponse, Embedding,
    FileStreamRequest, FileMetadata, ProcessingOptions,
    StatusRequest, ListModelsRequest
)
from proto.embed_pb2_grpc import EmbeddingServiceStub


class FastEmbedClient:
    """Client for the enhanced FastEmbed gRPC server."""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.address = f"{host}:{port}"
        self.channel = None
        self.stub = None
    
    async def connect(self):
        """Connect to the server."""
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = EmbeddingServiceStub(self.channel)
        print(f"Connected to FastEmbed server at {self.address}")
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.channel:
            await self.channel.close()
    
    async def get_status(self):
        """Get server status."""
        response = await self.stub.GetStatus(StatusRequest())
        print("\n=== Server Status ===")
        print(f"Version: {response.server_version}")
        print(f"Current Model: {response.current_model}")
        print(f"CUDA Available: {response.cuda_available}")
        print(f"Qdrant Connected: {response.qdrant_connected}")
        print(f"Uptime: {response.uptime_seconds} seconds")
        print("Configuration:")
        for key, value in response.configuration.items():
            print(f"  {key}: {value}")
        return response
    
    async def list_models(self):
        """List supported models."""
        response = await self.stub.ListModels(ListModelsRequest())
        print(f"\n=== Supported Models ({len(response.models)}) ===")
        for model in response.models[:10]:  # Show first 10
            print(f"- {model.model_name}")
            print(f"  Dimensions: {model.dimensions}")
            print(f"  Max Length: {model.max_length}")
            print(f"  Default Chunk Size: {model.default_chunk_size}")
            print(f"  Size: {model.size_gb:.3f} GB")
            print(f"  License: {model.license}")
            print()
        if len(response.models) > 10:
            print(f"... and {len(response.models) - 10} more models")
        return response
    
    async def get_embeddings(self, texts: list, model_name: str = ""):
        """Get embeddings for texts."""
        request = EmbeddingRequest(texts=texts, model_name=model_name)
        response = await self.stub.GetEmbeddings(request)
        
        print(f"\n=== Generated {len(response.embeddings)} Embeddings ===")
        for i, embedding in enumerate(response.embeddings):
            print(f"Text {i+1}: '{texts[i][:50]}{'...' if len(texts[i]) > 50 else ''}'")
            print(f"  Embedding dimension: {embedding.dimension}")
            print(f"  First 5 values: {embedding.vector[:5]}")
            print()
        
        return response
    
    async def process_file_stream(
        self,
        file_content: str,
        filename: str = "test.txt",
        model_name: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        store_in_qdrant: bool = False,
        collection_name: str = "documents",
        document_id: str = "",
        custom_metadata: dict = None
    ):
        """Process a file through streaming."""
        print(f"\n=== Processing File Stream: {filename} ===")
        print(f"Content length: {len(file_content)} characters")
        print(f"Model: {model_name or 'default'}")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"Store in Qdrant: {store_in_qdrant}")
        
        async def create_stream() -> AsyncGenerator[FileStreamRequest, None]:
            # Send metadata first
            metadata = FileMetadata(
                filename=filename,
                content_type="text/plain",
                file_size=len(file_content),
                custom_metadata=custom_metadata or {},
                document_id=document_id or f"doc_{int(asyncio.get_event_loop().time())}"
            )
            
            yield FileStreamRequest(
                metadata=metadata,
                model_name=model_name
            )
            
            # Send processing options
            processing_opts = ProcessingOptions(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                store_in_qdrant=store_in_qdrant,
                collection_name=collection_name
            )
            
            yield FileStreamRequest(options=processing_opts)
            
            # Send file content in chunks
            file_bytes = file_content.encode('utf-8')
            stream_chunk_size = 8192  # 8KB chunks
            
            for i in range(0, len(file_bytes), stream_chunk_size):
                chunk = file_bytes[i:i + stream_chunk_size]
                yield FileStreamRequest(chunk_data=chunk)
                print(f"Sent chunk {i//stream_chunk_size + 1}: {len(chunk)} bytes")
        
        # Send the stream and get response
        response = await self.stub.ProcessFileStream(create_stream())
        
        print(f"\n=== Processing Results ===")
        print(f"Success: {response.success}")
        print(f"Message: {response.message}")
        print(f"Chunks processed: {response.chunks_processed}")
        print(f"Embeddings created: {response.embeddings_created}")
        print(f"Points stored in Qdrant: {response.points_stored}")
        print(f"Document ID: {response.document_id}")
        print(f"Chunk IDs: {len(response.chunk_ids)} IDs")
        
        return response


async def main():
    """Main example function."""
    client = FastEmbedClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Get server status
        await client.get_status()
        
        # List supported models
        await client.list_models()
        
        # Test basic embeddings
        test_texts = [
            "This is a test sentence for embedding.",
            "FastEmbed provides high-performance text embeddings.",
            "Qdrant is a vector database for similarity search."
        ]
        
        await client.get_embeddings(test_texts)
        
        # Test file streaming with a longer text
        long_text = """
        FastEmbed is a lightweight, fast, and accurate library built for Retrieval Augmented Generation (RAG) & related tasks.
        
        It's designed to be:
        - Lightweight: No hidden dependencies. FastEmbed is built with a minimal set of dependencies.
        - Fast: FastEmbed is designed for speed. It's built on top of ONNX Runtime, which provides excellent performance.
        - Accurate: FastEmbed is built on top of state-of-the-art models.
        - Easy to use: FastEmbed is designed to be easy to use. It provides a simple API for embedding generation.
        
        FastEmbed supports a variety of models for text embedding, including:
        - BAAI/bge-small-en-v1.5: A small English model with 384 dimensions
        - BAAI/bge-base-en-v1.5: A base English model with 768 dimensions  
        - BAAI/bge-large-en-v1.5: A large English model with 1024 dimensions
        - sentence-transformers/all-MiniLM-L6-v2: A compact model with good performance
        
        When integrated with Qdrant, FastEmbed provides a complete solution for vector search and similarity matching.
        Qdrant is a vector database that enables efficient similarity search at scale.
        
        This text will be chunked and processed by the enhanced FastEmbed server, demonstrating the file streaming
        capabilities and Qdrant integration. Each chunk will be embedded and optionally stored in the vector database
        for later retrieval and similarity search.
        """ * 3  # Repeat to make it longer
        
        await client.process_file_stream(
            file_content=long_text,
            filename="fastembed_demo.txt",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=400,
            chunk_overlap=50,
            store_in_qdrant=False,  # Set to True if Qdrant is configured
            custom_metadata={
                "author": "FastEmbed Demo",
                "category": "documentation",
                "topic": "embedding"
            }
        )
        
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()}: {e.details()}")
        print("Make sure the FastEmbed server is running on localhost:50051")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
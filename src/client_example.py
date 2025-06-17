"""
Example client for testing the FastEmbed-Qdrant server.
"""

import asyncio
import aiohttp
import json


class FastEmbedClient:
    """Client for the FastEmbed HTTP server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def health_check(self):
        """Check server health."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Server is healthy: {data}")
                    return True
                else:
                    print(f"❌ Server health check failed: {response.status}")
                    return False
    
    async def get_status(self):
        """Get server status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print("\n=== Server Status ===")
                    print(f"Version: {data['server_version']}")
                    print(f"Current Model: {data['current_model']}")
                    print(f"CUDA Available: {data['cuda_available']}")
                    print(f"Qdrant Connected: {data['qdrant_connected']}")
                    print(f"Uptime: {data['uptime_seconds']} seconds")
                    return data
                else:
                    print(f"❌ Failed to get status: {response.status}")
                    return None
    
    async def get_embeddings(self, texts: list, model_name: str = None):
        """Get embeddings for texts."""
        payload = {"texts": texts}
        if model_name:
            payload["model_name"] = model_name
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"\n=== Generated {len(data['embeddings'])} Embeddings ===")
                    for i, embedding in enumerate(data['embeddings']):
                        text_preview = texts[i][:50] + ('...' if len(texts[i]) > 50 else '')
                        print(f"Text {i+1}: '{text_preview}'")
                        print(f"  Embedding dimension: {len(embedding)}")
                        print(f"  First 5 values: {embedding[:5]}")
                    return data
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to get embeddings: {response.status} - {error_text}")
                    return None
    
    async def process_file_content(
        self, 
        content: str, 
        filename: str = "test.txt", 
        model_name: str = None,
        chunk_size: int = 512,
        store_in_qdrant: bool = False
    ):
        """Process file content."""
        data = aiohttp.FormData()
        data.add_field('file', content, filename=filename, content_type='text/plain')
        
        if model_name:
            data.add_field('model_name', model_name)
        if chunk_size:
            data.add_field('chunk_size', str(chunk_size))
        if store_in_qdrant:
            data.add_field('store_in_qdrant', 'true')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/process-file",
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"\n=== File Processing Results ===")
                    print(f"Success: {result['success']}")
                    print(f"Message: {result['message']}")
                    print(f"Chunks processed: {result['chunks_processed']}")
                    print(f"Embeddings created: {result['embeddings_created']}")
                    print(f"Points stored in Qdrant: {result['points_stored']}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to process file: {response.status} - {error_text}")
                    return None


async def main():
    """Main example function."""
    print("🚀 Testing FastEmbed-Qdrant Server")
    
    client = FastEmbedClient()
    
    # Health check
    healthy = await client.health_check()
    if not healthy:
        print("❌ Server is not healthy. Make sure it's running on http://localhost:8000")
        return
    
    # Get server status
    await client.get_status()
    
    # Test basic embeddings
    test_texts = [
        "This is a test sentence for embedding.",
        "FastEmbed provides high-performance text embeddings.",
        "Qdrant is a vector database for similarity search."
    ]
    
    await client.get_embeddings(test_texts, model_name="BAAI/bge-base-en-v1.5")
    
    # Test file processing with longer content
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
    
    This text will be chunked and processed by the FastEmbed server, demonstrating the file processing
    capabilities and optional Qdrant integration. Each chunk will be embedded and optionally stored 
    in the vector database for later retrieval and similarity search.
    """ * 2
    
    await client.process_file_content(
        content=long_text,
        filename="fastembed_demo.txt",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=400,
        store_in_qdrant=False  # Set to True if Qdrant is configured
    )
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
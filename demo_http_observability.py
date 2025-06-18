#!/usr/bin/env python3
"""
Demo script for FastEmbed HTTP Server with Observability.
Shows structured logging and request correlation in action.
"""

import time
import json
import asyncio
import aiohttp
from pathlib import Path
import sys

# Add src to path for observability imports
src_path = Path(__file__).parent / 'src'  
sys.path.insert(0, str(src_path))

from observability.logger import setup_logging, get_logger
from observability.log_config import LogConfig

async def demo_http_client():
    """Demo HTTP client that tests the observability features."""
    
    # Setup logging for the demo client
    log_config = LogConfig.from_env()
    setup_logging(level='INFO', format_type='text', output_type='console')
    logger = get_logger('demo_client')
    
    base_url = "http://localhost:8000"
    
    logger.info("🚀 Starting FastEmbed HTTP Server Observability Demo")
    logger.info(f"📡 Connecting to: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health Check
        logger.info("🩺 Testing health endpoint...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Health check passed", status_code=response.status, 
                              response=data)
                else:
                    logger.error("❌ Health check failed", status_code=response.status)
        except Exception as e:
            logger.error("❌ Health check error", error=str(e))
            
        await asyncio.sleep(1)
        
        # Test 2: Get Status
        logger.info("📊 Testing status endpoint...")
        try:
            async with session.get(f"{base_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Status retrieved", status_code=response.status,
                              server_version=data.get('server_version'),
                              current_model=data.get('current_model'),
                              cuda_available=data.get('cuda_available'))
                else:
                    logger.error("❌ Status check failed", status_code=response.status)
        except Exception as e:
            logger.error("❌ Status check error", error=str(e))
            
        await asyncio.sleep(1)
        
        # Test 3: List Models
        logger.info("📋 Testing models endpoint...")
        try:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Models listed", status_code=response.status,
                              model_count=len(data))
                    if data:
                        logger.info("📝 Sample model", model_info=data[0])
                else:
                    logger.error("❌ Models listing failed", status_code=response.status)
        except Exception as e:
            logger.error("❌ Models listing error", error=str(e))
            
        await asyncio.sleep(1)
        
        # Test 4: Generate Embeddings
        logger.info("🧮 Testing embeddings endpoint...")
        try:
            payload = {
                "texts": [
                    "Hello, this is a test sentence for embedding generation.",
                    "FastEmbed with observability provides excellent monitoring capabilities.",
                    "Structured logging makes debugging and monitoring much easier."
                ],
                "model_name": "BAAI/bge-base-en-v1.5"
            }
            
            async with session.post(f"{base_url}/embeddings", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Embeddings generated", status_code=response.status,
                              embeddings_count=len(data.get('embeddings', [])),
                              model_name=data.get('model_name'))
                else:
                    logger.error("❌ Embeddings generation failed", 
                               status_code=response.status)
        except Exception as e:
            logger.error("❌ Embeddings generation error", error=str(e))
            
        await asyncio.sleep(1)
        
        # Test 5: File Processing (simulate error case)
        logger.info("📄 Testing file processing (error simulation)...")
        try:
            # Create a test file
            test_content = b"This is test content for file processing with observability logging."
            
            data = aiohttp.FormData()
            data.add_field('file', test_content, filename='test.txt', 
                          content_type='text/plain')
            data.add_field('model_name', 'BAAI/bge-base-en-v1.5')
            data.add_field('store_in_qdrant', 'true')
            data.add_field('collection_name', 'demo_collection')
            
            async with session.post(f"{base_url}/process-file", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ File processed", status_code=response.status,
                              chunks_processed=result.get('chunks_processed'),
                              embeddings_created=result.get('embeddings_created'),
                              points_stored=result.get('points_stored'))
                else:
                    logger.warning("⚠️ File processing failed (expected if no gRPC server)",
                                 status_code=response.status)
        except Exception as e:
            logger.warning("⚠️ File processing error (expected if no gRPC server)", 
                          error=str(e))
    
    logger.info("🎉 Demo completed! Check the server logs for structured observability data.")
    logger.info("📊 Server observability features demonstrated:")
    logger.info("  ✨ Request correlation IDs")
    logger.info("  ✨ Operation context tracking")  
    logger.info("  ✨ Performance timing")
    logger.info("  ✨ Error correlation")
    logger.info("  ✨ Structured JSON logging")


def main():
    """Main demo function."""
    print("=" * 60)
    print("🚀 FastEmbed HTTP Server Observability Demo")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("1. Start gRPC server: ./dev.sh run-grpc")
    print("2. Start HTTP server: ./dev.sh run-http-obs") 
    print("3. Then run this demo")
    print()
    
    try:
        asyncio.run(demo_http_client())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
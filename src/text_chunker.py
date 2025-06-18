"""
Text chunking utilities for processing file streams.
"""

import re
from typing import List, Generator, Dict, Any
from dataclasses import dataclass
import uuid


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    id: str
    text: str
    start_offset: int
    end_offset: int
    chunk_index: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class TextChunker:
    """Handles text chunking with overlaps and smart boundary detection."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(
        self, 
        text: str, 
        base_metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks, trying to break at sentence boundaries.
        
        Args:
            text: Text to chunk
            base_metadata: Base metadata to include in all chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
            
        base_metadata = base_metadata or {}
        chunks = []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            # Text is small enough to fit in one chunk
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                text=text,
                start_offset=0,
                end_offset=len(text),
                chunk_index=0,
                metadata=base_metadata.copy()
            )
            return [chunk]
        
        chunk_index = 0
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Try to find a good breaking point
                chunk_text = text[start:end]
                
                # Look for sentence boundaries within the last 10% of the chunk
                boundary_search_start = max(0, len(chunk_text) - self.chunk_size // 10)
                
                # Try to break at sentence boundaries
                sentence_breaks = list(re.finditer(r'[.!?]\s+', chunk_text[boundary_search_start:]))
                if sentence_breaks:
                    # Use the last sentence break
                    break_pos = boundary_search_start + sentence_breaks[-1].end()
                    chunk_text = chunk_text[:break_pos]
                else:
                    # Try to break at word boundaries
                    word_break = chunk_text.rfind(' ', boundary_search_start)
                    if word_break > boundary_search_start:
                        chunk_text = chunk_text[:word_break]
            
            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_size': len(chunk_text),
                'total_chunks': None,  # Will be updated later
                'char_start': start,
                'char_end': start + len(chunk_text)
            })
            
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                text=chunk_text.strip(),
                start_offset=start,
                end_offset=start + len(chunk_text),
                chunk_index=chunk_index,
                metadata=chunk_metadata
            )
            
            if chunk.text:  # Only add non-empty chunks
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next start position with overlap
            next_start = start + len(chunk_text) - self.chunk_overlap
            
            # Ensure we make progress (avoid infinite loops)
            if next_start <= start:
                next_start = start + max(1, len(chunk_text) // 2)
            
            start = next_start
        
        # Update total chunks count in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def chunk_streaming_text(
        self, 
        text_generator: Generator[str, None, None],
        base_metadata: Dict[str, Any] = None
    ) -> Generator[TextChunk, None, None]:
        """
        Process streaming text and yield chunks as they become available.
        
        Args:
            text_generator: Generator that yields text pieces
            base_metadata: Base metadata to include in all chunks
            
        Yields:
            TextChunk objects as they are created
        """
        base_metadata = base_metadata or {}
        buffer = ""
        chunk_index = 0
        total_offset = 0
        
        for text_piece in text_generator:
            buffer += text_piece
            
            # Process the buffer when it gets large enough
            while len(buffer) > self.chunk_size + self.chunk_overlap:
                # Find chunk boundary
                chunk_end = self.chunk_size
                
                # Try to find sentence boundary
                boundary_search_start = max(0, chunk_end - self.chunk_size // 10)
                sentence_breaks = list(re.finditer(r'[.!?]\s+', buffer[boundary_search_start:chunk_end]))
                
                if sentence_breaks:
                    chunk_end = boundary_search_start + sentence_breaks[-1].end()
                else:
                    # Try word boundary
                    word_break = buffer.rfind(' ', boundary_search_start, chunk_end)
                    if word_break > boundary_search_start:
                        chunk_end = word_break
                
                # Extract chunk text
                chunk_text = buffer[:chunk_end].strip()
                
                if chunk_text:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'chunk_size': len(chunk_text),
                        'char_start': total_offset,
                        'char_end': total_offset + len(chunk_text),
                        'is_streaming': True
                    })
                    
                    chunk = TextChunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        start_offset=total_offset,
                        end_offset=total_offset + len(chunk_text),
                        chunk_index=chunk_index,
                        metadata=chunk_metadata
                    )
                    
                    yield chunk
                    chunk_index += 1
                
                # Update buffer and offset
                overlap_start = max(0, chunk_end - self.chunk_overlap)
                buffer = buffer[overlap_start:]
                total_offset += overlap_start
        
        # Process remaining buffer
        if buffer.strip():
            chunk_text = buffer.strip()
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_size': len(chunk_text),
                'char_start': total_offset,
                'char_end': total_offset + len(chunk_text),
                'is_streaming': True,
                'is_final_chunk': True
            })
            
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                start_offset=total_offset,
                end_offset=total_offset + len(chunk_text),
                chunk_index=chunk_index,
                metadata=chunk_metadata
            )
            
            yield chunk
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


def create_chunker(chunk_size: int, chunk_overlap: int) -> TextChunker:
    """Factory function to create a text chunker."""
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
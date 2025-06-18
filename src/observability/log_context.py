"""Request correlation and context management for FastEmbed server."""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import structlog

from .logger import get_logger, set_request_context, clear_request_context, log_performance


class RequestContext:
    """Manages request context and correlation across the application."""
    
    def __init__(self, request_id: str, method: Optional[str] = None, 
                 user_id: Optional[str] = None, model: Optional[str] = None):
        self.request_id = request_id
        self.method = method
        self.user_id = user_id
        self.model = model
        self.start_time = time.time()
        self.logger = get_logger('request')
    
    def __enter__(self):
        set_request_context(
            req_id=self.request_id,
            uid=self.user_id,
            method=self.method,
            model=self.model
        )
        self.logger.debug(
            "Request started",
            method=self.method,
            user_id=self.user_id,
            model=self.model
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                "Request failed",
                duration=duration,
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                exc_info=True
            )
        else:
            self.logger.info(
                "Request completed",
                duration=duration
            )
        
        clear_request_context()


@contextmanager
def operation_context(operation_name: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
    """Context manager for tracking individual operations within a request."""
    logger = get_logger('operation')
    start_time = time.time()
    
    context = {'operation': operation_name, **kwargs}
    
    logger.debug("Operation started", **context)
    
    try:
        yield context
        duration = time.time() - start_time
        context['duration'] = duration  # Set duration in context
        
        # Remove duration from context before passing to logger to avoid collision
        log_context = {k: v for k, v in context.items() if k != 'duration'}
        logger.debug("Operation completed", duration=duration, **log_context)
        log_performance(logger, operation_name, duration, **kwargs)
    except Exception as e:
        duration = time.time() - start_time
        context['duration'] = duration  # Set duration in context even for errors
        
        # Remove duration from context before passing to logger to avoid collision  
        log_context = {k: v for k, v in context.items() if k != 'duration'}
        logger.error(
            "Operation failed",
            duration=duration,
            error=str(e),
            error_type=type(e).__name__,
            **log_context,
            exc_info=True
        )
        raise


def log_model_operation(operation: str, model_name: str, duration: Optional[float] = None, 
                       error: Optional[str] = None, **kwargs) -> None:
    """Log model-specific operations with standardized format."""
    logger = get_logger('model')
    
    log_data = {
        'operation': operation,
        'model': model_name,
        **kwargs
    }
    
    if duration is not None:
        log_data['duration'] = duration
    
    if error:
        logger.error("Model operation failed", error=error, **log_data)
    else:
        logger.info("Model operation", **log_data)


def log_qdrant_operation(operation: str, collection: str, points_count: Optional[int] = None,
                        duration: Optional[float] = None, error: Optional[str] = None, 
                        **kwargs) -> None:
    """Log Qdrant operations with standardized format."""
    logger = get_logger('qdrant')
    
    log_data = {
        'operation': operation,
        'collection': collection,
        **kwargs
    }
    
    if points_count is not None:
        log_data['points_count'] = points_count
    
    if duration is not None:
        log_data['duration'] = duration
    
    if error:
        logger.error("Qdrant operation failed", error=error, **log_data)
    else:
        logger.info("Qdrant operation", **log_data)


def log_file_processing(filename: str, size_bytes: int, chunks_count: int, 
                       duration: float, model: str, stored_to_qdrant: bool = False,
                       error: Optional[str] = None) -> None:
    """Log file processing operations with standardized format."""
    logger = get_logger('file_processing')
    
    log_data = {
        'filename': filename,
        'size_bytes': size_bytes,
        'chunks_count': chunks_count,
        'duration': duration,
        'model': model,
        'stored_to_qdrant': stored_to_qdrant
    }
    
    if error:
        logger.error("File processing failed", error=error, **log_data)
    else:
        logger.info("File processed successfully", **log_data)


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    from .logger import request_id
    return request_id.get()
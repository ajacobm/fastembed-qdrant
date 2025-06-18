"""Enhanced structured logging for FastEmbed server."""

import logging
import logging.handlers
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

# Request context tracking
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
method_name: ContextVar[Optional[str]] = ContextVar('method_name', default=None)
model_name: ContextVar[Optional[str]] = ContextVar('model_name', default=None)

# Global logger instance
_logger_configured = False


def add_context_fields(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request context fields to log entries."""
    event_dict['request_id'] = request_id.get()
    event_dict['user_id'] = user_id.get()
    event_dict['method'] = method_name.get()
    event_dict['model'] = model_name.get()
    event_dict['service'] = 'fastembed-server'
    event_dict['timestamp'] = time.time()
    return event_dict


def add_performance_fields(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add performance timing if available."""
    if 'duration' in event_dict:
        duration = event_dict['duration']
        if isinstance(duration, (int, float)):
            event_dict['duration_ms'] = round(duration * 1000, 2)
    return event_dict


def setup_logging(
    level: str = 'INFO',
    format_type: str = 'json',
    output_type: str = 'console',
    file_path: str = '/app/logs/fastembed.log',
    max_size: str = '100MB',
    backup_count: int = 5
) -> None:
    """Set up structured logging configuration."""
    global _logger_configured
    
    if _logger_configured:
        return
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_context_fields,
        add_performance_fields,
    ]
    
    if format_type == 'json':
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[]
    )
    
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if output_type == 'console':
        handler = logging.StreamHandler()
    elif output_type == 'file':
        # Create log directory if it doesn't exist
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert size string to bytes
        max_bytes = _parse_size(max_size)
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
    else:
        # For remote logging, use console for now
        handler = logging.StreamHandler()
    
    # Don't add formatter for structlog (it handles formatting)
    root_logger.addHandler(handler)
    
    _logger_configured = True


def _parse_size(size_str: str) -> int:
    """Parse size string like '100MB' into bytes."""
    size_str = size_str.upper().strip()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the specified name."""
    return structlog.get_logger(f'fastembed.{name}')


def set_request_context(
    req_id: Optional[str] = None,
    uid: Optional[str] = None,
    method: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """Set request context for logging correlation."""
    if not req_id:
        req_id = str(uuid.uuid4())
    
    request_id.set(req_id)
    if uid:
        user_id.set(uid)
    if method:
        method_name.set(method)
    if model:
        model_name.set(model)
    
    return req_id


def update_model_context(model: str) -> None:
    """Update model context for current request."""
    model_name.set(model)


def clear_request_context() -> None:
    """Clear the current request context."""
    request_id.set(None)
    user_id.set(None)
    method_name.set(None)
    model_name.set(None)


def log_performance(logger: structlog.BoundLogger, operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics for operations."""
    logger.info(
        "Performance metric",
        operation=operation,
        duration=duration,
        **kwargs
    )
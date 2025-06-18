"""FastEmbed Server Observability Package.

This package provides structured logging, metrics collection, and monitoring
capabilities for the FastEmbed server.

Components:
- logger: Enhanced structured logging with request context
- log_config: Environment-based logging configuration
- log_context: Request correlation and context management
"""

from .logger import (
    get_logger, 
    set_request_context, 
    clear_request_context, 
    setup_logging,
    update_model_context,
    log_performance
)
from .log_config import LogConfig
from .log_context import (
    RequestContext,
    operation_context,
    log_model_operation,
    log_qdrant_operation,
    log_file_processing
)

__all__ = [
    'get_logger',
    'set_request_context', 
    'clear_request_context',
    'setup_logging',
    'update_model_context',
    'log_performance',
    'LogConfig',
    'RequestContext',
    'operation_context',
    'log_model_operation',
    'log_qdrant_operation',
    'log_file_processing'
]
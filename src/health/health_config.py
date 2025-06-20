"""
Health configuration for FastEmbed Server.
Manages health check settings with container boundary awareness.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class HealthCheckLevel(Enum):
    """Health check detail levels for different consumers."""
    BASIC = "basic"          # Simple up/down for load balancers
    DETAILED = "detailed"    # Detailed status for monitoring systems  
    DIAGNOSTIC = "diagnostic"  # Full diagnostic info for troubleshooting


class ResponseFormat(Enum):
    """Health check response formats."""
    JSON = "json"
    PROMETHEUS = "prometheus"
    TEXT = "text"


@dataclass
class HealthConfig:
    """Configuration for health check system."""
    
    # Basic health check settings
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    startup_grace_period: int = 120  # seconds
    shutdown_grace_period: int = 30  # seconds
    
    # Container boundary settings
    enable_container_monitoring: bool = True
    container_memory_threshold: float = 85.0  # percentage
    container_cpu_threshold: float = 90.0  # percentage
    container_disk_threshold: float = 90.0  # percentage
    
    # gRPC health check settings
    grpc_health_enabled: bool = True
    grpc_health_timeout: float = 5.0  # seconds
    grpc_startup_timeout: int = 300  # seconds
    grpc_health_check_interval: int = 10  # seconds
    
    # HTTP health check settings
    http_health_enabled: bool = True
    http_health_timeout: float = 3.0  # seconds
    http_startup_timeout: int = 60  # seconds
    
    # Model health check settings
    model_health_enabled: bool = True
    model_health_timeout: float = 10.0  # seconds
    model_warmup_timeout: int = 180  # seconds
    
    # Qdrant health check settings
    qdrant_health_enabled: bool = True
    qdrant_health_timeout: float = 5.0  # seconds
    qdrant_connection_timeout: int = 30  # seconds
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    memory_threshold: float = 80.0  # percentage
    gpu_memory_threshold: float = 85.0  # percentage
    
    # Health check endpoints
    health_endpoints: Dict[str, HealthCheckLevel] = field(default_factory=lambda: {
        "/health": HealthCheckLevel.BASIC,
        "/health/detailed": HealthCheckLevel.DETAILED,  
        "/health/diagnostic": HealthCheckLevel.DIAGNOSTIC,
        "/readiness": HealthCheckLevel.BASIC,
        "/liveness": HealthCheckLevel.BASIC
    })
    
    # Default response format per endpoint
    endpoint_formats: Dict[str, ResponseFormat] = field(default_factory=lambda: {
        "/health": ResponseFormat.JSON,
        "/health/detailed": ResponseFormat.JSON,
        "/health/diagnostic": ResponseFormat.JSON,
        "/readiness": ResponseFormat.JSON,
        "/liveness": ResponseFormat.JSON,
        "/metrics/health": ResponseFormat.PROMETHEUS
    })
    
    # Health check dependencies (fail fast if these fail)
    critical_dependencies: List[str] = field(default_factory=lambda: [
        "grpc_server", "model_loaded"
    ])
    
    # Optional dependencies (don't fail overall health)
    optional_dependencies: List[str] = field(default_factory=lambda: [
        "qdrant_connection", "cuda_available"
    ])
    
    @classmethod
    def from_env(cls) -> "HealthConfig":
        """Create configuration from environment variables."""
        return cls(
            enable_health_checks=os.getenv("HEALTH_ENABLE", "true").lower() == "true",
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            startup_grace_period=int(os.getenv("HEALTH_STARTUP_GRACE", "120")),
            shutdown_grace_period=int(os.getenv("HEALTH_SHUTDOWN_GRACE", "30")),
            
            enable_container_monitoring=os.getenv("HEALTH_CONTAINER_MONITORING", "true").lower() == "true",
            container_memory_threshold=float(os.getenv("HEALTH_MEMORY_THRESHOLD", "85.0")),
            container_cpu_threshold=float(os.getenv("HEALTH_CPU_THRESHOLD", "90.0")),
            container_disk_threshold=float(os.getenv("HEALTH_DISK_THRESHOLD", "90.0")),
            
            grpc_health_enabled=os.getenv("HEALTH_GRPC_ENABLE", "true").lower() == "true",
            grpc_health_timeout=float(os.getenv("HEALTH_GRPC_TIMEOUT", "5.0")),
            grpc_startup_timeout=int(os.getenv("GRPC_STARTUP_TIMEOUT", "300")),
            grpc_health_check_interval=int(os.getenv("GRPC_HEALTH_CHECK_INTERVAL", "10")),
            
            http_health_enabled=os.getenv("HEALTH_HTTP_ENABLE", "true").lower() == "true",
            http_health_timeout=float(os.getenv("HEALTH_HTTP_TIMEOUT", "3.0")),
            http_startup_timeout=int(os.getenv("HTTP_STARTUP_TIMEOUT", "60")),
            
            model_health_enabled=os.getenv("HEALTH_MODEL_ENABLE", "true").lower() == "true",
            model_health_timeout=float(os.getenv("HEALTH_MODEL_TIMEOUT", "10.0")),
            model_warmup_timeout=int(os.getenv("MODEL_WARMUP_TIMEOUT", "180")),
            
            qdrant_health_enabled=os.getenv("HEALTH_QDRANT_ENABLE", "true").lower() == "true",
            qdrant_health_timeout=float(os.getenv("HEALTH_QDRANT_TIMEOUT", "5.0")),
            qdrant_connection_timeout=int(os.getenv("QDRANT_CONNECTION_TIMEOUT", "30")),
            
            enable_resource_monitoring=os.getenv("HEALTH_RESOURCE_MONITORING", "true").lower() == "true",
            memory_threshold=float(os.getenv("HEALTH_MEMORY_THRESHOLD", "80.0")),
            gpu_memory_threshold=float(os.getenv("HEALTH_GPU_MEMORY_THRESHOLD", "85.0"))
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if self.health_check_interval <= 0:
            errors.append("health_check_interval must be positive")
        
        if self.startup_grace_period < 0:
            errors.append("startup_grace_period must be non-negative")
            
        if not (0 < self.container_memory_threshold <= 100):
            errors.append("container_memory_threshold must be between 0 and 100")
            
        if not (0 < self.container_cpu_threshold <= 100):
            errors.append("container_cpu_threshold must be between 0 and 100")
            
        if self.grpc_health_timeout <= 0:
            errors.append("grpc_health_timeout must be positive")
            
        if self.http_health_timeout <= 0:
            errors.append("http_health_timeout must be positive")
        
        return errors
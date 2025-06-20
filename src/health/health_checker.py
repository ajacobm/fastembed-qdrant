"""
Comprehensive health checker for FastEmbed Server with Docker/Kubernetes boundary awareness.
Provides different health check levels for different consumers.
"""

import time
import asyncio
import grpc
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

from .health_config import HealthConfig, HealthCheckLevel, ResponseFormat
from .container_health import ContainerHealthMonitor, ResourceUsage
from ..observability.logger import get_logger
from ..observability.log_context import operation_context


class ComponentStatus(Enum):
    """Health status for individual components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class OverallStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"      # All critical components healthy
    DEGRADED = "degraded"    # Some non-critical issues
    UNHEALTHY = "unhealthy"  # Critical components failing
    STARTING = "starting"    # Still in startup phase


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: ComponentStatus
    message: str
    details: Dict[str, Any]
    last_check: float
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass 
class HealthStatus:
    """Complete health status response."""
    overall_status: OverallStatus
    message: str
    timestamp: float
    uptime_seconds: float
    version: str
    components: List[ComponentHealth]
    container_info: Dict[str, Any]
    resource_usage: Dict[str, Any]
    recommendations: List[str]
    
    def to_basic_response(self) -> Dict[str, Any]:
        """Convert to basic health check response (for load balancers)."""
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp,
            "message": self.message
        }
    
    def to_detailed_response(self) -> Dict[str, Any]:
        """Convert to detailed health check response (for monitoring)."""
        return {
            "status": self.overall_status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "components": {
                comp.name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms
                }
                for comp in self.components
            },
            "resource_usage": self.resource_usage,
            "recommendations": self.recommendations[:3]  # Limit for monitoring
        }
    
    def to_diagnostic_response(self) -> Dict[str, Any]:
        """Convert to full diagnostic response (for troubleshooting)."""
        return asdict(self)
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus metrics format."""
        lines = []
        
        # Overall health metric
        overall_value = 1 if self.overall_status == OverallStatus.HEALTHY else 0
        lines.append(f'fastembed_health_status{{service="fastembed"}} {overall_value}')
        
        # Component health metrics
        for comp in self.components:
            comp_value = 1 if comp.status == ComponentStatus.HEALTHY else 0
            lines.append(f'fastembed_component_health{{component="{comp.name}"}} {comp_value}')
            
            if comp.response_time_ms is not None:
                lines.append(f'fastembed_component_response_time_ms{{component="{comp.name}"}} {comp.response_time_ms}')
        
        # Resource usage metrics
        if "cpu_percent" in self.resource_usage:
            lines.append(f'fastembed_cpu_usage_percent {{}} {self.resource_usage["cpu_percent"]}')
        
        if "memory_percent" in self.resource_usage:
            lines.append(f'fastembed_memory_usage_percent {{}} {self.resource_usage["memory_percent"]}')
        
        # Uptime
        lines.append(f'fastembed_uptime_seconds {{}} {self.uptime_seconds}')
        
        return "\n".join(lines) + "\n"


class HealthChecker:
    """Comprehensive health checker with container boundary awareness."""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = get_logger("health_checker")
        self.container_monitor = ContainerHealthMonitor()
        self.start_time = time.time()
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub = None
        
        # Component check methods
        self._component_checkers = {
            "grpc_server": self._check_grpc_server,
            "model_loaded": self._check_model_loaded,
            "qdrant_connection": self._check_qdrant_connection,
            "container_resources": self._check_container_resources,
            "cuda_available": self._check_cuda_available
        }
    
    async def initialize(self, grpc_host: str = "localhost", grpc_port: int = 50051):
        """Initialize the health checker with gRPC connection."""
        try:
            self._grpc_channel = grpc.aio.insecure_channel(f'{grpc_host}:{grpc_port}')
            # Import here to avoid circular imports
            from ..proto.embed_pb2_grpc import EmbeddingServiceStub
            self._grpc_stub = EmbeddingServiceStub(self._grpc_channel)
            self.logger.info("Health checker initialized", extra={
                "grpc_host": grpc_host,
                "grpc_port": grpc_port
            })
        except Exception as e:
            self.logger.error("Failed to initialize health checker", extra={
                "error": str(e),
                "grpc_host": grpc_host, 
                "grpc_port": grpc_port
            })
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._grpc_channel:
            await self._grpc_channel.close()
            self._grpc_channel = None
            self._grpc_stub = None
    
    def is_in_startup_phase(self) -> bool:
        """Check if we're still in the startup grace period."""
        uptime = time.time() - self.start_time
        return uptime < self.config.startup_grace_period
    
    async def check_health(self, level: HealthCheckLevel = HealthCheckLevel.BASIC) -> HealthStatus:
        """Perform health check at the specified level."""
        start_time = time.time()
        
        with operation_context("health_check", level=level.value):
            # Determine which components to check based on level
            components_to_check = self._get_components_for_level(level)
            
            # Check all components
            component_results = []
            for component_name in components_to_check:
                if component_name in self._component_checkers:
                    try:
                        result = await self._component_checkers[component_name]()
                        component_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error checking component {component_name}", extra={
                            "component": component_name,
                            "error": str(e)
                        })
                        component_results.append(ComponentHealth(
                            name=component_name,
                            status=ComponentStatus.UNKNOWN,
                            message=f"Health check failed: {str(e)}",
                            details={},
                            last_check=time.time(),
                            error=str(e)
                        ))
            
            # Determine overall status
            overall_status = self._calculate_overall_status(component_results)
            
            # Get container and resource information
            container_info = {}
            resource_usage = {}
            
            if level in [HealthCheckLevel.DETAILED, HealthCheckLevel.DIAGNOSTIC]:
                try:
                    container_info = self.container_monitor.get_container_health_info()
                    usage = self.container_monitor.get_resource_usage()
                    resource_usage = {
                        "cpu_percent": usage.cpu_percent,
                        "memory_percent": usage.memory_percent,
                        "memory_used_mb": usage.memory_used_mb,
                        "memory_total_mb": usage.memory_total_mb,
                        "disk_percent": usage.disk_percent
                    }
                    if usage.gpu_memory_percent is not None:
                        resource_usage["gpu_memory_percent"] = usage.gpu_memory_percent
                        resource_usage["gpu_memory_used_mb"] = usage.gpu_memory_used_mb
                except Exception as e:
                    self.logger.warning("Failed to get container info", extra={"error": str(e)})
            
            # Generate recommendations
            recommendations = self._generate_recommendations(component_results, resource_usage)
            
            # Create health status
            health_status = HealthStatus(
                overall_status=overall_status,
                message=self._get_status_message(overall_status, component_results),
                timestamp=time.time(),
                uptime_seconds=time.time() - self.start_time,
                version="2.0.0",  # Should come from config
                components=component_results,
                container_info=container_info,
                resource_usage=resource_usage,
                recommendations=recommendations
            )
            
            # Log health check completion
            self.logger.info("Health check completed", extra={
                "level": level.value,
                "overall_status": overall_status.value,
                "check_duration_ms": round((time.time() - start_time) * 1000, 2),
                "components_checked": len(component_results)
            })
            
            return health_status
    
    def _get_components_for_level(self, level: HealthCheckLevel) -> List[str]:
        """Get list of components to check for the given level."""
        if level == HealthCheckLevel.BASIC:
            return ["grpc_server"]
        elif level == HealthCheckLevel.DETAILED:
            return ["grpc_server", "model_loaded", "qdrant_connection", "container_resources"]
        elif level == HealthCheckLevel.DIAGNOSTIC:
            return list(self._component_checkers.keys())
        else:
            return ["grpc_server"]
    
    async def _check_grpc_server(self) -> ComponentHealth:
        """Check gRPC server health."""
        start_check = time.time()
        
        if not self.config.grpc_health_enabled:
            return ComponentHealth(
                name="grpc_server",
                status=ComponentStatus.HEALTHY,
                message="gRPC health checking disabled",
                details={"enabled": False},
                last_check=time.time()
            )
        
        try:
            # Import here to avoid circular imports
            from ..proto.embed_pb2 import StatusRequest
            
            if not self._grpc_stub:
                raise Exception("gRPC client not initialized")
            
            # Call GetStatus with timeout
            response = await asyncio.wait_for(
                self._grpc_stub.GetStatus(StatusRequest()),
                timeout=self.config.grpc_health_timeout
            )
            
            response_time = round((time.time() - start_check) * 1000, 2)
            
            return ComponentHealth(
                name="grpc_server",
                status=ComponentStatus.HEALTHY,
                message="gRPC server responding",
                details={
                    "server_version": response.server_version,
                    "current_model": response.current_model,
                    "cuda_available": response.cuda_available,
                    "uptime_seconds": response.uptime_seconds
                },
                last_check=time.time(),
                response_time_ms=response_time
            )
            
        except asyncio.TimeoutError:
            return ComponentHealth(
                name="grpc_server",
                status=ComponentStatus.UNHEALTHY,
                message="gRPC server timeout",
                details={"timeout_seconds": self.config.grpc_health_timeout},
                last_check=time.time(),
                error="Timeout"
            )
        except grpc.RpcError as e:
            return ComponentHealth(
                name="grpc_server",
                status=ComponentStatus.UNHEALTHY,
                message=f"gRPC error: {e.code().name}",
                details={
                    "grpc_code": e.code().name,
                    "grpc_details": e.details()
                },
                last_check=time.time(),
                error=str(e)
            )
        except Exception as e:
            return ComponentHealth(
                name="grpc_server",
                status=ComponentStatus.UNHEALTHY,
                message=f"gRPC server check failed: {str(e)}",
                details={},
                last_check=time.time(),
                error=str(e)
            )
    
    async def _check_model_loaded(self) -> ComponentHealth:
        """Check if embedding model is properly loaded."""
        if not self.config.model_health_enabled:
            return ComponentHealth(
                name="model_loaded",
                status=ComponentStatus.HEALTHY,
                message="Model health checking disabled",
                details={"enabled": False},
                last_check=time.time()
            )
        
        try:
            # Import here to avoid circular imports
            from ..proto.embed_pb2 import StatusRequest
            
            if not self._grpc_stub:
                raise Exception("gRPC client not initialized")
            
            response = await asyncio.wait_for(
                self._grpc_stub.GetStatus(StatusRequest()),
                timeout=self.config.model_health_timeout
            )
            
            current_model = response.current_model
            
            if current_model and current_model != "No model loaded":
                return ComponentHealth(
                    name="model_loaded",
                    status=ComponentStatus.HEALTHY,
                    message=f"Model loaded: {current_model}",
                    details={
                        "model_name": current_model,
                        "cuda_available": response.cuda_available
                    },
                    last_check=time.time()
                )
            else:
                # If we're in startup phase, this might be expected
                if self.is_in_startup_phase():
                    return ComponentHealth(
                        name="model_loaded",
                        status=ComponentStatus.DEGRADED,
                        message="Model not loaded yet (still starting)",
                        details={"in_startup_phase": True},
                        last_check=time.time()
                    )
                else:
                    return ComponentHealth(
                        name="model_loaded",
                        status=ComponentStatus.UNHEALTHY,
                        message="No model loaded",
                        details={"current_model": current_model},
                        last_check=time.time()
                    )
        
        except Exception as e:
            return ComponentHealth(
                name="model_loaded",
                status=ComponentStatus.UNKNOWN,
                message=f"Could not check model status: {str(e)}",
                details={},
                last_check=time.time(),
                error=str(e)
            )
    
    async def _check_qdrant_connection(self) -> ComponentHealth:
        """Check Qdrant database connection."""
        if not self.config.qdrant_health_enabled:
            return ComponentHealth(
                name="qdrant_connection",
                status=ComponentStatus.HEALTHY,
                message="Qdrant health checking disabled",
                details={"enabled": False},
                last_check=time.time()
            )
        
        try:
            # Import here to avoid circular imports
            from ..proto.embed_pb2 import StatusRequest
            
            if not self._grpc_stub:
                raise Exception("gRPC client not initialized")
            
            response = await asyncio.wait_for(
                self._grpc_stub.GetStatus(StatusRequest()),
                timeout=self.config.qdrant_health_timeout
            )
            
            if response.qdrant_connected:
                return ComponentHealth(
                    name="qdrant_connection",
                    status=ComponentStatus.HEALTHY,
                    message="Qdrant connected",
                    details={"connected": True},
                    last_check=time.time()
                )
            else:
                return ComponentHealth(
                    name="qdrant_connection",
                    status=ComponentStatus.DEGRADED,
                    message="Qdrant not connected",
                    details={"connected": False},
                    last_check=time.time()
                )
        
        except Exception as e:
            return ComponentHealth(
                name="qdrant_connection",
                status=ComponentStatus.UNKNOWN,
                message=f"Could not check Qdrant status: {str(e)}",
                details={},
                last_check=time.time(),
                error=str(e)
            )
    
    async def _check_container_resources(self) -> ComponentHealth:
        """Check container resource usage."""
        if not self.config.enable_resource_monitoring:
            return ComponentHealth(
                name="container_resources",
                status=ComponentStatus.HEALTHY,
                message="Resource monitoring disabled",
                details={"enabled": False},
                last_check=time.time()
            )
        
        try:
            usage = self.container_monitor.get_resource_usage()
            
            # Check against thresholds
            constraints = self.container_monitor.is_resource_constrained({
                "memory": self.config.memory_threshold,
                "cpu": self.config.container_cpu_threshold,
                "disk": self.config.container_disk_threshold,
                "gpu_memory": self.config.gpu_memory_threshold
            })
            
            # Determine status based on constraints
            if any(constraints.values()):
                constrained_resources = [k for k, v in constraints.items() if v]
                status = ComponentStatus.DEGRADED if len(constrained_resources) == 1 else ComponentStatus.UNHEALTHY
                message = f"Resource constraints: {', '.join(constrained_resources)}"
            else:
                status = ComponentStatus.HEALTHY
                message = "Resources within healthy limits"
            
            # Check OOM risk
            oom_info = self.container_monitor.check_oom_risk()
            
            return ComponentHealth(
                name="container_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": usage.cpu_percent,
                    "memory_percent": usage.memory_percent,
                    "disk_percent": usage.disk_percent,
                    "constraints": constraints,
                    "oom_risk": oom_info["risk_level"],
                    "gpu_memory_percent": usage.gpu_memory_percent
                },
                last_check=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                name="container_resources",
                status=ComponentStatus.UNKNOWN,
                message=f"Could not check resource usage: {str(e)}",
                details={},
                last_check=time.time(),
                error=str(e)
            )
    
    async def _check_cuda_available(self) -> ComponentHealth:
        """Check CUDA availability."""
        try:
            # Import here to avoid circular imports
            from ..proto.embed_pb2 import StatusRequest
            
            if not self._grpc_stub:
                raise Exception("gRPC client not initialized")
            
            response = await self._grpc_stub.GetStatus(StatusRequest())
            
            if response.cuda_available:
                status = ComponentStatus.HEALTHY
                message = "CUDA available"
            else:
                status = ComponentStatus.DEGRADED
                message = "CUDA not available (CPU mode)"
            
            return ComponentHealth(
                name="cuda_available",
                status=status,
                message=message,
                details={"cuda_available": response.cuda_available},
                last_check=time.time()
            )
        
        except Exception as e:
            return ComponentHealth(
                name="cuda_available",
                status=ComponentStatus.UNKNOWN,
                message=f"Could not check CUDA status: {str(e)}",
                details={},
                last_check=time.time(),
                error=str(e)
            )
    
    def _calculate_overall_status(self, components: List[ComponentHealth]) -> OverallStatus:
        """Calculate overall status from component statuses."""
        if self.is_in_startup_phase():
            # During startup, be more lenient
            critical_components = [c for c in components if c.name in self.config.critical_dependencies]
            if any(c.status == ComponentStatus.UNHEALTHY for c in critical_components):
                return OverallStatus.UNHEALTHY
            else:
                return OverallStatus.STARTING
        
        # Normal operation - strict checking
        critical_components = [c for c in components if c.name in self.config.critical_dependencies]
        optional_components = [c for c in components if c.name in self.config.optional_dependencies]
        
        # Check critical components
        unhealthy_critical = [c for c in critical_components if c.status == ComponentStatus.UNHEALTHY]
        if unhealthy_critical:
            return OverallStatus.UNHEALTHY
        
        # Check for any degraded components
        degraded_components = [c for c in components if c.status == ComponentStatus.DEGRADED]
        if degraded_components:
            return OverallStatus.DEGRADED
        
        # All components healthy or unknown (non-critical)
        return OverallStatus.HEALTHY
    
    def _get_status_message(self, status: OverallStatus, components: List[ComponentHealth]) -> str:
        """Generate a human-readable status message."""
        if status == OverallStatus.HEALTHY:
            return "All systems operational"
        elif status == OverallStatus.STARTING:
            return "Service starting up"
        elif status == OverallStatus.DEGRADED:
            degraded = [c.name for c in components if c.status == ComponentStatus.DEGRADED]
            return f"Service degraded: {', '.join(degraded)}"
        elif status == OverallStatus.UNHEALTHY:
            unhealthy = [c.name for c in components if c.status == ComponentStatus.UNHEALTHY]
            return f"Service unhealthy: {', '.join(unhealthy)}"
        else:
            return "Status unknown"
    
    def _generate_recommendations(self, components: List[ComponentHealth], resource_usage: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on health status."""
        recommendations = []
        
        # Check for high resource usage
        if resource_usage:
            if resource_usage.get("memory_percent", 0) > 85:
                recommendations.append("High memory usage detected - consider freeing memory or increasing limits")
            
            if resource_usage.get("cpu_percent", 0) > 90:
                recommendations.append("High CPU usage detected - may impact response times")
            
            if resource_usage.get("gpu_memory_percent", 0) > 85:
                recommendations.append("High GPU memory usage - consider reducing model size or batch size")
        
        # Check component-specific issues
        for comp in components:
            if comp.status == ComponentStatus.UNHEALTHY:
                if comp.name == "grpc_server":
                    recommendations.append("gRPC server unhealthy - check server logs and restart if needed")
                elif comp.name == "model_loaded":
                    recommendations.append("No model loaded - verify model configuration and CUDA availability")
                elif comp.name == "qdrant_connection":
                    recommendations.append("Qdrant connection failed - check network connectivity and Qdrant status")
        
        # OOM risk recommendations
        try:
            oom_info = self.container_monitor.check_oom_risk()
            if oom_info["risk_level"] in ["high", "critical"]:
                recommendations.extend(oom_info.get("recommendations", []))
        except:
            pass
        
        return recommendations[:5]  # Limit to top 5 recommendations
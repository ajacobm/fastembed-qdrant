"""
Container health monitoring for Docker/Kubernetes environments.
Handles the container boundary concerns for health checking.
"""

import os
import time
import psutil
import json
import subprocess
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ContainerRuntime(Enum):
    """Detected container runtime."""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman" 
    NONE = "none"


@dataclass
class ResourceUsage:
    """System resource usage information."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


@dataclass
class ContainerInfo:
    """Container environment information."""
    runtime: ContainerRuntime
    container_id: Optional[str]
    container_name: Optional[str]
    image_name: Optional[str]
    pod_name: Optional[str]
    namespace: Optional[str]
    node_name: Optional[str]
    limits: Dict[str, Any]
    requests: Dict[str, Any]


class ContainerHealthMonitor:
    """Monitor container health and resource constraints with Docker/K8s boundary awareness."""
    
    def __init__(self):
        self.start_time = time.time()
        self.runtime_info = self._detect_container_runtime()
        
    def _detect_container_runtime(self) -> ContainerInfo:
        """Detect the container runtime and gather environment information."""
        
        # Check for Kubernetes environment
        if self._is_kubernetes():
            return self._get_kubernetes_info()
        
        # Check for Docker environment
        elif self._is_docker():
            return self._get_docker_info()
            
        # Check for Podman environment 
        elif self._is_podman():
            return self._get_podman_info()
            
        # Not in a container
        else:
            return ContainerInfo(
                runtime=ContainerRuntime.NONE,
                container_id=None,
                container_name=None,
                image_name=None,
                pod_name=None,
                namespace=None,
                node_name=None,
                limits={},
                requests={}
            )
    
    def _is_kubernetes(self) -> bool:
        """Check if running in Kubernetes."""
        return (
            os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount") or
            "KUBERNETES_SERVICE_HOST" in os.environ or
            "POD_NAME" in os.environ
        )
    
    def _is_docker(self) -> bool:
        """Check if running in Docker."""
        return (
            os.path.exists("/.dockerenv") or
            os.path.exists("/proc/1/cgroup") and
            any("docker" in line for line in open("/proc/1/cgroup", "r").readlines())
        )
    
    def _is_podman(self) -> bool:
        """Check if running in Podman."""
        return (
            os.path.exists("/run/.containerenv") or
            os.path.exists("/proc/1/cgroup") and
            any("libpod" in line for line in open("/proc/1/cgroup", "r").readlines())
        )
    
    def _get_kubernetes_info(self) -> ContainerInfo:
        """Get Kubernetes-specific container information."""
        limits = {}
        requests = {}
        
        # Read resource limits from environment or cgroups
        try:
            # Memory limit from cgroups v1 or v2
            memory_limit = self._read_cgroup_memory_limit()
            if memory_limit:
                limits["memory"] = memory_limit
                
            # CPU limit from environment or cgroups
            cpu_limit = self._read_cgroup_cpu_limit()
            if cpu_limit:
                limits["cpu"] = cpu_limit
                
        except Exception:
            pass  # Resource limits not available or readable
        
        return ContainerInfo(
            runtime=ContainerRuntime.KUBERNETES,
            container_id=os.environ.get("HOSTNAME"),  # Usually the pod name in K8s
            container_name=os.environ.get("HOSTNAME"),
            image_name=None,  # Not easily available in K8s
            pod_name=os.environ.get("POD_NAME", os.environ.get("HOSTNAME")),
            namespace=os.environ.get("POD_NAMESPACE", "default"),
            node_name=os.environ.get("NODE_NAME"),
            limits=limits,
            requests=requests
        )
    
    def _get_docker_info(self) -> ContainerInfo:
        """Get Docker-specific container information."""
        container_id = None
        
        # Try to get container ID from hostname or /proc/self/cgroup
        try:
            if os.path.exists("/proc/self/cgroup"):
                with open("/proc/self/cgroup", "r") as f:
                    for line in f:
                        if "docker" in line:
                            container_id = line.split("/")[-1].strip()
                            break
        except Exception:
            pass
        
        if not container_id:
            container_id = os.environ.get("HOSTNAME")
        
        # Read resource limits from cgroups
        limits = {}
        try:
            memory_limit = self._read_cgroup_memory_limit()
            if memory_limit:
                limits["memory"] = memory_limit
                
            cpu_limit = self._read_cgroup_cpu_limit()
            if cpu_limit:
                limits["cpu"] = cpu_limit
        except Exception:
            pass
        
        return ContainerInfo(
            runtime=ContainerRuntime.DOCKER,
            container_id=container_id,
            container_name=None,  # Not easily available without Docker API
            image_name=None,  # Not easily available without Docker API
            pod_name=None,
            namespace=None,
            node_name=None,
            limits=limits,
            requests={}
        )
    
    def _get_podman_info(self) -> ContainerInfo:
        """Get Podman-specific container information."""
        container_id = None
        
        # Read container info from /run/.containerenv if available
        if os.path.exists("/run/.containerenv"):
            try:
                with open("/run/.containerenv", "r") as f:
                    content = f.read()
                    # Parse container environment file
                    for line in content.split("\n"):
                        if line.startswith("name="):
                            container_id = line.split("=", 1)[1].strip('"')
                            break
            except Exception:
                pass
        
        if not container_id:
            container_id = os.environ.get("HOSTNAME")
        
        return ContainerInfo(
            runtime=ContainerRuntime.PODMAN,
            container_id=container_id,
            container_name=container_id,
            image_name=None,
            pod_name=None,
            namespace=None,
            node_name=None,
            limits={},  # Podman limits are harder to detect
            requests={}
        )
    
    def _read_cgroup_memory_limit(self) -> Optional[int]:
        """Read memory limit from cgroups (v1 or v2)."""
        try:
            # Try cgroups v2 first
            v2_path = "/sys/fs/cgroup/memory.max"
            if os.path.exists(v2_path):
                with open(v2_path, "r") as f:
                    limit = f.read().strip()
                    if limit != "max":
                        return int(limit)
            
            # Try cgroups v1
            v1_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            if os.path.exists(v1_path):
                with open(v1_path, "r") as f:
                    limit = int(f.read().strip())
                    # cgroups v1 often returns a very large number for unlimited
                    if limit < (2**63 - 1):  # Reasonable upper bound
                        return limit
                        
        except Exception:
            pass
        
        return None
    
    def _read_cgroup_cpu_limit(self) -> Optional[float]:
        """Read CPU limit from cgroups (v1 or v2)."""
        try:
            # Try cgroups v2 first
            v2_max_path = "/sys/fs/cgroup/cpu.max"
            if os.path.exists(v2_max_path):
                with open(v2_max_path, "r") as f:
                    content = f.read().strip()
                    if content != "max":
                        parts = content.split()
                        if len(parts) == 2:
                            quota, period = int(parts[0]), int(parts[1])
                            return quota / period  # CPU cores
            
            # Try cgroups v1
            quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
            period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
            
            if os.path.exists(quota_path) and os.path.exists(period_path):
                with open(quota_path, "r") as f:
                    quota = int(f.read().strip())
                with open(period_path, "r") as f:
                    period = int(f.read().strip())
                    
                if quota > 0:
                    return quota / period  # CPU cores
                    
        except Exception:
            pass
        
        return None
    
    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        memory_total_mb = memory.total / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / 1024 / 1024 / 1024
        disk_total_gb = disk.total / 1024 / 1024 / 1024
        
        # GPU memory usage (if available)
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Use first GPU
                    gpu = gpus[0]
                    gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_memory_total_mb = gpu.memoryTotal
            except Exception:
                pass  # GPU monitoring failed
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb
        )
    
    def get_uptime(self) -> float:
        """Get container uptime in seconds."""
        return time.time() - self.start_time
    
    def is_resource_constrained(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Check if resources are constrained against thresholds."""
        usage = self.get_resource_usage()
        
        constraints = {
            "memory": False,
            "cpu": False,
            "disk": False,
            "gpu_memory": False
        }
        
        if usage.memory_percent > thresholds.get("memory", 100):
            constraints["memory"] = True
            
        if usage.cpu_percent > thresholds.get("cpu", 100):
            constraints["cpu"] = True
            
        if usage.disk_percent > thresholds.get("disk", 100):
            constraints["disk"] = True
            
        if (usage.gpu_memory_percent is not None and 
            usage.gpu_memory_percent > thresholds.get("gpu_memory", 100)):
            constraints["gpu_memory"] = True
        
        return constraints
    
    def get_container_health_info(self) -> Dict[str, Any]:
        """Get comprehensive container health information."""
        usage = self.get_resource_usage()
        
        health_info = {
            "runtime": {
                "type": self.runtime_info.runtime.value,
                "container_id": self.runtime_info.container_id,
                "container_name": self.runtime_info.container_name,
                "image_name": self.runtime_info.image_name,
                "pod_name": self.runtime_info.pod_name,
                "namespace": self.runtime_info.namespace,
                "node_name": self.runtime_info.node_name,
            },
            "uptime": {
                "seconds": self.get_uptime(),
                "human": self._format_uptime(self.get_uptime())
            },
            "resources": {
                "cpu": {
                    "percent": usage.cpu_percent,
                    "limit": self.runtime_info.limits.get("cpu")
                },
                "memory": {
                    "percent": usage.memory_percent,
                    "used_mb": usage.memory_used_mb,
                    "total_mb": usage.memory_total_mb,
                    "limit": self.runtime_info.limits.get("memory")
                },
                "disk": {
                    "percent": usage.disk_percent,
                    "used_gb": usage.disk_used_gb,
                    "total_gb": usage.disk_total_gb
                }
            }
        }
        
        # Add GPU info if available
        if usage.gpu_memory_percent is not None:
            health_info["resources"]["gpu"] = {
                "memory_percent": usage.gpu_memory_percent,
                "memory_used_mb": usage.gpu_memory_used_mb,
                "memory_total_mb": usage.gpu_memory_total_mb
            }
        
        return health_info
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        uptime_int = int(seconds)
        
        days = uptime_int // 86400
        hours = (uptime_int % 86400) // 3600
        minutes = (uptime_int % 3600) // 60
        seconds = uptime_int % 60
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def check_oom_risk(self) -> Dict[str, Any]:
        """Check risk of Out-Of-Memory (OOM) kill in container."""
        usage = self.get_resource_usage()
        
        # Calculate OOM risk based on memory usage and container limits
        memory_limit = self.runtime_info.limits.get("memory")
        
        oom_info = {
            "risk_level": "unknown",
            "memory_usage_percent": usage.memory_percent,
            "container_has_limit": memory_limit is not None,
            "recommendations": []
        }
        
        if memory_limit:
            # Calculate usage against container limit
            limit_mb = memory_limit / 1024 / 1024
            usage_against_limit = (usage.memory_used_mb / limit_mb) * 100
            
            oom_info["memory_limit_mb"] = limit_mb
            oom_info["usage_against_limit_percent"] = usage_against_limit
            
            if usage_against_limit > 90:
                oom_info["risk_level"] = "critical"
                oom_info["recommendations"].append("Immediate memory cleanup or container restart recommended")
            elif usage_against_limit > 75:
                oom_info["risk_level"] = "high"
                oom_info["recommendations"].append("Monitor closely, consider increasing memory limit")
            elif usage_against_limit > 50:
                oom_info["risk_level"] = "medium"
                oom_info["recommendations"].append("Memory usage is elevated but manageable")
            else:
                oom_info["risk_level"] = "low"
        else:
            # No container limit, use system memory
            if usage.memory_percent > 90:
                oom_info["risk_level"] = "high"
                oom_info["recommendations"].append("High system memory usage, monitor for stability")
            elif usage.memory_percent > 75:
                oom_info["risk_level"] = "medium"
            else:
                oom_info["risk_level"] = "low"
        
        return oom_info
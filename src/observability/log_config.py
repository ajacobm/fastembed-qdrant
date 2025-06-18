"""Log configuration management for FastEmbed server."""

import os
from dataclasses import dataclass
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "text"]
LogOutput = Literal["console", "file", "remote"]


@dataclass
class LogConfig:
    """Configuration for logging behavior."""
    
    level: LogLevel = "INFO"
    format: LogFormat = "json"
    output: LogOutput = "console"
    file_path: str = "/app/logs/fastembed.log"
    max_size: str = "100MB"
    backup_count: int = 5
    remote_endpoint: str = ""
    remote_format: str = "json"
    
    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create LogConfig from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format=os.getenv("LOG_FORMAT", "json").lower(),
            output=os.getenv("LOG_OUTPUT", "console").lower(),
            file_path=os.getenv("LOG_FILE_PATH", "/app/logs/fastembed.log"),
            max_size=os.getenv("LOG_MAX_SIZE", "100MB"),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            remote_endpoint=os.getenv("LOG_REMOTE_ENDPOINT", ""),
            remote_format=os.getenv("LOG_REMOTE_FORMAT", "json"),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        
        valid_formats = ["json", "text"]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid log format: {self.format}. Must be one of {valid_formats}")
        
        valid_outputs = ["console", "file", "remote"]
        if self.output not in valid_outputs:
            raise ValueError(f"Invalid log output: {self.output}. Must be one of {valid_outputs}")
        
        if self.output == "remote" and not self.remote_endpoint:
            raise ValueError("Remote endpoint must be specified when using remote output")
        
        if self.backup_count < 0:
            raise ValueError("Backup count must be non-negative")
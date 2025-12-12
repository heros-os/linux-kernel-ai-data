"""
Configuration management for the Linux Kernel History Extraction Pipeline.
Uses Pydantic Settings for environment variable support and validation.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClickHouseSettings(BaseSettings):
    """ClickHouse database connection settings."""
    
    model_config = SettingsConfigDict(env_prefix="CLICKHOUSE_")
    
    host: str = Field(default="localhost", description="ClickHouse server host")
    port: int = Field(default=9000, description="ClickHouse native TCP port")
    http_port: int = Field(default=8123, description="ClickHouse HTTP port")
    database: str = Field(default="kernel_history", description="Database name")
    user: str = Field(default="default", description="Database user")
    password: str = Field(default="", description="Database password")
    
    # Batch insert settings
    batch_size: int = Field(default=10000, description="Records per batch insert")
    flush_interval: float = Field(default=5.0, description="Seconds between auto-flush")
    
    @property
    def connection_string(self) -> str:
        """Return connection string for clickhouse-driver."""
        return f"clickhouse://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RepositorySettings(BaseSettings):
    """Git repository settings."""
    
    model_config = SettingsConfigDict(env_prefix="REPO_")
    
    # Repository paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory to store cloned repositories"
    )
    
    # Mainline kernel repository
    kernel_url: str = Field(
        default="git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git",
        description="URL for the mainline Linux kernel repository"
    )
    kernel_dir: str = Field(default="linux.git", description="Local directory name for kernel repo")
    
    # Historical repositories for pre-Git history
    history_repos: list[dict] = Field(
        default=[
            {
                "name": "tglx-history",
                "url": "https://git.kernel.org/pub/scm/linux/kernel/git/tglx/history.git",
                "dir": "tglx-history.git"
            },
            {
                "name": "davej-history", 
                "url": "https://github.com/davej/history.git",
                "dir": "davej-history.git"
            }
        ],
        description="Historical repository URLs for pre-Git history"
    )
    
    # Graft points to connect histories
    # Format: {"modern_commit": "historical_commit"}
    graft_points: dict[str, str] = Field(
        default={},
        description="Graft points to connect modern and historical repositories"
    )
    
    @property
    def kernel_path(self) -> Path:
        """Full path to the kernel repository."""
        return self.data_dir / self.kernel_dir
    
    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v


class ExtractionSettings(BaseSettings):
    """Settings for the extraction pipeline."""
    
    model_config = SettingsConfigDict(env_prefix="EXTRACT_")
    
    # Parallelism
    num_workers: int = Field(
        default=0,  # 0 means auto-detect (CPU count - 2)
        description="Number of worker processes for extraction"
    )
    batch_size: int = Field(
        default=100,
        description="Number of commits per batch for workers"
    )
    queue_size: int = Field(
        default=10,
        description="Maximum number of batches in the work queue"
    )
    
    # Filtering
    exclude_merges: bool = Field(
        default=False,
        description="Exclude merge commits from extraction"
    )
    include_binary: bool = Field(
        default=False,
        description="Include binary file changes (not recommended)"
    )
    max_diff_size: int = Field(
        default=100000,  # 100KB
        description="Maximum diff size in bytes to include"
    )
    
    # File filtering
    include_extensions: list[str] = Field(
        default=[".c", ".h", ".S", ".rs", ".py", ".sh", ".pl", ".awk"],
        description="File extensions to include"
    )
    exclude_paths: list[str] = Field(
        default=["Documentation/", "tools/testing/"],
        description="Path prefixes to exclude"
    )
    
    # Resume support
    resume_from: Optional[str] = Field(
        default=None,
        description="Commit hash to resume extraction from"
    )
    checkpoint_interval: int = Field(
        default=10000,
        description="Commits between checkpoints"
    )
    
    @property
    def effective_workers(self) -> int:
        """Return the effective number of workers."""
        if self.num_workers > 0:
            return self.num_workers
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 2)


class ExportSettings(BaseSettings):
    """Settings for training data export."""
    
    model_config = SettingsConfigDict(env_prefix="EXPORT_")
    
    output_dir: Path = Field(
        default=Path("./exports"),
        description="Directory for exported training data"
    )
    
    # Format options
    format: str = Field(
        default="jsonl",
        description="Export format: jsonl, parquet"
    )
    output_mode: str = Field(
        default="diff",
        description="Output mode: 'diff' (unified diff) or 'full' (complete code)"
    )
    
    # Filtering for export
    min_lines_changed: int = Field(default=5, description="Minimum lines changed to include")
    max_lines_changed: int = Field(default=500, description="Maximum lines changed to include")
    exclude_subsystems: list[str] = Field(
        default=[],
        description="Subsystems to exclude from export"
    )
    include_subsystems: list[str] = Field(
        default=[],
        description="Subsystems to include (empty = all)"
    )
    
    # Sampling
    sample_ratio: float = Field(
        default=1.0,
        description="Ratio of data to sample (1.0 = all)"
    )
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v



class QualitySettings(BaseSettings):
    """Settings for quality scoring."""
    
    model_config = SettingsConfigDict(env_prefix="QUALITY_")
    
    # AI Scoring
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    model_name: str = Field(default="qwen2.5-coder:7b", description="Model to use for scoring")
    batch_size: int = Field(default=50, description="Batch size for AI scoring")
    
    # Thresholds
    min_heuristic_score: float = Field(default=30.0, description="Minimum heuristic score to run AI scoring")
    min_ai_score: float = Field(default=3.0, description="Minimum AI score to include in export")


class Settings(BaseSettings):
    """Main configuration container."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Sub-configurations
    clickhouse: ClickHouseSettings = Field(default_factory=ClickHouseSettings)
    repository: RepositorySettings = Field(default_factory=RepositorySettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.repository.data_dir.mkdir(parents=True, exist_ok=True)
        self.export.output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

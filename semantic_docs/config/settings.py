"""Configuration management for semantic document analysis system.

Provides centralized configuration for Claude Code integration with environment
variable overrides, JSON config files, and validation.
"""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticConfig(BaseModel):
    """Configuration settings for semantic engine operations."""

    # ChromaDB settings
    chroma_path: str = Field(
        default="./chroma_db", description="Path to ChromaDB database"
    )
    collection_name: str = Field(
        default="markdown_docs", description="ChromaDB collection name"
    )

    # Similarity thresholds
    redundancy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for redundancy detection",
    )
    placement_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for placement suggestions",
    )

    # API limits and performance
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum results per query"
    )
    api_timeout: float = Field(
        default=30.0, ge=1.0, description="OpenAI API timeout in seconds"
    )

    # Claude Code integration
    validation_timeout: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Maximum time for post-write validation",
    )
    enable_background_monitoring: bool = Field(
        default=False, description="Enable background file monitoring"
    )

    # Output formatting
    json_output: bool = Field(default=True, description="Format output as JSON")
    include_content_preview: bool = Field(
        default=True, description="Include content preview in similarity results"
    )
    preview_length: int = Field(
        default=200, ge=50, le=1000, description="Maximum length of content preview"
    )

    # Clustering configuration
    cluster_method: str = Field(
        default="kmeans",
        description="Default clustering method (kmeans or hierarchical)",
    )
    min_cluster_size: int = Field(
        default=2, ge=1, le=10, description="Minimum documents per cluster"
    )
    max_clusters: int = Field(
        default=10, ge=2, le=20, description="Maximum number of clusters to try"
    )

    # Drift detection thresholds
    entropy_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Entropy threshold for warning (higher = more scattered)",
    )
    coherence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Coherence threshold for warning (lower = less coherent)",
    )
    drift_check_frequency: int = Field(
        default=10, ge=1, description="Check drift every N commits/operations"
    )

    # Visualization settings
    cluster_visualization: bool = Field(
        default=True, description="Enable cluster visualization generation"
    )

    # AI Summarization settings
    ai_summarization_enabled: bool = Field(
        default=True, description="Enable AI-powered document summarization"
    )
    ai_model: str = Field(
        default="gpt-4-turbo",
        description="AI model for summarization and labeling",
    )
    summary_max_tokens: int = Field(
        default=150, ge=50, le=500, description="Maximum tokens for generated summaries"
    )
    max_content_labels: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of content labels per document",
    )
    ai_cache_enabled: bool = Field(
        default=True, description="Enable AI response caching to reduce costs"
    )
    ai_cache_dir: str = Field(
        default="./ai_cache", description="Directory for AI response cache"
    )

    model_config = ConfigDict(
        env_prefix="SEMANTIC_", case_sensitive=False, extra="ignore"
    )

    @field_validator("redundancy_threshold")
    @classmethod
    def redundancy_threshold_must_be_reasonable(cls, v):
        """Ensure redundancy threshold is in reasonable range."""
        if v < 0.7:
            raise ValueError(
                "Redundancy threshold should be >= 0.7 for meaningful detection"
            )
        return v

    @field_validator("placement_threshold")
    @classmethod
    def placement_threshold_must_be_lower_than_redundancy(cls, v, info):
        """Ensure placement threshold is lower than redundancy threshold."""
        if (
            info.data
            and "redundancy_threshold" in info.data
            and v >= info.data["redundancy_threshold"]
        ):
            raise ValueError(
                "Placement threshold must be lower than redundancy threshold"
            )
        return v

    @field_validator("cluster_method")
    @classmethod
    def validate_cluster_method(cls, v):
        """Ensure cluster method is supported."""
        if v not in ["kmeans", "hierarchical"]:
            raise ValueError("Cluster method must be 'kmeans' or 'hierarchical'")
        return v

    @field_validator("ai_model")
    @classmethod
    def validate_ai_model(cls, v):
        """Ensure AI model is supported."""
        supported_models = [
            "gpt-4.1",
            "gpt-4-1106-preview",
            "gpt-5",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        if v not in supported_models:
            raise ValueError(f"AI model must be one of: {', '.join(supported_models)}")
        return v


def load_config(config_file: str | None = None) -> SemanticConfig:
    """Load configuration from file and environment variables.

    Args:
        config_file: Optional path to JSON configuration file

    Returns:
        Validated configuration instance

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file specified but not found
    """
    config_data = {}

    # Load from JSON file if specified
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path) as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

    # Read environment variables and merge with config data
    env_data = {}
    for field_name in SemanticConfig.model_fields:
        env_var = f"SEMANTIC_{field_name.upper()}"
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Handle type conversion for common types
            field_info = SemanticConfig.model_fields[field_name]
            if field_info.annotation in [int, type(int)]:
                env_data[field_name] = int(env_value)
            elif field_info.annotation in [float, type(float)]:
                env_data[field_name] = float(env_value)
            elif field_info.annotation in [bool, type(bool)]:
                env_data[field_name] = env_value.lower() in ("true", "1", "yes", "on")
            else:
                env_data[field_name] = env_value

    # Environment variables override JSON settings
    config_data.update(env_data)

    try:
        return SemanticConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def get_default_config_path() -> Path:
    """Get default configuration file path."""
    return Path("scripts/config/semantic_config.json")


def create_default_config_file(path: str | None = None) -> Path:
    """Create a default configuration file.

    Args:
        path: Optional path for config file, defaults to semantic_config.json

    Returns:
        Path to created configuration file
    """
    if path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(path)

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create default config
    default_config = SemanticConfig()
    config_data = {
        "_comment": "Semantic engine configuration for Claude Code integration",
        "chroma_path": default_config.chroma_path,
        "collection_name": default_config.collection_name,
        "redundancy_threshold": default_config.redundancy_threshold,
        "placement_threshold": default_config.placement_threshold,
        "max_results": default_config.max_results,
        "api_timeout": default_config.api_timeout,
        "validation_timeout": default_config.validation_timeout,
        "enable_background_monitoring": default_config.enable_background_monitoring,
        "json_output": default_config.json_output,
        "include_content_preview": default_config.include_content_preview,
        "preview_length": default_config.preview_length,
        "cluster_method": default_config.cluster_method,
        "min_cluster_size": default_config.min_cluster_size,
        "max_clusters": default_config.max_clusters,
        "entropy_threshold": default_config.entropy_threshold,
        "coherence_threshold": default_config.coherence_threshold,
        "drift_check_frequency": default_config.drift_check_frequency,
        "cluster_visualization": default_config.cluster_visualization,
        "ai_summarization_enabled": default_config.ai_summarization_enabled,
        "ai_model": default_config.ai_model,
        "summary_max_tokens": default_config.summary_max_tokens,
        "max_content_labels": default_config.max_content_labels,
        "ai_cache_enabled": default_config.ai_cache_enabled,
        "ai_cache_dir": default_config.ai_cache_dir,
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_path


def validate_environment() -> dict[str, Any]:
    """Validate environment for semantic engine operation.

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "environment_vars": {},
    }

    # Check for required OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        validation_results["valid"] = False
        validation_results["errors"].append(
            "OPENAI_API_KEY environment variable required"
        )
    else:
        validation_results["environment_vars"]["OPENAI_API_KEY"] = "***set***"

    # Check for semantic-specific environment variables
    semantic_vars = [
        "SEMANTIC_CHROMA_PATH",
        "SEMANTIC_COLLECTION_NAME",
        "SEMANTIC_REDUNDANCY_THRESHOLD",
        "SEMANTIC_PLACEMENT_THRESHOLD",
        "SEMANTIC_MAX_RESULTS",
        "SEMANTIC_API_TIMEOUT",
        "SEMANTIC_VALIDATION_TIMEOUT",
        "SEMANTIC_ENABLE_BACKGROUND_MONITORING",
        "SEMANTIC_JSON_OUTPUT",
        "SEMANTIC_INCLUDE_CONTENT_PREVIEW",
        "SEMANTIC_PREVIEW_LENGTH",
    ]

    for var in semantic_vars:
        value = os.environ.get(var)
        if value:
            validation_results["environment_vars"][var] = value

    # Check ChromaDB path accessibility
    try:
        config = load_config()
        chroma_path = Path(config.chroma_path)
        if not chroma_path.exists():
            validation_results["warnings"].append(
                f"ChromaDB path does not exist: {chroma_path}"
            )
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Configuration validation failed: {e}")

    return validation_results

"""Test configuration management for semantic document analysis system.

Tests for semantic_config.py including validation, environment variable overrides,
JSON config file loading, and default value handling.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_docs.config.settings import (
    SemanticConfig,
    create_default_config_file,
    get_default_config_path,
    load_config,
    validate_environment,
)


class TestSemanticConfig:
    """Test SemanticConfig model validation and defaults."""

    def test_semantic_config_initialization_when_defaults_creates_valid_config(self):
        """Test SemanticConfig initialization with default values."""
        config = SemanticConfig()

        assert config.chroma_path == "./chroma_db"
        assert config.collection_name == "markdown_docs"
        assert config.redundancy_threshold == 0.85
        assert config.placement_threshold == 0.3
        assert config.max_results == 10
        assert config.api_timeout == 30.0
        assert config.validation_timeout == 2.0
        assert config.enable_background_monitoring is False
        assert config.json_output is True
        assert config.include_content_preview is True
        assert config.preview_length == 200

    def test_semantic_config_validation_when_redundancy_threshold_too_low_raises_error(
        self,
    ):
        """Test redundancy threshold validation fails for values < 0.7."""
        with pytest.raises(ValueError, match="Redundancy threshold should be >= 0.7"):
            SemanticConfig(redundancy_threshold=0.5)

    def test_semantic_config_validation_when_placement_threshold_higher_than_redundancy_raises_error(
        self,
    ):
        """Test placement threshold must be lower than redundancy threshold."""
        with pytest.raises(
            ValueError,
            match="Placement threshold must be lower than redundancy threshold",
        ):
            SemanticConfig(redundancy_threshold=0.8, placement_threshold=0.9)

    def test_semantic_config_validation_when_valid_thresholds_creates_config(self):
        """Test valid threshold configuration."""
        config = SemanticConfig(redundancy_threshold=0.85, placement_threshold=0.3)
        assert config.redundancy_threshold == 0.85
        assert config.placement_threshold == 0.3

    def test_semantic_config_validation_when_max_results_out_of_range_raises_error(
        self,
    ):
        """Test max_results validation with boundary values."""
        with pytest.raises(ValueError):
            SemanticConfig(max_results=0)

        with pytest.raises(ValueError):
            SemanticConfig(max_results=100)

    def test_semantic_config_validation_when_preview_length_out_of_range_raises_error(
        self,
    ):
        """Test preview_length validation with boundary values."""
        with pytest.raises(ValueError):
            SemanticConfig(preview_length=25)

        with pytest.raises(ValueError):
            SemanticConfig(preview_length=2000)

    @patch.dict(
        os.environ,
        {
            "SEMANTIC_CHROMA_PATH": "/custom/chroma",
            "SEMANTIC_COLLECTION_NAME": "test_docs",
            "SEMANTIC_REDUNDANCY_THRESHOLD": "0.9",
            "SEMANTIC_MAX_RESULTS": "20",
        },
    )
    def test_semantic_config_initialization_when_env_vars_overrides_defaults(self):
        """Test environment variable overrides work correctly."""
        config = load_config()

        assert config.chroma_path == "/custom/chroma"
        assert config.collection_name == "test_docs"
        assert config.redundancy_threshold == 0.9
        assert config.max_results == 20


class TestConfigLoading:
    """Test configuration loading from files and environment."""

    def test_load_config_when_no_file_returns_default_config(self):
        """Test loading config without file uses defaults."""
        config = load_config()
        assert isinstance(config, SemanticConfig)
        assert config.chroma_path == "./chroma_db"

    def test_load_config_when_valid_json_file_loads_successfully(self):
        """Test loading configuration from valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "chroma_path": "/test/chroma",
                "collection_name": "test_collection",
                "redundancy_threshold": 0.8,
            }
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.chroma_path == "/test/chroma"
            assert config.collection_name == "test_collection"
            assert config.redundancy_threshold == 0.8
        finally:
            os.unlink(temp_path)

    def test_load_config_when_nonexistent_file_raises_file_not_found(self):
        """Test loading config from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/config.json")

    def test_load_config_when_invalid_json_raises_value_error(self):
        """Test loading config from invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON in config file"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_when_invalid_config_values_raises_validation_error(self):
        """Test loading config with invalid values raises validation error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "redundancy_threshold": 0.5  # Too low
            }
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    @patch.dict(
        os.environ,
        {
            "SEMANTIC_CHROMA_PATH": "/env/chroma",
            "SEMANTIC_REDUNDANCY_THRESHOLD": "0.75",
        },
    )
    def test_load_config_when_env_vars_override_json_file_uses_env_values(self):
        """Test environment variables override JSON file values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"chroma_path": "/json/chroma", "redundancy_threshold": 0.8}
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.chroma_path == "/env/chroma"  # From environment
            assert config.redundancy_threshold == 0.75  # From environment
        finally:
            os.unlink(temp_path)


class TestConfigFileManagement:
    """Test configuration file creation and management."""

    def test_get_default_config_path_returns_expected_path(self):
        """Test default configuration path."""
        path = get_default_config_path()
        assert str(path) == "scripts/config/semantic_config.json"

    def test_create_default_config_file_when_no_path_creates_at_default_location(self):
        """Test creating default config file."""
        # Clean up any existing file first
        default_path = get_default_config_path()
        if default_path.exists():
            default_path.unlink()

        try:
            created_path = create_default_config_file()
            assert created_path.exists()
            assert created_path == default_path

            # Verify content is valid JSON
            with open(created_path) as f:
                config_data = json.load(f)
                assert "_comment" in config_data
                assert config_data["chroma_path"] == "./chroma_db"
                assert config_data["redundancy_threshold"] == 0.85
        finally:
            if default_path.exists():
                default_path.unlink()

    def test_create_default_config_file_when_custom_path_creates_at_specified_location(
        self,
    ):
        """Test creating config file at custom path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom_config.json"

            created_path = create_default_config_file(str(custom_path))
            assert created_path.exists()
            assert created_path == custom_path

            # Verify file can be loaded as valid config
            config = load_config(str(custom_path))
            assert isinstance(config, SemanticConfig)

    def test_create_default_config_file_when_directory_missing_creates_directory(self):
        """Test creating config file creates missing directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "deep" / "config.json"

            created_path = create_default_config_file(str(nested_path))
            assert created_path.exists()
            assert created_path.parent.exists()


class TestEnvironmentValidation:
    """Test environment validation functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_validate_environment_when_api_key_present_returns_valid(self):
        """Test environment validation with API key present."""
        result = validate_environment()

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["environment_vars"]["OPENAI_API_KEY"] == "***set***"

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_when_api_key_missing_returns_invalid(self):
        """Test environment validation without API key."""
        result = validate_environment()

        assert result["valid"] is False
        assert "OPENAI_API_KEY environment variable required" in result["errors"]

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "SEMANTIC_CHROMA_PATH": "/custom/path",
            "SEMANTIC_MAX_RESULTS": "15",
        },
    )
    def test_validate_environment_when_semantic_vars_present_includes_in_results(self):
        """Test environment validation includes semantic variables."""
        result = validate_environment()

        assert result["valid"] is True
        assert result["environment_vars"]["SEMANTIC_CHROMA_PATH"] == "/custom/path"
        assert result["environment_vars"]["SEMANTIC_MAX_RESULTS"] == "15"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.config.semantic_config.load_config")
    def test_validate_environment_when_config_loading_fails_returns_invalid(
        self, mock_load_config
    ):
        """Test environment validation when config loading fails."""
        mock_load_config.side_effect = ValueError("Config error")

        result = validate_environment()

        assert result["valid"] is False
        assert "Configuration validation failed: Config error" in result["errors"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("pathlib.Path.exists")
    def test_validate_environment_when_chroma_path_missing_includes_warning(
        self, mock_exists
    ):
        """Test environment validation warns about missing ChromaDB path."""
        mock_exists.return_value = False

        result = validate_environment()

        assert result["valid"] is True  # Not a critical error
        assert any(
            "ChromaDB path does not exist" in warning for warning in result["warnings"]
        )

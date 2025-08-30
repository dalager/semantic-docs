"""Test Claude Code hooks for semantic document analysis integration.

Tests for hook interface, JSON response formatting, error handling, timeout scenarios,
and integration with Claude Code's hook system.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from semantic_docs.config.settings import SemanticConfig
from semantic_docs.integrations.claude_hooks import ClaudeCodeHooks


class TestClaudeCodeHooksInitialization:
    """Test ClaudeCodeHooks initialization and configuration."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_hooks_initialization_when_default_config_creates_instance(
        self, mock_load_config, mock_engine_class
    ):
        """Test ClaudeCodeHooks initialization with default configuration."""
        # Mock configuration
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()

        assert hooks.config == config
        assert hooks.semantic_engine == mock_engine
        mock_engine_class.assert_called_once_with(config=config)

    @patch("scripts.claude_hooks.SemanticEngine")
    def test_hooks_initialization_when_custom_config_uses_provided_config(
        self, mock_engine_class
    ):
        """Test ClaudeCodeHooks initialization with custom configuration."""
        custom_config = SemanticConfig(redundancy_threshold=0.9, max_results=20)

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks(config=custom_config)

        assert hooks.config == custom_config
        assert hooks.config.redundancy_threshold == 0.9
        assert hooks.config.max_results == 20
        mock_engine_class.assert_called_once_with(config=custom_config)

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_hooks_initialization_when_engine_fails_raises_exception(
        self, mock_load_config, mock_engine_class
    ):
        """Test ClaudeCodeHooks initialization fails when semantic engine fails."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine failure
        mock_engine_class.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(Exception, match="ChromaDB connection failed"):
            ClaudeCodeHooks()


class TestDocumentValidation:
    """Test document validation functionality."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_no_redundancy_returns_success(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation with no redundancy detected."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": False,
            "similarity_threshold": 0.85,
            "similar_documents": [],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        assert result["status"] == "success"
        assert result["file_path"] == "test.md"
        assert "timestamp" in result
        assert "validation_time" in result
        assert result["redundancy_analysis"]["redundancy_detected"] is False
        assert "placement_analysis" in result
        assert "recommendations" in result

        # Should have success recommendation
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["type"] == "validation_success"

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_redundancy_detected_returns_warning(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation with redundancy detected."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine with redundancy
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": True,
            "similarity_threshold": 0.85,
            "similar_documents": [
                {
                    "file_path": "docs/existing.md",
                    "similarity_score": 0.92,
                    "content_preview": "Similar content...",
                }
            ],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 1,
                "directories_analyzed": 1,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        assert result["status"] == "success"
        assert result["redundancy_analysis"]["redundancy_detected"] is True

        # Should have redundancy warning
        redundancy_rec = [
            r for r in result["recommendations"] if r["type"] == "redundancy_warning"
        ][0]
        assert redundancy_rec["priority"] == "high"
        assert "docs/existing.md" in redundancy_rec["message"]
        assert redundancy_rec["similar_files"] == ["docs/existing.md"]

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_self_match_is_ignored(
        self, mock_load_config, mock_engine_class
    ):
        """Ensure that if the only similar document is the same file, it's ignored."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine with a self-match
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": True,
            "similarity_threshold": 0.85,
            "similar_documents": [
                {
                    "file_path": "/abs/path/docs/test.md",
                    "similarity_score": 0.99,
                    "content_preview": "Same file",
                }
            ],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 1,
                "directories_analyzed": 1,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        # Use a path that resolves to same as similar_documents entry
        result = hooks.validate_document("/abs/path/docs/test.md", "Some content")

        assert result["status"] == "success"
        # After filtering self-match, redundancy should not be detected
        assert result["redundancy_analysis"]["redundancy_detected"] is False, (
            "Self-match should be ignored from redundancy detection"
        )
        # And recommendations should not include redundancy_warning
        assert not any(
            r["type"] == "redundancy_warning" for r in result["recommendations"]
        )

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_placement_suggestion_returns_recommendation(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation with placement suggestion."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine with placement suggestion
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": False,
            "similarity_threshold": 0.85,
            "similar_documents": [],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [
                {
                    "directory": "docs/architecture",
                    "confidence": 0.8,
                    "similar_file_count": 3,
                    "similar_files": [
                        "docs/architecture/guide1.md",
                        "docs/architecture/guide2.md",
                    ],
                }
            ],
            "analysis_summary": {
                "total_similar_documents": 3,
                "directories_analyzed": 2,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document(
            "docs/development/test.md", "Architectural content"
        )

        assert result["status"] == "success"

        # Should have placement suggestion
        placement_rec = [
            r for r in result["recommendations"] if r["type"] == "placement_suggestion"
        ][0]
        assert placement_rec["priority"] == "medium"
        assert "docs/architecture" in placement_rec["message"]
        assert placement_rec["suggested_directory"] == "docs/architecture"
        assert placement_rec["confidence"] == 0.8

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_timeout_exceeded_includes_warning(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation timeout handling."""
        config = SemanticConfig(validation_timeout=0.5)  # Short timeout
        mock_load_config.return_value = config

        # Mock semantic engine with slow response
        mock_engine = Mock()

        def slow_detect_redundancy(*args, **kwargs):
            time.sleep(0.6)  # Exceed timeout
            return {
                "redundancy_detected": False,
                "similarity_threshold": 0.85,
                "similar_documents": [],
            }

        mock_engine.detect_redundancy.side_effect = slow_detect_redundancy
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        assert result["status"] == "success"
        assert "warning" in result
        assert "timeout" in result["warning"]
        assert result["validation_time"] > config.validation_timeout

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_engine_error_returns_error_status(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation error handling."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine error
        mock_engine = Mock()
        mock_engine.detect_redundancy.side_effect = Exception("API rate limit exceeded")
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        assert result["status"] == "error"
        assert result["file_path"] == "test.md"
        assert "API rate limit exceeded" in result["error"]
        assert "validation_time" in result


class TestPostWriteHook:
    """Test post-write hook functionality."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_post_write_hook_when_markdown_file_validates_content(
        self, mock_load_config, mock_engine_class
    ):
        """Test post-write hook processes markdown files correctly."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": False,
            "similarity_threshold": 0.85,
            "similar_documents": [],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Document\n\nThis is test content.")
            temp_path = f.name

        try:
            hooks = ClaudeCodeHooks()
            result = hooks.post_write_hook(temp_path)

            assert result["status"] == "success"
            assert result["file_path"] == temp_path
            mock_engine.detect_redundancy.assert_called_once()
            mock_engine.suggest_placement.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_post_write_hook_when_non_markdown_file_skips_processing(
        self, mock_load_config, mock_engine_class
    ):
        """Test post-write hook skips non-markdown files."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        # Create temporary non-markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            hooks = ClaudeCodeHooks()
            result = hooks.post_write_hook(temp_path)

            assert result["status"] == "skipped"
            assert result["message"] == "Not a markdown file"
            mock_engine.detect_redundancy.assert_not_called()
        finally:
            Path(temp_path).unlink()

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_post_write_hook_when_file_not_found_returns_error(
        self, mock_load_config, mock_engine_class
    ):
        """Test post-write hook handles missing files."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.post_write_hook("/nonexistent/file.md")

        assert result["status"] == "error"
        assert result["error"] == "File not found"


class TestBatchValidation:
    """Test batch validation functionality."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_batch_validate_when_multiple_files_processes_all(
        self, mock_load_config, mock_engine_class
    ):
        """Test batch validation processes multiple files."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": False,
            "similarity_threshold": 0.85,
            "similar_documents": [],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        # Create temporary markdown files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(f"# Test Document {i}\n\nContent {i}")
                temp_files.append(f.name)

        try:
            hooks = ClaudeCodeHooks()
            result = hooks.batch_validate(temp_files)

            assert "batch_id" in result
            assert result["total_files"] == 3
            assert len(result["results"]) == 3
            assert result["summary"]["success"] == 3
            assert result["summary"]["errors"] == 0
            assert "batch_time" in result

            # All results should be successful
            for file_result in result["results"]:
                assert file_result["status"] == "success"

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink()

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_batch_validate_when_mixed_results_summarizes_correctly(
        self, mock_load_config, mock_engine_class
    ):
        """Test batch validation with mixed success/error results."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine with redundancy for one file
        mock_engine = Mock()
        call_count = 0

        def mock_detect_redundancy(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second file has redundancy
                return {
                    "redundancy_detected": True,
                    "similarity_threshold": 0.85,
                    "similar_documents": [
                        {"file_path": "existing.md", "similarity_score": 0.9}
                    ],
                }
            return {
                "redundancy_detected": False,
                "similarity_threshold": 0.85,
                "similar_documents": [],
            }

        mock_engine.detect_redundancy.side_effect = mock_detect_redundancy
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        # Create temporary files (including one that doesn't exist)
        temp_files = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Document 1")
            temp_files.append(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Document 2")
            temp_files.append(f.name)

        temp_files.append("/nonexistent/file.md")  # This will cause an error

        try:
            hooks = ClaudeCodeHooks()
            result = hooks.batch_validate(temp_files)

            assert result["total_files"] == 3
            assert result["summary"]["success"] == 2
            assert result["summary"]["errors"] == 1
            assert result["summary"]["redundancy_detected"] == 1

        finally:
            for temp_file in temp_files[:-1]:  # Skip the nonexistent file
                if Path(temp_file).exists():
                    Path(temp_file).unlink()


class TestSystemStatus:
    """Test system status functionality."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    @patch("scripts.claude_hooks.validate_environment")
    def test_get_system_status_when_healthy_returns_complete_status(
        self, mock_validate_env, mock_load_config, mock_engine_class
    ):
        """Test system status with healthy environment."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock healthy environment
        mock_validate_env.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "environment_vars": {"OPENAI_API_KEY": "***set***"},
        }

        # Mock semantic engine with stats
        mock_engine = Mock()
        mock_engine.get_collection_stats.return_value = {
            "total_documents": 83,
            "collection_name": "markdown_docs",
            "embedding_model": "text-embedding-3-large",
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        status = hooks.get_system_status()

        assert status["status"] == "healthy"
        assert "timestamp" in status
        assert status["environment"]["valid"] is True
        assert status["collection"]["total_documents"] == 83
        assert "configuration" in status
        assert status["configuration"]["redundancy_threshold"] == 0.85

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    @patch("scripts.claude_hooks.validate_environment")
    def test_get_system_status_when_degraded_returns_degraded_status(
        self, mock_validate_env, mock_load_config, mock_engine_class
    ):
        """Test system status with degraded environment."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock degraded environment
        mock_validate_env.return_value = {
            "valid": False,
            "errors": ["ChromaDB path does not exist"],
            "warnings": [],
            "environment_vars": {},
        }

        mock_engine = Mock()
        mock_engine.get_collection_stats.return_value = {"total_documents": 0}
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        status = hooks.get_system_status()

        assert status["status"] == "degraded"
        assert status["environment"]["valid"] is False
        assert len(status["environment"]["errors"]) > 0


class TestJSONOutputFormatting:
    """Test JSON output format compatibility with Claude Code."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_output_when_serialized_produces_valid_json(
        self, mock_load_config, mock_engine_class
    ):
        """Test document validation output can be serialized to valid JSON."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine
        mock_engine = Mock()
        mock_engine.detect_redundancy.return_value = {
            "redundancy_detected": False,
            "similarity_threshold": 0.85,
            "similar_documents": [],
        }
        mock_engine.suggest_placement.return_value = {
            "placement_suggestions": [],
            "analysis_summary": {
                "total_similar_documents": 0,
                "directories_analyzed": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        # Should be serializable to JSON
        json_output = json.dumps(result)
        assert json_output is not None

        # Should be deserializable
        parsed_result = json.loads(json_output)
        assert parsed_result["status"] == "success"
        assert "recommendations" in parsed_result

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_system_status_output_when_serialized_produces_valid_json(
        self, mock_load_config, mock_engine_class
    ):
        """Test system status output JSON compatibility."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        mock_engine = Mock()
        mock_engine.get_collection_stats.return_value = {
            "total_documents": 83,
            "collection_name": "markdown_docs",
        }
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()

        with patch("scripts.claude_hooks.validate_environment") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": [], "warnings": []}
            status = hooks.get_system_status()

        # Should be serializable to JSON
        json_output = json.dumps(status)
        parsed_status = json.loads(json_output)
        assert parsed_status["status"] in ["healthy", "degraded", "error"]


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_validate_document_when_exception_returns_error_response(
        self, mock_load_config, mock_engine_class
    ):
        """Test validation handles exceptions gracefully."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine that raises exception
        mock_engine = Mock()
        mock_engine.detect_redundancy.side_effect = Exception("Network timeout")
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        result = hooks.validate_document("test.md", "Test content")

        assert result["status"] == "error"
        assert "Network timeout" in result["error"]
        assert "validation_time" in result

    @patch("scripts.claude_hooks.SemanticEngine")
    @patch("scripts.claude_hooks.load_config")
    def test_get_system_status_when_exception_returns_error_status(
        self, mock_load_config, mock_engine_class
    ):
        """Test system status handles exceptions gracefully."""
        config = SemanticConfig()
        mock_load_config.return_value = config

        # Mock semantic engine that raises exception
        mock_engine = Mock()
        mock_engine.get_collection_stats.side_effect = Exception(
            "Database connection failed"
        )
        mock_engine_class.return_value = mock_engine

        hooks = ClaudeCodeHooks()
        status = hooks.get_system_status()

        assert status["status"] == "error"
        assert "Database connection failed" in status["error"]
        assert "timestamp" in status

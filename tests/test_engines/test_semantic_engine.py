"""Test ChromaDB-based semantic document analysis functionality.

Tests for semantic_engine.py using ChromaDB infrastructure for document
similarity, redundancy detection, and placement suggestions.
"""

import os
from unittest.mock import Mock, patch

import pytest


class TestSemanticEngine:
    """Test ChromaDB-based semantic engine functionality."""

    def test_semantic_engine_import_when_module_exists_imports_successfully(self):
        """Test that semantic_engine module can be imported."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        assert SemanticEngine is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    @patch("semantic_docs.engines.semantic_engine.load_config")
    def test_semantic_engine_initialization_when_valid_config_creates_instance(
        self, mock_load_config, mock_client
    ):
        """Test SemanticEngine initialization with valid configuration."""
        from semantic_docs.config.settings import SemanticConfig
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock configuration
        mock_config = SemanticConfig()
        mock_load_config.return_value = mock_config

        # Mock ChromaDB components
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine(
            chroma_path="./test_chroma", collection_name="test_docs"
        )
        assert engine is not None
        assert engine.collection_name == "test_docs"

    @patch("semantic_docs.engines.semantic_engine.load_config")
    def test_semantic_engine_initialization_when_missing_api_key_raises_error(
        self, mock_load_config
    ):
        """Test SemanticEngine initialization fails without API key."""
        from semantic_docs.config.settings import SemanticConfig
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock configuration
        mock_config = SemanticConfig()
        mock_load_config.return_value = mock_config

        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="OPENAI_API_KEY environment variable required"
            ):
                SemanticEngine()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_semantic_engine_initialization_when_collection_missing_raises_error(
        self, mock_client
    ):
        """Test SemanticEngine initialization fails when collection doesn't exist."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock ChromaDB client that raises ValueError for missing collection
        mock_client_instance = Mock()
        mock_client_instance.get_collection.side_effect = ValueError(
            "Collection not found"
        )
        mock_client.return_value = mock_client_instance

        with pytest.raises(ValueError, match="Collection 'markdown_docs' not found"):
            SemanticEngine()


class TestDocumentSimilarity:
    """Test document similarity search functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_find_similar_documents_when_valid_query_returns_results(self, mock_client):
        """Test finding similar documents with valid query."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock ChromaDB collection with query results
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Test document content", "Another document"]],
            "distances": [[0.1, 0.3]],  # 0.9 and 0.7 similarity
            "metadatas": [
                [{"file_path": "docs/test.md"}, {"file_path": "docs/other.md"}]
            ],
            "ids": [["doc1", "doc2"]],
        }

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        results = engine.find_similar_documents("test query", similarity_threshold=0.8)

        assert len(results) == 1  # Only first doc meets 0.8 threshold
        assert results[0]["similarity"] == 0.9
        assert results[0]["file_path"] == "docs/test.md"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_find_similar_documents_when_empty_query_raises_error(self, mock_client):
        """Test finding similar documents fails with empty query."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()

        with pytest.raises(ValueError, match="Query text cannot be empty"):
            engine.find_similar_documents("")


class TestRedundancyDetection:
    """Test redundancy detection functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_detect_redundancy_when_similar_content_found_returns_redundancy(
        self, mock_client
    ):
        """Test redundancy detection finds similar content."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock similar documents found
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Similar document content"]],
            "distances": [[0.1]],  # 0.9 similarity
            "metadatas": [[{"file_path": "docs/similar.md"}]],
            "ids": [["doc1"]],
        }

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        result = engine.detect_redundancy("Test content", threshold=0.85)

        assert result["redundancy_detected"] is True
        assert result["similarity_threshold"] == 0.85
        assert len(result["similar_documents"]) == 1
        assert result["similar_documents"][0]["file_path"] == "docs/similar.md"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_detect_redundancy_when_no_similar_content_returns_no_redundancy(
        self, mock_client
    ):
        """Test redundancy detection with no similar content."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock no similar documents found
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "ids": [[]],
        }

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        result = engine.detect_redundancy("Unique content", threshold=0.85)

        assert result["redundancy_detected"] is False
        assert len(result["similar_documents"]) == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_detect_redundancy_when_empty_content_raises_error(self, mock_client):
        """Test redundancy detection fails with empty content."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()

        with pytest.raises(ValueError, match="Document content cannot be empty"):
            engine.detect_redundancy("")


class TestPlacementSuggestions:
    """Test document placement suggestion functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_suggest_placement_when_similar_docs_found_returns_suggestions(
        self, mock_client
    ):
        """Test placement suggestions based on similar documents."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock similar documents in specific directories
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Doc 1 content", "Doc 2 content", "Doc 3 content"]],
            "distances": [[0.2, 0.3, 0.4]],  # 0.8, 0.7, 0.6 similarity
            "metadatas": [
                [
                    {"file_path": "docs/development/guide1.md"},
                    {"file_path": "docs/development/guide2.md"},
                    {"file_path": "docs/architecture/arch1.md"},
                ]
            ],
            "ids": [["doc1", "doc2", "doc3"]],
        }

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        result = engine.suggest_placement("New development guide content")

        assert len(result["placement_suggestions"]) > 0

        # Should suggest docs/development as top choice (2 similar docs)
        top_suggestion = result["placement_suggestions"][0]
        assert "docs/development" in top_suggestion["directory"]
        assert top_suggestion["similar_file_count"] == 2

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_suggest_placement_when_empty_content_raises_error(self, mock_client):
        """Test placement suggestion fails with empty content."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()

        with pytest.raises(ValueError, match="Document content cannot be empty"):
            engine.suggest_placement("")


class TestCollectionStats:
    """Test ChromaDB collection statistics functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_get_collection_stats_when_collection_healthy_returns_stats(
        self, mock_client
    ):
        """Test getting collection statistics."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock collection with count method
        mock_collection = Mock()
        mock_collection.count.return_value = 42

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        stats = engine.get_collection_stats()

        assert stats["total_documents"] == 42
        assert stats["collection_name"] == "markdown_docs"
        assert stats["embedding_model"] == "text-embedding-3-large"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("chromadb.PersistentClient")
    def test_get_collection_stats_when_collection_error_returns_error_info(
        self, mock_client
    ):
        """Test getting collection statistics handles errors."""
        from semantic_docs.engines.semantic_engine import SemanticEngine

        # Mock collection that raises error on count
        mock_collection = Mock()
        mock_collection.count.side_effect = Exception("Database error")

        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        engine = SemanticEngine()
        stats = engine.get_collection_stats()

        assert "error" in stats
        assert "Database error" in stats["error"]

"""ChromaDB-based semantic document analysis engine.

Leverages existing ChromaDB infrastructure for semantic document analysis,
redundancy detection, and placement recommendations for Claude Code integration.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from semantic_docs.config.settings import SemanticConfig, load_config


class SemanticEngine:
    """Semantic document analyzer using existing ChromaDB infrastructure."""

    def __init__(
        self,
        config: SemanticConfig | None = None,
        chroma_path: str | None = None,
        collection_name: str | None = None,
    ):
        """Initialize semantic engine with ChromaDB.

        Args:
            config: Optional SemanticConfig instance, will load default if not provided
            chroma_path: Optional override for ChromaDB path
            collection_name: Optional override for collection name

        Raises:
            ValueError: If OpenAI API key not found or ChromaDB not accessible
        """
        # Load configuration
        if config is None:
            config = load_config()
        self.config = config

        # Override config values if provided
        if chroma_path is not None:
            self.config.chroma_path = chroma_path
        if collection_name is not None:
            self.config.collection_name = collection_name

        # Setup logging
        self.logger = logging.getLogger("semantic_engine")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.chroma_path = Path(self.config.chroma_path)
        self.collection_name = self.config.collection_name

        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY environment variable not found")
            raise ValueError("OPENAI_API_KEY environment variable required")

        # Initialize ChromaDB client
        try:
            self.logger.info(f"Initializing ChromaDB client at {self.chroma_path}")
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))

            # Create OpenAI embedding function with timeout
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, model_name="text-embedding-3-large"
            )

            # Get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.openai_ef
            )

            # Log collection stats
            doc_count = self.collection.count()
            self.logger.info(
                f"Connected to collection '{self.collection_name}' with {doc_count} documents"
            )

        except ValueError as e:
            self.logger.error(f"Collection '{self.collection_name}' not found: {e}")
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                f"Run scripts/index_markdown_chromadb.py first to create the document index."
            ) from e
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise

    def find_similar_documents(
        self,
        query_text: str,
        similarity_threshold: float | None = None,
        max_results: int | None = None,
        label_filters: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find documents similar to the given query text.

        Args:
            query_text: Text to find similar documents for
            similarity_threshold: Minimum similarity score (0-1), uses config default if not provided
            max_results: Maximum number of results to return, uses config default if not provided
            label_filters: Optional list of labels to filter results by

        Returns:
            List of similar documents with metadata and similarity scores
        """
        if not query_text.strip():
            self.logger.error("Empty query text provided")
            raise ValueError("Query text cannot be empty")

        # Use config defaults if not provided
        if similarity_threshold is None:
            similarity_threshold = self.config.redundancy_threshold
        if max_results is None:
            max_results = self.config.max_results

        start_time = time.time()
        self.logger.debug(
            f"Searching for similar documents with threshold {similarity_threshold}"
        )

        try:
            results = self.collection.query(
                query_texts=[query_text], n_results=max_results
            )
            query_time = time.time() - start_time
            self.logger.debug(f"ChromaDB query completed in {query_time:.3f}s")

            similar_docs = self._process_query_results(results, similarity_threshold)
            similar_docs = self._apply_label_filtering(similar_docs, label_filters)

            self._log_search_results(
                start_time, similar_docs, similarity_threshold, label_filters
            )
            return similar_docs

        except Exception as e:
            self.logger.error(f"Error finding similar documents: {e}")
            raise

    def _process_query_results(
        self, results: dict, similarity_threshold: float
    ) -> list[dict[str, Any]]:
        """Process ChromaDB query results and apply similarity filtering."""
        similar_docs = []

        if not (results["documents"] and results["distances"]):
            return similar_docs

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]

        distance_threshold = self._calculate_distance_threshold(similarity_threshold)

        for doc, distance, metadata, doc_id in zip(
            documents, distances, metadatas, ids, strict=False
        ):
            if distance <= distance_threshold:
                similarity_display = 1.0 / (1.0 + distance)
                similar_docs.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "similarity": similarity_display,
                        "distance": distance,
                        "metadata": metadata or {},
                        "file_path": self._extract_file_path(metadata),
                    }
                )

        return similar_docs

    def _calculate_distance_threshold(self, similarity_threshold: float) -> float:
        """Calculate distance threshold from similarity threshold."""
        return (
            (1.0 - similarity_threshold) / similarity_threshold
            if similarity_threshold > 0
            else float("inf")
        )

    def _extract_file_path(self, metadata: dict | None) -> str:
        """Extract file path from metadata."""
        if not metadata:
            return ""
        return metadata.get("filepath", metadata.get("file_path", ""))

    def _apply_label_filtering(
        self, similar_docs: list[dict], label_filters: list[str] | None
    ) -> list[dict]:
        """Apply label filtering to search results."""
        if not label_filters:
            return similar_docs

        filtered_docs = []
        for doc in similar_docs:
            doc_labels = doc["metadata"].get("labels", [])
            if doc_labels and any(label in doc_labels for label in label_filters):
                filtered_docs.append(doc)

        self.logger.debug(
            f"Label filtering reduced results from {len(similar_docs)} to {len(filtered_docs)} documents"
        )
        return filtered_docs

    def _log_search_results(
        self,
        start_time: float,
        similar_docs: list,
        similarity_threshold: float,
        label_filters: list[str] | None,
    ):
        """Log search results summary."""
        total_time = time.time() - start_time
        filter_info = f" (labels: {', '.join(label_filters)})" if label_filters else ""
        self.logger.info(
            f"Found {len(similar_docs)} similar documents (threshold={similarity_threshold:.2f}){filter_info} in {total_time:.3f}s"
        )

    def detect_redundancy(
        self, document_content: str, threshold: float | None = None
    ) -> dict[str, Any]:
        """Detect redundant content in existing documents.

        Args:
            document_content: Content to check for redundancy
            threshold: Similarity threshold for redundancy detection, uses config default if not provided

        Returns:
            Redundancy analysis results
        """
        if not document_content.strip():
            self.logger.error(
                "Empty document content provided for redundancy detection"
            )
            raise ValueError("Document content cannot be empty")

        # Use config default if not provided
        if threshold is None:
            threshold = self.config.redundancy_threshold

        start_time = time.time()
        self.logger.debug(f"Detecting redundancy with threshold {threshold}")

        try:
            similar_docs = self.find_similar_documents(
                document_content, similarity_threshold=threshold, max_results=5
            )

            redundancy_found = len(similar_docs) > 0

            # Format for Claude Code integration
            result = {
                "redundancy_detected": redundancy_found,
                "similarity_threshold": threshold,
                "similar_documents": [],
            }

            if redundancy_found:
                for doc in similar_docs:
                    preview_length = self.config.preview_length
                    content_preview = (
                        (
                            doc["content"][:preview_length] + "..."
                            if len(doc["content"]) > preview_length
                            else doc["content"]
                        )
                        if self.config.include_content_preview
                        else ""
                    )

                    result["similar_documents"].append(
                        {
                            "file_path": doc["file_path"],
                            "similarity_score": doc["similarity"],
                            "content_preview": content_preview,
                        }
                    )

            processing_time = time.time() - start_time
            self.logger.info(
                f"Redundancy detection completed: {'Found' if redundancy_found else 'No'} redundancy "
                f"({len(similar_docs)} similar docs) in {processing_time:.3f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error detecting redundancy: {e}")
            raise

    def suggest_placement(
        self, document_content: str, candidate_directories: list[str] | None = None
    ) -> dict[str, Any]:
        """Suggest optimal placement for new documentation.

        Args:
            document_content: Content to find placement for
            candidate_directories: Optional list of directories to consider

        Returns:
            Placement recommendations with confidence scores
        """
        if not document_content.strip():
            self.logger.error(
                "Empty document content provided for placement suggestion"
            )
            raise ValueError("Document content cannot be empty")

        start_time = time.time()
        self.logger.debug("Analyzing document placement options")

        try:
            # Find similar documents for placement guidance
            similar_docs = self.find_similar_documents(
                document_content,
                similarity_threshold=self.config.placement_threshold,
                max_results=10,
            )

            # Analyze directory patterns from similar documents
            directory_scores = {}
            for doc in similar_docs:
                file_path = doc["file_path"]
                if file_path:
                    directory = str(Path(file_path).parent)
                    similarity = doc["similarity"]

                    if directory not in directory_scores:
                        directory_scores[directory] = {
                            "score": 0.0,
                            "count": 0,
                            "files": [],
                        }

                    directory_scores[directory]["score"] += similarity
                    directory_scores[directory]["count"] += 1
                    directory_scores[directory]["files"].append(file_path)

            # Calculate average scores and rank directories
            placement_suggestions = []
            for directory, data in directory_scores.items():
                avg_score = data["score"] / data["count"] if data["count"] > 0 else 0.0

                # Skip if no candidate directories specified or directory matches
                if candidate_directories and not any(
                    directory.startswith(candidate)
                    for candidate in candidate_directories
                ):
                    continue

                placement_suggestions.append(
                    {
                        "directory": directory,
                        "confidence": avg_score,
                        "similar_file_count": data["count"],
                        "similar_files": data["files"][:3],  # Show top 3 similar files
                    }
                )

            # Sort by confidence
            placement_suggestions.sort(key=lambda x: x["confidence"], reverse=True)

            result = {
                "placement_suggestions": placement_suggestions[:5],  # Top 5 suggestions
                "analysis_summary": {
                    "total_similar_documents": len(similar_docs),
                    "directories_analyzed": len(directory_scores),
                },
            }

            processing_time = time.time() - start_time
            self.logger.info(
                f"Placement analysis completed: {len(placement_suggestions)} suggestions "
                f"from {len(similar_docs)} similar docs in {processing_time:.3f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error suggesting placement: {e}")
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the ChromaDB collection.

        Returns:
            Collection statistics and metadata
        """
        try:
            self.logger.debug("Retrieving collection statistics")
            count = self.collection.count()

            stats = {
                "total_documents": count,
                "collection_name": self.collection_name,
                "chroma_path": str(self.chroma_path),
                "embedding_model": "text-embedding-3-large",
                "config": {
                    "redundancy_threshold": self.config.redundancy_threshold,
                    "placement_threshold": self.config.placement_threshold,
                    "max_results": self.config.max_results,
                    "validation_timeout": self.config.validation_timeout,
                },
            }

            self.logger.info(
                f"Collection stats retrieved: {count} documents in '{self.collection_name}'"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {
                "error": f"Failed to get collection stats: {str(e)}",
                "collection_name": self.collection_name,
            }

"""Base indexer functionality shared between CLI and incremental indexers.

Provides common ChromaDB initialization, document processing, and AI summarization
capabilities to eliminate code duplication between different indexing approaches.
"""

import logging
import os
import re
import time
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from semantic_docs.config.settings import SemanticConfig, load_config
from semantic_docs.services.ai_summarizer import AISummarizer


class BaseIndexer(ABC):
    """Base class for document indexers with shared functionality."""

    def __init__(
        self,
        config: SemanticConfig | None = None,
        chroma_path: str = None,
        collection_name: str = None,
    ):
        """Initialize the base indexer.

        Args:
            config: Optional SemanticConfig instance, will load default if not provided
            chroma_path: Override ChromaDB path from config
            collection_name: Override collection name from config
        """
        # Load configuration
        if config is None:
            config = load_config()
        self.config = config

        # Setup paths
        self.chroma_path = Path(chroma_path or self.config.chroma_path)
        self.collection_name = collection_name or self.config.collection_name
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Default exclude patterns
        self.exclude_patterns = {
            "*/venv/*",
            "*/endpoint_loadtesting/anna/*",
            "*/endpoint_loadtesting/reports/*",
            "*/site-packages/*",
            "*/node_modules/*",
            "*/.git/*",
            "*/.pytest_cache/*",
            "*/chroma_db/*",
            "*/__pycache__/*",
            ".clinerules",
            ".claude",
        }

        # Initialize ChromaDB
        self._init_chromadb()

        # Initialize AI summarizer if enabled
        self.ai_summarizer = None
        if self.config.ai_summarization_enabled:
            try:
                self.ai_summarizer = AISummarizer(self.config)
                self.logger.info("AI summarization enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI summarizer: {e}")
                self.logger.info("Will use fallback text extraction instead")

    def _init_chromadb(self):
        """Initialize ChromaDB client and embedding function."""
        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with your OpenAI API key."
            )

        # Create OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-large"
        )

        # Create persistent client
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.logger.debug(f"ChromaDB initialized at: {self.chroma_path}")

    def get_or_create_collection(self, rebuild: bool = False):
        """Get or create the markdown documents collection.

        Args:
            rebuild: If True, delete existing collection first

        Returns:
            ChromaDB collection instance
        """
        if rebuild:
            try:
                self.client.delete_collection(self.collection_name)
                self.logger.info(f"Deleted existing collection: {self.collection_name}")
            except ValueError:
                pass  # Collection didn't exist

        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.openai_ef
            )
            self.logger.info(
                f"Connected to existing collection '{self.collection_name}'"
            )
        except (ValueError, Exception):
            # Collection doesn't exist, create it
            # ChromaDB may throw different exceptions depending on version
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.openai_ef,
                metadata={"description": "Hudpin Whisper project documentation"},
            )
            self.logger.info(f"Created new collection '{self.collection_name}'")

        return self.collection

    def should_include_file(
        self, file_path: Path, additional_excludes: set[str] | None = None
    ) -> bool:
        """Check if a file should be included in the index.

        Args:
            file_path: Path to the file to check
            additional_excludes: Additional exclude patterns to apply

        Returns:
            True if file should be indexed, False otherwise
        """
        # Must be a markdown file
        if file_path.suffix.lower() not in [".md", ".markdown"]:
            return False

        # Must exist
        if not file_path.exists():
            return False

        # Convert to relative path for pattern matching
        try:
            # Resolve both paths to handle relative references correctly
            resolved_file = file_path.resolve()
            resolved_project = self.project_root.resolve()
            rel_path = resolved_file.relative_to(resolved_project)
        except ValueError:
            return False  # File is outside project root

        rel_path_str = str(rel_path)

        # Check exclude patterns
        all_excludes = self.exclude_patterns.copy()
        if additional_excludes:
            all_excludes.update(additional_excludes)

        for pattern in all_excludes:
            # Convert glob pattern to simple string matching
            if pattern.replace("*", "").replace("/", "") in rel_path_str.replace(
                "/", ""
            ):
                return False

        return True

    def extract_heading(self, content: str) -> str:
        """Extract the first markdown heading from content.

        Args:
            content: Document content

        Returns:
            First heading found or "Untitled" if none
        """
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                # Remove # symbols and clean up
                heading = re.sub(r"^#+\s*", "", line).strip()
                if heading:
                    return heading
        return "Untitled"

    def _extract_lead_fallback(self, content: str) -> str:
        """Extract lead text as fallback when AI summarization is unavailable.

        Args:
            content: Document content

        Returns:
            Lead text extracted from first few non-heading lines
        """
        lead_lines = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and len(lead_lines) < 5:
                # Remove markdown formatting for lead
                clean_line = line.replace("*", "").replace("_", "").replace("`", "")
                lead_lines.append(clean_line)

        return " ".join(lead_lines)[:300]  # Limit lead length

    def get_relative_path(self, file_path: Path) -> Path:
        """Get relative path from project root, handling resolution correctly.

        Args:
            file_path: Absolute or relative file path

        Returns:
            Path relative to project root
        """
        try:
            resolved_file = file_path.resolve()
            resolved_project = self.project_root.resolve()
            return resolved_file.relative_to(resolved_project)
        except ValueError:
            # File outside project root, use as-is
            return file_path

    def _extract_folder_context(self, rel_path: Path) -> str:
        """Extract semantic context from folder hierarchy.

        Args:
            rel_path: Relative path to the document

        Returns:
            Folder context string for embedding
        """
        # Get all path components except the filename
        path_parts = rel_path.parts[:-1]  # Exclude filename

        if not path_parts:
            return ""

        # Create weighted folder context
        folder_context_parts = []

        for i, part in enumerate(path_parts):
            # Clean up folder names (remove common prefixes/suffixes, handle separators)
            clean_part = part.replace("_", " ").replace("-", " ").strip()

            # Skip generic folder names that don't add semantic value
            if clean_part.lower() in {
                "docs",
                "doc",
                "documentation",
                "files",
                "assets",
            }:
                continue

            # Weight deeper folders higher (they're more specific)
            weight = 1.0 + (
                i * 0.5
            )  # Root folder = 1.0, deeper folders get higher weight

            # Add multiple instances for weighting (simple but effective)
            repeat_count = int(weight)
            folder_context_parts.extend([clean_part] * repeat_count)

        return " ".join(folder_context_parts) if folder_context_parts else ""

    def create_document_metadata(self, file_path: Path, content: str) -> dict[str, Any]:
        """Create document metadata for indexing.

        Args:
            file_path: Path to the document file
            content: Document content

        Returns:
            Dictionary containing document ID, embedding content, and metadata
        """
        # Get relative path for consistent IDs
        rel_path = self.get_relative_path(file_path)
        file_id = str(rel_path)

        # Extract heading and folder context
        heading = self.extract_heading(content)
        folder_context = self._extract_folder_context(rel_path)

        # Generate AI summary and labels if available, fallback to lead extraction
        summary = ""
        labels = []
        embedding_content = content  # Default to full content

        if self.ai_summarizer:
            try:
                summary, labels = self.ai_summarizer.generate_summary_and_labels(
                    content, str(rel_path)
                )
                self.logger.debug(
                    f"Generated AI summary for {rel_path}: {len(summary)} chars, {len(labels)} labels"
                )

                # Use AI summary for embeddings if available
                if summary:
                    # Combine heading, folder context, summary, and labels for embedding
                    # This creates a focused representation of the document's key concepts and location
                    embedding_parts = [heading]
                    if folder_context:
                        embedding_parts.append(folder_context)
                    embedding_parts.append(summary)
                    if labels:
                        embedding_parts.append(" ".join(labels))
                    embedding_content = " | ".join(
                        part for part in embedding_parts if part
                    )

                    self.logger.debug(
                        f"Using AI summary for embedding ({len(embedding_content)} chars vs {len(content)} original)"
                    )

            except Exception as e:
                self.logger.warning(f"AI summarization failed for {rel_path}: {e}")
                # Fallback to lead extraction
                summary = self._extract_lead_fallback(content)
                # Use truncated content for embedding as fallback
                embedding_content = self._create_fallback_embedding_content(
                    content, heading, folder_context
                )
        else:
            # Fallback to lead extraction when AI summarization is disabled
            summary = self._extract_lead_fallback(content)
            # Use truncated content for embedding as fallback
            embedding_content = self._create_fallback_embedding_content(
                content, heading, folder_context
            )

        # File stats
        stat = file_path.stat()
        last_updated = datetime.fromtimestamp(stat.st_mtime).isoformat()
        file_size = stat.st_size
        directory = str(rel_path.parent) if rel_path.parent != Path(".") else "root"

        return {
            "id": file_id,
            "content": embedding_content,  # This is what gets embedded
            "metadata": {
                "filepath": file_id,
                "heading": heading,
                "summary": summary,
                "labels": ", ".join(labels) if labels else "",
                "last_updated": last_updated,
                "file_size": file_size,
                "directory": directory,
                "folder_context": folder_context,
                "folder_depth": len(rel_path.parts) - 1,  # Exclude filename
                "folder_path_components": ", ".join(rel_path.parts[:-1])
                if len(rel_path.parts) > 1
                else "",
                "indexed_at": datetime.now().isoformat(),
                "ai_generated": bool(self.ai_summarizer and labels),
                "original_content": content,  # Store original content in metadata
                "embedding_source": "ai_summary"
                if self.ai_summarizer and summary
                else "fallback",
            },
        }

    def _create_fallback_embedding_content(
        self, content: str, heading: str, folder_context: str = ""
    ) -> str:
        """Create embedding content when AI summarization is not available.

        Args:
            content: Original document content
            heading: Document heading
            folder_context: Semantic folder context

        Returns:
            Focused content for embedding generation
        """
        # Extract first few paragraphs for embedding instead of full content
        lead_text = self._extract_lead_fallback(content)

        # Combine heading, folder context, and lead text
        embedding_parts = [heading]
        if folder_context:
            embedding_parts.append(folder_context)
        embedding_parts.append(lead_text)
        embedding_content = " | ".join(part for part in embedding_parts if part)

        # Ensure it's not too long (aim for ~500 tokens max)
        if len(embedding_content) > 2000:  # ~500 tokens
            embedding_content = embedding_content[:2000] + "..."

        self.logger.debug(
            f"Using fallback content for embedding ({len(embedding_content)} chars vs {len(content)} original)"
        )

        return embedding_content

    def read_file_content(self, file_path: Path) -> str | None:
        """Read file content with error handling.

        Args:
            file_path: Path to the file to read

        Returns:
            File content as string, or None if reading failed
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def truncate_content(self, content: str, max_tokens: int = 8000) -> str:
        """Truncate content to fit within token limit.

        Args:
            content: Content to truncate
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated content with indicator if truncated
        """
        estimated_tokens = self.estimate_tokens(content)
        if estimated_tokens <= max_tokens:
            return content

        # Truncate to approximately max_tokens * 4 characters
        max_chars = max_tokens * 4
        truncated = content[:max_chars]

        # Try to truncate at a sensible boundary (end of paragraph)
        last_double_newline = truncated.rfind("\n\n")
        if last_double_newline > max_chars * 0.8:  # If we can keep 80% of content
            truncated = truncated[:last_double_newline]

        return truncated + "\n\n[Content truncated for indexing]"

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "chroma_path": str(self.chroma_path),
                "embedding_model": "text-embedding-3-large",
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def log_processing_time(self, operation: str, file_path: Path, start_time: float):
        """Log processing time for an operation.

        Args:
            operation: Name of the operation performed
            file_path: Path to the file processed
            start_time: Start time from time.time()
        """
        elapsed_time = time.time() - start_time
        rel_path = self.get_relative_path(file_path)
        self.logger.info(f"{operation} {rel_path} in {elapsed_time:.3f}s")

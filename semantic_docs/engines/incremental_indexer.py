"""Incremental document indexer for ChromaDB integration.

Provides functionality to add, update, and remove individual documents from the ChromaDB
index without rebuilding the entire collection. Designed for real-time indexing of
new markdown files created or updated by Claude Code.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from semantic_docs.config.settings import SemanticConfig, load_config
from semantic_docs.engines.base_indexer import BaseIndexer


class IncrementalIndexer(BaseIndexer):
    """Handles incremental updates to the ChromaDB document index."""

    def __init__(self, config: SemanticConfig | None = None):
        """Initialize the incremental indexer.

        Args:
            config: Optional SemanticConfig instance, will load default if not provided
        """
        # Initialize base indexer
        super().__init__(config=config)

        # Initialize collection (connect to existing or create new)
        self.get_or_create_collection(rebuild=False)

        # Override project root for this specific indexer
        self.project_root = self.project_root  # Keep the base class value

        # Log initialization status
        if self.ai_summarizer:
            self.logger.info("AI summarization enabled")
        else:
            if self.config.ai_summarization_enabled:
                self.logger.warning(
                    "AI summarization configured but failed to initialize"
                )
                self.logger.info("Continuing without AI summarization")
            else:
                self.logger.info("AI summarization disabled in configuration")

    def should_index_file(self, file_path: str | Path) -> bool:
        """Determine if a file should be indexed.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be indexed, False otherwise
        """
        file_path = Path(file_path)

        # Use the base class method with incremental indexer specific excludes
        additional_excludes = {
            "*/endpoint_loadtesting/*",
        }

        return self.should_include_file(file_path, additional_excludes)

    def extract_document_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Extract metadata from a markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary containing document metadata
        """
        file_path = Path(file_path)

        # Read file content
        content = self.read_file_content(file_path)
        if content is None:
            self.logger.error(f"Failed to read {file_path}")
            raise OSError(f"Failed to read {file_path}")

        # Use base class method to create metadata
        return self.create_document_metadata(file_path, content)

    def add_document(self, file_path: str | Path) -> bool:
        """Add a new document to the index.

        Args:
            file_path: Path to the markdown file to add

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        start_time = time.time()

        if not self.should_index_file(file_path):
            self.logger.warning(
                f"File {file_path} should not be indexed (excluded or not markdown)"
            )
            return False

        try:
            # Extract document data
            doc_data = self.extract_document_metadata(file_path)

            # Check if document already exists
            existing_docs = self.collection.get(ids=[doc_data["id"]])
            if existing_docs["ids"]:
                self.logger.info(
                    f"Document {doc_data['id']} already exists, updating instead"
                )
                return self.update_document(file_path)

            # Add to collection
            self.collection.add(
                documents=[doc_data["content"]],
                metadatas=[doc_data["metadata"]],
                ids=[doc_data["id"]],
            )

            self.log_processing_time("Added document", file_path, start_time)
            return True

        except Exception as e:
            self.logger.error(f"Failed to add document {file_path}: {e}")
            return False

    def update_document(self, file_path: str | Path) -> bool:
        """Update an existing document in the index.

        Args:
            file_path: Path to the markdown file to update

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        start_time = time.time()

        if not self.should_index_file(file_path):
            self.logger.warning(f"File {file_path} should not be indexed")
            return False

        try:
            # Extract document data
            doc_data = self.extract_document_metadata(file_path)

            # Check if document exists
            existing_docs = self.collection.get(ids=[doc_data["id"]])
            if not existing_docs["ids"]:
                self.logger.info(
                    f"Document {doc_data['id']} doesn't exist, adding instead"
                )
                return self.add_document(file_path)

            # Update existing document
            self.collection.update(
                documents=[doc_data["content"]],
                metadatas=[doc_data["metadata"]],
                ids=[doc_data["id"]],
            )

            self.log_processing_time("Updated document", file_path, start_time)
            return True

        except Exception as e:
            self.logger.error(f"Failed to update document {file_path}: {e}")
            return False

    def remove_document(self, file_path: str | Path) -> bool:
        """Remove a document from the index.

        Args:
            file_path: Path to the markdown file to remove

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)

        try:
            # Get relative path for consistent ID using base class method
            rel_path = self.get_relative_path(file_path)
            doc_id = str(rel_path)

            # Check if document exists
            existing_docs = self.collection.get(ids=[doc_id])
            if not existing_docs["ids"]:
                self.logger.warning(f"Document {doc_id} not found in index")
                return False

            # Remove from collection
            self.collection.delete(ids=[doc_id])

            self.logger.info(f"Removed document {doc_id} from index")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove document {file_path}: {e}")
            return False

    def upsert_document(self, file_path: str | Path) -> bool:
        """Add or update a document in the index (upsert operation).

        Args:
            file_path: Path to the markdown file to upsert

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)

        if not self.should_index_file(file_path):
            return False

        try:
            # Extract document data
            doc_data = self.extract_document_metadata(file_path)

            # ChromaDB upsert - adds if not exists, updates if exists
            self.collection.upsert(
                documents=[doc_data["content"]],
                metadatas=[doc_data["metadata"]],
                ids=[doc_data["id"]],
            )

            self.logger.info(f"Upserted document {doc_data['id']} to index")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upsert document {file_path}: {e}")
            return False

    def get_document_info(self, file_path: str | Path) -> dict[str, Any] | None:
        """Get information about a document in the index.

        Args:
            file_path: Path to the markdown file

        Returns:
            Document information if found, None otherwise
        """
        file_path = Path(file_path)

        try:
            # Get relative path for consistent ID using base class method
            rel_path = self.get_relative_path(file_path)
            doc_id = str(rel_path)

            # Get document from collection
            result = self.collection.get(ids=[doc_id], include=["metadatas"])

            if result and result["ids"] and result["metadatas"]:
                return {"id": result["ids"][0], "metadata": result["metadatas"][0]}
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get document info for {file_path}: {e}")
            return None

    def is_document_indexed(self, file_path: str | Path) -> bool:
        """Check if a document is already indexed.

        Args:
            file_path: Path to the markdown file

        Returns:
            True if document is indexed, False otherwise
        """
        return self.get_document_info(file_path) is not None

    def needs_update(self, file_path: str | Path) -> bool:
        """Check if a document needs to be updated in the index.

        Args:
            file_path: Path to the markdown file

        Returns:
            True if document needs update, False otherwise
        """
        file_path = Path(file_path)

        if not Path(file_path).exists():
            return False

        doc_info = self.get_document_info(file_path)
        if not doc_info:
            return True  # Not indexed, needs to be added

        # Check if file is newer than indexed version
        try:
            file_mtime = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
            indexed_time = datetime.fromisoformat(doc_info["metadata"]["last_updated"])
            return file_mtime > indexed_time
        except (KeyError, ValueError):
            return True  # Can't determine, assume needs update

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


def main():
    """CLI interface for incremental indexing operations."""
    args = _parse_cli_arguments()
    _setup_logging(args.verbose)

    try:
        config = load_config(args.config) if args.config else load_config()
        indexer = IncrementalIndexer(config)
        _execute_command(args, indexer)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _parse_cli_arguments():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Incremental document indexer for ChromaDB"
    )
    parser.add_argument(
        "command", choices=["add", "update", "remove", "upsert", "info", "stats"]
    )
    parser.add_argument("-f", "--file", help="File path to operate on")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def _setup_logging(verbose: bool):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _execute_command(args, indexer):
    """Execute the specified command."""
    if args.command == "stats":
        _handle_stats_command(indexer)
    elif args.command in ["add", "update", "remove", "upsert", "info"]:
        _handle_file_command(args, indexer)


def _handle_stats_command(indexer):
    """Handle the stats command."""
    stats = indexer.get_collection_stats()
    print(f"Collection: {stats.get('collection_name', 'unknown')}")
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"ChromaDB path: {stats.get('chroma_path', 'unknown')}")


def _handle_file_command(args, indexer):
    """Handle file-based commands."""
    if not args.file:
        print(f"Error: --file required for {args.command} command")
        sys.exit(1)

    command_handlers = {
        "add": lambda: indexer.add_document(args.file),
        "update": lambda: indexer.update_document(args.file),
        "remove": lambda: indexer.remove_document(args.file),
        "upsert": lambda: indexer.upsert_document(args.file),
        "info": lambda: indexer.get_document_info(args.file),
    }

    if args.command == "info":
        info = command_handlers[args.command]()
        _display_document_info(info)
    else:
        success = command_handlers[args.command]()
        print(f"{args.command.title()} {'succeeded' if success else 'failed'}")


def _display_document_info(info):
    """Display document information."""
    if info:
        print(f"Document ID: {info['id']}")
        print(f"Last updated: {info['metadata'].get('last_updated', 'unknown')}")
        print(f"Indexed at: {info['metadata'].get('indexed_at', 'unknown')}")
        print(f"Directory: {info['metadata'].get('directory', 'unknown')}")
    else:
        print("Document not found in index")


if __name__ == "__main__":
    main()

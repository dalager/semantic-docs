#!/usr/bin/env python3
"""
Markdown Documentation Indexer using ChromaDB

This script indexes all markdown files in the project using ChromaDB with OpenAI embeddings.
It creates a searchable vector database of the documentation and exports metadata to JSON.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python scripts/index_markdown_chromadb.py [options]

Options:
    --rebuild     Clear existing collection and rebuild from scratch
    --dry-run     List files that would be processed without indexing
    --limit N     Process only the first N files (for testing)
    --exclude     Additional exclude patterns (comma-separated)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from semantic_docs.engines.base_indexer import BaseIndexer


class MarkdownIndexer(BaseIndexer):
    """Indexes markdown files using ChromaDB with OpenAI embeddings."""

    def __init__(
        self, chroma_path: str = "./chroma_db", collection_name: str = "markdown_docs"
    ):
        # Initialize base indexer with custom paths
        super().__init__(chroma_path=chroma_path, collection_name=collection_name)

        # Add CLI-specific exclude patterns
        self.exclude_patterns.update(
            {
                "*/endpoint_loadtesting/anna/*",
                "*/endpoint_loadtesting/reports/*",
                ".clinerules",
                ".claude",
            }
        )

        # Print initialization status for CLI
        if self.ai_summarizer:
            print("AI summarization enabled")
        else:
            if self.config.ai_summarization_enabled:
                print("Warning: AI summarization configured but failed to initialize")
                print("Will use fallback text extraction instead")
            else:
                print("AI summarization disabled in configuration")

    def get_or_create_collection(self, rebuild: bool = False):
        """Get or create the markdown documents collection."""
        # Call base method
        collection = super().get_or_create_collection(rebuild)

        # Print CLI-friendly messages
        if rebuild:
            print(f"Deleted existing collection: {self.collection_name}")
        print(f"ChromaDB initialized at: {self.chroma_path}")
        print(f"Using collection: {self.collection_name}")

        return collection

    def find_markdown_files(self, additional_excludes: set[str] = None) -> list[Path]:
        """Find all markdown files that should be indexed."""
        markdown_files = []

        # Search recursively from project root
        for md_file in self.project_root.rglob("*.md"):
            if self.should_include_file(md_file, additional_excludes):
                markdown_files.append(md_file)

        # Sort by modification time (newest first)
        markdown_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return markdown_files

    def extract_lead(self, content: str, max_lines: int = 10) -> str:
        """Extract first N non-empty lines as plaintext preview."""
        lines = content.split("\n")
        lead_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove markdown formatting
                import re

                clean_line = re.sub(r"[*_`]", "", line)
                clean_line = re.sub(
                    r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_line
                )  # Links
                lead_lines.append(clean_line)

                if len(lead_lines) >= max_lines:
                    break

        return " ".join(lead_lines)

    def process_file(self, file_path: Path) -> dict | None:
        """Process a single markdown file and extract metadata."""
        # Read file content
        content = self.read_file_content(file_path)
        if content is None:
            return None

        # Create document metadata using base method
        doc_data = self.create_document_metadata(file_path, content)

        # Print CLI-friendly progress for AI processing
        if self.ai_summarizer and doc_data and doc_data["metadata"]["ai_generated"]:
            rel_path = self.get_relative_path(file_path)
            summary_len = len(doc_data["metadata"]["summary"])
            labels_count = (
                len(doc_data["metadata"]["labels"].split(", "))
                if doc_data["metadata"]["labels"]
                else 0
            )
            print(
                f"Generated AI summary for {rel_path}: {summary_len} chars, {labels_count} labels"
            )

        return doc_data

    def index_files(self, files: list[Path], limit: int = None) -> list[dict]:
        """Index files into ChromaDB and return metadata."""
        if limit:
            files = files[:limit]

        indexed_docs = []
        batch_size = 10
        max_tokens_per_doc = 8000

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            batch_docs, oversized_docs = self._process_file_batch(
                batch, max_tokens_per_doc
            )

            # Index regular batch
            if batch_docs:
                self._index_batch_documents(batch_docs, indexed_docs, i, batch_size)

            # Index oversized documents
            self._index_oversized_documents(oversized_docs, indexed_docs)

        return indexed_docs

    def _process_file_batch(
        self, batch: list[Path], max_tokens: int
    ) -> tuple[list[dict], list[dict]]:
        """Process a batch of files and separate by size."""
        batch_docs = []
        oversized_docs = []

        for file_path in batch:
            doc = self.process_file(file_path)
            if doc:
                estimated_tokens = self.estimate_tokens(doc["content"])
                if estimated_tokens > max_tokens:
                    print(f"⚠️  Large document ({estimated_tokens} tokens): {doc['id']}")
                    doc["content"] = self.truncate_content(doc["content"], max_tokens)
                    oversized_docs.append(doc)
                else:
                    batch_docs.append(doc)

        return batch_docs, oversized_docs

    def _index_batch_documents(
        self,
        batch_docs: list[dict],
        indexed_docs: list[dict],
        batch_index: int,
        batch_size: int,
    ):
        """Index a batch of regular-sized documents."""
        batch_ids = [doc["id"] for doc in batch_docs]
        batch_contents = [doc["content"] for doc in batch_docs]
        batch_metadatas = [doc["metadata"] for doc in batch_docs]

        try:
            self.collection.add(
                ids=batch_ids,
                documents=batch_contents,
                metadatas=batch_metadatas,
            )
            indexed_docs.extend(batch_docs)
            print(
                f"Indexed batch {batch_index // batch_size + 1}: {len(batch_docs)} files"
            )
        except Exception as e:
            print(f"❌ Error indexing batch: {e}")
            self._index_documents_individually(batch_docs, indexed_docs)

    def _index_documents_individually(self, docs: list[dict], indexed_docs: list[dict]):
        """Index documents one by one as fallback."""
        for doc in docs:
            try:
                self.collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]],
                )
                indexed_docs.append(doc)
                print(f"✅ Individually indexed: {doc['id']}")
            except Exception as individual_error:
                print(f"❌ Failed to index {doc['id']}: {individual_error}")

    def _index_oversized_documents(
        self, oversized_docs: list[dict], indexed_docs: list[dict]
    ):
        """Index oversized documents individually."""
        for doc in oversized_docs:
            try:
                self.collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]],
                )
                indexed_docs.append(doc)
                print(f"✅ Indexed large document (truncated): {doc['id']}")
            except Exception as e:
                print(f"❌ Failed to index large document {doc['id']}: {e}")

    def export_to_json(
        self, indexed_docs: list[dict], output_path: str = "documentindex.json"
    ):
        """Export indexed document metadata to JSON."""
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_files": len(indexed_docs),
                "chroma_collection": self.collection_name,
                "embedding_model": "text-embedding-3-large",
            },
            "documents": [doc["metadata"] for doc in indexed_docs],
        }

        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Exported metadata to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Index markdown files using ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing collection and rebuild from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without indexing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Process only the first N files (for testing)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        metavar="PATTERNS",
        help="Additional exclude patterns (comma-separated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="documentindex.json",
        help="Output JSON file path (default: documentindex.json)",
    )

    args = parser.parse_args()

    # Parse additional excludes
    additional_excludes = set()
    if args.exclude:
        additional_excludes = {p.strip() for p in args.exclude.split(",")}

    try:
        # Initialize indexer
        indexer = MarkdownIndexer()

        # Find files
        print("Discovering markdown files...")
        files = indexer.find_markdown_files(additional_excludes)

        if args.limit:
            files = files[: args.limit]

        print(f"Found {len(files)} markdown files to process")

        # Dry run - just list files
        if args.dry_run:
            print("\nFiles that would be processed:")
            for i, file_path in enumerate(files, 1):
                rel_path = file_path.relative_to(indexer.project_root)
                file_size = file_path.stat().st_size
                print(f"{i:3d}. {rel_path} ({file_size} bytes)")
            return

        # Initialize collection
        _ = indexer.get_or_create_collection(rebuild=args.rebuild)

        # Index files
        print(f"\nIndexing {len(files)} files...")
        indexed_docs = indexer.index_files(files, args.limit)

        # Export to JSON
        export_file = indexer.export_to_json(indexed_docs, args.output)

        print(f"\n✅ Successfully indexed {len(indexed_docs)} documents")
        print(f"   ChromaDB collection: {indexer.collection_name}")
        print(f"   JSON export: {export_file}")

    except KeyboardInterrupt:
        print("\n❌ Indexing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Semantic Search CLI for document discovery and similarity analysis.

This CLI provides semantic search capabilities over the indexed markdown documentation,
allowing natural language queries and file-based similarity searches.
"""

import argparse
import json
import sys
from pathlib import Path

from semantic_docs import SemanticEngine, load_config


class SemanticSearchCLI:
    """CLI interface for semantic search operations."""

    def __init__(self, config_path: str | None = None):
        """Initialize the semantic search CLI.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = load_config(config_path) if config_path else load_config()
        self.engine = SemanticEngine(self.config)

    def search(
        self,
        query: str,
        max_results: int = 10,
        threshold: float = 0.3,
        output_format: str = "human",
        labels: list[str] | None = None,
    ) -> None:
        """Perform semantic search with natural language query.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            threshold: Similarity threshold (0-1)
            output_format: Output format (human, json, paths)
            labels: Optional list of labels to filter results by
        """
        try:
            results = self.engine.find_similar_documents(
                query,
                max_results=max_results,
                similarity_threshold=threshold,
                label_filters=labels,
            )

            self._output_search_results(results, query, threshold, output_format)

        except Exception as e:
            print(f"Search failed: {e}", file=sys.stderr)
            sys.exit(1)

    def _output_search_results(
        self, results: list, query: str, threshold: float, output_format: str
    ):
        """Output search results in the specified format."""
        if output_format == "json":
            print(json.dumps(results, indent=2))
        elif output_format == "paths":
            self._output_paths_format(results)
        else:  # human format
            self._output_human_format(results, query, threshold)

    def _output_paths_format(self, results: list):
        """Output results as file paths only."""
        for result in results:
            print(result.get("file_path", "unknown"))

    def _output_human_format(self, results: list, query: str, threshold: float):
        """Output results in human-readable format."""
        if not results:
            print(f"No documents found matching '{query}' (threshold: {threshold})")
            return

        print(f"Found {len(results)} documents matching '{query}':\n")
        for i, result in enumerate(results, 1):
            self._display_search_result(result, i)

    def _display_search_result(self, result: dict, index: int):
        """Display a single search result."""
        file_path = result.get("file_path", "unknown")
        similarity = result.get("similarity", 0)
        distance = result.get("distance", 0)

        print(f"{index:2d}. {file_path}")
        print(f"    Similarity: {similarity:.3f} (distance: {distance:.3f})")

        metadata = result.get("metadata", {})
        ai_generated = metadata.get("ai_generated", False)

        self._display_summary_or_preview(result, metadata, ai_generated)
        self._display_labels(metadata, ai_generated)
        print()

    def _display_summary_or_preview(
        self, result: dict, metadata: dict, ai_generated: bool
    ):
        """Display either AI summary or content preview."""
        summary = metadata.get("summary", "")

        if summary:
            status = "🤖 AI Summary" if ai_generated else "📝 Summary"
            print(f"    {status}: {summary}")
        else:
            content = result.get("content", "")
            if content and len(content) > 100:
                preview = content[:200].replace("\n", " ").strip()
                print(f"    Preview: {preview}...")
            elif content:
                print(f"    Preview: {content}")

    def _display_labels(self, metadata: dict, ai_generated: bool):
        """Display labels if available."""
        labels = metadata.get("labels", [])
        if labels:
            status = "🏷️  Labels" if ai_generated else "📋 Labels"
            labels_str = labels if isinstance(labels, str) else ", ".join(labels)
            print(f"    {status}: {labels_str}")

    def find_similar(
        self,
        file_path: str,
        max_results: int = 10,
        threshold: float = 0.3,
        output_format: str = "human",
    ) -> None:
        """Find documents similar to the given file.

        Args:
            file_path: Path to reference file
            max_results: Maximum number of results to return
            threshold: Similarity threshold (0-1)
            output_format: Output format (human, json, paths)
        """
        file_path = Path(file_path)

        self._validate_file_path(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
            results = self._get_filtered_similar_results(
                content, file_path, max_results, threshold
            )
            self._output_similarity_results(
                results, file_path, threshold, output_format
            )

        except Exception as e:
            print(f"Similarity search failed: {e}", file=sys.stderr)
            sys.exit(1)

    def _validate_file_path(self, file_path: Path):
        """Validate the input file path."""
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist", file=sys.stderr)
            sys.exit(1)

        if file_path.suffix.lower() not in [".md", ".markdown"]:
            print(f"Warning: {file_path} is not a markdown file")

    def _get_filtered_similar_results(
        self, content: str, file_path: Path, max_results: int, threshold: float
    ) -> list:
        """Get similar documents with original file filtered out."""
        results = self.engine.find_similar_documents(
            content, max_results=max_results, similarity_threshold=threshold
        )

        # Filter out the original file if it's in results
        return [
            r
            for r in results
            if r.get("metadata", {}).get("file_path") != str(file_path)
        ]

    def _output_similarity_results(
        self, results: list, file_path: Path, threshold: float, output_format: str
    ):
        """Output similarity search results in the specified format."""
        if output_format == "json":
            output = {
                "reference_file": str(file_path),
                "similar_documents": results,
            }
            print(json.dumps(output, indent=2))
        elif output_format == "paths":
            self._output_paths_format(results)
        else:  # human format
            self._output_similarity_human_format(results, file_path, threshold)

    def _output_similarity_human_format(
        self, results: list, file_path: Path, threshold: float
    ):
        """Output similarity results in human-readable format."""
        if not results:
            print(
                f"No similar documents found for '{file_path}' (threshold: {threshold})"
            )
            return

        print(f"Found {len(results)} documents similar to '{file_path}':\n")
        for i, result in enumerate(results, 1):
            self._display_search_result(result, i)

    def get_stats(self) -> None:
        """Display collection statistics."""
        try:
            stats = self.engine.get_collection_stats()
            print(f"Collection: {stats.get('collection_name', 'unknown')}")
            print(f"Total documents: {stats.get('total_documents', 0)}")
            print(f"ChromaDB path: {stats.get('chroma_path', 'unknown')}")
            print(f"Embedding model: {stats.get('embedding_model', 'unknown')}")
        except Exception as e:
            print(f"Failed to get stats: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic search over indexed markdown documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with natural language query
  semantic-docs search "opentelemetry configuration"
  semantic-docs search "testing patterns" --max-results 5 --threshold 0.5

  # Search with label filtering
  semantic-docs search "authentication" --labels API Security
  semantic-docs search "database queries" --labels Database Reference

  # Find similar documents to a file
  semantic-docs similar -f docs/architecture/CORE_CONCEPTS.md
  semantic-docs similar -f my-doc.md --output json

  # Get collection statistics
  semantic-docs stats
        """,
    )

    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold 0-1 (default: 0.3, lower = more results)",
    )
    parser.add_argument(
        "--output",
        choices=["human", "json", "paths"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Filter results by labels (space-separated list, e.g., --labels API Testing)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search with natural language query"
    )
    search_parser.add_argument("query", help="Natural language search query")

    # Similar command
    similar_parser = subparsers.add_parser(
        "similar", help="Find documents similar to a file"
    )
    similar_parser.add_argument(
        "-f", "--file", required=True, help="Path to reference file"
    )

    # Stats command
    _ = subparsers.add_parser("stats", help="Show collection statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    try:
        cli = SemanticSearchCLI(args.config)
    except Exception as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute command
    if args.command == "search":
        cli.search(
            args.query,
            max_results=args.max_results,
            threshold=args.threshold,
            output_format=args.output,
            labels=args.labels,
        )
    elif args.command == "similar":
        cli.find_similar(
            args.file,
            max_results=args.max_results,
            threshold=args.threshold,
            output_format=args.output,
        )
    elif args.command == "stats":
        cli.get_stats()


if __name__ == "__main__":
    main()

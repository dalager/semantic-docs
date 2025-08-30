"""Claude Code integration hooks for semantic document analysis.

Provides hook interface for real-time document analysis, redundancy detection,
and placement suggestions during Claude Code markdown file operations.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent

from semantic_docs.config.settings import (
    SemanticConfig,
    load_config,
    validate_environment,
)
from semantic_docs.engines.cluster_engine import DocumentClusterEngine
from semantic_docs.engines.drift_detector import DriftDetector
from semantic_docs.engines.incremental_indexer import IncrementalIndexer
from semantic_docs.engines.semantic_engine import SemanticEngine


class ClaudeCodeHooks:
    """Hook interface for Claude Code semantic document analysis integration."""

    def __init__(self, config: SemanticConfig | None = None):
        """Initialize Claude Code hooks.

        Args:
            config: Optional SemanticConfig instance, will load default if not provided
        """
        # Load configuration
        if config is None:
            config = load_config()
        self.config = config

        # Setup logging
        self.logger = logging.getLogger("claude_hooks")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Initialize semantic engine
        try:
            self.semantic_engine = SemanticEngine(config=self.config)
            self.logger.info("Claude Code hooks initialized successfully")

            # Initialize cluster engine for drift detection (optional)
            self.cluster_engine = None
            self.drift_detector = None
            self._validation_count = 0
            try:
                self.cluster_engine = DocumentClusterEngine(
                    config=self.config, semantic_engine=self.semantic_engine
                )
                self.drift_detector = DriftDetector(config=self.config)
                self.logger.info("Clustering drift detection enabled")
            except Exception as cluster_error:
                self.logger.warning(
                    f"Clustering drift detection disabled: {cluster_error}"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize semantic engine: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4

    def truncate_content(self, content: str, max_tokens: int = 8000) -> str:
        """Truncate content to fit within token limit."""
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

        return truncated + "\n\n[Content truncated for validation]"

    def validate_document(
        self, file_path: str, content: str, auto_index: bool = True
    ) -> dict[str, Any]:
        """Validate a document for redundancy and suggest optimal placement.

        Args:
            file_path: Path to the document being validated
            content: Content of the document
            auto_index: Whether to automatically index the document after validation

        Returns:
            Validation result with redundancy detection and placement suggestions
        """
        start_time = time.time()
        self.logger.info(f"Validating document: {file_path}")

        try:
            validation_content, truncation_data = self._prepare_content_for_validation(
                content, file_path
            )

            analysis_results = self._perform_document_analysis(
                validation_content, file_path
            )

            validation_result = self._build_validation_result(
                file_path, start_time, analysis_results, truncation_data
            )

            self._process_drift_analysis(validation_result, analysis_results)
            self._handle_auto_indexing(auto_index, file_path, validation_result)
            self._finalize_validation_result(file_path, start_time, validation_result)

            return validation_result

        except Exception as e:
            return self._create_error_result(file_path, start_time, e)

    def _prepare_content_for_validation(
        self, content: str, file_path: str
    ) -> tuple[str, dict[str, Any]]:
        """Prepare content for validation, handling truncation if necessary."""
        max_tokens_for_validation = 8000
        estimated_tokens = self.estimate_tokens(content)
        validation_content = content
        was_truncated = False

        if estimated_tokens > max_tokens_for_validation:
            self.logger.warning(
                f"Large document ({estimated_tokens} tokens): {file_path} - truncating for validation"
            )
            validation_content = self.truncate_content(
                content, max_tokens_for_validation
            )
            was_truncated = True

        return validation_content, {
            "was_truncated": was_truncated,
            "estimated_tokens": estimated_tokens,
            "max_tokens_for_validation": max_tokens_for_validation,
        }

    def _perform_document_analysis(
        self, validation_content: str, file_path: str
    ) -> dict[str, Any]:
        """Perform redundancy detection and placement analysis."""
        # Perform redundancy detection
        redundancy_result = self.semantic_engine.detect_redundancy(
            validation_content, threshold=self.config.redundancy_threshold
        )

        # Filter out self-matches
        redundancy_result = self._filter_self_matches(redundancy_result, file_path)

        # Perform placement analysis
        placement_result = self.semantic_engine.suggest_placement(validation_content)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            file_path, redundancy_result, placement_result
        )

        return {
            "redundancy_result": redundancy_result,
            "placement_result": placement_result,
            "recommendations": recommendations,
        }

    def _filter_self_matches(
        self, redundancy_result: dict[str, Any], file_path: str
    ) -> dict[str, Any]:
        """Filter out self-matches from redundancy results."""
        try:
            original_similars = redundancy_result.get("similar_documents", [])
            if isinstance(original_similars, list) and original_similars:
                filtered_similars: list[dict[str, Any]] = []
                for doc in original_similars:
                    doc_path = doc.get("file_path") if isinstance(doc, dict) else None
                    if not self._is_same_file(file_path, doc_path):
                        filtered_similars.append(doc)

                # Only update if anything changed
                if len(filtered_similars) != len(original_similars):
                    redundancy_result["similar_documents"] = filtered_similars
                    redundancy_result["redundancy_detected"] = (
                        len(filtered_similars) > 0
                    )
        except Exception:
            # Be resilient: if any issue occurs during filtering, proceed without failing validation
            pass

        return redundancy_result

    def _build_validation_result(
        self,
        file_path: str,
        start_time: float,
        analysis_results: dict[str, Any],
        truncation_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the base validation result dictionary."""
        recommendations = analysis_results["recommendations"]

        # Add truncation warning if content was truncated
        if truncation_data["was_truncated"]:
            truncation_warning = self._create_truncation_warning(truncation_data)
            recommendations.insert(0, truncation_warning)

        return {
            "file_path": file_path,
            "timestamp": time.time(),
            "validation_time": time.time() - start_time,
            "redundancy_analysis": analysis_results["redundancy_result"],
            "placement_analysis": analysis_results["placement_result"],
            "recommendations": recommendations,
            "status": "success",
            "indexed": False,
            "truncated": truncation_data["was_truncated"],
            "original_tokens": (
                truncation_data["estimated_tokens"]
                if truncation_data["was_truncated"]
                else None
            ),
        }

    def _create_truncation_warning(
        self, truncation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a truncation warning recommendation."""
        return {
            "type": "truncation_warning",
            "priority": "medium",
            "message": (
                f"Document truncated for validation "
                f"(original: {truncation_data['estimated_tokens']} tokens, "
                f"validated: {truncation_data['max_tokens_for_validation']} tokens)"
            ),
            "action": "Validation performed on truncated content - results may not reflect full document",
        }

    def _process_drift_analysis(
        self, validation_result: dict[str, Any], analysis_results: dict[str, Any]
    ) -> None:
        """Process drift analysis if needed and update validation result."""
        drift_analysis = self._check_drift_if_needed()
        if not drift_analysis:
            return

        validation_result["drift_analysis"] = drift_analysis

        # Add drift warnings to recommendations
        drift_recommendations = self._generate_drift_recommendations(drift_analysis)
        recommendations = analysis_results["recommendations"]
        recommendations.extend(drift_recommendations)
        validation_result["recommendations"] = recommendations

    def _handle_auto_indexing(
        self, auto_index: bool, file_path: str, validation_result: dict[str, Any]
    ) -> None:
        """Handle automatic indexing if enabled."""
        if not auto_index:
            return

        try:
            recommendations = validation_result["recommendations"]
            has_high_priority_issues = self._has_high_priority_issues(recommendations)

            if not has_high_priority_issues:
                self._perform_indexing(file_path, validation_result)
            else:
                self.logger.info(
                    f"Skipping indexing for {file_path} due to high-priority issues"
                )

        except Exception as index_error:
            self.logger.error(f"Indexing error for {file_path}: {index_error}")
            validation_result["index_error"] = str(index_error)

    def _has_high_priority_issues(self, recommendations: list[dict[str, Any]]) -> bool:
        """Check if there are any high-priority recommendations."""
        return any(rec.get("priority") == "high" for rec in recommendations)

    def _perform_indexing(
        self, file_path: str, validation_result: dict[str, Any]
    ) -> None:
        """Perform document indexing."""
        indexer = IncrementalIndexer(self.config)
        if indexer.upsert_document(file_path):
            validation_result["indexed"] = True
            self.logger.info(f"Document {file_path} successfully indexed")
        else:
            self.logger.warning(f"Failed to index document {file_path}")

    def _finalize_validation_result(
        self, file_path: str, start_time: float, validation_result: dict[str, Any]
    ) -> None:
        """Finalize validation result with timing and timeout checks."""
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Document validation completed for {file_path} in {elapsed_time:.3f}s"
        )

        # Check if validation exceeded timeout
        if elapsed_time > self.config.validation_timeout:
            self.logger.warning(
                f"Validation exceeded timeout ({elapsed_time:.3f}s > {self.config.validation_timeout}s)"
            )
            validation_result["warning"] = "Validation exceeded configured timeout"

    def _create_error_result(
        self, file_path: str, start_time: float, error: Exception
    ) -> dict[str, Any]:
        """Create an error validation result."""
        self.logger.error(f"Error validating document {file_path}: {error}")
        return {
            "file_path": file_path,
            "timestamp": time.time(),
            "validation_time": time.time() - start_time,
            "status": "error",
            "error": str(error),
            "indexed": False,
        }

    def _is_same_file(self, a: str | None, b: str | None) -> bool:
        """Robustly determine if two paths refer to the same file.

        Tries absolute resolution and falls back to project-root-relative resolution
        to account for metadata storing relative paths while the validator sees
        absolute paths.

        Args:
            a: First file path
            b: Second file path

        Returns:
            True if both paths resolve to the same location, else False.
        """
        if not a or not b:
            return False

        try:
            pa = Path(a)
            pb = Path(b)

            # Resolve A
            ra = pa if pa.is_absolute() else (project_root / pa)
            ra = ra.resolve()

            # Resolve B
            rb = pb if pb.is_absolute() else (project_root / pb)
            rb = rb.resolve()

            return ra == rb
        except Exception:
            # As a conservative fallback, compare normalized strings
            try:
                return os.path.abspath(str(a)) == os.path.abspath(str(b))
            except Exception:
                return False

    def _generate_recommendations(
        self, file_path: str, redundancy_result: dict, placement_result: dict
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on analysis results.

        Args:
            file_path: Path to the document
            redundancy_result: Results from redundancy detection
            placement_result: Results from placement analysis

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Redundancy recommendations
        if redundancy_result.get("redundancy_detected", False):
            similar_docs = redundancy_result.get("similar_documents", [])
            if similar_docs:
                top_similar = similar_docs[0]
                recommendations.append(
                    {
                        "type": "redundancy_warning",
                        "priority": "high",
                        "message": f"Similar content detected in {top_similar['file_path']} "
                        f"(similarity: {top_similar['similarity_score']:.2f})",
                        "action": "Consider consolidating or referencing existing documentation",
                        "similar_files": [doc["file_path"] for doc in similar_docs[:3]],
                    }
                )

        # Placement recommendations
        placement_suggestions = placement_result.get("placement_suggestions", [])
        if placement_suggestions:
            current_dir = str(Path(file_path).parent)
            top_suggestion = placement_suggestions[0]

            if (
                top_suggestion["directory"] != current_dir
                and top_suggestion["confidence"] > 0.6
            ):
                recommendations.append(
                    {
                        "type": "placement_suggestion",
                        "priority": "medium",
                        "message": f"Consider placing this document in {top_suggestion['directory']} "
                        f"(confidence: {top_suggestion['confidence']:.2f})",
                        "action": f"Move to {top_suggestion['directory']} for better organization",
                        "suggested_directory": top_suggestion["directory"],
                        "confidence": top_suggestion["confidence"],
                    }
                )

        # No issues found
        if not recommendations:
            recommendations.append(
                {
                    "type": "validation_success",
                    "priority": "info",
                    "message": "No redundancy or placement issues detected",
                    "action": "Document appears to be well-placed and unique",
                }
            )

        return recommendations

    def _check_drift_if_needed(self) -> dict[str, Any] | None:
        """Check for documentation drift periodically based on configuration.

        Returns:
            Drift analysis results if check is performed, None otherwise
        """
        # Only check if clustering is enabled and it's time to check
        if not self.cluster_engine or not self.drift_detector:
            return None

        self._validation_count += 1

        # Check drift every N validations as configured
        if self._validation_count % self.config.drift_check_frequency != 0:
            return None

        try:
            self.logger.info(
                f"Performing drift check (validation #{self._validation_count})"
            )

            # Get health metrics quickly first
            health_metrics = self.cluster_engine.get_cluster_health_metrics()

            if "error" in health_metrics:
                self.logger.warning(f"Drift check failed: {health_metrics['error']}")
                return None

            # Only do full analysis if there are potential issues
            health_status = health_metrics.get("drift_status", "unknown")
            entropy_score = health_metrics.get("entropy_score", 0)
            coherence_score = health_metrics.get("coherence_score", 1)

            # Skip expensive analysis if everything looks healthy
            if (
                health_status == "healthy"
                and entropy_score < self.config.entropy_threshold
                and coherence_score >= self.config.coherence_threshold
            ):
                return {
                    "quick_check": True,
                    "status": "healthy",
                    "entropy_score": entropy_score,
                    "coherence_score": coherence_score,
                }

            # Perform detailed drift analysis if issues detected
            self.logger.info(
                "Performing detailed drift analysis due to potential issues"
            )
            clusters_data = self.cluster_engine.create_clusters(include_analysis=True)

            if "error" in clusters_data:
                return None

            return clusters_data.get("drift_analysis", {})

        except Exception as e:
            self.logger.error(f"Error during drift check: {e}")
            return None

    def _generate_drift_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on drift analysis results.

        Args:
            drift_analysis: Drift analysis results

        Returns:
            List of recommendation dictionaries
        """
        if not drift_analysis:
            return []

        # Handle quick check results first
        if drift_analysis.get("quick_check"):
            return self._generate_quick_check_recommendations(drift_analysis)

        # Generate comprehensive recommendations
        recommendations = []
        recommendations.extend(self._generate_status_recommendations(drift_analysis))
        recommendations.extend(self._generate_metric_recommendations(drift_analysis))
        recommendations.extend(
            self._generate_organizational_recommendations(drift_analysis)
        )
        recommendations.extend(self._generate_velocity_recommendations(drift_analysis))

        return recommendations

    def _generate_quick_check_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations for quick check results."""
        status = drift_analysis.get("status", "unknown")

        if status != "healthy":
            return []

        entropy_score = drift_analysis.get("entropy_score", 0)
        coherence_score = drift_analysis.get("coherence_score", 1)

        return [
            {
                "type": "drift_status",
                "priority": "info",
                "message": "Documentation organization is healthy",
                "action": f"Entropy: {entropy_score:.2f}, Coherence: {coherence_score:.2f}",
            }
        ]

    def _generate_status_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on drift status."""
        status = drift_analysis.get("status", "unknown")

        if status == "critical":
            return [
                {
                    "type": "drift_critical",
                    "priority": "high",
                    "message": "Critical documentation drift detected",
                    "action": "Consider running 'semantic-cluster suggest-reorg' to fix organization issues",
                }
            ]

        return []

    def _generate_metric_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on entropy and coherence metrics."""
        status = drift_analysis.get("status", "unknown")

        if status != "warning":
            return []

        recommendations = []
        entropy_score = drift_analysis.get("entropy_score", 0)
        coherence_score = drift_analysis.get("coherence_score", 1)

        # Check entropy threshold
        if entropy_score > self.config.entropy_threshold:
            recommendations.append(
                {
                    "type": "drift_entropy",
                    "priority": "medium",
                    "message": f"High entropy detected ({entropy_score:.2f} > {self.config.entropy_threshold:.2f})",
                    "action": "Related documents are scattered across directories",
                }
            )

        # Check coherence threshold
        if coherence_score < self.config.coherence_threshold:
            recommendations.append(
                {
                    "type": "drift_coherence",
                    "priority": "medium",
                    "message": f"Low coherence detected ({coherence_score:.2f} < {self.config.coherence_threshold:.2f})",
                    "action": "Documents in same directories are not semantically related",
                }
            )

        return recommendations

    def _generate_organizational_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations for organizational issues."""
        recommendations = []

        scattered_clusters = drift_analysis.get("scattered_clusters", 0)
        if scattered_clusters > 0:
            recommendations.append(
                {
                    "type": "scattered_clusters",
                    "priority": "medium",
                    "message": f"{scattered_clusters} clusters are scattered across multiple directories",
                    "action": "Run 'semantic-cluster analyze' to see detailed cluster analysis",
                }
            )

        consolidation_opportunities = drift_analysis.get(
            "consolidation_opportunities", 0
        )
        if consolidation_opportunities > 0:
            recommendations.append(
                {
                    "type": "consolidation_opportunities",
                    "priority": "low",
                    "message": f"{consolidation_opportunities} consolidation opportunities found",
                    "action": "Run 'semantic-cluster suggest-reorg' to see reorganization suggestions",
                }
            )

        return recommendations

    def _generate_velocity_recommendations(
        self, drift_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations for drift velocity warnings."""
        drift_velocity = drift_analysis.get("drift_velocity", {})

        if not drift_velocity:
            return []

        overall_velocity = drift_velocity.get("overall_drift_velocity", 0)
        if overall_velocity <= 0.1:  # Not significant drift velocity
            return []

        return [
            {
                "type": "drift_velocity",
                "priority": "medium",
                "message": f"Documentation drift is accelerating (velocity: {overall_velocity:.3f})",
                "action": "Monitor organization more frequently or implement preventive measures",
            }
        ]

    def get_drift_health_status(self) -> dict[str, Any]:
        """Get current drift health status without full analysis.

        Returns:
            Current health status and basic metrics
        """
        if not self.cluster_engine:
            return {
                "status": "disabled",
                "message": "Drift detection not available",
            }

        try:
            health_metrics = self.cluster_engine.get_cluster_health_metrics()

            if "error" in health_metrics:
                return {
                    "status": "error",
                    "message": health_metrics["error"],
                }

            return {
                "status": health_metrics.get("drift_status", "unknown"),
                "health_score": health_metrics.get("health_score", 0),
                "entropy_score": health_metrics.get("entropy_score", 0),
                "coherence_score": health_metrics.get("coherence_score", 0),
                "total_documents": health_metrics.get("total_documents", 0),
                "total_clusters": health_metrics.get("total_clusters", 0),
                "validation_count": self._validation_count,
                "last_check": (
                    self._validation_count // self.config.drift_check_frequency
                )
                * self.config.drift_check_frequency,
                "next_check": (
                    (self._validation_count // self.config.drift_check_frequency) + 1
                )
                * self.config.drift_check_frequency,
            }

        except Exception as e:
            self.logger.error(f"Error getting drift health status: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def post_write_hook(self, file_path: str) -> dict[str, Any]:
        """Claude Code post-write hook for document validation.

        This method is called by Claude Code after writing a markdown file.

        Args:
            file_path: Path to the file that was written

        Returns:
            Validation result formatted for Claude Code consumption
        """
        self.logger.debug(f"Post-write hook triggered for: {file_path}")

        try:
            # Read the file content
            if not Path(file_path).exists():
                return {
                    "file_path": file_path,
                    "status": "error",
                    "error": "File not found",
                }

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Only process markdown files
            if not file_path.lower().endswith((".md", ".markdown")):
                return {
                    "file_path": file_path,
                    "status": "skipped",
                    "message": "Not a markdown file",
                }

            # Validate the document
            return self.validate_document(file_path, content)

        except Exception as e:
            self.logger.error(f"Post-write hook error for {file_path}: {e}")
            return {"file_path": file_path, "status": "error", "error": str(e)}

    def batch_validate(self, file_paths: list[str]) -> dict[str, Any]:
        """Validate multiple files in batch for efficiency.

        Args:
            file_paths: List of file paths to validate

        Returns:
            Batch validation results
        """
        start_time = time.time()
        self.logger.info(f"Starting batch validation of {len(file_paths)} files")

        results = {
            "batch_id": f"batch_{int(time.time())}",
            "timestamp": time.time(),
            "total_files": len(file_paths),
            "results": [],
            "summary": {
                "success": 0,
                "errors": 0,
                "warnings": 0,
                "redundancy_detected": 0,
            },
        }

        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    result = self.post_write_hook(file_path)
                    results["results"].append(result)

                    # Update summary
                    if result["status"] == "success":
                        results["summary"]["success"] += 1
                        if result.get("redundancy_analysis", {}).get(
                            "redundancy_detected", False
                        ):
                            results["summary"]["redundancy_detected"] += 1
                        if result.get("warning"):
                            results["summary"]["warnings"] += 1
                    else:
                        results["summary"]["errors"] += 1
                else:
                    error_result = {
                        "file_path": file_path,
                        "status": "error",
                        "error": "File not found",
                    }
                    results["results"].append(error_result)
                    results["summary"]["errors"] += 1

            except Exception as e:
                error_result = {
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e),
                }
                results["results"].append(error_result)
                results["summary"]["errors"] += 1

        elapsed_time = time.time() - start_time
        results["batch_time"] = elapsed_time

        self.logger.info(
            f"Batch validation completed: {results['summary']['success']} success, "
            f"{results['summary']['errors']} errors in {elapsed_time:.3f}s"
        )

        return results

    def get_system_status(self) -> dict[str, Any]:
        """Get system status and health information.

        Returns:
            System status including configuration and collection stats
        """
        try:
            # Validate environment
            env_validation = validate_environment()

            # Get collection stats
            collection_stats = self.semantic_engine.get_collection_stats()

            # Get drift health status
            drift_status = self.get_drift_health_status()

            return {
                "status": "healthy" if env_validation["valid"] else "degraded",
                "timestamp": time.time(),
                "environment": env_validation,
                "collection": collection_stats,
                "drift_detection": drift_status,
                "configuration": {
                    "redundancy_threshold": self.config.redundancy_threshold,
                    "placement_threshold": self.config.placement_threshold,
                    "validation_timeout": self.config.validation_timeout,
                    "max_results": self.config.max_results,
                    "entropy_threshold": self.config.entropy_threshold,
                    "coherence_threshold": self.config.coherence_threshold,
                    "drift_check_frequency": self.config.drift_check_frequency,
                    "cluster_method": self.config.cluster_method,
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"status": "error", "timestamp": time.time(), "error": str(e)}


def main():
    """CLI entry point for Claude Code hooks."""

    parser = _create_cli_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        config = _load_config(args.config)
        hooks = ClaudeCodeHooks(config)
        result = _execute_cli_command(args, hooks)
        _output_result(result, args.json)

    except Exception as e:
        _handle_error(e, args.json)
        sys.exit(1)


def _create_cli_parser() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser."""
    parser = argparse.ArgumentParser(description="Claude Code semantic analysis hooks")
    parser.add_argument("command", choices=["validate", "batch", "status", "test"])
    parser.add_argument("--file", "-f", help="File path for validation")
    parser.add_argument(
        "--files", "-F", nargs="+", help="Multiple file paths for batch validation"
    )
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser


def _setup_logging(verbose: bool) -> None:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _load_config(config_path: str | None) -> Any:
    """Load configuration from file or defaults."""
    return load_config(config_path) if config_path else load_config()


def _execute_cli_command(
    args: argparse.Namespace, hooks: ClaudeCodeHooks
) -> dict[str, Any]:
    """Execute the specified CLI command."""
    command_handlers = {
        "validate": lambda: _handle_validate_command(args, hooks),
        "batch": lambda: _handle_batch_command(args, hooks),
        "status": lambda: hooks.get_system_status(),
        "test": lambda: _handle_test_command(hooks),
    }

    return command_handlers[args.command]()


def _handle_validate_command(
    args: argparse.Namespace, hooks: ClaudeCodeHooks
) -> dict[str, Any]:
    """Handle validate command."""
    if not args.file:
        print("Error: --file required for validate command", file=sys.stderr)
        sys.exit(1)

    return hooks.post_write_hook(args.file)


def _handle_batch_command(
    args: argparse.Namespace, hooks: ClaudeCodeHooks
) -> dict[str, Any]:
    """Handle batch command."""
    if not args.files:
        print("Error: --files required for batch command", file=sys.stderr)
        sys.exit(1)

    return hooks.batch_validate(args.files)


def _handle_test_command(hooks: ClaudeCodeHooks) -> dict[str, Any]:
    """Handle test command."""
    test_content = """
    # Test Document

    This is a test document for validating the semantic analysis system.
    It contains sample content to verify redundancy detection and placement suggestions.
    """
    return hooks.validate_document("test.md", test_content)


def _output_result(result: dict[str, Any], as_json: bool) -> None:
    """Output result in specified format."""
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        _output_human_readable(result)


def _output_human_readable(result: dict[str, Any]) -> None:
    """Output result in human-readable format."""
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("status") != "success":
        return

    print(f"Validation time: {result.get('validation_time', 0):.3f}s")

    recommendations = result.get("recommendations")
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec['priority'].upper()}: {rec['message']}")


def _handle_error(error: Exception, as_json: bool) -> None:
    """Handle and output error."""
    error_result = {"status": "error", "error": str(error)}
    if as_json:
        print(json.dumps(error_result, indent=2))
    else:
        print(f"Error: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()

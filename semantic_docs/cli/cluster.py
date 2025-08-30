"""Semantic document clustering CLI for drift detection and organization analysis.

Provides commands to analyze document clustering, detect drift, and suggest
reorganization to prevent documentation entropy over time.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from semantic_docs.config.settings import load_config, validate_environment
from semantic_docs.engines.cluster_engine import DocumentClusterEngine
from semantic_docs.engines.drift_detector import DriftDetector


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging for clustering CLI.

    Args:
        verbose: Enable debug logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("semantic_cluster")
    logger.setLevel(log_level)

    # Only add handler if none exists
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_health_status(health_score: float, status: str) -> str:
    """Format health status with color coding.

    Args:
        health_score: Health score (0-100)
        status: Status string

    Returns:
        Formatted status string
    """
    if status == "healthy":
        return f"🟢 {status.upper()} (score: {health_score}/100)"
    elif status == "warning":
        return f"🟡 {status.upper()} (score: {health_score}/100)"
    else:  # critical or error
        return f"🔴 {status.upper()} (score: {health_score}/100)"


def analyze_clusters(
    output_format: str = "human",
    n_clusters: int | None = None,
    method: str = "kmeans",
    save_results: str | None = None,
    verbose: bool = False,
) -> int:
    """Analyze current document clustering and detect drift.

    Args:
        output_format: Output format ('human', 'json', 'report')
        n_clusters: Number of clusters (auto-detect if None)
        method: Clustering method ('kmeans' or 'hierarchical')
        save_results: Optional file path to save results
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger = setup_logging(verbose)

    try:
        # Validate environment
        env_valid = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return 1

        # Load configuration
        config = load_config()

        # Initialize cluster engine
        logger.info("Initializing cluster engine...")
        cluster_engine = DocumentClusterEngine(config)

        # Create clusters with analysis
        logger.info(f"Analyzing clusters using {method} method...")
        clusters_data = cluster_engine.create_clusters(
            n_clusters=n_clusters,
            method=method,
            include_analysis=True,
        )

        if "error" in clusters_data:
            logger.error(f"Clustering failed: {clusters_data['error']}")
            return 1

        # Save results if requested
        if save_results:
            output_path = Path(save_results)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clusters_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")

        # Format output
        if output_format == "json":
            print(json.dumps(clusters_data, indent=2))
        elif output_format == "report":
            _generate_detailed_report(clusters_data)
        else:  # human
            _print_human_readable_analysis(clusters_data)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def health_check(
    output_format: str = "human",
    verbose: bool = False,
) -> int:
    """Perform health check on documentation organization.

    Args:
        output_format: Output format ('human', 'json')
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = healthy/warning, 1 = critical/error)
    """
    logger = setup_logging(verbose)

    try:
        # Validate environment
        env_valid = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return 1

        # Load configuration
        config = load_config()

        # Initialize cluster engine
        cluster_engine = DocumentClusterEngine(config)

        # Get health metrics
        logger.info("Checking documentation health...")
        health_metrics = cluster_engine.get_cluster_health_metrics()

        if "error" in health_metrics:
            logger.error(f"Health check failed: {health_metrics['error']}")
            return 1

        # Format output
        if output_format == "json":
            print(json.dumps(health_metrics, indent=2))
        else:
            _print_health_summary(health_metrics)

        # Return appropriate exit code
        status = health_metrics.get("drift_status", "unknown")
        return 1 if status == "critical" else 0

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def suggest_reorganization(
    output_format: str = "human",
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Suggest reorganization to improve documentation structure.

    Args:
        output_format: Output format ('human', 'json')
        dry_run: Show suggestions without executing
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger = setup_logging(verbose)

    try:
        # Validate environment
        env_valid = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return 1

        # Load configuration
        config = load_config()

        # Initialize engines
        cluster_engine = DocumentClusterEngine(config)
        _ = DriftDetector(config)  # Initialize for potential future use

        # Analyze current state
        logger.info("Analyzing current organization...")
        clusters_data = cluster_engine.create_clusters(include_analysis=True)

        if "error" in clusters_data:
            logger.error(f"Analysis failed: {clusters_data['error']}")
            return 1

        drift_analysis = clusters_data.get("drift_analysis", {})

        # Get reorganization suggestions
        suggestions = drift_analysis.get("consolidation_suggestions", [])

        if not suggestions:
            print("✅ No reorganization needed - documentation is well organized!")
            return 0

        # Format output
        if output_format == "json":
            print(json.dumps({"suggestions": suggestions}, indent=2))
        else:
            _print_reorganization_suggestions(suggestions, dry_run)

        return 0

    except Exception as e:
        logger.error(f"Reorganization analysis failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def visualize_clusters(
    output_file: str | None = None,
    verbose: bool = False,
) -> int:
    """Create visualization of document clusters.

    Args:
        output_file: Output file path for visualization
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger = setup_logging(verbose)

    try:
        # Validate environment
        env_valid = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return 1

        # Load configuration
        config = load_config()

        # Initialize cluster engine
        cluster_engine = DocumentClusterEngine(config)

        # Create clusters
        logger.info("Generating clusters for visualization...")
        clusters_data = cluster_engine.create_clusters(include_analysis=False)

        if "error" in clusters_data:
            logger.error(f"Clustering failed: {clusters_data['error']}")
            return 1

        # Create visualization
        output_path = output_file or "document_clusters_visualization.png"
        result_path = cluster_engine.visualize_clusters(clusters_data, output_path)

        if result_path:
            print(f"✅ Visualization saved to: {result_path}")
            return 0
        else:
            logger.warning(
                "Visualization creation failed (matplotlib may not be available)"
            )
            return 1

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def _print_human_readable_analysis(clusters_data: dict[str, Any]) -> None:
    """Print human-readable cluster analysis."""
    metadata = clusters_data.get("metadata", {})
    clusters = clusters_data.get("clusters", {})
    drift_analysis = clusters_data.get("drift_analysis", {})

    print("\n" + "=" * 60)
    print("📊 SEMANTIC DOCUMENT CLUSTERING ANALYSIS")
    print("=" * 60)

    # Basic stats
    print("\n📈 Overview:")
    print(f"   Total Documents: {metadata.get('total_documents', 0)}")
    print(f"   Total Clusters: {metadata.get('total_clusters', 0)}")
    print(f"   Method: {metadata.get('clustering_method', 'unknown')}")
    print(f"   Generated: {metadata.get('generated_at', 'unknown')}")

    # Drift analysis
    if drift_analysis:
        print("\n🎯 Organization Health:")
        health_score = drift_analysis.get("health_score", 0)
        status = drift_analysis.get("status", "unknown")
        print(f"   Status: {format_health_status(health_score, status)}")
        print(
            f"   Entropy Score: {drift_analysis.get('entropy_score', 0):.3f} (lower is better)"
        )
        print(
            f"   Coherence Score: {drift_analysis.get('coherence_score', 0):.3f} (higher is better)"
        )

        scattered_count = drift_analysis.get("scattered_clusters", 0)
        if scattered_count > 0:
            print(f"   ⚠️  Scattered Clusters: {scattered_count}")

        suggestions_count = drift_analysis.get("consolidation_opportunities", 0)
        if suggestions_count > 0:
            print(f"   💡 Consolidation Opportunities: {suggestions_count}")

    # Cluster details
    print("\n📁 Cluster Details:")
    for cluster_id, cluster_info in clusters.items():
        description = cluster_info.get("description", f"Cluster {cluster_id}")
        doc_count = cluster_info.get("document_count", 0)
        print(f"   {cluster_id}: {description} ({doc_count} docs)")


def _print_health_summary(health_metrics: dict[str, Any]) -> None:
    """Print health summary in human-readable format."""
    print("\n" + "=" * 50)
    print("🏥 DOCUMENTATION HEALTH CHECK")
    print("=" * 50)

    total_docs = health_metrics.get("total_documents", 0)
    total_clusters = health_metrics.get("total_clusters", 0)
    avg_cluster_size = health_metrics.get("average_cluster_size", 0)

    print("\n📊 Basic Metrics:")
    print(f"   Total Documents: {total_docs}")
    print(f"   Total Clusters: {total_clusters}")
    print(f"   Average Cluster Size: {avg_cluster_size}")

    # Health status
    health_score = health_metrics.get("health_score", 0)
    status = health_metrics.get("drift_status", "unknown")

    print("\n🎯 Health Status:")
    print(f"   Overall: {format_health_status(health_score, status)}")

    # Detailed metrics if available
    if "entropy_score" in health_metrics:
        print(f"   Entropy: {health_metrics['entropy_score']:.3f}")
    if "coherence_score" in health_metrics:
        print(f"   Coherence: {health_metrics['coherence_score']:.3f}")


def _print_reorganization_suggestions(suggestions: list, dry_run: bool) -> None:
    """Print reorganization suggestions in human-readable format."""
    print("\n" + "=" * 60)
    print("💡 DOCUMENTATION REORGANIZATION SUGGESTIONS")
    print("=" * 60)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")

    for i, suggestion in enumerate(suggestions, 1):
        cluster_id = suggestion.get("cluster_id", "unknown")
        target_dir = suggestion.get("target_directory", "unknown")
        benefit = suggestion.get("consolidation_benefit", "")
        moves = suggestion.get("moves", [])

        print(f"\n{i}. Cluster {cluster_id} Consolidation")
        print(f"   Target Directory: {target_dir}")
        print(f"   Benefit: {benefit}")
        print(f"   Files to move: {len(moves)}")

        for move in moves[:3]:  # Show first 3 moves
            current = move.get("current_path", "")
            suggested = move.get("suggested_path", "")
            print(f"     • {current} → {suggested}")

        if len(moves) > 3:
            print(f"     ... and {len(moves) - 3} more files")


def _generate_detailed_report(clusters_data: dict[str, Any]) -> None:
    """Generate detailed report format output."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
# Semantic Document Clustering Report
Generated: {timestamp}

## Summary
""")

    metadata = clusters_data.get("metadata", {})
    drift_analysis = clusters_data.get("drift_analysis", {})

    print(f"- **Total Documents**: {metadata.get('total_documents', 0)}")
    print(f"- **Total Clusters**: {metadata.get('total_clusters', 0)}")
    print(f"- **Clustering Method**: {metadata.get('clustering_method', 'unknown')}")

    if drift_analysis:
        health_score = drift_analysis.get("health_score", 0)
        status = drift_analysis.get("status", "unknown")
        print(f"- **Health Status**: {status} ({health_score}/100)")
        print(f"- **Entropy Score**: {drift_analysis.get('entropy_score', 0):.3f}")
        print(f"- **Coherence Score**: {drift_analysis.get('coherence_score', 0):.3f}")

    # Detailed cluster information
    clusters = clusters_data.get("clusters", {})
    print("\n## Cluster Details\n")

    for cluster_id, cluster_info in clusters.items():
        description = cluster_info.get("description", f"Cluster {cluster_id}")
        doc_count = cluster_info.get("document_count", 0)
        documents = cluster_info.get("documents", [])

        print(f"### {description}")
        print(f"**Documents**: {doc_count}\n")

        for doc in documents[:10]:  # Show first 10 documents
            filepath = doc.get("filepath", "unknown")
            heading = doc.get("heading", "")
            print(f"- `{filepath}`{': ' + heading if heading else ''}")

        if len(documents) > 10:
            print(f"- ... and {len(documents) - 10} more documents")
        print()


def analyze_labels(
    output_format: str = "human",
    include_suggestions: bool = False,
    save_results: str | None = None,
    verbose: bool = False,
) -> int:
    """Analyze label distribution and coherence across clusters.

    Args:
        output_format: Output format ('human' or 'json')
        include_suggestions: Include reorganization suggestions
        save_results: Optional file path to save results
        verbose: Enable verbose logging

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging(verbose)

    try:
        # Initialize components
        engines = _initialize_label_analysis_engines(logger)
        if engines is None:
            return 1
        cluster_engine, drift_detector = engines

        # Get clusters
        clusters = _get_analysis_clusters(cluster_engine, logger)
        if clusters is None:
            return 1

        # Perform analysis
        analysis_results = _perform_label_analysis(
            drift_detector, clusters, include_suggestions, logger
        )
        if analysis_results is None:
            return 1
        label_analysis, suggestions = analysis_results

        # Generate results
        results = _build_analysis_results(
            label_analysis, suggestions, include_suggestions
        )

        # Handle output
        _handle_analysis_output(
            results,
            output_format,
            save_results,
            label_analysis,
            suggestions,
            include_suggestions,
            logger,
        )

        logger.info("Label analysis completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Label analysis failed: {e}")
        return 1


def _initialize_label_analysis_engines(logger):
    """Initialize engines for label analysis."""
    env_check = validate_environment()
    if not env_check["valid"]:
        logger.error("Environment validation failed:")
        for error in env_check["errors"]:
            logger.error(f"  - {error}")
        return None

    config = load_config()
    cluster_engine = DocumentClusterEngine(config)
    drift_detector = DriftDetector(config)
    return cluster_engine, drift_detector


def _get_analysis_clusters(cluster_engine, logger):
    """Get and transform clusters for analysis."""
    logger.info("Performing clustering for label analysis...")
    clustering_results = cluster_engine.create_clusters(include_analysis=False)

    if "error" in clustering_results:
        logger.error(f"Clustering failed: {clustering_results['error']}")
        return None

    raw_clusters = clustering_results["clusters"]
    if not raw_clusters:
        logger.error("No clusters found - ensure documents are indexed")
        return None

    return _transform_clusters_format(raw_clusters)


def _transform_clusters_format(raw_clusters):
    """Transform cluster format for drift detector."""
    clusters = {}
    for cluster_id_str, cluster_info in raw_clusters.items():
        cluster_id = int(cluster_id_str)
        documents = cluster_info.get("documents", [])

        transformed_docs = [{"metadata": doc} for doc in documents]
        clusters[cluster_id] = transformed_docs

    return clusters


def _perform_label_analysis(drift_detector, clusters, include_suggestions, logger):
    """Perform label distribution analysis."""
    logger.info("Analyzing label distribution and coherence...")
    label_analysis = drift_detector.analyze_label_distribution(clusters)

    if "error" in label_analysis:
        logger.error(f"Label analysis failed: {label_analysis['error']}")
        return None

    suggestions = []
    if include_suggestions:
        logger.info("Generating label-based reorganization suggestions...")
        suggestions = drift_detector.suggest_label_based_reorganization(
            label_analysis, clusters
        )

    return label_analysis, suggestions


def _build_analysis_results(label_analysis, suggestions, include_suggestions):
    """Build results dictionary."""
    return {
        "timestamp": datetime.now().isoformat(),
        "label_analysis": label_analysis,
        "reorganization_suggestions": suggestions if include_suggestions else [],
    }


def _handle_analysis_output(
    results,
    output_format,
    save_results,
    label_analysis,
    suggestions,
    include_suggestions,
    logger,
):
    """Handle saving and displaying results."""
    if save_results:
        _save_analysis_results(results, save_results, logger)

    if output_format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        _display_label_analysis_human(label_analysis, suggestions, include_suggestions)


def _save_analysis_results(results, save_results, logger):
    """Save analysis results to file."""
    try:
        with open(save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {save_results}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def _display_label_analysis_human(
    analysis: dict[str, Any], suggestions: list[dict[str, Any]], show_suggestions: bool
) -> None:
    """Display label analysis results in human-readable format."""
    print("\n🏷️  LABEL DISTRIBUTION ANALYSIS")
    print("=" * 50)

    _display_summary_metrics(analysis)
    _display_label_frequency(analysis)
    _display_scattered_labels(analysis)
    _display_diverse_clusters(analysis)
    _display_suggestions(suggestions, show_suggestions)


def _display_summary_metrics(analysis: dict[str, Any]) -> None:
    """Display summary metrics section."""
    print("\n📊 Summary:")
    print(f"   Total unique labels: {analysis['total_unique_labels']}")
    print(f"   Documents with labels: {analysis['total_docs_with_labels']}")
    print(f"   Label coverage: {analysis['label_coverage']:.1f}%")
    print(f"   Average label coherence: {analysis['average_label_coherence']:.3f}")


def _display_label_frequency(analysis: dict[str, Any]) -> None:
    """Display label frequency distribution."""
    print("\n🔢 Most Common Labels:")
    freq_dist = analysis.get("label_frequency_distribution", {})
    for i, (label, count) in enumerate(list(freq_dist.items())[:10]):
        print(f"   {i + 1:2d}. {label}: {count} occurrences")


def _display_scattered_labels(analysis: dict[str, Any]) -> None:
    """Display scattered labels information."""
    scattered = analysis.get("scattered_labels", [])
    if not scattered:
        return

    print("\n⚠️  Scattered Labels (appear in many clusters):")
    for label_info in scattered[:5]:
        label = label_info["label"]
        cluster_count = label_info["cluster_count"]
        coherence = label_info["coherence_score"]
        print(f"   • {label}: {cluster_count} clusters (coherence: {coherence:.3f})")


def _display_diverse_clusters(analysis: dict[str, Any]) -> None:
    """Display most diverse clusters information."""
    diverse = analysis.get("most_diverse_clusters", [])
    if not diverse:
        return

    print("\n🌀 Most Label-Diverse Clusters:")
    for cluster_info in diverse[:3]:
        cluster_id = cluster_info["cluster_id"]
        entropy = cluster_info["label_entropy"]
        unique_labels = cluster_info["unique_labels"]
        labels = list(cluster_info["labels"].keys())[:5]  # Top 5 labels
        print(
            f"   • Cluster {cluster_id}: entropy={entropy:.2f}, {unique_labels} unique labels"
        )
        print(f"     Labels: {', '.join(labels)}")


def _display_suggestions(
    suggestions: list[dict[str, Any]], show_suggestions: bool
) -> None:
    """Display reorganization suggestions section."""
    if not show_suggestions:
        print("\n💡 Use --suggestions to see reorganization recommendations")
        return

    if not suggestions:
        return

    print("\n💡 REORGANIZATION SUGGESTIONS")
    print("=" * 50)

    for i, suggestion in enumerate(suggestions[:5], 1):
        _display_single_suggestion(suggestion, i)


def _display_single_suggestion(suggestion: dict[str, Any], index: int) -> None:
    """Display a single reorganization suggestion."""
    stype = suggestion["type"]
    priority = suggestion["priority"]
    reason = suggestion["reason"]

    priority_icon = _get_priority_icon(priority)

    print(f"\n{priority_icon} Suggestion {index} ({priority} priority):")
    print(f"   Type: {stype.replace('_', ' ').title()}")
    print(f"   Reason: {reason}")

    _display_suggestion_details(stype, suggestion)


def _get_priority_icon(priority: str) -> str:
    """Get icon for priority level."""
    return "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"


def _display_suggestion_details(stype: str, suggestion: dict[str, Any]) -> None:
    """Display type-specific suggestion details."""
    if stype == "split_diverse_cluster":
        _display_split_cluster_details(suggestion)
    elif stype == "consolidate_scattered_label":
        _display_consolidate_label_details(suggestion)
    elif stype == "create_label_combination_category":
        _display_combination_category_details(suggestion)


def _display_split_cluster_details(suggestion: dict[str, Any]) -> None:
    """Display details for split cluster suggestion."""
    cluster_id = suggestion["cluster_id"]
    primary = suggestion["primary_label"]
    splits = suggestion["suggested_splits"]

    print(f"   Action: Split cluster {cluster_id}")
    print(f"   Primary label: {primary}")
    if splits:
        print(f"   Suggested splits: {', '.join(splits)}")


def _display_consolidate_label_details(suggestion: dict[str, Any]) -> None:
    """Display details for consolidate label suggestion."""
    label = suggestion["label"]
    primary_cluster = suggestion["primary_cluster"]
    from_clusters = suggestion["consolidate_from_clusters"]

    print(f"   Action: Consolidate '{label}' documents")
    print(f"   Target cluster: {primary_cluster}")
    if from_clusters:
        print(f"   From clusters: {', '.join(map(str, from_clusters))}")


def _display_combination_category_details(suggestion: dict[str, Any]) -> None:
    """Display details for label combination category suggestion."""
    combo = suggestion["label_combination"]
    suggested_dir = suggestion["suggested_directory"]

    print(f"   Action: Create category for {' + '.join(combo)}")
    print(f"   Suggested directory: {suggested_dir}")


def compare_structure(
    output_format: str = "human",
    visualize: bool = False,
    save_results: str | None = None,
    verbose: bool = False,
) -> int:
    """Compare folder structure with semantic clusters.

    Args:
        output_format: Output format ('human', 'json')
        visualize: Create visualization of comparison
        save_results: Optional file path to save results
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger = setup_logging(verbose)

    try:
        # Validate environment
        env_valid = validate_environment()
        if not env_valid:
            logger.error("Environment validation failed")
            return 1

        # Load configuration
        config = load_config()

        # Initialize cluster engine
        logger.info("Initializing cluster engine...")
        cluster_engine = DocumentClusterEngine(config)

        # Perform folder-cluster alignment analysis
        logger.info("Analyzing folder-cluster alignment...")
        alignment_data = cluster_engine.analyze_folder_cluster_alignment()

        if "error" in alignment_data:
            logger.error(f"Alignment analysis failed: {alignment_data['error']}")
            return 1

        # Create visualization if requested
        visualization_path = None
        if visualize:
            logger.info("Creating folder-cluster comparison visualization...")
            visualization_path = cluster_engine.visualize_folder_cluster_comparison(
                alignment_data, "folder_cluster_comparison.png"
            )
            if visualization_path:
                print(f"📊 Visualization saved to: {visualization_path}")
            else:
                logger.warning("Visualization creation failed")

        # Save results if requested
        if save_results:
            output_path = Path(save_results)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(alignment_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")

        # Format output
        if output_format == "json":
            print(json.dumps(alignment_data, indent=2))
        else:  # human
            _print_alignment_analysis(alignment_data)

        return 0

    except Exception as e:
        logger.error(f"Structure comparison failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def _get_cluster_name(cluster_id: int, cluster_metrics: list[dict[str, Any]]) -> str:
    """Get a meaningful cluster name with dominant folder."""
    for cluster in cluster_metrics:
        if cluster["cluster_id"] == cluster_id:
            dominant_folder = cluster.get("dominant_folder", "unknown")
            return f"Cluster {cluster_id} ({dominant_folder})"
    return f"Cluster {cluster_id}"


def _print_alignment_analysis(alignment_data: dict[str, Any]) -> None:
    """Print folder-cluster alignment analysis in human-readable format."""
    metadata = alignment_data.get("metadata", {})
    overall = alignment_data.get("overall_alignment", {})
    folder_metrics = alignment_data.get("folder_metrics", [])
    cluster_metrics = alignment_data.get("cluster_metrics", [])
    misaligned_docs = alignment_data.get("misaligned_documents", [])

    print("\n" + "=" * 65)
    print("🏗️  FOLDER-CLUSTER STRUCTURE COMPARISON")
    print("=" * 65)

    # Overview
    print("\n📊 Overview:")
    print(f"   Total Folders: {metadata.get('total_folders', 0)}")
    print(f"   Total Clusters: {metadata.get('total_clusters', 0)}")
    print(f"   Total Documents: {metadata.get('total_documents', 0)}")
    print(f"   Generated: {metadata.get('generated_at', 'unknown')}")

    # Overall alignment
    print("\n🎯 Overall Alignment:")
    alignment_score = overall.get("alignment_score", 0)
    quality = overall.get("alignment_quality", "unknown")
    purity = overall.get("weighted_average_purity", 0)
    homogeneity = overall.get("weighted_average_homogeneity", 0)

    # Format quality with emoji
    quality_icons = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}
    quality_icon = quality_icons.get(quality.lower(), "⚪")

    print(
        f"   {quality_icon} Quality: {quality.upper()} (score: {alignment_score:.3f})"
    )
    print(f"   📁 Folder Purity: {purity:.3f} (how focused folders are)")
    print(f"   🗂️  Cluster Homogeneity: {homogeneity:.3f} (how unified clusters are)")

    # Top problematic folders (lowest purity)
    print("\n📁 Folders Needing Attention (lowest purity first):")
    sorted_folders = sorted(folder_metrics, key=lambda x: x["purity"])
    for i, folder in enumerate(sorted_folders[:5], 1):
        purity_score = folder["purity"]
        cluster_count = folder["num_clusters_spanned"]
        doc_count = folder["total_documents"]

        purity_icon = (
            "🔴" if purity_score < 0.5 else "🟡" if purity_score < 0.8 else "🟢"
        )
        print(f"   {i}. {purity_icon} {folder['folder']} - purity: {purity_score:.2f}")
        print(f"      ({doc_count} docs across {cluster_count} clusters)")

    # Top problematic clusters (lowest homogeneity)
    print("\n🗂️  Clusters Needing Attention (lowest homogeneity first):")
    sorted_clusters = sorted(cluster_metrics, key=lambda x: x["homogeneity"])
    for i, cluster in enumerate(sorted_clusters[:5], 1):
        homogeneity_score = cluster["homogeneity"]
        folder_count = cluster["num_folders_represented"]
        doc_count = cluster["total_documents"]

        homogeneity_icon = (
            "🔴"
            if homogeneity_score < 0.5
            else "🟡"
            if homogeneity_score < 0.8
            else "🟢"
        )
        dominant_folder = cluster.get("dominant_folder", "unknown")
        print(
            f"   {i}. {homogeneity_icon} Cluster {cluster['cluster_id']} ({dominant_folder}) - homogeneity: {homogeneity_score:.2f}"
        )
        print(f"      ({doc_count} docs from {folder_count} folders)")

    # Misaligned documents
    if misaligned_docs:
        print(f"\n⚠️  Misaligned Documents ({len(misaligned_docs)} found):")
        for i, doc in enumerate(misaligned_docs[:10], 1):  # Show first 10
            filepath = doc["filepath"]
            current_folder = doc["current_folder"]
            current_cluster = doc["current_cluster"]
            dominant_cluster = doc["folder_dominant_cluster"]
            alignment_ratio = doc["alignment_ratio"]

            current_cluster_name = _get_cluster_name(current_cluster, cluster_metrics)
            dominant_cluster_name = _get_cluster_name(dominant_cluster, cluster_metrics)

            print(f"   {i}. {filepath}")
            print(f"      📁 In folder: {current_folder} → 🗂️  {current_cluster_name}")
            print(
                f"      💡 Folder majority is in {dominant_cluster_name} (alignment: {alignment_ratio:.1%})"
            )

        if len(misaligned_docs) > 10:
            print(
                f"      ... and {len(misaligned_docs) - 10} more misaligned documents"
            )
    else:
        print("\n✅ No significantly misaligned documents found!")

    # Summary recommendations
    print("\n💡 Recommendations:")
    if alignment_score >= 0.8:
        print(
            "   🎉 Excellent alignment! Your folder structure matches semantic clusters well."
        )
    elif alignment_score >= 0.6:
        print("   👍 Good alignment with room for minor improvements.")
        print(
            "   📝 Consider consolidating scattered folders or splitting diverse ones."
        )
    elif alignment_score >= 0.4:
        print("   ⚠️  Fair alignment - several opportunities for improvement.")
        print("   📋 Review folders with low purity and clusters with low homogeneity.")
        print("   🔄 Consider reorganizing misaligned documents.")
    else:
        print("   🚨 Poor alignment - significant reorganization recommended.")
        print("   📚 Folder structure doesn't match semantic content organization.")
        print("   🔧 Consider major restructuring based on cluster analysis.")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic document clustering for drift detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze                 # Analyze current clustering
  %(prog)s health                  # Quick health check
  %(prog)s suggest-reorg           # Show reorganization suggestions
  %(prog)s visualize -o plot.png   # Create visualization
  %(prog)s analyze-labels          # Analyze label distribution
  %(prog)s analyze-labels --suggestions  # Include label-based reorg suggestions
  %(prog)s compare-structure       # Compare folder vs cluster alignment
  %(prog)s compare-structure --visualize  # Include alignment visualization
  %(prog)s analyze --save results.json  # Save results to file
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze document clustering and detect drift"
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        choices=["human", "json", "report"],
        default="human",
        help="Output format (default: human)",
    )
    analyze_parser.add_argument(
        "--clusters",
        "-c",
        type=int,
        help="Number of clusters (auto-detect if not specified)",
    )
    analyze_parser.add_argument(
        "--method",
        "-m",
        choices=["kmeans", "hierarchical"],
        default="kmeans",
        help="Clustering method (default: kmeans)",
    )
    analyze_parser.add_argument("--save", "-s", help="Save results to JSON file")
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Health command
    health_parser = subparsers.add_parser(
        "health", help="Check documentation organization health"
    )
    health_parser.add_argument(
        "--output",
        "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    health_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Suggest reorganization command
    suggest_parser = subparsers.add_parser(
        "suggest-reorg", help="Suggest documentation reorganization"
    )
    suggest_parser.add_argument(
        "--output",
        "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    suggest_parser.add_argument(
        "--dry-run", action="store_true", help="Show suggestions without executing"
    )
    suggest_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create cluster visualization")
    viz_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: document_clusters_visualization.png)",
    )
    viz_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Label analysis command
    label_parser = subparsers.add_parser(
        "analyze-labels", help="Analyze label distribution and coherence"
    )
    label_parser.add_argument(
        "--output",
        "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    label_parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Include reorganization suggestions based on labels",
    )
    label_parser.add_argument("--save", "-s", help="Save results to JSON file")
    label_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Compare structure command
    compare_parser = subparsers.add_parser(
        "compare-structure", help="Compare folder structure with semantic clusters"
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    compare_parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Create visualization of folder-cluster comparison",
    )
    compare_parser.add_argument("--save", "-s", help="Save results to JSON file")
    compare_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "analyze":
            return analyze_clusters(
                output_format=args.output,
                n_clusters=args.clusters,
                method=args.method,
                save_results=args.save,
                verbose=args.verbose,
            )
        elif args.command == "health":
            return health_check(
                output_format=args.output,
                verbose=args.verbose,
            )
        elif args.command == "suggest-reorg":
            return suggest_reorganization(
                output_format=args.output,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        elif args.command == "visualize":
            return visualize_clusters(
                output_file=args.output,
                verbose=args.verbose,
            )
        elif args.command == "analyze-labels":
            return analyze_labels(
                output_format=args.output,
                include_suggestions=args.suggestions,
                save_results=args.save,
                verbose=args.verbose,
            )
        elif args.command == "compare-structure":
            return compare_structure(
                output_format=args.output,
                visualize=args.visualize,
                save_results=args.save,
                verbose=args.verbose,
            )
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())

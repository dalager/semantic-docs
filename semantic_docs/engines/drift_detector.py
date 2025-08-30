"""Documentation drift detection and entropy analysis engine.

Analyzes document organization patterns and detects when documentation
drifts from optimal semantic clustering, providing metrics for preventing
documentation entropy over time.
"""

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from semantic_docs.config.settings import SemanticConfig


class LabelData(NamedTuple):
    """Container for collected label data."""

    all_labels: Counter
    cluster_labels: dict[int, dict[str, int]]
    label_clusters: dict[str, set[int]]
    total_docs_with_labels: int


class CoherenceData(NamedTuple):
    """Container for label coherence analysis results."""

    label_coherence_scores: dict[str, dict[str, Any]]
    scattered_labels: list[dict[str, Any]]
    avg_coherence: float


class DriftDetector:
    """Detects documentation drift and calculates organization metrics."""

    def __init__(self, config: SemanticConfig):
        """Initialize drift detector.

        Args:
            config: SemanticConfig instance
        """
        self.config = config

        # Setup logging
        self.logger = logging.getLogger("drift_detector")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calculate_entropy_score(
        self,
        clusters: dict[int, list[dict]],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Calculate Shannon entropy of document distribution across directories.

        Higher entropy indicates more scattered organization - documents that
        should be together semantically are spread across many directories.

        Args:
            clusters: Dictionary mapping cluster_id to document list
            embeddings: Document embeddings array
            labels: Cluster labels array

        Returns:
            Entropy score (0.0 = perfectly organized, 1.0 = maximum scatter)
        """
        try:
            # Group documents by directory for each cluster
            cluster_directory_distributions = {}

            for cluster_id, docs in clusters.items():
                directories = []
                for doc in docs:
                    filepath = doc.get("filepath", "")
                    if filepath:
                        directory = str(Path(filepath).parent)
                        directories.append(directory)

                if directories:
                    # Count directory occurrences for this cluster
                    dir_counts = Counter(directories)
                    total_docs_in_cluster = len(directories)

                    # Calculate entropy for this cluster's directory distribution
                    cluster_entropy = 0.0
                    for count in dir_counts.values():
                        probability = count / total_docs_in_cluster
                        if probability > 0:
                            cluster_entropy -= probability * math.log2(probability)

                    # Normalize by maximum possible entropy for this cluster size
                    max_entropy = (
                        math.log2(len(dir_counts)) if len(dir_counts) > 1 else 0
                    )
                    normalized_entropy = (
                        cluster_entropy / max_entropy if max_entropy > 0 else 0
                    )

                    cluster_directory_distributions[cluster_id] = {
                        "entropy": cluster_entropy,
                        "normalized_entropy": normalized_entropy,
                        "directory_count": len(dir_counts),
                        "document_count": total_docs_in_cluster,
                        "directories": dict(dir_counts),
                    }

            # Calculate overall entropy score as weighted average
            total_entropy = 0.0
            total_weight = 0.0

            for cluster_id, dist_info in cluster_directory_distributions.items():
                weight = dist_info["document_count"]
                cluster_entropy = dist_info["normalized_entropy"]
                total_entropy += weight * cluster_entropy
                total_weight += weight

            overall_entropy = total_entropy / total_weight if total_weight > 0 else 0.0

            self.logger.debug(f"Calculated entropy score: {overall_entropy:.3f}")
            return overall_entropy

        except Exception as e:
            self.logger.error(f"Error calculating entropy score: {e}")
            return 0.0

    def calculate_coherence_score(
        self,
        clusters: dict[int, list[dict]],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Calculate coherence score measuring how well directories contain similar content.

        Higher coherence means documents in the same directory are semantically similar.

        Args:
            clusters: Dictionary mapping cluster_id to document list
            embeddings: Document embeddings array
            labels: Cluster labels array

        Returns:
            Coherence score (0.0 = no coherence, 1.0 = perfect coherence)
        """
        try:
            directory_groups, doc_id_to_embedding = self._build_directory_mappings(
                clusters, embeddings, labels
            )
            directory_coherences = self._calculate_directory_coherences(
                directory_groups, doc_id_to_embedding
            )
            overall_coherence = self._calculate_weighted_coherence(
                directory_coherences, directory_groups
            )

            self.logger.debug(f"Calculated coherence score: {overall_coherence:.3f}")
            return overall_coherence

        except Exception as e:
            self.logger.error(f"Error calculating coherence score: {e}")
            return 0.0

    def _build_directory_mappings(
        self,
        clusters: dict[int, list[dict]],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
        """Build directory groupings and document-to-embedding mappings."""
        directory_groups = defaultdict(list)
        doc_id_to_embedding = {}

        for cluster_id, docs in clusters.items():
            for doc in docs:
                doc_id = doc["id"]
                filepath = doc.get("filepath", "")

                if not (filepath and doc_id):
                    continue

                directory = str(Path(filepath).parent)
                directory_groups[directory].append(doc_id)

                # Find embedding for this document
                embedding = self._find_document_embedding(
                    doc_id, clusters, embeddings, labels
                )
                if embedding is not None:
                    doc_id_to_embedding[doc_id] = embedding

        return directory_groups, doc_id_to_embedding

    def _find_document_embedding(
        self,
        target_doc_id: str,
        clusters: dict[int, list[dict]],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray | None:
        """Find embedding for a specific document ID."""
        for i, (other_doc, _) in enumerate(
            zip(clusters.values(), labels, strict=False)
        ):
            for other_doc_data in other_doc:
                if other_doc_data["id"] == target_doc_id:
                    return embeddings[i]
        return None

    def _calculate_directory_coherences(
        self,
        directory_groups: dict[str, list[str]],
        doc_id_to_embedding: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Calculate coherence score for each directory."""
        directory_coherences = {}

        for directory, doc_ids in directory_groups.items():
            coherence = self._calculate_single_directory_coherence(
                doc_ids, doc_id_to_embedding
            )
            directory_coherences[directory] = coherence

        return directory_coherences

    def _calculate_single_directory_coherence(
        self, doc_ids: list[str], doc_id_to_embedding: dict[str, np.ndarray]
    ) -> float:
        """Calculate coherence for a single directory."""
        if len(doc_ids) < 2:
            return 1.0

        dir_embeddings = self._get_valid_embeddings(doc_ids, doc_id_to_embedding)

        if len(dir_embeddings) < 2:
            return 1.0

        similarities = self._calculate_pairwise_similarities(dir_embeddings)
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Convert from [-1, 1] to [0, 1] range
        return (avg_similarity + 1) / 2

    def _get_valid_embeddings(
        self, doc_ids: list[str], doc_id_to_embedding: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Get valid embeddings for a list of document IDs."""
        return [
            doc_id_to_embedding[doc_id]
            for doc_id in doc_ids
            if doc_id in doc_id_to_embedding
        ]

    def _calculate_pairwise_similarities(
        self, embeddings: list[np.ndarray]
    ) -> list[float]:
        """Calculate pairwise cosine similarities between embeddings."""
        similarities = []
        embeddings_array = np.array(embeddings)

        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                similarity = self._cosine_similarity(
                    embeddings_array[i], embeddings_array[j]
                )
                if similarity is not None:
                    similarities.append(similarity)

        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float | None:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return None

    def _calculate_weighted_coherence(
        self,
        directory_coherences: dict[str, float],
        directory_groups: dict[str, list[str]],
    ) -> float:
        """Calculate overall coherence as weighted average by document count."""
        total_coherence = 0.0
        total_docs = 0

        for directory, coherence in directory_coherences.items():
            doc_count = len(directory_groups[directory])
            total_coherence += coherence * doc_count
            total_docs += doc_count

        return total_coherence / total_docs if total_docs > 0 else 0.0

    def detect_scattered_clusters(
        self, clusters: dict[int, list[dict]], min_scatter_threshold: int = 3
    ) -> list[dict[str, Any]]:
        """Detect clusters that are scattered across too many directories.

        Args:
            clusters: Dictionary mapping cluster_id to document list
            min_scatter_threshold: Minimum directories to consider scattered

        Returns:
            List of scattered cluster information
        """
        scattered_clusters = []

        for cluster_id, docs in clusters.items():
            # Count unique directories for this cluster
            directories = set()
            doc_count = 0

            for doc in docs:
                filepath = doc.get("filepath", "")
                if filepath:
                    directory = str(Path(filepath).parent)
                    directories.add(directory)
                    doc_count += 1

            # Check if this cluster is scattered
            if len(directories) >= min_scatter_threshold and doc_count > 1:
                directory_counts = defaultdict(int)
                for doc in docs:
                    filepath = doc.get("filepath", "")
                    if filepath:
                        directory = str(Path(filepath).parent)
                        directory_counts[directory] += 1

                scattered_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "document_count": doc_count,
                        "directory_count": len(directories),
                        "directories": dict(directory_counts),
                        "scatter_ratio": len(directories) / doc_count,
                    }
                )

        # Sort by scatter ratio (most scattered first)
        scattered_clusters.sort(key=lambda x: x["scatter_ratio"], reverse=True)

        self.logger.info(f"Found {len(scattered_clusters)} scattered clusters")
        return scattered_clusters

    def suggest_consolidation(
        self, scattered_clusters: list[dict[str, Any]], clusters: dict[int, list[dict]]
    ) -> list[dict[str, Any]]:
        """Suggest consolidation moves for scattered clusters.

        Args:
            scattered_clusters: List of scattered cluster information
            clusters: Original clusters data

        Returns:
            List of consolidation suggestions
        """
        suggestions = []

        for scattered_info in scattered_clusters:
            cluster_id = scattered_info["cluster_id"]
            cluster_docs = clusters.get(cluster_id, [])

            if not cluster_docs:
                continue

            # Find the directory with the most documents in this cluster
            directory_counts = scattered_info["directories"]
            target_directory = max(directory_counts.keys(), key=directory_counts.get)
            max_count = directory_counts[target_directory]

            # Find documents that should be moved
            moves = []
            for doc in cluster_docs:
                filepath = doc.get("filepath", "")
                if filepath:
                    current_directory = str(Path(filepath).parent)
                    if current_directory != target_directory:
                        filename = Path(filepath).name
                        suggested_path = str(Path(target_directory) / filename)
                        moves.append(
                            {
                                "current_path": filepath,
                                "suggested_path": suggested_path,
                                "reason": f"Consolidate cluster {cluster_id} documents",
                            }
                        )

            if moves:
                suggestions.append(
                    {
                        "cluster_id": cluster_id,
                        "target_directory": target_directory,
                        "consolidation_benefit": f"{max_count}/{scattered_info['document_count']} docs already there",
                        "moves": moves,
                        "estimated_entropy_reduction": scattered_info["scatter_ratio"]
                        * 0.1,  # Rough estimate
                    }
                )

        self.logger.info(f"Generated {len(suggestions)} consolidation suggestions")
        return suggestions

    def calculate_drift_velocity(
        self,
        current_metrics: dict[str, float],
        previous_metrics: dict[str, float] | None = None,
        time_delta: float = 1.0,
    ) -> dict[str, float]:
        """Calculate the velocity of documentation drift.

        Args:
            current_metrics: Current entropy/coherence metrics
            previous_metrics: Previous metrics for comparison
            time_delta: Time period between measurements (in days/commits)

        Returns:
            Dictionary with drift velocity metrics
        """
        if previous_metrics is None:
            return {
                "entropy_velocity": 0.0,
                "coherence_velocity": 0.0,
                "overall_drift_velocity": 0.0,
            }

        # Calculate rate of change
        entropy_change = current_metrics.get("entropy_score", 0) - previous_metrics.get(
            "entropy_score", 0
        )
        coherence_change = current_metrics.get(
            "coherence_score", 0
        ) - previous_metrics.get("coherence_score", 0)

        entropy_velocity = entropy_change / time_delta
        coherence_velocity = coherence_change / time_delta

        # Overall drift velocity (entropy increasing is bad, coherence decreasing is bad)
        overall_drift_velocity = entropy_velocity - coherence_velocity

        return {
            "entropy_velocity": entropy_velocity,
            "coherence_velocity": coherence_velocity,
            "overall_drift_velocity": overall_drift_velocity,
        }

    def analyze_clusters(
        self,
        clusters: dict[int, list[dict]],
        embeddings: np.ndarray,
        labels: np.ndarray,
        previous_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive drift analysis on document clusters.

        Args:
            clusters: Dictionary mapping cluster_id to document list
            embeddings: Document embeddings array
            labels: Cluster labels array
            previous_analysis: Previous analysis for drift velocity calculation

        Returns:
            Complete drift analysis results
        """
        self.logger.info("Performing comprehensive drift analysis")

        try:
            # Calculate core metrics
            entropy_score = self.calculate_entropy_score(clusters, embeddings, labels)
            coherence_score = self.calculate_coherence_score(
                clusters, embeddings, labels
            )

            # Detect organizational issues
            scattered_clusters = self.detect_scattered_clusters(clusters)
            consolidation_suggestions = self.suggest_consolidation(
                scattered_clusters, clusters
            )

            # Calculate drift velocity if previous analysis available
            current_metrics = {
                "entropy_score": entropy_score,
                "coherence_score": coherence_score,
            }

            velocity_metrics = {}
            if previous_analysis:
                prev_metrics = {
                    "entropy_score": previous_analysis.get("entropy_score", 0),
                    "coherence_score": previous_analysis.get("coherence_score", 0),
                }
                velocity_metrics = self.calculate_drift_velocity(
                    current_metrics, prev_metrics
                )

            # Determine overall status
            status = self._determine_drift_status(
                entropy_score, coherence_score, len(scattered_clusters)
            )

            # Calculate organization health score (0-100)
            health_score = self._calculate_health_score(
                entropy_score, coherence_score, len(scattered_clusters), len(clusters)
            )

            analysis_result = {
                "entropy_score": round(entropy_score, 3),
                "coherence_score": round(coherence_score, 3),
                "health_score": round(health_score, 1),
                "status": status,
                "scattered_clusters": len(scattered_clusters),
                "consolidation_opportunities": len(consolidation_suggestions),
                "scattered_cluster_details": scattered_clusters[
                    :5
                ],  # Top 5 most scattered
                "consolidation_suggestions": consolidation_suggestions[
                    :3
                ],  # Top 3 suggestions
            }

            # Add velocity metrics if available
            if velocity_metrics:
                analysis_result["drift_velocity"] = velocity_metrics

            # Add thresholds for context
            analysis_result["thresholds"] = {
                "entropy_warning": 0.7,
                "entropy_critical": 0.85,
                "coherence_warning": 0.4,
                "coherence_critical": 0.25,
            }

            self.logger.info(
                f"Drift analysis complete: entropy={entropy_score:.3f}, "
                f"coherence={coherence_score:.3f}, status={status}"
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in drift analysis: {e}")
            return {
                "error": str(e),
                "entropy_score": 0.0,
                "coherence_score": 0.0,
                "status": "error",
            }

    def _determine_drift_status(
        self, entropy_score: float, coherence_score: float, scattered_count: int
    ) -> str:
        """Determine overall drift status based on metrics.

        Args:
            entropy_score: Entropy metric (0-1)
            coherence_score: Coherence metric (0-1)
            scattered_count: Number of scattered clusters

        Returns:
            Status string: 'healthy', 'warning', 'critical'
        """
        # Critical conditions
        if entropy_score > 0.85 or coherence_score < 0.25:
            return "critical"

        # Warning conditions
        if entropy_score > 0.7 or coherence_score < 0.4 or scattered_count > 3:
            return "warning"

        return "healthy"

    def _calculate_health_score(
        self,
        entropy_score: float,
        coherence_score: float,
        scattered_count: int,
        total_clusters: int,
    ) -> float:
        """Calculate overall documentation health score (0-100).

        Args:
            entropy_score: Entropy metric (0-1)
            coherence_score: Coherence metric (0-1)
            scattered_count: Number of scattered clusters
            total_clusters: Total number of clusters

        Returns:
            Health score (0-100, higher is better)
        """
        # Base score from entropy and coherence (70% weight)
        entropy_contribution = (1.0 - entropy_score) * 40  # Lower entropy is better
        coherence_contribution = coherence_score * 30  # Higher coherence is better

        # Organization penalty (20% weight)
        scatter_ratio = scattered_count / total_clusters if total_clusters > 0 else 0
        organization_contribution = (1.0 - min(scatter_ratio, 1.0)) * 20

        # Bonus for good organization (10% weight)
        bonus = 10 if entropy_score < 0.5 and coherence_score > 0.7 else 0

        health_score = (
            entropy_contribution
            + coherence_contribution
            + organization_contribution
            + bonus
        )

        return max(0, min(100, health_score))

    def analyze_label_distribution(
        self, clusters: dict[int, list[dict]]
    ) -> dict[str, Any]:
        """Analyze how labels are distributed across clusters.

        Args:
            clusters: Dictionary mapping cluster_id to document list

        Returns:
            Label distribution analysis results
        """
        self.logger.info("Analyzing label distribution across clusters")

        try:
            label_data = self._collect_label_data(clusters)
            coherence_data = self._calculate_label_coherence_metrics(label_data)
            diversity_data = self._calculate_cluster_label_diversity(
                label_data.cluster_labels
            )
            analysis_result = self._build_label_analysis_result(
                label_data, coherence_data, diversity_data, clusters
            )

            self._log_label_analysis_completion(label_data, coherence_data)
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in label distribution analysis: {e}")
            return {"error": str(e)}

    def _collect_label_data(self, clusters: dict[int, list[dict]]) -> "LabelData":
        """Collect and organize all label data from clusters."""
        all_labels = Counter()
        cluster_labels = {}
        label_clusters = defaultdict(set)
        total_docs_with_labels = 0

        for cluster_id, docs in clusters.items():
            cluster_label_count = Counter()

            for doc in docs:
                doc_labels_list = self._extract_document_labels(doc)

                if doc_labels_list:
                    total_docs_with_labels += 1
                    for label in doc_labels_list:
                        if label:  # Skip empty labels
                            all_labels[label] += 1
                            cluster_label_count[label] += 1
                            label_clusters[label].add(cluster_id)

            cluster_labels[cluster_id] = dict(cluster_label_count)

        return LabelData(
            all_labels, cluster_labels, label_clusters, total_docs_with_labels
        )

    def _extract_document_labels(self, doc: dict) -> list[str]:
        """Extract labels from a document's metadata."""
        metadata = doc.get("metadata", {})
        doc_labels = metadata.get("labels", "")

        # Convert comma-separated string to list
        if isinstance(doc_labels, str) and doc_labels.strip():
            return [label.strip() for label in doc_labels.split(",")]
        elif isinstance(doc_labels, list):
            # Fallback for old format
            return doc_labels
        else:
            return []

    def _calculate_label_coherence_metrics(
        self, label_data: "LabelData"
    ) -> "CoherenceData":
        """Calculate coherence metrics for all labels."""
        label_coherence_scores = {}
        scattered_labels = []

        for label, cluster_set in label_data.label_clusters.items():
            cluster_count = len(cluster_set)
            label_frequency = label_data.all_labels[label]

            coherence = 1.0 / cluster_count if cluster_count > 0 else 0.0
            label_coherence_scores[label] = {
                "coherence_score": round(coherence, 3),
                "cluster_count": cluster_count,
                "total_occurrences": label_frequency,
                "clusters": list(cluster_set),
            }

            # Identify scattered labels
            if cluster_count > 3 and label_frequency > 5:
                scattered_labels.append(
                    {
                        "label": label,
                        "cluster_count": cluster_count,
                        "total_occurrences": label_frequency,
                        "coherence_score": round(coherence, 3),
                    }
                )

        avg_coherence = self._calculate_average_coherence(label_coherence_scores)
        return CoherenceData(label_coherence_scores, scattered_labels, avg_coherence)

    def _calculate_average_coherence(
        self, label_coherence_scores: dict[str, dict]
    ) -> float:
        """Calculate average label coherence across all labels."""
        if not label_coherence_scores:
            return 0.0

        total_coherence = sum(
            data["coherence_score"] for data in label_coherence_scores.values()
        )
        return total_coherence / len(label_coherence_scores)

    def _calculate_cluster_label_diversity(
        self, cluster_labels: dict[int, dict]
    ) -> dict[int, dict]:
        """Calculate label diversity for each cluster using Shannon entropy."""
        cluster_diversity = {}

        for cluster_id, label_dist in cluster_labels.items():
            if not label_dist:
                continue

            entropy = self._calculate_shannon_entropy(label_dist)
            cluster_diversity[cluster_id] = {
                "label_entropy": round(entropy, 3),
                "unique_labels": len(label_dist),
                "total_label_instances": sum(label_dist.values()),
                "labels": label_dist,
            }

        return cluster_diversity

    def _calculate_shannon_entropy(self, label_distribution: dict[str, int]) -> float:
        """Calculate Shannon entropy for a label distribution."""
        total_labels = sum(label_distribution.values())
        if total_labels == 0:
            return 0.0

        entropy = 0.0
        for count in label_distribution.values():
            prob = count / total_labels
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def _build_label_analysis_result(
        self,
        label_data: "LabelData",
        coherence_data: "CoherenceData",
        diversity_data: dict[int, dict],
        clusters: dict[int, list[dict]],
    ) -> dict[str, Any]:
        """Build the final analysis result dictionary."""
        scattered_labels = sorted(
            coherence_data.scattered_labels,
            key=lambda x: x["cluster_count"],
            reverse=True,
        )
        most_diverse_clusters = sorted(
            diversity_data.items(),
            key=lambda x: x[1]["label_entropy"],
            reverse=True,
        )[:5]

        label_coverage = self._calculate_label_coverage(label_data, clusters)

        return {
            "total_unique_labels": len(label_data.all_labels),
            "total_docs_with_labels": label_data.total_docs_with_labels,
            "label_coverage": label_coverage,
            "average_label_coherence": round(coherence_data.avg_coherence, 3),
            "scattered_labels": scattered_labels[:10],
            "most_diverse_clusters": [
                {"cluster_id": cid, **data} for cid, data in most_diverse_clusters
            ],
            "label_frequency_distribution": dict(label_data.all_labels.most_common(20)),
            "cluster_label_distributions": label_data.cluster_labels,
            "label_coherence_scores": coherence_data.label_coherence_scores,
        }

    def _calculate_label_coverage(
        self, label_data: "LabelData", clusters: dict[int, list[dict]]
    ) -> float:
        """Calculate percentage of documents that have labels."""
        if not clusters:
            return 0.0

        total_docs = sum(len(docs) for docs in clusters.values())
        return round(label_data.total_docs_with_labels / total_docs * 100, 1)

    def _log_label_analysis_completion(
        self, label_data: "LabelData", coherence_data: "CoherenceData"
    ) -> None:
        """Log completion of label analysis."""
        self.logger.info(
            f"Label analysis complete: {len(label_data.all_labels)} unique labels, "
            f"{len(coherence_data.scattered_labels)} scattered labels, "
            f"{coherence_data.avg_coherence:.3f} avg coherence"
        )

    def suggest_label_based_reorganization(
        self, label_analysis: dict[str, Any], clusters: dict[int, list[dict]]
    ) -> list[dict[str, Any]]:
        """Suggest reorganization based on label patterns.

        Args:
            label_analysis: Results from analyze_label_distribution
            clusters: Original clusters data

        Returns:
            List of label-based reorganization suggestions
        """
        try:
            suggestions = []

            # Generate different types of suggestions
            suggestions.extend(self._suggest_cluster_splits(label_analysis, clusters))
            suggestions.extend(
                self._suggest_label_consolidation(label_analysis, clusters)
            )
            suggestions.extend(
                self._suggest_label_combination_categories(label_analysis)
            )

            # Sort and finalize suggestions
            final_suggestions = self._prioritize_suggestions(suggestions)

            self.logger.info(
                f"Generated {len(final_suggestions)} label-based reorganization suggestions"
            )
            return final_suggestions

        except Exception as e:
            self.logger.error(f"Error generating label-based suggestions: {e}")
            return []

    def _suggest_cluster_splits(
        self, label_analysis: dict[str, Any], clusters: dict[int, list[dict]]
    ) -> list[dict[str, Any]]:
        """Suggest splitting clusters with high label diversity."""
        suggestions = []
        diverse_clusters = label_analysis.get("most_diverse_clusters", [])

        for cluster_info in diverse_clusters[:3]:  # Top 3 most diverse
            suggestion = self._create_cluster_split_suggestion(cluster_info, clusters)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _create_cluster_split_suggestion(
        self, cluster_info: dict[str, Any], clusters: dict[int, list[dict]]
    ) -> dict[str, Any] | None:
        """Create a suggestion for splitting a diverse cluster."""
        cluster_id = cluster_info["cluster_id"]
        entropy = cluster_info["label_entropy"]
        labels = cluster_info["labels"]

        if not (entropy > 2.0 and len(labels) > 3):
            return None

        # Find the dominant label groups
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
        primary_label = sorted_labels[0][0]
        secondary_labels = [label for label, count in sorted_labels[1:3] if count > 1]

        return {
            "type": "split_diverse_cluster",
            "cluster_id": cluster_id,
            "reason": f"Cluster has high label diversity (entropy: {entropy:.2f})",
            "primary_label": primary_label,
            "suggested_splits": secondary_labels,
            "affected_documents": len(clusters.get(cluster_id, [])),
            "priority": "high" if entropy > 3.0 else "medium",
        }

    def _suggest_label_consolidation(
        self, label_analysis: dict[str, Any], clusters: dict[int, list[dict]]
    ) -> list[dict[str, Any]]:
        """Suggest consolidating scattered labels."""
        suggestions = []
        scattered_labels = label_analysis.get("scattered_labels", [])

        for label_info in scattered_labels[:5]:  # Top 5 most scattered
            suggestion = self._create_label_consolidation_suggestion(
                label_info, clusters
            )
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _create_label_consolidation_suggestion(
        self, label_info: dict[str, Any], clusters: dict[int, list[dict]]
    ) -> dict[str, Any] | None:
        """Create a suggestion for consolidating a scattered label."""
        label = label_info["label"]
        cluster_count = label_info["cluster_count"]

        if cluster_count <= 4:  # Not highly scattered
            return None

        label_cluster_sizes = self._calculate_label_cluster_sizes(label, clusters)

        if not label_cluster_sizes:
            return None

        primary_cluster = max(label_cluster_sizes, key=label_cluster_sizes.get)
        other_clusters = [
            cid for cid in label_cluster_sizes.keys() if cid != primary_cluster
        ]

        return {
            "type": "consolidate_scattered_label",
            "label": label,
            "reason": f"Label '{label}' scattered across {cluster_count} clusters",
            "primary_cluster": primary_cluster,
            "consolidate_from_clusters": other_clusters[:3],  # Top 3 source clusters
            "total_affected_docs": label_info["total_occurrences"],
            "priority": "high" if cluster_count > 6 else "medium",
        }

    def _calculate_label_cluster_sizes(
        self, label: str, clusters: dict[int, list[dict]]
    ) -> dict[int, int]:
        """Calculate how many documents with the given label exist in each cluster."""
        label_cluster_sizes = {}

        for cluster_id, docs in clusters.items():
            count = sum(
                1 for doc in docs if label in doc.get("metadata", {}).get("labels", [])
            )
            if count > 0:
                label_cluster_sizes[cluster_id] = count

        return label_cluster_sizes

    def _suggest_label_combination_categories(
        self, label_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Suggest creating new categories for frequent label combinations."""
        suggestions = []
        cluster_labels = label_analysis.get("cluster_label_distributions", {})

        label_combinations = self._find_label_combinations(cluster_labels)

        for combo, frequency in label_combinations.most_common(3):
            suggestion = self._create_label_combination_suggestion(combo, frequency)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _find_label_combinations(
        self, cluster_labels: dict[int, dict[str, int]]
    ) -> Counter:
        """Find frequently co-occurring label combinations."""
        label_combinations = Counter()

        for cluster_id, labels in cluster_labels.items():
            if len(labels) < 2:
                continue

            # Find co-occurring labels
            label_list = [label for label, count in labels.items() if count > 1]
            if len(label_list) >= 2:
                combo = tuple(sorted(label_list[:2]))  # Most frequent pair
                label_combinations[combo] += 1

        return label_combinations

    def _create_label_combination_suggestion(
        self, combo: tuple[str, str], frequency: int
    ) -> dict[str, Any] | None:
        """Create a suggestion for a label combination."""
        if frequency < 2:  # Must appear in at least 2 clusters
            return None

        return {
            "type": "create_label_combination_category",
            "label_combination": list(combo),
            "reason": f"Labels {combo} frequently appear together",
            "frequency": frequency,
            "suggested_directory": f"docs/{'-'.join(combo).lower().replace(' ', '-')}",
            "priority": "medium",
        }

    def _prioritize_suggestions(
        self, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sort suggestions by priority and impact."""
        priority_order = {"high": 3, "medium": 2, "low": 1}
        return sorted(
            suggestions,
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True,
        )

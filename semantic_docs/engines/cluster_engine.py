"""Semantic document clustering engine for drift detection and organization analysis.

Provides clustering capabilities to identify semantic groupings in documentation
and compare them against actual file organization to detect drift and entropy.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np

# Optional imports for clustering and visualization
try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    # Always use a non-interactive backend to avoid DISPLAY/X issues (e.g., WSL/CI)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None

from semantic_docs.config.settings import SemanticConfig, load_config
from semantic_docs.engines.semantic_engine import SemanticEngine


class DocumentClusterEngine:
    """Groups documents into semantic clusters and analyzes organization drift."""

    def __init__(
        self,
        config: SemanticConfig | None = None,
        semantic_engine: SemanticEngine | None = None,
    ):
        """Initialize cluster engine.

        Args:
            config: Optional SemanticConfig instance
            semantic_engine: Optional SemanticEngine instance to reuse
        """
        if config is None:
            config = load_config()
        self.config = config

        # Initialize semantic engine if not provided
        if semantic_engine is None:
            self.semantic_engine = SemanticEngine(config)
        else:
            self.semantic_engine = semantic_engine

        # Setup logging
        self.logger = logging.getLogger("cluster_engine")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Check for scikit-learn availability
        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not available. Clustering functionality will be limited."
            )

    def get_all_documents_with_embeddings(
        self,
    ) -> tuple[list[str], np.ndarray, list[dict]]:
        """Retrieve all documents with their embeddings and metadata.

        Returns:
            Tuple of (document_ids, embeddings_array, metadatas)
        """
        try:
            # Ensure pyplot is bound within this scope for static analysis
            # Local import to satisfy static analysis and ensure plt is bound here
            collection = self.semantic_engine.collection
            results = collection.get(include=["embeddings", "metadatas", "documents"])

            ids = results["ids"]
            embeddings_list = results["embeddings"]
            if embeddings_list is not None and len(embeddings_list) > 0:
                embeddings = np.array(embeddings_list)
            else:
                embeddings = np.array([])
            metadatas = results["metadatas"] if results["metadatas"] else []

            self.logger.info(f"Retrieved {len(ids)} documents with embeddings")
            return ids, embeddings, metadatas

        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            raise

    def find_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 10
    ) -> int:
        """Find optimal number of clusters using silhouette analysis.

        Args:
            embeddings: Document embeddings array
            max_clusters: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "sklearn not available, using default cluster count of 5"
            )
            return min(5, len(embeddings) // 2)

        if len(embeddings) < 4:
            return 2  # Minimum clusters for small datasets

        max_k = min(max_clusters, len(embeddings) - 1)
        silhouette_scores = []

        self.logger.info("Finding optimal number of clusters...")
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Skip if any cluster is too small
            unique, counts = np.unique(cluster_labels, return_counts=True)
            if min(counts) < 1:
                continue

            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append((k, score))
            self.logger.debug(f"  k={k}: silhouette score = {score:.3f}")

        if not silhouette_scores:
            return 2

        # Find k with highest silhouette score
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k

    def cluster_documents(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
        method: str = "kmeans",
        min_cluster_size: int = 2,
    ) -> np.ndarray:
        """Cluster documents using the specified method.

        Args:
            embeddings: Document embeddings array
            n_clusters: Number of clusters (auto-detect if None)
            method: Clustering method ('kmeans' or 'hierarchical')
            min_cluster_size: Minimum cluster size

        Returns:
            Cluster labels array
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("sklearn required for clustering")
            raise ImportError("scikit-learn is required for clustering functionality")

        # Auto-detect optimal number of clusters if not specified
        if n_clusters is None:
            self.logger.info("Finding optimal number of clusters...")
            n_clusters = self.find_optimal_clusters(embeddings)

        self.logger.info(
            f"Clustering {len(embeddings)} documents into {n_clusters} clusters using {method}"
        )

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clusterer.fit_predict(embeddings)

        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        self.logger.info(f"Cluster sizes: {dict(zip(unique, counts, strict=False))}")

        return labels

    def generate_cluster_descriptions(
        self,
        clusters: dict[int, list[dict]],
    ) -> dict[int, str]:
        """Generate descriptive names for each cluster based on content.

        Args:
            clusters: Dictionary mapping cluster_id to list of document metadata

        Returns:
            Dictionary mapping cluster_id to description string
        """
        descriptions = {}
        theme_keywords = self._get_theme_keywords()
        used_names = set()

        for cluster_id, docs in clusters.items():
            all_text = self._extract_cluster_text(docs)
            theme_scores = self._calculate_theme_scores(all_text, theme_keywords)
            cluster_labels = self._extract_cluster_labels(docs)

            base_description = self._generate_description(theme_scores, cluster_labels)

            # Ensure unique names
            description = self._ensure_unique_name(
                base_description, cluster_id, len(docs), used_names
            )
            used_names.add(description)

            descriptions[cluster_id] = description

        return descriptions

    def _get_theme_keywords(self) -> dict[str, list[str]]:
        """Get theme keywords for cluster analysis including folder-based themes."""
        return {
            "testing": [
                "test",
                "pytest",
                "mock",
                "fixture",
                "coverage",
                "unit",
                "integration",
            ],
            "configuration": [
                "config",
                "settings",
                "environment",
                "setup",
                "env",
                "variable",
            ],
            "architecture": [
                "architecture",
                "design",
                "pattern",
                "structure",
                "system",
                "component",
            ],
            "api": [
                "api",
                "fastapi",
                "endpoint",
                "http",
                "service",
                "request",
                "response",
            ],
            "deployment": [
                "docker",
                "deployment",
                "container",
                "build",
                "production",
                "cicd",
            ],
            "observability": [
                "telemetry",
                "metrics",
                "logging",
                "monitoring",
                "otel",
                "trace",
            ],
            "security": ["auth", "authentication", "oidc", "security", "jwt", "token"],
            "models": ["model", "whisper", "transcription", "ai", "ml", "inference"],
            "development": [
                "dev",
                "development",
                "workflow",
                "process",
                "task",
                "makefile",
            ],
            "reference": [
                "reference",
                "guide",
                "troubleshooting",
                "setup",
                "installation",
            ],
            "core_concepts": [
                "concepts",
                "overview",
                "principles",
                "fundamentals",
                "basics",
            ],
            # Folder-based themes (these are detected from folder paths)
            "guides": [
                "guides",
                "guide",
                "tutorial",
                "walkthrough",
                "howto",
                "instructions",
            ],
            "reports": [
                "reports",
                "report",
                "analysis",
                "metrics",
                "summary",
                "findings",
            ],
            "processes": [
                "processes",
                "process",
                "workflow",
                "procedure",
                "methodology",
            ],
        }

    def _extract_cluster_text(self, docs: list[dict]) -> str:
        """Extract text content from cluster documents including folder context."""
        headings = [doc.get("heading", "") for doc in docs]
        file_paths = [doc.get("filepath", "") for doc in docs]

        # Extract folder context and components for better theme detection
        folder_contexts = []
        folder_components = []

        for doc in docs:
            metadata = doc.get("metadata", {})

            # Add folder context (semantic folder names)
            folder_context = metadata.get("folder_context", "")
            if folder_context:
                folder_contexts.append(folder_context)

            # Add individual folder path components
            components = metadata.get("folder_path_components", "")
            if components:
                folder_components.extend(components.split(", "))

        # Combine all text sources with folder information weighted more heavily
        all_text_parts = []
        all_text_parts.extend(headings)
        all_text_parts.extend(file_paths)

        # Weight folder context by adding multiple times (simple but effective)
        all_text_parts.extend(
            folder_contexts * 2
        )  # Double weight for semantic folder context
        all_text_parts.extend(folder_components)  # Single weight for path components

        return " ".join(part for part in all_text_parts if part).lower()

    def _calculate_theme_scores(
        self, text: str, theme_keywords: dict[str, list[str]]
    ) -> dict[str, int]:
        """Calculate theme scores based on keyword frequency."""
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(text.count(keyword) for keyword in keywords)
            if score > 0:
                theme_scores[theme] = score
        return theme_scores

    def _extract_cluster_labels(self, docs: list[dict]) -> list[str]:
        """Extract labels from documents in the cluster."""
        cluster_labels = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            doc_labels = metadata.get("labels", "")

            if isinstance(doc_labels, str) and doc_labels.strip():
                cluster_labels.extend(
                    [label.strip() for label in doc_labels.split(",")]
                )
            elif isinstance(doc_labels, list):
                cluster_labels.extend(doc_labels)

        return cluster_labels

    def _generate_description(
        self, theme_scores: dict[str, int], cluster_labels: list[str]
    ) -> str:
        """Generate cluster description from themes and labels."""
        from collections import Counter

        # Get top themes (sorted by score)
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        top_themes = [theme for theme, score in sorted_themes[:2] if score > 0]

        # Get most common labels
        if cluster_labels:
            label_counts = Counter(cluster_labels)
            top_labels = [
                label for label, count in label_counts.most_common(2) if label.strip()
            ]
        else:
            top_labels = []

        # Generate description with priority: labels > themes > fallback
        if top_labels and top_themes:
            # Combine primary label with theme context
            primary_label = top_labels[0]
            primary_theme = top_themes[0]

            # Use label if it's more specific than theme
            if primary_label.lower() not in primary_theme.lower():
                return f"{primary_label} ({primary_theme.title()})"
            else:
                return primary_theme.title()

        elif top_labels:
            # Use labels when no strong theme
            if len(top_labels) == 1:
                return top_labels[0]
            else:
                return f"{top_labels[0]} & {top_labels[1]}"

        elif top_themes:
            # Use theme when no labels
            if len(top_themes) == 1:
                return top_themes[0].title()
            else:
                return f"{top_themes[0].title()} & {top_themes[1].title()}"

        else:
            # Fallback for unclear content
            return "General Content"

    def _ensure_unique_name(
        self, base_name: str, cluster_id: int, doc_count: int, used_names: set[str]
    ) -> str:
        """Ensure cluster name is unique by adding suffixes if needed."""
        full_name = f"{base_name} ({doc_count} docs)"

        if full_name not in used_names:
            return full_name

        # Try with cluster ID suffix
        name_with_id = f"{base_name} #{cluster_id} ({doc_count} docs)"
        if name_with_id not in used_names:
            return name_with_id

        # Final fallback with numeric suffix
        counter = 2
        while f"{base_name} ({counter}) ({doc_count} docs)" in used_names:
            counter += 1
        return f"{base_name} ({counter}) ({doc_count} docs)"

    def create_clusters(
        self,
        n_clusters: int | None = None,
        method: str = "kmeans",
        min_cluster_size: int = 2,
        include_analysis: bool = True,
    ) -> dict[str, Any]:
        """Create semantic clusters of documents with optional drift analysis.

        Args:
            n_clusters: Number of clusters (auto-detect if None)
            method: Clustering method
            min_cluster_size: Minimum cluster size
            include_analysis: Whether to include drift analysis

        Returns:
            Dictionary with clustering results and optional analysis
        """
        try:
            # Get all documents and embeddings
            ids, embeddings, metadatas = self.get_all_documents_with_embeddings()

            if len(embeddings) < 2:
                self.logger.warning("Not enough documents for clustering")
                return {"error": "Need at least 2 documents for clustering"}

            # Perform clustering
            labels = self.cluster_documents(
                embeddings, n_clusters, method, min_cluster_size
            )

            # Group documents by cluster
            clusters = {}
            for i, (doc_id, metadata, label) in enumerate(
                zip(ids, metadatas, labels, strict=False)
            ):
                if label not in clusters:
                    clusters[label] = []

                doc_info = {"id": doc_id}
                if metadata:
                    doc_info.update(metadata)
                clusters[label].append(doc_info)

            # Generate cluster descriptions
            descriptions = self.generate_cluster_descriptions(clusters)

            # Create basic result
            result = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_documents": len(ids),
                    "total_clusters": len(clusters),
                    "clustering_method": method,
                    "embedding_model": "text-embedding-3-large",
                },
                "clusters": {},
            }

            # Format clusters for output
            for cluster_id in sorted(clusters.keys()):
                result["clusters"][str(cluster_id)] = {
                    "description": descriptions[cluster_id],
                    "document_count": len(clusters[cluster_id]),
                    "documents": clusters[cluster_id],
                }

            # Add drift analysis if requested
            if include_analysis:
                from semantic_docs.engines.drift_detector import DriftDetector

                drift_detector = DriftDetector(self.config)
                drift_analysis = drift_detector.analyze_clusters(
                    clusters, embeddings, labels
                )
                result["drift_analysis"] = drift_analysis

            return result

        except Exception as e:
            self.logger.error(f"Error creating clusters: {e}")
            raise

    def visualize_clusters(
        self,
        clusters_data: dict[str, Any],
        output_file: str = "document_clusters_visualization.png",
    ) -> str | None:
        """Create a 2D visualization of document clusters.

        Args:
            clusters_data: Clustering results from create_clusters()
            output_file: Output file path

        Returns:
            Output file path if successful, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Skipping visualization.")
            return None

        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available. Skipping visualization.")
            return None

        try:
            # Get embeddings and labels
            ids, embeddings, metadatas = self.get_all_documents_with_embeddings()
            embeddings_array = np.array(embeddings)

            # Reduce dimensions to 2D using PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings_array)

            # Create color map (denser palette suitable for many clusters)
            n_clusters = len(clusters_data["clusters"])
            if n_clusters <= 20:
                cmap = plt.get_cmap("tab20")
            else:
                # Fallback to HSV when more than 20 clusters
                cmap = plt.get_cmap("hsv")
            colors = cmap(np.linspace(0, 1, n_clusters))

            # Create the plot with a 4:2 (2:1) landscape aspect ratio
            plt.figure(figsize=(12, 8))

            # Plot each cluster
            for i, (cluster_id, cluster_info) in enumerate(
                clusters_data["clusters"].items()
            ):
                cluster_docs = cluster_info["documents"]
                cluster_indices = [ids.index(doc["id"]) for doc in cluster_docs]

                cluster_embeddings = embeddings_2d[cluster_indices]

                plt.scatter(
                    cluster_embeddings[:, 0],
                    cluster_embeddings[:, 1],
                    c=[colors[i]],
                    label=cluster_info["description"],
                    alpha=0.85,
                    s=100,
                    linewidths=0,
                    edgecolors="none",
                )

                # Add document labels for small clusters
                if len(cluster_embeddings) <= 3:
                    for j, (x, y) in enumerate(cluster_embeddings):
                        doc = cluster_docs[j]
                        heading = doc.get("heading", doc.get("filepath", ""))

                        # Include label information if available
                        metadata = doc.get("metadata", {})
                        doc_labels = metadata.get("labels", [])

                        if doc_labels:
                            # Show heading + primary label
                            primary_label = doc_labels[0] if doc_labels else ""
                            label_text = f"{heading[:15]}... [{primary_label}]"
                        else:
                            label_text = (
                                heading[:20] + "..." if len(heading) > 20 else heading
                            )

                        plt.annotate(
                            label_text,
                            (x, y),
                            xytext=(12, 0),
                            textcoords="offset points",
                            fontsize=7,
                            alpha=0.7,
                        )

            plt.xlabel(
                f"First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)",
                fontsize=10,
            )
            plt.ylabel(
                f"Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)",
                fontsize=10,
            )
            plt.title("Semantic Clusters of Documentation", fontsize=12, pad=10)
            # Compact legend at the top center across multiple columns for dense layout
            legend_cols = min(max(n_clusters, 1), 6)
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=legend_cols,
                fontsize=8,
                frameon=False,
                markerscale=0.8,
                handlelength=1.5,
                columnspacing=0.8,
                handletextpad=0.4,
            )
            plt.grid(True, alpha=0.2, linewidth=0.3)
            plt.margins(x=0.02, y=0.05)
            plt.tight_layout(pad=1.1)
            plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
            self.logger.info(f"Visualization saved to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return None

    def get_cluster_health_metrics(self) -> dict[str, Any]:
        """Get health metrics for the current document clustering.

        Returns:
            Dictionary with health metrics
        """
        try:
            # Create clusters for analysis
            clusters_data = self.create_clusters(include_analysis=True)

            if "error" in clusters_data:
                return {"error": clusters_data["error"]}

            # Extract key metrics
            total_docs = clusters_data["metadata"]["total_documents"]
            total_clusters = clusters_data["metadata"]["total_clusters"]

            # Basic clustering health
            avg_cluster_size = total_docs / total_clusters if total_clusters > 0 else 0

            # Get drift analysis if available
            drift_analysis = clusters_data.get("drift_analysis", {})

            health_metrics = {
                "total_documents": total_docs,
                "total_clusters": total_clusters,
                "average_cluster_size": round(avg_cluster_size, 2),
                "clustering_health": "healthy"
                if 2 <= avg_cluster_size <= 10
                else "needs_attention",
            }

            # Add drift metrics if available
            if drift_analysis:
                health_metrics.update(
                    {
                        "entropy_score": drift_analysis.get("entropy_score"),
                        "coherence_score": drift_analysis.get("coherence_score"),
                        "drift_status": drift_analysis.get("status", "unknown"),
                    }
                )

            return health_metrics

        except Exception as e:
            self.logger.error(f"Error getting health metrics: {e}")
            return {"error": str(e)}

    def analyze_folder_cluster_alignment(
        self,
        clusters_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze how well folder structure aligns with semantic clusters.

        Args:
            clusters_data: Optional existing cluster data to reuse

        Returns:
            Dictionary with folder-cluster alignment analysis
        """
        try:
            # Get clusters if not provided
            if clusters_data is None:
                clusters_data = self.create_clusters(include_analysis=False)
                if "error" in clusters_data:
                    return {"error": clusters_data["error"]}

            clusters = clusters_data.get("clusters", {})
            if not clusters:
                return {"error": "No clusters found"}

            # Build folder-cluster distribution matrix
            folder_cluster_matrix = {}  # folder -> {cluster_id: doc_count}
            cluster_folder_matrix = {}  # cluster_id -> {folder: doc_count}
            total_docs_by_folder = {}  # folder -> total_doc_count
            total_docs_by_cluster = {}  # cluster_id -> total_doc_count

            # Process all documents
            for cluster_id_str, cluster_info in clusters.items():
                cluster_id = int(cluster_id_str)
                documents = cluster_info.get("documents", [])
                total_docs_by_cluster[cluster_id] = len(documents)
                cluster_folder_matrix[cluster_id] = {}

                for doc in documents:
                    # Extract folder information
                    directory = doc.get("directory", "root")

                    # Normalize directory path for consistency
                    if directory == "." or directory == "":
                        directory = "root"

                    # Update matrices
                    if directory not in folder_cluster_matrix:
                        folder_cluster_matrix[directory] = {}
                        total_docs_by_folder[directory] = 0

                    if cluster_id not in folder_cluster_matrix[directory]:
                        folder_cluster_matrix[directory][cluster_id] = 0
                    if directory not in cluster_folder_matrix[cluster_id]:
                        cluster_folder_matrix[cluster_id][directory] = 0

                    folder_cluster_matrix[directory][cluster_id] += 1
                    cluster_folder_matrix[cluster_id][directory] += 1
                    total_docs_by_folder[directory] += 1

            # Calculate alignment metrics
            folder_metrics = self._calculate_folder_metrics(
                folder_cluster_matrix, total_docs_by_folder
            )
            cluster_metrics = self._calculate_cluster_metrics(
                cluster_folder_matrix, total_docs_by_cluster
            )
            overall_metrics = self._calculate_overall_alignment(
                folder_cluster_matrix,
                cluster_folder_matrix,
                total_docs_by_folder,
                total_docs_by_cluster,
            )

            # Identify misaligned documents
            misaligned_docs = self._identify_misaligned_documents(
                clusters, folder_cluster_matrix
            )

            return {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_folders": len(folder_cluster_matrix),
                    "total_clusters": len(cluster_folder_matrix),
                    "total_documents": sum(total_docs_by_folder.values()),
                },
                "folder_metrics": folder_metrics,
                "cluster_metrics": cluster_metrics,
                "overall_alignment": overall_metrics,
                "misaligned_documents": misaligned_docs,
                "distribution_matrix": {
                    "folder_to_cluster": folder_cluster_matrix,
                    "cluster_to_folder": cluster_folder_matrix,
                },
            }

        except Exception as e:
            self.logger.error(f"Error analyzing folder-cluster alignment: {e}")
            return {"error": str(e)}

    def _calculate_folder_metrics(
        self,
        folder_cluster_matrix: dict[str, dict[int, int]],
        total_docs_by_folder: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Calculate purity and dispersion metrics for each folder."""
        folder_metrics = []

        for folder, cluster_counts in folder_cluster_matrix.items():
            total_docs = total_docs_by_folder[folder]

            # Find dominant cluster
            dominant_cluster = max(cluster_counts.items(), key=lambda x: x[1])
            dominant_cluster_id, dominant_count = dominant_cluster

            # Calculate purity (% of folder's docs in dominant cluster)
            purity = dominant_count / total_docs if total_docs > 0 else 0

            # Calculate dispersion (number of clusters this folder spans)
            num_clusters = len(cluster_counts)

            # Calculate entropy (measure of how scattered the folder is)
            folder_entropy = 0.0
            if total_docs > 0:
                for count in cluster_counts.values():
                    if count > 0:
                        prob = count / total_docs
                        folder_entropy -= prob * np.log2(prob)

            folder_metrics.append(
                {
                    "folder": folder,
                    "total_documents": total_docs,
                    "dominant_cluster": dominant_cluster_id,
                    "dominant_cluster_docs": dominant_count,
                    "purity": round(purity, 3),
                    "num_clusters_spanned": num_clusters,
                    "entropy": round(folder_entropy, 3),
                    "cluster_distribution": dict(cluster_counts),
                }
            )

        # Sort by purity (lowest first - most problematic)
        return sorted(folder_metrics, key=lambda x: x["purity"])

    def _calculate_cluster_metrics(
        self,
        cluster_folder_matrix: dict[int, dict[str, int]],
        total_docs_by_cluster: dict[int, int],
    ) -> list[dict[str, Any]]:
        """Calculate homogeneity metrics for each cluster."""
        cluster_metrics = []

        for cluster_id, folder_counts in cluster_folder_matrix.items():
            total_docs = total_docs_by_cluster[cluster_id]

            # Find dominant folder
            dominant_folder = max(folder_counts.items(), key=lambda x: x[1])
            dominant_folder_name, dominant_count = dominant_folder

            # Calculate homogeneity (% of cluster from dominant folder)
            homogeneity = dominant_count / total_docs if total_docs > 0 else 0

            # Calculate diversity (number of folders this cluster draws from)
            num_folders = len(folder_counts)

            # Calculate entropy (measure of how diverse the cluster is)
            cluster_entropy = 0.0
            if total_docs > 0:
                for count in folder_counts.values():
                    if count > 0:
                        prob = count / total_docs
                        cluster_entropy -= prob * np.log2(prob)

            cluster_metrics.append(
                {
                    "cluster_id": cluster_id,
                    "total_documents": total_docs,
                    "dominant_folder": dominant_folder_name,
                    "dominant_folder_docs": dominant_count,
                    "homogeneity": round(homogeneity, 3),
                    "num_folders_represented": num_folders,
                    "entropy": round(cluster_entropy, 3),
                    "folder_distribution": dict(folder_counts),
                }
            )

        # Sort by homogeneity (lowest first - most problematic)
        return sorted(cluster_metrics, key=lambda x: x["homogeneity"])

    def _calculate_overall_alignment(
        self,
        folder_cluster_matrix: dict[str, dict[int, int]],
        cluster_folder_matrix: dict[int, dict[str, int]],
        total_docs_by_folder: dict[str, int],
        total_docs_by_cluster: dict[int, int],
    ) -> dict[str, Any]:
        """Calculate overall alignment metrics."""
        total_docs = sum(total_docs_by_folder.values())

        # Calculate weighted average purity and homogeneity
        weighted_purity = 0.0
        weighted_homogeneity = 0.0

        for folder, cluster_counts in folder_cluster_matrix.items():
            folder_total = total_docs_by_folder[folder]
            dominant_count = max(cluster_counts.values()) if cluster_counts else 0
            purity = dominant_count / folder_total if folder_total > 0 else 0
            weight = folder_total / total_docs
            weighted_purity += purity * weight

        for cluster_id, folder_counts in cluster_folder_matrix.items():
            cluster_total = total_docs_by_cluster[cluster_id]
            dominant_count = max(folder_counts.values()) if folder_counts else 0
            homogeneity = dominant_count / cluster_total if cluster_total > 0 else 0
            weight = cluster_total / total_docs
            weighted_homogeneity += homogeneity * weight

        # Calculate alignment score (harmonic mean of purity and homogeneity)
        alignment_score = 0.0
        if weighted_purity > 0 and weighted_homogeneity > 0:
            alignment_score = (
                2
                * (weighted_purity * weighted_homogeneity)
                / (weighted_purity + weighted_homogeneity)
            )

        return {
            "weighted_average_purity": round(weighted_purity, 3),
            "weighted_average_homogeneity": round(weighted_homogeneity, 3),
            "alignment_score": round(alignment_score, 3),
            "alignment_quality": self._get_alignment_quality(alignment_score),
            "total_unique_folders": len(folder_cluster_matrix),
            "total_unique_clusters": len(cluster_folder_matrix),
        }

    def _get_alignment_quality(self, alignment_score: float) -> str:
        """Get qualitative assessment of alignment score."""
        if alignment_score >= 0.8:
            return "excellent"
        elif alignment_score >= 0.6:
            return "good"
        elif alignment_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _identify_misaligned_documents(
        self, clusters: dict[str, Any], folder_cluster_matrix: dict[str, dict[int, int]]
    ) -> list[dict[str, Any]]:
        """Identify documents that might be better placed in different folders."""
        misaligned_docs = []

        # For each document, check if it's in a minority within its folder
        for cluster_id_str, cluster_info in clusters.items():
            cluster_id = int(cluster_id_str)
            documents = cluster_info.get("documents", [])

            for doc in documents:
                directory = doc.get("directory", "root")
                if directory == "." or directory == "":
                    directory = "root"

                filepath = doc.get("filepath", "")

                # Check if this document is part of the dominant cluster for its folder
                folder_clusters = folder_cluster_matrix.get(directory, {})
                if not folder_clusters:
                    continue

                dominant_cluster_id = max(folder_clusters.items(), key=lambda x: x[1])[
                    0
                ]
                total_docs_in_folder = sum(folder_clusters.values())
                docs_in_current_cluster = folder_clusters.get(cluster_id, 0)

                # Consider document misaligned if it's in a minority cluster for its folder
                minority_threshold = (
                    0.3  # Less than 30% of folder's docs in this cluster
                )
                if total_docs_in_folder > 1:  # Only for folders with multiple docs
                    alignment_ratio = docs_in_current_cluster / total_docs_in_folder
                    if (
                        alignment_ratio < minority_threshold
                        and cluster_id != dominant_cluster_id
                    ):
                        misaligned_docs.append(
                            {
                                "filepath": filepath,
                                "current_folder": directory,
                                "current_cluster": cluster_id,
                                "folder_dominant_cluster": dominant_cluster_id,
                                "alignment_ratio": round(alignment_ratio, 3),
                                "suggestion": f"Consider moving to cluster {dominant_cluster_id} majority",
                            }
                        )

        # Sort by alignment ratio (most misaligned first)
        return sorted(misaligned_docs, key=lambda x: x["alignment_ratio"])

    def visualize_folder_cluster_comparison(
        self,
        alignment_data: dict[str, Any] | None = None,
        output_file: str = "folder_cluster_comparison.png",
    ) -> str | None:
        """Create visualization comparing folder structure to semantic clusters.

        Args:
            alignment_data: Optional existing alignment data to reuse
            output_file: Output file path

        Returns:
            Output file path if successful, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Skipping visualization.")
            return None

        try:
            # Get alignment data if not provided
            if alignment_data is None:
                alignment_data = self.analyze_folder_cluster_alignment()
                if "error" in alignment_data:
                    self.logger.error(
                        f"Failed to get alignment data: {alignment_data['error']}"
                    )
                    return None

            # Extract data for visualization
            folder_metrics = alignment_data.get("folder_metrics", [])
            cluster_metrics = alignment_data.get("cluster_metrics", [])
            overall = alignment_data.get("overall_alignment", {})
            distribution_matrix = alignment_data.get("distribution_matrix", {})

            if not folder_metrics or not cluster_metrics:
                self.logger.warning("No data available for visualization")
                return None

            # Create comprehensive visualization with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                "Folder-Cluster Alignment Analysis", fontsize=16, fontweight="bold"
            )

            # 1. Folder Purity Distribution (top left)
            self._plot_folder_purity(axes[0, 0], folder_metrics)

            # 2. Cluster Homogeneity Distribution (top right)
            self._plot_cluster_homogeneity(axes[0, 1], cluster_metrics)

            # 3. Folder-Cluster Heatmap (bottom left)
            self._plot_alignment_heatmap(axes[1, 0], distribution_matrix)

            # 4. Alignment Metrics Summary (bottom right)
            self._plot_alignment_summary(
                axes[1, 1], overall, alignment_data["metadata"]
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2)
            self.logger.info(
                f"Folder-cluster comparison visualization saved to: {output_file}"
            )
            return output_file

        except Exception as e:
            self.logger.error(f"Error creating folder-cluster visualization: {e}")
            return None

    def _plot_folder_purity(self, ax, folder_metrics: list[dict[str, Any]]) -> None:
        """Plot folder purity distribution."""
        if not folder_metrics:
            ax.text(0.5, 0.5, "No folder data available", ha="center", va="center")
            ax.set_title("Folder Purity Distribution")
            return

        # Sort by purity ascending (lowest first) and limit to top 10 for readability
        sorted_folders = sorted(folder_metrics, key=lambda x: x["purity"])
        top_folders = sorted_folders[:10]
        folder_names = [
            f"{m['folder'][:15]}..." if len(m["folder"]) > 15 else m["folder"]
            for m in top_folders
        ]
        purities = [m["purity"] for m in top_folders]

        # Color bars based on purity level
        colors = [
            "#ff4444" if p < 0.5 else "#ffaa00" if p < 0.8 else "#44aa44"
            for p in purities
        ]

        # Use reversed range so lowest purity (first in list) appears at TOP of chart
        y_positions = list(range(len(folder_names)))[::-1]
        bars = ax.barh(y_positions, purities, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, purity in zip(bars, purities, strict=False):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{purity:.2f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(folder_names, fontsize=9)
        ax.set_xlabel("Purity Score (1.0 = all docs in same cluster)")
        ax.set_title("Folder Purity Distribution\n(Lowest purity first)", fontsize=12)
        ax.set_xlim(0, 1.1)
        ax.grid(axis="x", alpha=0.3)

    def _plot_cluster_homogeneity(
        self, ax, cluster_metrics: list[dict[str, Any]]
    ) -> None:
        """Plot cluster homogeneity distribution."""
        if not cluster_metrics:
            ax.text(0.5, 0.5, "No cluster data available", ha="center", va="center")
            ax.set_title("Cluster Homogeneity Distribution")
            return

        # Sort by homogeneity ascending (lowest first) and limit to top 10 for readability
        sorted_clusters = sorted(cluster_metrics, key=lambda x: x["homogeneity"])
        top_clusters = sorted_clusters[:10]
        cluster_names = [
            f"Cluster {m['cluster_id']} ({m['dominant_folder']})" for m in top_clusters
        ]
        homogeneities = [m["homogeneity"] for m in top_clusters]

        # Color bars based on homogeneity level
        colors = [
            "#ff4444" if h < 0.5 else "#ffaa00" if h < 0.8 else "#44aa44"
            for h in homogeneities
        ]

        # Use reversed range so lowest homogeneity (first in list) appears at TOP of chart
        y_positions = list(range(len(cluster_names)))[::-1]
        bars = ax.barh(y_positions, homogeneities, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, homogeneity in zip(bars, homogeneities, strict=False):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{homogeneity:.2f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(cluster_names, fontsize=9)
        ax.set_xlabel("Homogeneity Score (1.0 = all docs from same folder)")
        ax.set_title(
            "Cluster Homogeneity Distribution\n(Lowest homogeneity first)", fontsize=12
        )
        ax.set_xlim(0, 1.1)
        ax.grid(axis="x", alpha=0.3)

    def _plot_alignment_heatmap(self, ax, distribution_matrix: dict[str, Any]) -> None:
        """Plot folder-cluster alignment as a heatmap."""
        folder_to_cluster = distribution_matrix.get("folder_to_cluster", {})

        if not folder_to_cluster:
            ax.text(
                0.5, 0.5, "No distribution data available", ha="center", va="center"
            )
            ax.set_title("Folder-Cluster Distribution Heatmap")
            return

        # Prepare data for heatmap
        folders = list(folder_to_cluster.keys())
        all_clusters = set()
        for cluster_dict in folder_to_cluster.values():
            all_clusters.update(cluster_dict.keys())
        clusters = sorted(list(all_clusters))

        # Create matrix
        matrix = np.zeros((len(folders), len(clusters)))
        for i, folder in enumerate(folders):
            for j, cluster in enumerate(clusters):
                matrix[i, j] = folder_to_cluster[folder].get(cluster, 0)

        # Normalize by row (folder) to show proportions
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / row_sums

        # Create heatmap
        im = ax.imshow(matrix_normalized, cmap="YlOrRd", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
        ax.set_yticks(range(len(folders)))
        ax.set_yticklabels(
            [f[:20] + "..." if len(f) > 20 else f for f in folders], fontsize=8
        )

        # Add colorbar
        if plt is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Proportion of folder in cluster", fontsize=9)

        ax.set_xlabel("Clusters")
        ax.set_ylabel("Folders")
        ax.set_title("Folder-Cluster Distribution\n(Normalized by folder)", fontsize=12)

    def _plot_alignment_summary(
        self, ax, overall: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        """Plot overall alignment metrics summary."""
        ax.axis("off")  # Turn off axis for text display

        # Overall metrics
        alignment_score = overall.get("alignment_score", 0)
        alignment_quality = overall.get("alignment_quality", "unknown")
        purity = overall.get("weighted_average_purity", 0)
        homogeneity = overall.get("weighted_average_homogeneity", 0)

        # Metadata
        total_folders = metadata.get("total_folders", 0)
        total_clusters = metadata.get("total_clusters", 0)
        total_docs = metadata.get("total_documents", 0)

        # Create summary text
        summary_text = f"""
ALIGNMENT SUMMARY

Overall Alignment Score: {alignment_score:.3f}
Quality Assessment: {alignment_quality.upper()}

Detailed Metrics:
• Weighted Avg Purity: {purity:.3f}
• Weighted Avg Homogeneity: {homogeneity:.3f}

Structure Overview:
• Total Folders: {total_folders}
• Total Clusters: {total_clusters}
• Total Documents: {total_docs}

Interpretation:
• Purity: How focused each folder is
• Homogeneity: How unified each cluster is
• Alignment: Overall folder↔cluster match
        """

        # Color code the quality assessment
        color_map = {
            "excellent": "#44aa44",
            "good": "#88aa44",
            "fair": "#aaaa44",
            "poor": "#aa4444",
        }
        text_color = color_map.get(alignment_quality.lower(), "#000000")

        ax.text(
            0.05,
            0.95,
            summary_text.strip(),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Add a colored box for the quality assessment
        if MATPLOTLIB_AVAILABLE and patches is not None:
            quality_y = 0.85
            ax.add_patch(
                patches.Rectangle(
                    (0.35, quality_y - 0.02),
                    0.25,
                    0.04,
                    transform=ax.transAxes,
                    facecolor=text_color,
                    alpha=0.3,
                )
            )

        ax.set_title("Alignment Quality Assessment", fontsize=12, fontweight="bold")

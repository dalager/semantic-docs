"""Semantic Documentation Analysis System.

A comprehensive system for semantic document analysis, redundancy detection,
and intelligent placement suggestions for documentation workflows.
"""

__version__ = "1.0.0"
__author__ = "Christian Dalager"

from semantic_docs.config.settings import SemanticConfig, load_config
from semantic_docs.engines.cluster_engine import DocumentClusterEngine
from semantic_docs.engines.drift_detector import DriftDetector
from semantic_docs.engines.incremental_indexer import IncrementalIndexer
from semantic_docs.engines.semantic_engine import SemanticEngine
from semantic_docs.integrations.claude_hooks import ClaudeCodeHooks

__all__ = [
    "SemanticEngine",
    "IncrementalIndexer",
    "DocumentClusterEngine",
    "DriftDetector",
    "SemanticConfig",
    "load_config",
    "ClaudeCodeHooks",
]

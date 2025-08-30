# Semantic Documentation Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 📖 **Read the blog post**: [Building a Semantic Documentation System](/semantic-docs-blogpost.md) - Learn about the architecture and design decisions behind this project.

AI-powered semantic analysis system for project documentation using ChromaDB embeddings, document clustering, and drift detection to maintain documentation quality and organization at scale.

## Key Features

- **Semantic Search**: Natural language queries across all documentation
- **Redundancy Detection**: Real-time identification of duplicate content  
- **Drift Analysis**: Monitor and prevent documentation entropy
- **Claude Code Integration**: Seamless IDE workflow integration
- **Cluster Visualization**: Visual insights into documentation structure

## Quick Start

```bash
# Install
pip install -e "./semantic-docs/[cli,clustering]"

# Configure OpenAI API
export OPENAI_API_KEY="your-api-key-here"

# Index documentation
semantic-index

# Search
semantic-docs search "authentication setup"

# Check health
semantic-cluster health
```

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API Key
- Git repository with markdown documentation

### Setup

```bash
# Install with all features
pip install -e "./semantic-docs/[dev,cli,clustering]"

# Configure API key
export OPENAI_API_KEY="your-api-key-here"

# Index documentation
semantic-index

# Verify installation
semantic-validate health
```

## Command Line Tools

### `semantic-docs` - Semantic Search

```bash
# Natural language search
semantic-docs search "how to configure authentication"
semantic-docs search "testing patterns with mocks"

# Advanced search options
semantic-docs --max-results 10 --threshold 0.7 search "API security"
semantic-docs --output json search "database setup" | jq '.results[0].file_path'

# Find similar documents
semantic-docs similar -f docs/new-feature.md
```

### `semantic-cluster` - Clustering & Drift Detection

```bash
# Health check
semantic-cluster health
# Output: Health Score: 72.4% - ATTENTION NEEDED

# Full analysis
semantic-cluster analyze
semantic-cluster analyze --output json --save cluster-report.json

# Reorganization suggestions
semantic-cluster suggest-reorg

# Visualization
semantic-cluster visualize -o document_clusters.png

# Folder-cluster alignment
semantic-cluster compare-structure --visualize
```

**Key Metrics:**
- **Folder Purity**: Document focus within folders (1.0 = perfect)
- **Cluster Homogeneity**: Cluster unity (1.0 = perfect)  
- **Alignment Score**: Overall structure quality (>0.8 excellent, >0.6 good)

### `semantic-validate` - Document Validation

```bash
# Validate new document
semantic-validate validate -f docs/new-guide.md

# Batch validation
semantic-validate batch docs/file1.md docs/file2.md docs/file3.md

# Watch directory
semantic-validate watch -d docs/ &

# System health
semantic-validate health
```

### `semantic-index` - Index Management

```bash
# Initial indexing
semantic-index

# Rebuild index
semantic-index --rebuild

# Dry run
semantic-index --dry-run --limit 10
```

## Python API

```python
from semantic_docs import SemanticEngine, load_config
from semantic_docs.engines.cluster_engine import DocumentClusterEngine
from semantic_docs.engines.drift_detector import DriftDetector

# Initialize
config = load_config()
engine = SemanticEngine(config)

# Search
results = engine.find_similar_documents(
    "authentication and authorization setup",
    max_results=10,
    similarity_threshold=0.7
)

# Validation
redundancy = engine.detect_redundancy("Your new document content here")
placement = engine.suggest_placement("New feature documentation")

# Clustering
cluster_engine = DocumentClusterEngine(config)
analysis = cluster_engine.analyze_clustering()

# Drift detection
drift_detector = DriftDetector(config)
health = drift_detector.calculate_health_score()
```

## Workflow Examples

### Writing New Documentation

```bash
# 1. Research existing content
semantic-docs search "authentication setup workflow"

# 2. Validate new document
semantic-validate validate -f docs/authentication/new-oauth-guide.md

# 3. Check organizational impact
semantic-cluster health
```

### CI/CD Integration

```yaml
# .github/workflows/documentation-health.yml
name: Documentation Health Check
on: [push, pull_request]

jobs:
  doc-health:
    steps:
      - name: Check Documentation Organization
        run: |
          semantic-cluster health --output json > health-report.json
          HEALTH_SCORE=$(cat health-report.json | jq '.health_score')
          
          if (( $(echo "$HEALTH_SCORE < 0.7" | bc -l) )); then
            echo "Documentation health score below threshold: $HEALTH_SCORE"
            semantic-cluster suggest-reorg
            exit 1
          fi
```

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional
export SEMANTIC_MAX_RESULTS="10"
export SEMANTIC_REDUNDANCY_THRESHOLD="0.85"
export SEMANTIC_PLACEMENT_THRESHOLD="0.30"
export SEMANTIC_CLUSTER_METHOD="kmeans"
export SEMANTIC_CHROMA_PATH="./chroma_db"
```

### Configuration File

```yaml
# semantic-docs.yaml
openai_api_key: "sk-your-key-here"
redundancy_threshold: 0.85
cluster_method: "kmeans"
entropy_threshold: 0.7
chroma_path: "./chroma_db"
```

## Performance

- **Search**: 0.5-1.0s average response time
- **Validation**: <2.0s per document
- **Clustering**: 3-5s for ~100 documents
- **Indexing**: 1-2 documents/second
- **Memory**: ~200MB steady state

## Troubleshooting

### Common Issues

**OpenAI API Key Not Found**
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

**ChromaDB Collection Not Found**
```bash
semantic-index --rebuild
```

**Clustering Analysis Fails**
```bash
pip install -e "./semantic-docs/[clustering]"
semantic-cluster analyze --clusters 3 --method kmeans
```

### Debug Commands

```bash
# Test integration
venv/bin/python -m semantic_docs.integrations.claude_hooks test

# Check configuration
venv/bin/python -c "from semantic_docs import load_config; print(load_config())"

# Performance profiling
time semantic-cluster analyze
```

## Project Structure

```
semantic_docs/
├── cli/                     # Command-line interfaces
├── config/                  # Configuration management
├── engines/                 # Core processing engines
│   ├── cluster_engine.py    # Document clustering
│   ├── drift_detector.py    # Drift detection
│   ├── incremental_indexer.py  # Incremental indexing
│   └── semantic_engine.py  # Main semantic processing
├── integrations/            # External integrations
└── services/                # Support services
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
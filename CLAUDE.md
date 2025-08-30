# Semantic Documentation Analysis System

A comprehensive AI-powered semantic document analysis tool with ChromaDB and OpenAI embeddings for intelligent documentation workflows.

## Overview

This project provides semantic search, redundancy detection, drift analysis, and intelligent placement suggestions for markdown documentation. It uses vector embeddings to understand document content semantically rather than just through keyword matching.

## Setup

### Environment

The project uses uv for dependency management. Set up the development environment:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment and install dependencies  
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,cli,clustering]"
```

### Required Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"

# Optional configuration (uses defaults if not set)
export SEMANTIC_CHROMA_PATH="./chroma_db"
export SEMANTIC_COLLECTION_NAME="documentation"
export SEMANTIC_REDUNDANCY_THRESHOLD="0.8"
export SEMANTIC_PLACEMENT_THRESHOLD="0.7"
export SEMANTIC_MAX_RESULTS="10"
```

## Development Commands

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Run type checking  
make type-check

# Build package
make build

# Clean build artifacts
make clean
```

## CLI Commands

### Semantic Search
```bash
# Search documentation with natural language
semantic-docs search "authentication patterns"
semantic-docs search "API configuration" --max-results 5 --threshold 0.5

# Find similar documents
semantic-docs similar -f docs/architecture.md
semantic-docs similar -f my-doc.md --output json

# Get collection statistics
semantic-docs stats
```

### Document Indexing
```bash
# Index all markdown files
semantic-index

# Rebuild index from scratch
semantic-index --rebuild

# Test indexing (dry run)
semantic-index --dry-run --limit 10
```

### Document Validation
```bash
# Validate single document
semantic-validate validate -f docs/new-feature.md

# Batch validate multiple files
semantic-validate batch -f docs/file1.md docs/file2.md

# Health check
semantic-validate health

# Watch directory for changes
semantic-validate watch -d docs/
```

### Clustering Analysis
```bash
# Analyze document clustering
semantic-cluster analyze

# Check documentation health
semantic-cluster health

# Get reorganization suggestions
semantic-cluster suggest-reorg

# Create cluster visualization
semantic-cluster visualize -o clusters.png

# Analyze label distribution
semantic-cluster analyze-labels --suggestions

# Compare folder structure with semantic clusters
semantic-cluster compare-structure --visualize
```

## Core Features

### 1. Semantic Search Engine (`semantic_docs.engines.semantic_engine`)
- Vector-based document similarity search
- Natural language query processing
- Redundancy detection with configurable thresholds
- Intelligent document placement suggestions

### 2. Incremental Indexer (`semantic_docs.engines.incremental_indexer`)
- Efficient incremental document indexing
- File system monitoring for automatic updates
- Metadata extraction and embedding generation
- ChromaDB integration for vector storage

### 3. Clustering Engine (`semantic_docs.engines.cluster_engine`)
- K-means and hierarchical clustering
- Document drift detection
- Cluster visualization capabilities
- Directory structure analysis

### 4. Claude Code Integration (`semantic_docs.integrations.claude_hooks`)
- Post-write validation workflows
- Background monitoring
- Integration with Claude Code IDE

### 5. Configuration System (`semantic_docs.config.settings`)
- Environment-based configuration
- JSON config file support
- Validation and health checking

## API Usage

```python
from semantic_docs import SemanticEngine, load_config

# Load configuration
config = load_config()

# Initialize semantic engine
engine = SemanticEngine(config)

# Search documents
results = engine.find_similar_documents("authentication setup")

# Detect redundancy
redundancy = engine.detect_redundancy("New content to check...")

# Get placement suggestions
suggestions = engine.suggest_placement("New document content")
```

## Project Structure

```
semantic_docs/
├── __init__.py              # Main exports
├── cli/                     # Command-line interfaces
│   ├── cluster.py          # Clustering commands
│   ├── indexer.py          # Indexing commands
│   ├── search.py           # Search commands
│   └── validator.py        # Validation commands
├── config/
│   └── settings.py         # Configuration management
├── engines/                # Core processing engines
│   ├── base_indexer.py     # Base indexer class
│   ├── cluster_engine.py   # Document clustering
│   ├── drift_detector.py   # Semantic drift detection
│   ├── incremental_indexer.py  # Incremental indexing
│   └── semantic_engine.py  # Main semantic processing
├── integrations/           # External integrations
│   ├── claude_hooks.py     # Claude Code integration
│   └── pre_commit_validator.py  # Pre-commit hooks
└── services/
    └── ai_summarizer.py    # AI-powered summarization
```

## Configuration

### Default Config File Location
`~/.config/semantic-docs/config.json`

### Sample Configuration
```json
{
  "chroma_path": "./chroma_db",
  "collection_name": "documentation", 
  "redundancy_threshold": 0.8,
  "placement_threshold": 0.7,
  "max_results": 10,
  "api_timeout": 30.0,
  "validation_timeout": 5.0,
  "enable_background_monitoring": false,
  "json_output": false,
  "include_content_preview": true,
  "preview_length": 200
}
```

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=semantic_docs --cov-report=html

# Run specific test modules
pytest tests/test_engines/
pytest tests/test_config/

# Run tests with verbose output
pytest -v
```

## Notes for Claude Code

- The CLI commands are installed as console scripts and available system-wide after installation
- The project uses ChromaDB for vector storage - ensure sufficient disk space for large document collections
- OpenAI API key is required for embedding generation and AI features
- Pre-commit hooks are configured for code quality and semantic document validation
- The project supports both programmatic API usage and CLI interaction
- All configuration can be done via environment variables or JSON config files
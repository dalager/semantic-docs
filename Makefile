.PHONY: install install-dev test lint format type-check clean build help

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  format       Format code"
	@echo "  type-check   Run type checker"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"

install:
	pip install .

install-dev:
	pip install -e ".[dev,cli,clustering]"

test:
	pytest

lint:
	ruff check semantic_docs/ tests/

format:
	ruff format semantic_docs/ tests/
	ruff check --fix semantic_docs/ tests/

type-check:
	mypy semantic_docs/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build
"""AI-powered document summarization and labeling service.

Provides GPT-4 based summarization and content labeling for markdown documents
to enhance search quality and document discovery in the semantic-docs system.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from semantic_docs.config.settings import SemanticConfig


class AISummarizer:
    """AI-powered document summarizer and content labeler using GPT-4."""

    def __init__(self, config: SemanticConfig):
        """Initialize the AI summarizer.

        Args:
            config: SemanticConfig instance with AI settings

        Raises:
            ValueError: If OpenAI API key is not available
        """
        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for AI summarization"
            )

        self.client = OpenAI(api_key=api_key)
        self.config = config
        self.model = config.ai_model

        # Setup logging
        self.logger = logging.getLogger("ai_summarizer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Setup cache if enabled
        self.cache_enabled = config.ai_cache_enabled
        if self.cache_enabled:
            self.cache_dir = Path(config.ai_cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.logger.info(f"AI cache enabled at: {self.cache_dir}")
        else:
            self.cache_dir = None

        self.logger.info(f"AI summarizer initialized with model: {self.model}")

    def _get_cache_key(self, content: str, file_path: str) -> str:
        """Generate cache key based on content and file path.

        Args:
            content: Document content
            file_path: Path to the document

        Returns:
            SHA256 hash as cache key
        """
        # Include model version and config in cache key
        cache_data = {
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "file_path": file_path,
            "model": self.model,
            "max_tokens": self.config.summary_max_tokens,
            "max_labels": self.config.max_content_labels,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> tuple[str, list[str]] | None:
        """Load cached summary and labels if available.

        Args:
            cache_key: Cache key to look up

        Returns:
            Tuple of (summary, labels) if cached, None otherwise
        """
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)

            # Check if cache is still valid (within 30 days)
            cache_age = time.time() - data.get("timestamp", 0)
            if cache_age > 30 * 24 * 60 * 60:  # 30 days
                cache_file.unlink()  # Remove old cache
                return None

            return data.get("summary", ""), data.get("labels", [])

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Invalid cache file {cache_file}: {e}")
            cache_file.unlink()  # Remove invalid cache
            return None

    def _save_to_cache(self, cache_key: str, summary: str, labels: list[str]) -> None:
        """Save summary and labels to cache.

        Args:
            cache_key: Cache key
            summary: Generated summary
            labels: Generated labels
        """
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            "summary": summary,
            "labels": labels,
            "timestamp": time.time(),
            "model": self.model,
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_file}: {e}")

    def _prepare_content(self, content: str) -> str:
        """Prepare content for AI processing by limiting size and cleaning.

        Args:
            content: Raw document content

        Returns:
            Processed content ready for AI analysis
        """
        # Remove excessive whitespace
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
                cleaned_lines.append("")

        cleaned_content = "\n".join(cleaned_lines)

        # Limit content size to avoid token limits (approximate 4000 tokens = 16000 chars)
        if len(cleaned_content) > 16000:
            # Truncate but try to keep complete sections
            truncated = cleaned_content[:16000]
            # Find last complete paragraph/section
            last_double_newline = truncated.rfind("\n\n")
            if last_double_newline > 12000:  # Keep at least 75% of content
                truncated = truncated[:last_double_newline]
            cleaned_content = truncated + "\n\n[Content truncated for analysis]"

        return cleaned_content

    def generate_summary_and_labels(
        self, content: str, file_path: str
    ) -> tuple[str, list[str]]:
        """Generate both summary and labels in a single API call for efficiency.

        Args:
            content: Document content to analyze
            file_path: Path to the document (for context and caching)

        Returns:
            Tuple of (summary, labels)

        Raises:
            Exception: If API call fails after retries
        """
        # Check cache first
        cache_key = self._get_cache_key(content, file_path)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            self.logger.debug(f"Using cached result for {file_path}")
            return cached_result

        # Prepare content for processing
        processed_content = self._prepare_content(content)

        # Create system prompt with clear instructions
        system_prompt = """Analyze technical documentation and return JSON with:
- "summary": 2-3 sentence summary of main purpose and key points
- "labels": 3-5 labels from: API, Testing, Configuration, Architecture, Development, Deployment, Security, Performance, Troubleshooting, Reference, Guide, Tutorial, Database, Monitoring, CI/CD, Documentation, Tools, Setup, Authentication, Observability

Be concise and accurate for developer search."""

        # Create user prompt with document context
        user_prompt = f"""Analyze this technical documentation:

File: {file_path}

Content:
{processed_content}

Provide a JSON response with summary and labels."""

        start_time = time.time()

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=self.config.summary_max_tokens
                + 100,  # Add buffer for JSON structure
                timeout=30,  # 30 second timeout
            )

            # Check for truncated response
            if response.choices[0].finish_reason == "length":
                self.logger.warning(
                    f"Response truncated for {file_path} - increasing token limit"
                )
                # Retry with higher token limit
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=self.config.summary_max_tokens * 2,  # Double the limit
                    timeout=30,
                )

            # Parse response
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                raise ValueError(
                    f"Empty response from AI model. Finish reason: {response.choices[0].finish_reason}"
                )

            result = json.loads(response_content)
            summary = result.get("summary", "").strip()
            labels = result.get("labels", [])

            # Validate and clean results
            if not summary:
                raise ValueError("No summary generated")

            # Ensure labels is a list and limit count
            if not isinstance(labels, list):
                labels = []
            labels = [str(label).strip() for label in labels if str(label).strip()]
            labels = labels[: self.config.max_content_labels]

            # Log successful processing
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"AI processed {file_path}: {len(summary)} chars summary, "
                f"{len(labels)} labels in {elapsed_time:.2f}s (finish_reason: {response.choices[0].finish_reason})"
            )

            # Cache the results
            self._save_to_cache(cache_key, summary, labels)

            return summary, labels

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from AI for {file_path}: {e}")
            raise ValueError(f"AI returned invalid JSON: {e}")

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(
                f"AI processing failed for {file_path} after {elapsed_time:.2f}s: {e}"
            )
            raise

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the AI cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            return {"cache_enabled": False}

        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_enabled": True,
            "cache_dir": str(self.cache_dir),
            "cached_documents": len(cache_files),
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "model": self.model,
        }

    def clear_cache(self) -> bool:
        """Clear the AI cache.

        Returns:
            True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.logger.info("AI cache cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

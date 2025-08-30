"""Post-write validation workflow for Claude Code integration.

Real-time document analysis script optimized for <2s validation time requirement.
Provides redundancy detection and placement suggestions for markdown files written by Claude Code.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent

from semantic_docs.config.settings import load_config, validate_environment
from semantic_docs.integrations.claude_hooks import ClaudeCodeHooks


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging for post-write validation.

    Args:
        verbose: Enable debug logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("post_write_validator")
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


def validate_file(
    file_path: str, config_file: str | None = None, timeout: float | None = None
) -> dict:
    """Validate a single file for redundancy and placement.

    Args:
        file_path: Path to the file to validate
        config_file: Optional path to configuration file
        timeout: Optional timeout override in seconds

    Returns:
        Validation result dictionary
    """
    logger = logging.getLogger("post_write_validator")
    start_time = time.time()

    try:
        # Load configuration
        config = load_config(config_file)

        # Override timeout if provided
        if timeout is not None:
            config.validation_timeout = timeout

        # Initialize hooks
        hooks = ClaudeCodeHooks(config)

        # Validate the file
        result = hooks.post_write_hook(file_path)

        # Add execution metadata
        result["execution_metadata"] = {
            "execution_time": time.time() - start_time,
            "timeout_limit": config.validation_timeout,
            "within_timeout": (time.time() - start_time) <= config.validation_timeout,
            "validator_version": "1.0.0",
        }

        elapsed_time = time.time() - start_time
        logger.info(f"Validation completed in {elapsed_time:.3f}s")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Validation failed after {elapsed_time:.3f}s: {e}")
        return {
            "file_path": file_path,
            "status": "error",
            "error": str(e),
            "execution_metadata": {
                "execution_time": elapsed_time,
                "timeout_limit": timeout or 2.0,
                "within_timeout": elapsed_time <= (timeout or 2.0),
                "validator_version": "1.0.0",
            },
        }


def validate_batch(
    file_paths: list[str], config_file: str | None = None, timeout: float | None = None
) -> dict:
    """Validate multiple files in batch.

    Args:
        file_paths: List of file paths to validate
        config_file: Optional path to configuration file
        timeout: Optional timeout override in seconds

    Returns:
        Batch validation result dictionary
    """
    logger = logging.getLogger("post_write_validator")
    start_time = time.time()

    try:
        # Load configuration
        config = load_config(config_file)

        # Override timeout if provided
        if timeout is not None:
            config.validation_timeout = timeout

        # Initialize hooks
        hooks = ClaudeCodeHooks(config)

        # Validate batch
        result = hooks.batch_validate(file_paths)

        # Add execution metadata
        result["execution_metadata"] = {
            "execution_time": time.time() - start_time,
            "timeout_limit": config.validation_timeout,
            "within_timeout": (time.time() - start_time) <= config.validation_timeout,
            "validator_version": "1.0.0",
            "files_per_second": len(file_paths) / (time.time() - start_time)
            if (time.time() - start_time) > 0
            else 0,
        }

        elapsed_time = time.time() - start_time
        logger.info(
            f"Batch validation of {len(file_paths)} files completed in {elapsed_time:.3f}s"
        )

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Batch validation failed after {elapsed_time:.3f}s: {e}")
        return {
            "total_files": len(file_paths),
            "status": "error",
            "error": str(e),
            "execution_metadata": {
                "execution_time": elapsed_time,
                "timeout_limit": timeout or 2.0,
                "within_timeout": elapsed_time <= (timeout or 2.0),
                "validator_version": "1.0.0",
            },
        }


def check_system_health() -> dict:
    """Check system health and readiness for validation.

    Returns:
        System health status dictionary
    """
    logger = logging.getLogger("post_write_validator")
    start_time = time.time()

    try:
        # Check environment
        env_status = validate_environment()

        # Try to initialize hooks
        config = load_config()
        hooks = ClaudeCodeHooks(config)
        system_status = hooks.get_system_status()

        # Add health check metadata
        health_result = {
            "status": "healthy"
            if env_status["valid"] and system_status["status"] == "healthy"
            else "unhealthy",
            "timestamp": time.time(),
            "environment": env_status,
            "system": system_status,
            "health_check_time": time.time() - start_time,
            "validator_version": "1.0.0",
        }

        logger.info(f"Health check completed: {health_result['status']}")
        return health_result

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "health_check_time": time.time() - start_time,
            "validator_version": "1.0.0",
        }


def watch_directory(
    directory: str, config_file: str | None = None, timeout: float | None = None
) -> None:
    """Watch directory for markdown file changes and validate automatically.

    Args:
        directory: Directory to watch for changes
        config_file: Optional path to configuration file
        timeout: Optional timeout override in seconds
    """
    logger = logging.getLogger("post_write_validator")

    try:
        import watchdog.events
        import watchdog.observers
    except ImportError:
        logger.error(
            "watchdog package required for directory watching. Install with: pip install watchdog"
        )
        sys.exit(1)

    class MarkdownHandler(watchdog.events.FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith((".md", ".markdown")):
                logger.info(f"Detected change in {event.src_path}")
                result = validate_file(event.src_path, config_file, timeout)

                # Print result in JSON format for programmatic consumption
                print(json.dumps(result, indent=2))

                # Print summary for human consumption
                if result["status"] == "success":
                    recommendations = result.get("recommendations", [])
                    high_priority = [
                        r for r in recommendations if r.get("priority") == "high"
                    ]
                    if high_priority:
                        logger.warning(
                            f"High priority issues found in {event.src_path}"
                        )
                        for rec in high_priority:
                            logger.warning(f"  {rec['message']}")
                    else:
                        logger.info(f"No issues found in {event.src_path}")
                else:
                    logger.error(
                        f"Validation failed for {event.src_path}: {result.get('error', 'Unknown error')}"
                    )

    observer = watchdog.observers.Observer()
    observer.schedule(MarkdownHandler(), directory, recursive=True)
    observer.start()

    logger.info(
        f"Watching directory {directory} for markdown file changes. Press Ctrl+C to stop."
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped watching directory")

    observer.join()


def format_output(result: dict, format_type: str = "json") -> str:
    """Format validation result for output.

    Args:
        result: Validation result dictionary
        format_type: Output format ("json", "human", or "claude")

    Returns:
        Formatted output string
    """
    formatters = {
        "json": _format_json_output,
        "human": _format_human_output,
        "claude": _format_claude_output,
    }

    if format_type not in formatters:
        raise ValueError(f"Unknown format type: {format_type}")

    return formatters[format_type](result)


def _format_json_output(result: dict) -> str:
    """Format result as JSON."""
    return json.dumps(result, indent=2)


def _format_human_output(result: dict) -> str:
    """Format result for human-readable output."""
    lines = [f"Status: {result.get('status', 'unknown')}"]

    status = result.get("status")
    if status == "success":
        lines.extend(_format_success_details(result))
    elif status == "error":
        lines.append(f"Error: {result.get('error', 'Unknown error')}")

    return "\n".join(lines)


def _format_success_details(result: dict) -> list[str]:
    """Format success status details for human output."""
    lines = []

    exec_meta = result.get("execution_metadata", {})
    lines.append(f"Validation time: {exec_meta.get('execution_time', 0):.3f}s")

    recommendations = result.get("recommendations", [])
    if recommendations:
        lines.append("\nRecommendations:")
        lines.extend(_format_recommendations_for_human(recommendations))
    else:
        lines.append("No recommendations")

    return lines


def _format_recommendations_for_human(recommendations: list[dict]) -> list[str]:
    """Format recommendations for human-readable output."""
    lines = []
    for rec in recommendations:
        priority = rec.get("priority", "info").upper()
        message = rec.get("message", "")
        lines.append(f"  [{priority}] {message}")
    return lines


def _format_claude_output(result: dict) -> str:
    """Format result specifically for Claude Code consumption."""
    claude_result = {
        "validation_status": result.get("status"),
        "file_path": result.get("file_path"),
        "recommendations": [],
    }

    if result.get("status") == "success":
        recommendations = result.get("recommendations", [])
        claude_result["recommendations"] = _format_claude_recommendations(
            recommendations
        )

    return json.dumps(claude_result, indent=2)


def _format_claude_recommendations(recommendations: list[dict]) -> list[dict]:
    """Format recommendations for Claude Code format."""
    claude_recs = []

    for rec in recommendations:
        claude_rec = {
            "type": rec.get("type"),
            "severity": rec.get("priority"),
            "message": rec.get("message"),
            "action": rec.get("action"),
        }

        # Add optional fields if present
        if "similar_files" in rec:
            claude_rec["related_files"] = rec["similar_files"]
        if "suggested_directory" in rec:
            claude_rec["suggested_location"] = rec["suggested_directory"]

        claude_recs.append(claude_rec)

    return claude_recs


def main():
    """Main entry point for post-write validation workflow."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    if not args.quiet:
        setup_logging(args.verbose)

    logger = logging.getLogger("post_write_validator")

    try:
        result = _execute_command(args, parser)
        _handle_exit_code(result)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        if not args.quiet:
            logger.error(f"Validation workflow failed: {e}")
        sys.exit(1)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Post-write validation workflow for Claude Code integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python scripts/post_write_validator.py validate -f docs/new-feature.md

  # Validate multiple files
  python scripts/post_write_validator.py batch -f docs/file1.md docs/file2.md

  # Check system health
  python scripts/post_write_validator.py health

  # Watch directory for changes
  python scripts/post_write_validator.py watch -d docs/

  # Use custom timeout
  python scripts/post_write_validator.py validate -f docs/test.md -t 1.0
        """,
    )

    _add_parser_arguments(parser)
    return parser


def _add_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all command line arguments to the parser."""
    parser.add_argument(
        "command",
        choices=["validate", "batch", "health", "watch"],
        help="Validation command to execute",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        help="File path(s) to validate (can be specified multiple times)",
    )
    parser.add_argument("-d", "--directory", help="Directory to watch for changes")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument(
        "-t", "--timeout", type=float, help="Validation timeout in seconds"
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["json", "human", "claude"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress logging output"
    )


def _execute_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Execute the specified command and return the result."""
    command_handlers = {
        "validate": _handle_validate_command,
        "batch": _handle_batch_command,
        "health": _handle_health_command,
        "watch": _handle_watch_command,
    }

    handler = command_handlers[args.command]
    return handler(args, parser)


def _handle_validate_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
    """Handle the validate command."""
    if not args.files or len(args.files) != 1:
        parser.error("validate command requires exactly one file (-f)")

    result = validate_file(args.files[0], args.config, args.timeout)
    print(format_output(result, args.output))
    return result


def _handle_batch_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
    """Handle the batch command."""
    if not args.files:
        parser.error("batch command requires at least one file (-f)")

    result = validate_batch(args.files, args.config, args.timeout)
    print(format_output(result, args.output))
    return result


def _handle_health_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
    """Handle the health command."""
    result = check_system_health()
    print(format_output(result, args.output))
    return result


def _handle_watch_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
    """Handle the watch command."""
    if not args.directory:
        parser.error("watch command requires directory (-d)")

    if not Path(args.directory).exists():
        parser.error(f"Directory does not exist: {args.directory}")

    watch_directory(args.directory, args.config, args.timeout)
    # Watch command doesn't return a result dict
    return {"status": "success"}


def _handle_exit_code(result: dict) -> None:
    """Handle exit code based on result."""
    if isinstance(result, dict):
        sys.exit(0 if result.get("status") in ["success", "healthy"] else 1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

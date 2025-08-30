#!/usr/bin/env python3
"""Pre-commit wrapper for semantic document validation.

Handles multiple files passed by pre-commit and validates each one individually.
"""

import subprocess
import sys


def main():
    """Validate each markdown file passed as arguments."""
    if len(sys.argv) < 2:
        print("No files to validate")
        sys.exit(0)

    files_to_validate = sys.argv[1:]
    failed_files = []

    for file_path in files_to_validate:
        # Only process markdown files
        if not file_path.lower().endswith((".md", ".markdown")):
            continue

        # Run validation
        try:
            result = subprocess.run(
                [
                    "venv/bin/python",
                    "-m",
                    "semantic_docs.cli.validator",
                    "validate",
                    "-f",
                    file_path,
                    "--output",
                    "human",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                failed_files.append(file_path)
                print(f"❌ Validation failed for {file_path}")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
            else:
                print(f"✅ Validation passed for {file_path}")

        except Exception as e:
            failed_files.append(file_path)
            print(f"❌ Error validating {file_path}: {e}")

    if failed_files:
        print(f"\n❌ Validation failed for {len(failed_files)} file(s)")
        sys.exit(1)
    else:
        print(
            f"\n✅ All {len(files_to_validate)} markdown files validated successfully"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

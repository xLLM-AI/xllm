#!/usr/bin/env bash
set -euo pipefail

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "$1 is not installed, please run:"
        echo "  pip install pre-commit"
        echo "  pre-commit install"
        exit 1
    fi
}

check_command pre-commit

if [[ "${1:-}" == "ci" ]]; then
    pre-commit run --all-files --hook-stage manual --show-diff-on-failure
else
    pre-commit run --all-files
fi

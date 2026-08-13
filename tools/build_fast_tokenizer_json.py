#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build a fast tokenizer.json from local HuggingFace BPE tokenizer files.

Some DiT checkpoints ship CLIP/GPT-style BPE files (`vocab.json` and
`merges.txt`) without a generated `tokenizer.json`. xLLM's fast tokenizer
runtime consumes `tokenizer.json`, so run this tool once during model
preparation instead of building the tokenizer inside the C++ server process.

Usage:
    python tools/build_fast_tokenizer_json.py \
        --tokenizer-dir /models/FLUX.1-dev/tokenizer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from scripts.logger import logger

REQUIRED_FILES = ["tokenizer_config.json", "vocab.json", "merges.txt"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tokenizer.json from local vocab.json and merges.txt files.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Directory containing tokenizer_config.json, vocab.json, and merges.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write tokenizer.json. Defaults to --tokenizer-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing tokenizer.json.",
    )
    return parser.parse_args()


def _validate_input(tokenizer_dir: Path) -> None:
    if not tokenizer_dir.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {tokenizer_dir}")

    missing_files = [name for name in REQUIRED_FILES if not (tokenizer_dir / name).exists()]
    if missing_files:
        raise ValueError(f"Missing required tokenizer files under {tokenizer_dir}: {', '.join(missing_files)}")


def main() -> None:
    args = _parse_args()
    tokenizer_dir = args.tokenizer_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else tokenizer_dir

    _validate_input(tokenizer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_json_path = output_dir / "tokenizer.json"
    if tokenizer_json_path.exists() and not args.overwrite:
        raise ValueError(f"{tokenizer_json_path} already exists. Use --overwrite to replace it.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.save_pretrained(output_dir)

    if not tokenizer_json_path.exists():
        raise RuntimeError(f"Failed to create {tokenizer_json_path}")

    logger.info(f"Created {tokenizer_json_path}")


if __name__ == "__main__":
    main()

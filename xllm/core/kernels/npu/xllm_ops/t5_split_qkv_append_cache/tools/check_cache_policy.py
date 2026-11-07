# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject cache-policy experiments absent from the pinned 910_93 implementation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from scripts.logger import logger

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--cann-root", type=Path, required=True)
args = parser.parse_args()
pattern = re.compile(r"SetL2CacheHint\s*\(\s*AscendC::CacheMode::CACHE_MODE_PERSISTENT\s*\)")
uses = [
    str(p)
    for p in (args.source_root / "op_kernel").rglob("*")
    if p.is_file() and p.suffix in (".h", ".cpp") and pattern.search(p.read_text(encoding="utf-8"))
]
if uses:
    header = args.cann_root / "aarch64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h"
    text = header.read_text(encoding="utf-8")
    start = text.index("#else\ntemplate<class T, CacheRwMode rwMode")
    end = text.index("return addr;", start)
    body = text[start:end]
    if "__NPU_ARCH__ == 2201" in body and "CACHE_MODE_PERSISTENT" not in body:
        parser.exit(
            2,
            "Unsupported cache-policy experiment: target 2201 L2CacheAlter treats PERSISTENT as NORMAL; inspect target implementation before benchmarking.\n",
        )
    parser.exit(2, "Unrecognized PERSISTENT implementation: review target semantics before enabling this experiment.\n")
logger.info("Cache policy guard: no unsupported PERSISTENT hint requested")

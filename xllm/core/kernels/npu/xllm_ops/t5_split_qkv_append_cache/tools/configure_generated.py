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

"""Configure the official msopgen scaffold, fail if expected knobs disappear."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from scripts.logger import logger

root = Path(sys.argv[1]).resolve()
path = root / "CMakePresets.json"
config = json.loads(path.read_text(encoding="utf-8"))
changed = set()
for preset in config.get("configurePresets", []):
    variables = preset.get("cacheVariables", {})
    for key, value in {
        "ASCEND_COMPUTE_UNIT": "ascend910_93",
        "vendor_name": "t5qkv",
        "ENABLE_SOURCE_PACKAGE": False,
        "ENABLE_BINARY_PACKAGE": True,
    }.items():
        if key in variables:
            old = variables[key]
            if isinstance(old, dict):
                old["value"] = value
            else:
                variables[key] = value
            changed.add(key)
if "ASCEND_COMPUTE_UNIT" not in changed or "vendor_name" not in changed:
    raise RuntimeError("Unexpected msopgen scaffold; inspect CMakePresets.json before adapting")
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
logger.info("Configured: %s", sorted(changed))

# Make the standalone project root available before generated subdirectories.
cmake_path = root / "CMakeLists.txt"
cmake_text = cmake_path.read_text(encoding="utf-8")
minimum = re.search(r"cmake_minimum_required\s*\([^)]*\)", cmake_text, re.IGNORECASE)
if minimum is None:
    raise RuntimeError("Unexpected msopgen scaffold: missing cmake_minimum_required")
cmake_text = (
    cmake_text[: minimum.end()] + '\ninclude_directories("${CMAKE_CURRENT_SOURCE_DIR}")' + cmake_text[minimum.end() :]
)
cmake_path.write_text(cmake_text, encoding="utf-8", newline="\n")

# Opbuild compiles host sources in a separate Make invocation. It does not
# inherit the directory's include_directories setting.
host_path = root / "op_host/CMakeLists.txt"
host_text = host_path.read_text(encoding="utf-8")
codegen_calls = list(re.finditer(r"\bnpu_op_code_gen\s*\(", host_text))
if len(codegen_calls) != 1:
    raise RuntimeError("Unexpected msopgen scaffold: expected one npu_op_code_gen call")
position = codegen_calls[0].end()
host_text = host_text[:position] + '\n    COMPILE_OPTIONS "-I${CMAKE_CURRENT_SOURCE_DIR}/.."' + host_text[position:]
host_path.write_text(host_text, encoding="utf-8", newline="\n")

# Kernel packaging also invokes its compiler outside normal CMake targets.
kernel_path = root / "op_kernel/CMakeLists.txt"
kernel_text = kernel_path.read_text(encoding="utf-8")
kernel_calls = list(re.finditer(r"\bnpu_op_kernel_sources\s*\(\s*(\w+)", kernel_text))
if len(kernel_calls) != 1:
    raise RuntimeError("Unexpected msopgen scaffold: expected one npu_op_kernel_sources call")
position = kernel_calls[0].start()
target = kernel_calls[0].group(1)
kernel_text = (
    kernel_text[:position]
    + f'npu_op_kernel_options({target} ALL OPTIONS "-I${{CMAKE_CURRENT_SOURCE_DIR}}/..")\n\n'
    + kernel_text[position:]
)
kernel_path.write_text(kernel_text, encoding="utf-8", newline="\n")

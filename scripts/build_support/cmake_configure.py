# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


CMAKE_CONFIGURE_FINGERPRINT = ".xllm_cmake_configure_fingerprint"

_CMAKE_ENVIRONMENT_KEYS = frozenset(
    {
        "CC",
        "CFLAGS",
        "CXX",
        "CXXFLAGS",
        "DEPENDENCES_ROOT",
        "LDFLAGS",
        "LD_LIBRARY_PATH",
        "PATH",
        "PKG_CONFIG_PATH",
        "TRITON_BINARY_PATH",
    }
)
_CMAKE_ENVIRONMENT_PREFIXES = (
    "ASCEND_",
    "ATB_",
    "CMAKE_",
    "CUDA_",
    "DCU_",
    "ILU_",
    "MACA_",
    "MLU_",
    "MUSA_",
    "NPU_",
    "PYTORCH_",
    "ROCM_",
    "TORCH_",
    "VCPKG_",
    "XLLM_",
)


def get_cmake_configure_fingerprint(
    command: Sequence[str], env: Mapping[str, str]
) -> str:
    """Returns a stable fingerprint for inputs not tracked by Ninja."""
    configure_environment = {
        key: value
        for key, value in sorted(env.items())
        if key in _CMAKE_ENVIRONMENT_KEYS
        or key.startswith(_CMAKE_ENVIRONMENT_PREFIXES)
    }
    payload = json.dumps(
        {
            "command": list(command),
            "environment": configure_environment,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cmake_configure_required(build_dir: str, fingerprint: str) -> bool:
    """Returns whether the existing Ninja build tree must be configured."""
    build_path = Path(build_dir)
    required_paths = (
        build_path / "CMakeCache.txt",
        build_path / "build.ninja",
        build_path / CMAKE_CONFIGURE_FINGERPRINT,
    )
    if not all(path.is_file() for path in required_paths):
        return True

    try:
        recorded_fingerprint = required_paths[-1].read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError):
        return True
    return recorded_fingerprint != fingerprint


def record_cmake_configure_fingerprint(build_dir: str, fingerprint: str) -> None:
    """Records a successful CMake configuration atomically."""
    fingerprint_path = Path(build_dir) / CMAKE_CONFIGURE_FINGERPRINT
    temporary_path = fingerprint_path.with_suffix(".tmp")
    temporary_path.write_text(f"{fingerprint}\n", encoding="utf-8")
    os.replace(temporary_path, fingerprint_path)

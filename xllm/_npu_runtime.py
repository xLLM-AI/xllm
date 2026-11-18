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
# ==============================================================================

"""Configure the custom NPU operators shipped beside the server binary."""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path


def _prepend_env_path(name: str, path: Path) -> None:
    existing = os.environ.get(name)
    paths = [str(path)]
    if existing is not None:
        paths.extend(existing.split(os.pathsep))
    os.environ[name] = os.pathsep.join(dict.fromkeys(paths))


def configure_npu_runtime(binary_path: str) -> Path | None:
    """Select the operator bundle belonging to the binary about to be run.

    The source-tree launcher can select an installed binary or an explicit
    ``--binary-path``. Only inspect that binary's directory, so an unrelated
    launcher's bundled operators cannot override its runtime dependencies.
    Return the validated API library for callers loading an extension in the
    current process; changing LD_LIBRARY_PATH only affects future processes.
    """
    binary = Path(binary_path)
    kernel_vendor = binary.absolute().parent.parent / "xllm_kernel/ascendc_opp/xllm_kernel"
    if kernel_vendor.is_dir():
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", kernel_vendor)

    opp_root = binary.resolve().parent / "opp"
    if not os.path.lexists(opp_root):
        return

    vendor_root = opp_root / "vendors" / "custom_xllm_math"
    library_dir = vendor_root / "op_api" / "lib"
    required_library = library_dir / "libcust_opapi.so"
    if not required_library.is_file():
        raise RuntimeError(f"Incomplete bundled xllm_ops: required library is missing: {required_library}")
    for name in ("op_impl", "op_proto"):
        required_dir = vendor_root / name
        if not required_dir.is_dir() or not any(required_dir.iterdir()):
            raise RuntimeError(f"Incomplete bundled xllm_ops: required directory is missing or empty: {required_dir}")

    _prepend_env_path("LD_LIBRARY_PATH", library_dir)
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_root)
    return required_library


@functools.cache
def preload_npu_ops(library_path: Path) -> ctypes.CDLL:
    """Keep the bundled API loaded before resolving the extension's dependencies.

    glibc retains the library search path from process startup. An absolute
    dlopen selects this bundle before an older library on that search path;
    global visibility also makes its symbols available to operator plugins.
    Keep the handle alive for the lifetime of the process.
    """
    return ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)

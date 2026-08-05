# Copyright 2026 The xLLM Authors.
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

"""Import scaffolding shared by the Python executor tests.

Importing ``xllm.python`` binds the active platform's kernel package, and that
package declares FakeTensor contracts for operators that only exist once the
compiled library is loaded. A bare test runner has neither, so the modules that
exercise layers, attention backends and the executor need a stand-in package.

Installing it once, here, is what keeps the test modules from overwriting each
other's ``sys.modules["xllm.python"]``: whoever ran first used to win, and a
stand-in without ``__path__`` turned every later source-tree import into a mock.

Tests that need the real binding run it in a subprocess, where the operator
schemas are declared first and this stand-in does not apply.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"

# The names layers, attention backends and runners reach through the binding.
_KERNEL_NAMES = (
    "fused_add_rms_norm",
    "prepare_row_parallel_weight",
    "reshape_paged_cache",
    "rms_norm",
    "update_decode_graph_metadata",
)
_COLLECTIVE_NAMES = ("all_gather", "all_gather_variable", "all_reduce_")


def _install_python_package_stub() -> None:
    kernels = types.ModuleType("xllm.python.kernels")
    for name in _KERNEL_NAMES:
        setattr(kernels, name, MagicMock())
    # Returns a bool rather than a truthy mock, so a caller can branch on it.
    kernels.supports_cutlass_moe = MagicMock(return_value=False)

    distributed = types.ModuleType("xllm.python.distributed")
    for name in _COLLECTIVE_NAMES:
        setattr(distributed, name, MagicMock())

    package = types.ModuleType("xllm.python")
    package.__path__ = [str(_PYTHON_ROOT)]
    package.kernels = kernels
    package.distributed = distributed

    sys.modules["xllm.python"] = package
    sys.modules["xllm.python.kernels"] = kernels
    sys.modules["xllm.python.distributed"] = distributed


_install_python_package_stub()

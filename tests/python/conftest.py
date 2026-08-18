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

"""Package stubs for pure-Python tests that run without platform kernels."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"


def _install_python_package_stub() -> None:
    kernels = types.ModuleType("xllm.python.kernels")
    distributed = types.ModuleType("xllm.python.distributed")

    package = types.ModuleType("xllm.python")
    # Keep source submodules importable without executing the real package binding.
    package.__path__ = [str(_PYTHON_ROOT)]
    package.kernels = kernels
    package.distributed = distributed

    sys.modules["xllm.python"] = package
    sys.modules["xllm.python.kernels"] = kernels
    sys.modules["xllm.python.distributed"] = distributed


_install_python_package_stub()

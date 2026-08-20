# Copyright 2025-2026 The xLLM Authors.
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

"""Python-defined model graphs executed by xLLM's C++ worker.

Importing this package is intentionally runtime-neutral. The embedded C++
bootstrap calls :func:`initialize_runtime` after registering native operators
and before importing a Python model module. That call selects the platform
kernel package and publishes it as :mod:`xllm.python.kernels`.

Keeping ordinary package import free of platform and native-operator side
effects lets build tools import leaf DSL modules such as
``xllm.python.kernels_npu.tilelang`` before the xLLM binary exists.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

_runtime_kernels: ModuleType | None = None


def initialize_runtime() -> None:
    """Initialize and publish the active platform's Python kernel package."""

    global _runtime_kernels
    if _runtime_kernels is not None:
        return

    from xllm.python.platform import current_platform

    if current_platform.is_cuda():
        backend_name = "xllm.python.kernels_cuda"
    elif current_platform.is_npu():
        backend_name = "xllm.python.kernels_npu"
    else:
        device_type = current_platform.device_type()
        raise ImportError(
            f"no Python kernel package for platform '{device_type}'; add "
            f"xllm/python/kernels_{device_type}/ with the APIs required by "
            "that platform's supported models"
        )

    backend = importlib.import_module(backend_name)
    backend_initializer = getattr(backend, "_initialize_runtime", None)
    if backend_initializer is not None:
        backend_initializer()

    globals()["kernels"] = backend
    sys.modules[f"{__name__}.kernels"] = backend
    _runtime_kernels = backend


def __getattr__(name: str) -> Any:
    if name == "kernels":
        raise RuntimeError(
            "xllm.python runtime is not initialized; the embedded C++ "
            "bootstrap must call xllm.python.initialize_runtime() before "
            "importing model or layer modules"
        )
    if name in {"get_model_class", "register_model"}:
        registry = importlib.import_module("xllm.python.registry")
        value = getattr(registry, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_model_class", "initialize_runtime", "kernels", "register_model"]

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

"""SDK-free operation and backend discovery metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module

from xllm_kernel.registry import OpContract, PreparedKernel


@dataclass(frozen=True)
class BackendModule:
    module: str
    device: str
    requires: tuple[str, ...]
    implementations: tuple[str, ...] = ()


@dataclass
class Operation:
    contract: OpContract
    backends: tuple[BackendModule, ...]
    prepared: PreparedKernel | None = field(default=None, init=False, repr=False)

    @property
    def implementations(self) -> tuple[str, ...]:
        """Expose CLI choices without importing device backends."""
        return tuple(name for backend in self.backends for name in backend.implementations)

    def get_prepared(self) -> PreparedKernel:
        """Return the startup binding without performing selection."""
        if self.prepared is None:
            raise RuntimeError("call xllm_kernel.initialize before using an operator")
        return self.prepared


_BUILTINS = ("xllm_kernel.ops.rms_norm",)


def load_operations() -> tuple[Operation, ...]:
    """Import operation definitions without importing their device backends."""
    return tuple(import_module(module).OPERATION for module in _BUILTINS)

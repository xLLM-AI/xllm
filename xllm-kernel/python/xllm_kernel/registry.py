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

"""Preparation-only registry. It never participates in model execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

_Kernel = TypeVar("_Kernel", bound=Callable[..., Any])


@dataclass(frozen=True)
class OpContract:
    operation: str
    version: int
    mutates: tuple[str, ...]
    aliases_inputs: bool
    numerical_mode: str


@dataclass(frozen=True)
class KernelSpec:
    name: str
    contract: OpContract
    device: str
    execution_modes: tuple[str, ...]
    input_domain: str
    priority: int = 0
    solution: str = "native"
    variant: str = ""
    binding: str = ""


@dataclass(frozen=True)
class PreparedKernel:
    """Immutable binding; callers retain it for their graph's entire lifetime.

    ``input_domain`` describes the selected operator's domain. These adapters
    does not specialize on shape or cache a plan prepared for a sample tensor.
    Native Torch dispatch remains responsible for tensor argument validation.
    """

    spec: KernelSpec
    function: Callable[..., Any]
    generation: int
    excluded: tuple[tuple[str, str], ...]


class KernelRegistry:
    _instance: KernelRegistry | None = None

    def __init__(self) -> None:
        self._entries: dict[str, tuple[KernelSpec, Callable[..., Any]]] = {}
        self._frozen = False
        self._generation = 0

    @classmethod
    def get(cls) -> KernelRegistry:
        """Return the process registry populated by backend module imports."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, spec: KernelSpec, function: Callable[..., Any]) -> None:
        if self._frozen:
            raise RuntimeError("kernel registry is frozen; rebuild the runtime and graphs to change it")
        if spec.name in self._entries:
            raise ValueError(f"kernel {spec.name!r} is already registered")
        self._entries[spec.name] = (spec, function)
        self._generation += 1

    def specs(self) -> tuple[KernelSpec, ...]:
        """Return registered metadata without exposing mutable registry state."""
        return tuple(spec for spec, _ in self._entries.values())

    def freeze(self) -> None:
        self._frozen = True

    def prepare(
        self,
        contract: OpContract,
        *,
        device: str,
        execution_mode: str,
        implementation: str | None = None,
        solution: str | None = None,
    ) -> PreparedKernel:
        if not self._frozen:
            raise RuntimeError("freeze the kernel registry before preparing a kernel")
        candidates = []
        excluded = []
        for spec, function in sorted(self._entries.values(), key=lambda entry: entry[0].name):
            reason = ""
            if spec.contract != contract:
                reason = "operation contract mismatch"
            elif spec.device != device:
                reason = f"requires device {spec.device}"
            elif execution_mode not in spec.execution_modes:
                reason = f"does not support execution mode {execution_mode}"
            elif solution is not None and spec.solution != solution:
                reason = f"explicit solution is {solution}"
            elif implementation is not None and spec.name != implementation:
                reason = f"explicit implementation is {implementation}"
            if reason:
                excluded.append((spec.name, reason))
                continue
            candidates.append((spec, function))
        if not candidates:
            reasons = "; ".join(f"{name}: {reason}" for name, reason in excluded)
            raise ValueError(
                f"no kernel for {contract.operation} on {device} ({execution_mode}), "
                f"implementation={implementation!r}, solution={solution!r}; {reasons or 'registry is empty'}"
            )
        spec, function = min(candidates, key=lambda entry: (-entry[0].priority, entry[0].name))
        return PreparedKernel(spec, function, self._generation, tuple(excluded))


def register_kernel(
    *,
    name: str,
    contract: OpContract,
    device: str,
    execution_modes: tuple[str, ...],
    input_domain: str,
    solution: str,
    priority: int = 0,
    variant: str = "",
    binding: str = "",
) -> Callable[[_Kernel], _Kernel]:
    """Declare a kernel's metadata and register its original callable.

    Backend modules apply this at module scope after resolving dependencies.
    Importing available backends populates ``KernelRegistry.get()`` before
    initialization freezes it. Returning the callable unchanged preserves its
    identity, signature and execution path; the decorator adds no forward
    wrapper and defines no Torch schema.
    """
    spec = KernelSpec(
        name=name,
        contract=contract,
        device=device,
        execution_modes=execution_modes,
        input_domain=input_domain,
        priority=priority,
        solution=solution,
        variant=variant,
        binding=binding,
    )

    def _register(function: _Kernel) -> _Kernel:
        KernelRegistry.get().register(spec, function)
        return function

    return _register

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
from typing import Any


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
    def __init__(self) -> None:
        self._entries: dict[str, tuple[KernelSpec, Callable[..., Any]]] = {}
        self._frozen = False
        self._generation = 0

    def register(self, spec: KernelSpec, function: Callable[..., Any]) -> None:
        if self._frozen:
            raise RuntimeError("kernel registry is frozen; rebuild the runtime and graphs to change it")
        if spec.name in self._entries:
            raise ValueError(f"kernel {spec.name!r} is already registered")
        self._entries[spec.name] = (spec, function)
        self._generation += 1

    def freeze(self) -> None:
        self._frozen = True

    def prepare(
        self,
        contract: OpContract,
        *,
        device: str,
        execution_mode: str,
        implementation: str | None = None,
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
                f"implementation={implementation!r}; {reasons or 'registry is empty'}"
            )
        spec, function = min(candidates, key=lambda entry: (-entry[0].priority, entry[0].name))
        return PreparedKernel(spec, function, self._generation, tuple(excluded))

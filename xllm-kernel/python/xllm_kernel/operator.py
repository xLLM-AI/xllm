# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Lazy, explicit preparation of an operation's independently owned backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from xllm_kernel.registry import KernelRegistry, KernelSpec, OpContract, PreparedKernel


@dataclass(frozen=True)
class Backend:
    """Separate the kernel solution from its binding and implementation variant."""

    solution: str
    loader: str
    input_domain: str
    execution_modes: tuple[str, ...] = ("eager", "aclgraph")
    variant: str = ""
    binding: str = ""

    def implementation_name(self, operation: str) -> str:
        suffix = f".{self.variant}" if self.variant else ""
        return f"npu.{self.solution}.{operation}{suffix}"


@dataclass(frozen=True)
class Operator:
    """Prepare only the requested provider; never probe or fall back at execution."""

    contract: OpContract
    backends: tuple[Backend, ...]

    @property
    def implementations(self) -> tuple[str, ...]:
        return tuple(backend.implementation_name(self.contract.operation) for backend in self.backends)

    def prepare(self, device: str, mode: str, implementation: str) -> PreparedKernel:
        if device != "npu":
            raise ValueError(f"{self.contract.operation} requires device='npu'")
        names = dict(zip(self.implementations, self.backends, strict=True))
        if len(names) != len(self.backends):
            raise ValueError(f"duplicate implementation names for {self.contract.operation}")
        if implementation not in names:
            raise ValueError(f"unsupported implementation {implementation!r}; choose from {tuple(names)}")
        backend = names[implementation]
        if mode not in backend.execution_modes:
            raise ValueError(f"{implementation} does not support {mode}")
        module, symbol = backend.loader.split(":", 1)
        function = getattr(import_module(module), symbol)()
        if not callable(function):
            raise TypeError(f"{backend.loader} must return a callable")
        registry = KernelRegistry()
        registry.register(
            KernelSpec(
                implementation,
                self.contract,
                device,
                backend.execution_modes,
                backend.input_domain,
                solution=backend.solution,
                variant=backend.variant,
                binding=backend.binding,
            ),
            function,
        )
        registry.freeze()
        return registry.prepare(self.contract, device=device, execution_mode=mode, implementation=implementation)

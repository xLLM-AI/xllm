# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Operation-owned cases consumed by the shared numerical and benchmark tools."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from xllm_kernel.registry import PreparedKernel


@dataclass(frozen=True)
class PreparedCase:
    """Bind inputs and validation without exposing operator arguments to runners."""

    function: Callable[[], Any]
    validate: Callable[[Callable[[], Any]], dict[str, Any]]
    metadata: dict[str, Any]


class KernelCase(Protocol):
    @property
    def case_id(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    def unsupported_reason(self, plan: PreparedKernel) -> str | None: ...

    def prepare(self, plan: PreparedKernel) -> PreparedCase: ...


@dataclass(frozen=True)
class OperationSuite:
    """Declare CLI options, cases and implementation preparation for an operation."""

    operation: str
    implementations: tuple[str, ...]
    add_arguments: Callable[[argparse.ArgumentParser], None]
    cases: Callable[[argparse.Namespace], Iterable[KernelCase]]
    prepare: Callable[[str, str, str], PreparedKernel]

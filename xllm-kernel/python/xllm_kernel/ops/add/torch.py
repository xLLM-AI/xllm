# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""add implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(value: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return torch.add(value, residual)

    return run

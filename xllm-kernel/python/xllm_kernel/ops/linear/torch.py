# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""linear implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight)

    return run

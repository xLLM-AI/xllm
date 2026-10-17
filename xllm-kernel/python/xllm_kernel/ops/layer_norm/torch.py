# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""layer_norm implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.nn.functional.layer_norm(value, (value.shape[-1],), weight, bias, eps)

    return run

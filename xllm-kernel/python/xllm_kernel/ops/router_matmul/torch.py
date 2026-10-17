# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""router_matmul implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = torch.empty((value.shape[0], weight.shape[0]), dtype=torch.float32, device=value.device)
        return torch.addmm(output, value, weight.T, beta=0, alpha=1, out=output)

    return run

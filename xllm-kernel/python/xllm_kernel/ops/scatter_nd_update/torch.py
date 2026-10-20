# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""scatter_nd_update implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    def run(value: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> None:
        value.index_put_(tuple(indices.unbind(-1)), updates, accumulate=False)

    return run

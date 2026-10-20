# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sfa_dcp_remap implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(indices: torch.Tensor, block_size: int, shards: int, rank: int) -> torch.Tensor:
        valid = (indices >= 0) & ((indices // block_size) % shards == rank)
        mapped = (indices // (block_size * shards)) * block_size + indices % block_size
        positions = torch.arange(indices.shape[-1], device=indices.device).expand_as(indices)
        order = torch.where(valid, positions, positions + indices.shape[-1]).argsort(-1)
        return torch.where(valid, mapped, -1).gather(-1, order)

    return run

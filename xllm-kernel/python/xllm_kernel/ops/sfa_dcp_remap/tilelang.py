# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sfa_dcp_remap implementations supplied by tilelang; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("sfa_dcp_remap_out")

    def run(indices: torch.Tensor, block_size: int, shards: int, rank: int) -> torch.Tensor:
        if not 0 < indices.numel() // 2048 <= 256 or indices.shape[-1] != 2048:
            raise ValueError("TileLang remap requires 1..256 rows of topk=2048")
        if block_size <= 0 or not 0 <= rank < shards:
            raise ValueError("invalid shard parameters")
        output = torch.empty_like(indices)
        scratch = torch.empty_like(indices)
        return op(indices, block_size, shards, rank, output, scratch)

    return run

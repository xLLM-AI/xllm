# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""lightning_indexer implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        weights: torch.Tensor,
        query_lengths: torch.Tensor,
        key_lengths: torch.Tensor,
        block_table: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        keys = key_cache[block_table[0].long()].reshape(-1, key_cache.shape[-1])
        scores = torch.einsum("thd,kd->thk", query.float(), keys.float()).relu()
        scores = (scores * weights.float()[..., None]).sum(1)
        positions = torch.arange(keys.shape[0], device=query.device)
        limits = key_lengths[0] - query.shape[0] + torch.arange(query.shape[0], device=query.device) + 1
        scores = scores.masked_fill(positions[None] >= limits[:, None], float("-inf"))
        return scores.topk(topk, -1).indices.to(torch.int32).unsqueeze(1)

    return run

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sparse_flash_attention implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    def run(
        query: torch.Tensor,
        cache: torch.Tensor,
        indices: torch.Tensor,
        query_lengths: torch.Tensor,
        key_lengths: torch.Tensor,
        block_table: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = cache[block_table[0].long()].reshape(-1, cache.shape[-1])
        ropes = key_rope[block_table[0].long()].reshape(-1, key_rope.shape[-1])
        selected = indices[:, 0].long()
        valid = (selected >= 0) & (
            selected < key_lengths[0] - query.shape[0] + torch.arange(query.shape[0], device=query.device)[:, None] + 1
        )
        safe = selected.clamp(min=0)
        values = keys[safe].float()
        logits = (
            torch.einsum("thd,tkd->thk", query.float(), values)
            + torch.einsum("thd,tkd->thk", query_rope.float(), ropes[safe].float())
        ) * scale
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        maximum = logits.max(-1).values
        exp = (logits - maximum[..., None]).exp()
        total = exp.sum(-1)
        output = torch.einsum("thk,tkd->thd", exp / total[..., None], values).to(query.dtype)
        return output, maximum.unsqueeze(0), total.unsqueeze(0)

    return run

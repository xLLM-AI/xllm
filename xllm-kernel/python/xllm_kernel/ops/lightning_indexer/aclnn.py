# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""lightning_indexer implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("lightning_indexer")

    def run(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        weights: torch.Tensor,
        query_lengths: torch.Tensor,
        key_lengths: torch.Tensor,
        block_table: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        return op(
            query,
            key_cache,
            weights,
            query_lengths,
            key_lengths,
            block_table,
            "TND",
            "PA_BSND",
            topk,
            3,
            9223372036854775807,
            9223372036854775807,
            False,
        )

    return run

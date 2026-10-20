# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Alias-preserving metadata for the caller-owned indexer output."""

import torch

from xllm_kernel.fake.lightning_indexer import check
from xllm_kernel.fake.validation import tensor


def _fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
    output: torch.Tensor,
) -> torch.Tensor:
    check(query, key_cache, weights, query_lengths, key_lengths, block_table, topk)
    tensor(output, 3, (torch.int32,))
    torch._check(output.shape == (query.shape[0], 1, topk) and output.device == query.device and output.is_contiguous())
    return output


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_lightning_indexer_out")(_fake)

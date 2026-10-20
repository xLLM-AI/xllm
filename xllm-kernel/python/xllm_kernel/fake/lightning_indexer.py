# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Pinned paged indexer metadata; actual lengths and page IDs are caller preconditions."""

import torch

from xllm_kernel.fake.validation import pair, tensor


def check(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> None:
    tensor(query, 3)
    pair(query, key_cache, 4)
    pair(query, weights, 2)
    torch._check(0 < query.shape[0] <= 128 and query.shape[1:] == (32, 128))
    torch._check(key_cache.shape[1:] == (128, 1, 128) and weights.shape == query.shape[:2])
    for value, rank in ((query_lengths, 1), (key_lengths, 1), (block_table, 2)):
        tensor(value, rank, (torch.int32,))
        torch._check(value.device == query.device and value.is_contiguous())
    torch._check(query_lengths.shape == key_lengths.shape == (1,))
    torch._check(block_table.shape[0] == 1 and 0 < block_table.shape[1] <= key_cache.shape[0])
    torch._check(topk == 2048 and block_table.shape[1] * 128 >= topk)


def _fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    check(query, key_cache, weights, query_lengths, key_lengths, block_table, topk)
    return query.new_empty((query.shape[0], 1, topk), dtype=torch.int32)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_lightning_indexer")(_fake)

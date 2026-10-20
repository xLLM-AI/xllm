# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Metadata for the caller-owned sparse decode output."""

import math

import torch

from xllm_kernel.fake.validation import pair, tensor


def _fake(
    query: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    block_table: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    tensor(query, 3, (torch.bfloat16,))
    pair(query, cache, 4)
    pair(query, query_rope, 3)
    pair(query, key_rope, 4)
    pair(query, output, 3)
    torch._check(query.shape[0] == 1 and 0 < query.shape[1] <= 32 and query.shape[2] == 512)
    torch._check(cache.shape[1:] == (128, 1, 512) and query_rope.shape == (1, query.shape[1], 64))
    torch._check(key_rope.shape == (cache.shape[0], 128, 1, 64))
    for value, rank in ((indices, 3), (query_lengths, 1), (key_lengths, 1), (block_table, 2)):
        tensor(value, rank, (torch.int32,))
        torch._check(value.device == query.device and value.is_contiguous())
    torch._check(indices.shape == (1, 1, 2048) and query_lengths.shape == key_lengths.shape == (1,))
    torch._check(block_table.shape[0] == 1 and 0 < block_table.shape[1] <= cache.shape[0])
    torch._check(output.shape == query.shape and output.is_contiguous())
    torch._check(math.isfinite(scale) and 0 < scale <= torch.finfo(torch.float32).max)
    return output


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_sparse_flash_attention_out")(_fake)

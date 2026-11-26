# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Host-side zigzag context-parallel index construction."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_cp_context(
    q_seq_lens: Sequence[int],
    kv_seq_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    torch.Tensor,
    list[int],
    int,
]:
    """Return the rank's zigzag index tensors and packed segment lengths.

    In tuple order, results are shard indices, safe gather indices, valid
    mask, restore indices, query indices, KV gather indices, cumulative query
    and KV lengths, original sequence indices, absolute segment KV lengths,
    and the padded local row count. This runs during eager prefill preparation.
    """
    cp_size = index(cp_size)
    cp_rank = index(cp_rank)
    if cp_size <= 1:
        raise ValueError("CP context cp_size must be greater than 1")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"CP context cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if len(q_seq_lens) != len(kv_seq_lens):
        raise ValueError("CP query and KV lengths must describe the same number of requests")

    query_lengths = [index(length) for length in q_seq_lens]
    kv_lengths = [index(length) for length in kv_seq_lens]
    for request_id, (query_length, kv_length) in enumerate(zip(query_lengths, kv_lengths, strict=True)):
        if query_length < 0:
            raise ValueError(f"CP request {request_id}: query length must be nonnegative")
        if kv_length < query_length:
            raise ValueError(f"CP request {request_id}: KV length must be at least the query length")

    import torch

    from xllm_kernel import _npu_binding  # noqa: F401

    return torch.ops.xllm_kernel.build_cp_context(query_lengths, kv_lengths, cp_size, cp_rank, device)

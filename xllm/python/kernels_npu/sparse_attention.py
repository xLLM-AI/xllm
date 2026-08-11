# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU sparse-attention kernels."""

from __future__ import annotations

import torch


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
) -> torch.Tensor:
    """Select the key blocks each query attends to.

    Args:
        query: Query tensor laid out as ``layout_query``.
        key: Key cache laid out as ``layout_key``.
        weights: Per-head indexer weights.
        query_seq_lengths: Query length of every sequence, or ``None``.
        key_seq_lengths: Key length of every sequence, or ``None``.
        block_table: Paged key-cache block table, or ``None``.
        layout_query: Query layout, ``"TND"`` or ``"BSND"``.
        layout_key: Key layout, for example ``"PA_BSND"``.
        selected_count: Key blocks kept per query.
        sparse_mode: Sparse masking mode.
        pre_tokens: Tokens visible before the query position.
        next_tokens: Tokens visible after the query position.
        return_value: Whether to also return the indexer scores.

    Returns:
        Selected key indices of dtype ``torch.int32``.
    """
    return torch.ops.xllm_ops.lightning_indexer(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        layout_query,
        layout_key,
        selected_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )


def scatter_nd_update(
    value: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """Write ``updates`` into ``value`` at ``indices``, in place.

    Args:
        value: Destination tensor, updated in place.
        indices: Index of every updated row, shape ``[num_updates, 1]``.
        updates: Rows written into ``value``.
    """
    torch.ops.xllm_ops.scatter_nd_update(value, indices, updates)


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PR #6 SparseFlashAttention (3 outputs / 8 attrs, supports rope=None).

    Dispatches to the custom_transformer vendor op via direct dlopen in the
    C++ impl, bypassing the global aclnn resolver so it does not disturb the
    other ops served by custom_xllm_math. Pass ``query_rope=None`` /
    ``key_rope=None`` for the no-rope path.

    Returns:
        ``(attention_out, softmax_max, softmax_sum)``. ``attention_out`` has
        the shape and dtype of ``query``; the two softmax tensors are empty
        (shape ``{0}``, float32) unless ``return_softmax_lse`` is set.
    """
    return torch.ops.xllm_ops.sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
    )


__all__ = [
    "lightning_indexer",
    "scatter_nd_update",
    "sparse_flash_attention",
]

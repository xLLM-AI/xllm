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


def lightning_indexer_out(
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
    sparse_indices_out: torch.Tensor,
    sparse_values_out: torch.Tensor,
) -> torch.Tensor:
    """Select key blocks and write the results to caller-owned buffers."""
    return torch.ops.xllm_ops.lightning_indexer_out(
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
        sparse_indices_out,
        sparse_values_out,
    )


def quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    metadata: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    selected_count: int,
    cmp_ratio: int = 1,
) -> torch.Tensor:
    """Run INT8 LightningIndexer with per-token Q/K dequant scales."""
    indices, _ = torch.ops.xllm_ops.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        0,
        0,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        metadata,
        "TND",
        "PA_BSND",
        selected_count,
        3,
        9223372036854775807,
        9223372036854775807,
        cmp_ratio,
        False,
    )
    return indices


def quant_lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sparse_count: int,
    cmp_ratio: int,
) -> torch.Tensor:
    """Create reusable tiling metadata for QuantLightningIndexer."""
    return torch.ops.xllm_ops.quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        0,
        0,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        actual_seq_lengths_key.numel(),
        max_seqlen_q,
        max_seqlen_k,
        "TND",
        "PA_BSND",
        sparse_count,
        3,
        9223372036854775807,
        9223372036854775807,
        cmp_ratio,
        str(actual_seq_lengths_query.device),
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
) -> torch.Tensor:
    """Attend to the key blocks selected by :func:`lightning_indexer`.

    Args:
        query: Query tensor laid out as ``layout_query``.
        key: Key cache laid out as ``layout_kv``.
        value: Value cache laid out as ``layout_kv``.
        sparse_indices: Key blocks selected per query.
        block_table: Paged cache block table, or ``None``.
        actual_seq_lengths_query: Query length of every sequence, or ``None``.
        actual_seq_lengths_kv: Key length of every sequence, or ``None``.
        query_rope: Rotary part of the query, or ``None``.
        key_rope: Rotary part of the key, or ``None``.
        scale_value: Softmax scale.
        sparse_block_size: Keys per selected block.
        layout_query: Query layout, ``"TND"`` or ``"BSND"``.
        layout_kv: Key and value layout.
        sparse_mode: Sparse masking mode.

    Returns:
        Attention output with the shape and dtype of ``query``.
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
    )


def sparse_flash_attention_out(
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
    output: torch.Tensor,
) -> torch.Tensor:
    """Attend to selected blocks and write the output into ``output``."""
    return torch.ops.xllm_ops.sparse_flash_attention_out(
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
        output,
    )


__all__ = [
    "lightning_indexer",
    "lightning_indexer_out",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "scatter_nd_update",
    "sparse_flash_attention",
    "sparse_flash_attention_out",
]

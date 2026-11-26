# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Packed paged sparse indexing, metadata generation and cache updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

ASCENDC_OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, False, name),
        (
            BackendModule(
                module="xllm_kernel.ops.attention.dsa.indexer.ascendc",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.ascendc.{name}",),
            ),
        ),
    )
    for name, mutates in (
        ("quant_lightning_indexer", ()),
        ("quant_lightning_indexer_metadata", ()),
        ("scatter_nd_update", ("value",)),
    )
)
_INDEXER, _METADATA, _SCATTER = ASCENDC_OPERATIONS

LIGHTNING_INDEXER_OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, bool(mutates), "lightning_indexer"),
        (
            BackendModule(
                module="xllm_kernel.ops.attention.dsa.indexer.npu",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.aclnn.{name}",),
            ),
        ),
    )
    for name, mutates in (
        ("lightning_indexer", ()),
        ("lightning_indexer_out", ("sparse_indices_out", "sparse_values_out")),
    )
)
_LIGHTNING_INDEXER, _LIGHTNING_INDEXER_OUT = LIGHTNING_INDEXER_OPERATIONS
OPERATIONS = (*ASCENDC_OPERATIONS, *LIGHTNING_INDEXER_OPERATIONS)


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
    """Return int32 top-k token offsets for packed queries and a paged quantized key cache."""
    return _INDEXER.get_prepared().function(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        metadata,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        selected_count,
        cmp_ratio,
    )


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
    """Produce the device metadata consumed by the packed quantized indexer."""
    return _METADATA.get_prepared().function(
        num_heads_q,
        num_heads_k,
        head_dim,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        max_seqlen_q,
        max_seqlen_k,
        sparse_count,
        cmp_ratio,
    )


def scatter_nd_update(value: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> None:
    """Update cache rows in place with int32 or int64 indices, skipping negative padding indices."""
    _SCATTER.get_prepared().function(value, indices, updates)


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
        return_value: Whether CANN also computes the indexer scores.

    Returns:
        Selected key indices of dtype ``torch.int32``.
    """
    return _LIGHTNING_INDEXER.get_prepared().function(
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
    return _LIGHTNING_INDEXER_OUT.get_prepared().function(
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

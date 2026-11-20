# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Packed paged sparse indexing, metadata generation and cache updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, False, name),
        (
            BackendModule(
                module="xllm_kernel.ops.attention.sparse_indexing.ascendc",
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
_INDEXER, _METADATA, _SCATTER = OPERATIONS


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

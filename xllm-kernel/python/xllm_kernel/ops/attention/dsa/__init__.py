# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Sparse attention over KV entries selected by the DSA indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

ASCENDC_OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, aliases, name),
        (
            BackendModule(
                module="xllm_kernel.ops.attention.dsa.ascendc",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.ascendc.{name}",),
            ),
        ),
    )
    for name, mutates, aliases in (
        ("sparse_flash_attention_lse", (), False),
        ("sparse_flash_attention_lse_out", ("attention_output",), True),
    )
)
_SPARSE_FLASH_ATTENTION_LSE, _SPARSE_FLASH_ATTENTION_LSE_OUT = ASCENDC_OPERATIONS

SFA_DCP_REMAP_TOPK = 2048
SFA_DCP_REMAP_MAX_TOKENS = 256
SFA_DCP_REMAP_OUT_CONTRACT = OpContract("sfa_dcp_remap_out", 1, ("out", "idx_scratch"), True, "stable_sparse_remap")
SFA_DCP_REMAP_OUT_OPERATION = Operation(
    SFA_DCP_REMAP_OUT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.attention.dsa.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "tilelang", "triton"),
            implementations=("npu.tilelang.sfa_dcp_remap_out",),
        ),
    ),
)
OPERATIONS = (*ASCENDC_OPERATIONS, SFA_DCP_REMAP_OUT_OPERATION)


def has_sfa_dcp_remap_out() -> bool:
    """Return whether the worker prepared the fused DCP index remap."""
    return SFA_DCP_REMAP_OUT_OPERATION.prepared is not None


def sparse_flash_attention_lse(
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
    """Run the packaged RoPE sparse attention and return optional softmax statistics."""
    return _SPARSE_FLASH_ATTENTION_LSE.get_prepared().function(
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


def sparse_flash_attention_lse_out(
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
    """Run sparse attention directly into a caller-owned output buffer."""
    return _SPARSE_FLASH_ATTENTION_LSE_OUT.get_prepared().function(
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
        9223372036854775807,
        9223372036854775807,
        2,
        False,
        output,
    )


def sfa_dcp_remap_out(
    topk_indices: torch.Tensor,
    physical_block_size: int,
    shard_size: int,
    shard_rank: int,
    out: torch.Tensor,
    idx_scratch: torch.Tensor,
) -> torch.Tensor:
    """Compact this rank's local offsets in index order and pad the tail with -1."""
    return SFA_DCP_REMAP_OUT_OPERATION.get_prepared().function(
        topk_indices, physical_block_size, shard_size, shard_rank, out, idx_scratch
    )

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Sparse attention over KV entries selected by the DSA indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

OPERATIONS = tuple(
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
_SPARSE_FLASH_ATTENTION_LSE, _SPARSE_FLASH_ATTENTION_LSE_OUT = OPERATIONS


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

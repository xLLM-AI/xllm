# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Packaged compressed KV, shared cache attention and RoPE sparse attention."""

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
                module="xllm_kernel.ops.attention.sparse_attention.ascendc",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.ascendc.{name}",),
            ),
        ),
    )
    for name, mutates, aliases in (
        ("compressor", ("kv_state", "score_state"), False),
        ("sparse_attn_sharedkv", (), False),
        ("sparse_attn_sharedkv_metadata", (), False),
        ("sparse_flash_attention_lse", (), False),
        ("sparse_flash_attention_lse_out", ("attention_output",), True),
    )
)
(
    _COMPRESSOR,
    _SPARSE_ATTN_SHAREDKV,
    _SPARSE_ATTN_SHAREDKV_METADATA,
    _SPARSE_FLASH_ATTENTION_LSE,
    _SPARSE_FLASH_ATTENTION_LSE_OUT,
) = OPERATIONS


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    kv_block_table: torch.Tensor | None,
    score_block_table: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    enable_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool KV along the token axis by ``cmp_ratio`` (NSA-style compressor).

    ``kv_state`` and ``score_state`` are updated in place.

    Returns ``(cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd)``; only
    ``cmp_kv`` is consumed by the DSA path.
    """
    # C++ moves DSA metadata to the active device before dispatch. Keep this
    # adapter deterministic; experimental clone/noalias paths do not belong in
    # the public binding.
    kv_block_table = kv_block_table.to(x.device) if kv_block_table is not None else None
    score_block_table = score_block_table.to(x.device) if score_block_table is not None else None
    cu_seqlens = cu_seqlens.to(x.device) if cu_seqlens is not None else None
    seqused = seqused.to(x.device) if seqused is not None else None
    start_pos = start_pos.to(x.device) if start_pos is not None else None
    return _COMPRESSOR.get_prepared().function(
        x,
        wkv,
        wgate,
        kv_state,
        score_state,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        kv_block_table,
        score_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
        norm_eps,
        rotary_mode,
        enable_grad,
    )


def sparse_attn_sharedkv(
    q: torch.Tensor,
    ori_kv: torch.Tensor | None,
    cmp_kv: torch.Tensor | None,
    ori_sparse_indices: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    ori_block_table: torch.Tensor | None,
    cmp_block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    sinks: torch.Tensor | None,
    metadata: torch.Tensor | None,
    softmax_scale: float,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    return_softmax_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage sparse attention over original and compressed KV."""
    return _SPARSE_ATTN_SHAREDKV.get_prepared().function(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        return_softmax_lse,
    )


def sparse_attn_sharedkv_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    ori_topk: int,
    cmp_topk: int,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    has_ori_kv: bool,
    has_cmp_kv: bool,
) -> torch.Tensor:
    """Build the AICPU tiling metadata for :func:`sparse_attn_sharedkv`."""
    return _SPARSE_ATTN_SHAREDKV_METADATA.get_prepared().function(
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )


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

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

"""NPU DeepSeek-V4 DSA kernels.

These wrap the AscendC operators registered as ``torch.ops.xllm_ops.*`` by
``core/kernels/npu/npu_ops_library.cpp``. They drive the two-stage sparse
attention (original + compressed KV), the KV compressor, the quantized
lightning indexer, and the HyperConnection pre/post used by DeepSeek-V4's DSA
attention path.
"""

from __future__ import annotations

import torch


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HyperConnection pre: mix hc_mult streams into one sub-block input.

    Returns ``(attn_input, post, comb)`` where post/comb feed ``hc_post``.
    """
    return torch.ops.xllm_ops.hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """HyperConnection post: combine sub-block output with the residual streams."""
    return torch.ops.xllm_ops.hc_post(x, residual, post, comb)


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
    return torch.ops.xllm_ops.compressor(
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
    return torch.ops.xllm_ops.sparse_attn_sharedkv(
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
    return torch.ops.xllm_ops.sparse_attn_sharedkv_metadata(
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

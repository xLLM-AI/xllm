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

"""NPU Triton kernel for the GLM-5.3-Flash MTP eh-input normalization.

Fuses ``cat(RMSNorm(embed, enorm_w), RMSNorm(carried, hnorm_w))`` — the two
draft-input RMSNorms plus the concat that feeds ``eh_proj`` — into one launch,
replacing two RMSNorm ops + a ``torch.cat``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_eh_norm_kernel(
    embed_ptr,
    carried_ptr,
    enorm_weight_ptr,
    hnorm_weight_ptr,
    output_ptr,
    num_rows,
    stride_embed_row,
    stride_carried_row,
    stride_output_row,
    eps,
    FEATURE_DIM: tl.constexpr,
    BLOCK_FEATURE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """Per-row RMSNorm of embed and carried, written side by side into [.., 2H]."""
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    features = tl.arange(0, BLOCK_FEATURE)
    row_mask = rows[:, None] < num_rows
    feature_mask = features[None, :] < FEATURE_DIM
    mask = row_mask & feature_mask

    embed = tl.load(embed_ptr + rows[:, None] * stride_embed_row + features[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    inv_e = 1.0 / tl.sqrt(tl.sum(tl.where(mask, embed * embed, 0.0), axis=1) / FEATURE_DIM + eps)
    enorm_w = tl.load(enorm_weight_ptr + features, mask=features < FEATURE_DIM, other=0.0).to(tl.float32)
    out_e = embed * inv_e[:, None] * enorm_w[None, :]
    tl.store(output_ptr + rows[:, None] * stride_output_row + features[None, :], out_e, mask=mask)

    carried = tl.load(carried_ptr + rows[:, None] * stride_carried_row + features[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    inv_h = 1.0 / tl.sqrt(tl.sum(tl.where(mask, carried * carried, 0.0), axis=1) / FEATURE_DIM + eps)
    hnorm_w = tl.load(hnorm_weight_ptr + features, mask=features < FEATURE_DIM, other=0.0).to(tl.float32)
    out_h = carried * inv_h[:, None] * hnorm_w[None, :]
    tl.store(
        output_ptr + rows[:, None] * stride_output_row + FEATURE_DIM + features[None, :],
        out_h,
        mask=mask,
    )


def fused_eh_norm(
    embed: torch.Tensor,
    carried: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute ``cat(RMSNorm(embed, enorm_w), RMSNorm(carried, hnorm_w))`` in one kernel.

    ``embed`` and ``carried`` share shape ``[..., H]``; the result is
    ``[..., 2H]`` (embed norm first, carried norm second) ready for ``eh_proj``.
    """
    if embed.shape != carried.shape:
        raise ValueError("embed and carried must have identical shapes")
    if embed.dtype != carried.dtype:
        raise ValueError("embed and carried must have identical dtypes")

    original_shape = embed.shape
    feature_dim = embed.shape[-1]
    embed_2d = embed.reshape(-1, feature_dim)
    carried_2d = carried.reshape(-1, feature_dim)
    if embed_2d.stride(-1) != 1:
        embed_2d = embed_2d.contiguous()
    if carried_2d.stride(-1) != 1:
        carried_2d = carried_2d.contiguous()
    enorm_weight = enorm_weight.contiguous()
    hnorm_weight = hnorm_weight.contiguous()
    if enorm_weight.shape != (feature_dim,) or hnorm_weight.shape != (feature_dim,):
        raise ValueError("norm weights must match the final input dimension")

    block_feature = min(65536 // embed.element_size(), triton.next_power_of_2(feature_dim))
    if feature_dim > block_feature:
        raise RuntimeError("fused eh-norm does not support feature dimensions >= 64 KiB")

    num_rows = embed_2d.shape[0]
    rows_per_block = max(1, min(32, 4096 // block_feature))
    output = torch.empty(num_rows, 2 * feature_dim, dtype=embed.dtype, device=embed.device)
    grid = (triton.cdiv(num_rows, rows_per_block),)
    _fused_eh_norm_kernel[grid](
        embed_2d,
        carried_2d,
        enorm_weight,
        hnorm_weight,
        output,
        num_rows,
        embed_2d.stride(0),
        carried_2d.stride(0),
        output.stride(0),
        eps,
        FEATURE_DIM=feature_dim,
        BLOCK_FEATURE=block_feature,
        ROWS_PER_BLOCK=rows_per_block,
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], 2 * feature_dim)


__all__ = ["fused_eh_norm"]

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

"""FLA-compatible Triton L2 normalization.

Adapted from vLLM's vendored flash-linear-attention implementation.
"""

import os

import torch
import triton
import triton.language as tl

_BLOCK_TOKEN_CHOICES = [8, 16, 32, 64, 128]
_USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]],
    key=["feature_dim"],
)
@triton.jit
def _l2_norm_single_row_kernel(
    input_ptr,
    output_ptr,
    feature_dim,
    BLOCK_FEATURE: tl.constexpr,
    eps,
):
    token = tl.program_id(0)
    input_ptr += token * feature_dim
    output_ptr += token * feature_dim
    features = tl.arange(0, BLOCK_FEATURE)
    mask = features < feature_dim
    values = tl.load(input_ptr + features, mask=mask, other=0.0).to(tl.float32)
    inverse_norm = 1 / tl.sqrt(tl.sum(values * values, axis=0) + eps)
    tl.store(output_ptr + features, values * inverse_norm, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_TOKEN": block_token}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for block_token in _BLOCK_TOKEN_CHOICES
    ],
    key=["FEATURE_DIM"],
)
@triton.jit(do_not_specialize=["num_token_blocks"])
def _l2_norm_block_kernel(
    input_ptr,
    output_ptr,
    eps,
    num_token_blocks,
    num_tokens,
    FEATURE_DIM: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
    BLOCK_FEATURE: tl.constexpr,
):
    token_block = tl.program_id(0)
    input_block = tl.make_block_ptr(
        input_ptr,
        (num_tokens, FEATURE_DIM),
        (FEATURE_DIM, 1),
        (token_block * BLOCK_TOKEN, 0),
        (BLOCK_TOKEN, BLOCK_FEATURE),
        (1, 0),
    )
    values = tl.load(input_block, boundary_check=(0, 1)).to(tl.float32)
    inverse_norm = 1 / tl.sqrt(tl.sum(values * values, axis=1) + eps)
    normalized = values * inverse_norm[:, None]
    output_block = tl.make_block_ptr(
        output_ptr,
        (num_tokens, FEATURE_DIM),
        (FEATURE_DIM, 1),
        (token_block * BLOCK_TOKEN, 0),
        (BLOCK_TOKEN, BLOCK_FEATURE),
        (1, 0),
    )
    tl.store(
        output_block,
        normalized.to(output_block.dtype.element_ty),
        boundary_check=(0, 1),
    )


@triton.jit
def _l2_norm_multi_row_kernel(
    input_ptr,
    output_ptr,
    eps,
    num_rows,
    feature_dim: tl.constexpr,
    BLOCK_FEATURE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_ROW
    rows = row_start + tl.arange(0, BLOCK_ROW)[:, None]
    row_mask = rows < num_rows
    features = tl.arange(0, BLOCK_FEATURE)[None, :]
    feature_mask = features < feature_dim
    mask = row_mask & feature_mask
    offsets = features + feature_dim * rows
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(tl.where(row_mask, values * values, 0), 1)[:, None]
    tl.store(
        output_ptr + offsets,
        values * tl.rsqrt(square_sum + eps),
        mask=mask,
    )


def l2_norm(
    value: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize the final dimension using vLLM's FLA Triton kernels."""
    original_shape = value.shape
    value_2d = value.view(-1, value.shape[-1])
    output = torch.empty_like(value_2d)
    num_rows, feature_dim = value_2d.shape
    max_fused_size = 65536 // value.element_size()
    block_feature = min(max_fused_size, triton.next_power_of_2(feature_dim))
    if feature_dim > block_feature:
        raise RuntimeError("L2 norm does not support feature dimensions >= 64 KiB")

    if not _USE_DEFAULT_FLA_NORM:
        block_row = 32
        _l2_norm_multi_row_kernel[(triton.cdiv(num_rows, block_row),)](
            value_2d,
            output,
            eps,
            num_rows,
            feature_dim,
            block_feature,
            block_row,
        )
    elif feature_dim <= 512:
        num_token_blocks = triton.cdiv(num_rows, 2048)

        def _grid(meta: dict[str, int]) -> tuple[int]:
            return (triton.cdiv(num_rows, meta["BLOCK_TOKEN"]),)

        _l2_norm_block_kernel[_grid](
            value_2d,
            output,
            eps,
            num_token_blocks=num_token_blocks,
            num_tokens=num_rows,
            FEATURE_DIM=feature_dim,
            BLOCK_FEATURE=block_feature,
        )
    else:
        _l2_norm_single_row_kernel[(num_rows,)](
            value_2d,
            output,
            eps=eps,
            feature_dim=feature_dim,
            BLOCK_FEATURE=block_feature,
        )
    return output.view(original_shape)

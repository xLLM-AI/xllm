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

"""NPU Triton kernels for GLM-style RMS normalization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_sigmoid_gated_kernel(
    value_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    num_rows,
    stride_value_row,
    stride_gate_row,
    stride_output_row,
    eps,
    FEATURE_DIM: tl.constexpr,
    BLOCK_FEATURE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """Apply one independent RMS reduction to each row, then sigmoid-gate."""
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    features = tl.arange(0, BLOCK_FEATURE)
    row_mask = rows[:, None] < num_rows
    feature_mask = features[None, :] < FEATURE_DIM
    mask = row_mask & feature_mask

    value_offsets = rows[:, None] * stride_value_row + features[None, :]
    values = tl.load(value_ptr + value_offsets, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(tl.where(mask, values * values, 0.0), axis=1)
    inverse_rms = 1.0 / tl.sqrt(square_sum / FEATURE_DIM + eps)

    weights = tl.load(
        weight_ptr + features,
        mask=features < FEATURE_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized = values * inverse_rms[:, None] * weights[None, :]

    gate_offsets = rows[:, None] * stride_gate_row + features[None, :]
    gates = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    output = normalized * tl.sigmoid(gates)

    output_offsets = rows[:, None] * stride_output_row + features[None, :]
    tl.store(output_ptr + output_offsets, output, mask=mask)


def rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute ``RMSNorm(value) * sigmoid(gate)`` in one NPU kernel.

    GLM-5.3-Flash uses this operation on KDA heads with a 128-element feature
    dimension. Processing a fixed number of rows per Triton program keeps the
    reduction tree for each row independent of the decode batch size.
    """
    if value.shape != gate.shape:
        raise ValueError("value and gate must have identical shapes")
    if value.dtype != gate.dtype:
        raise ValueError("value and gate must have identical dtypes")

    original_shape = value.shape
    value_2d = value.reshape(-1, value.shape[-1])
    if value_2d.stride(-1) != 1:
        value_2d = value_2d.contiguous()
    num_rows, feature_dim = value_2d.shape
    block_feature = min(65536 // value.element_size(), triton.next_power_of_2(feature_dim))
    if feature_dim > block_feature:
        raise RuntimeError("RMSNorm does not support feature dimensions >= 64 KiB")
    gate_2d = gate.reshape(-1, gate.shape[-1])
    if gate_2d.stride(-1) != 1:
        gate_2d = gate_2d.contiguous()
    weight = weight.contiguous()
    if weight.shape != (feature_dim,):
        raise ValueError("weight must match the final input dimension")

    rows_per_block = 32
    output = torch.empty_like(value_2d)
    grid = (triton.cdiv(num_rows, rows_per_block),)
    _rms_norm_sigmoid_gated_kernel[grid](
        value_2d,
        gate_2d,
        weight,
        output,
        num_rows,
        value_2d.stride(0),
        gate_2d.stride(0),
        output.stride(0),
        eps,
        FEATURE_DIM=feature_dim,
        BLOCK_FEATURE=block_feature,
        ROWS_PER_BLOCK=rows_per_block,
        num_warps=4,
    )
    return output.reshape(original_shape)


__all__ = ["rms_norm_sigmoid_gated"]

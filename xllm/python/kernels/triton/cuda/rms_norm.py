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

"""FLA-compatible gated RMS normalization.

Adapted from vLLM's vendored flash-linear-attention layernorm guard.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_gated_kernel(
    value_ptr,
    output_ptr,
    weight_ptr,
    gate_ptr,
    inverse_rms_ptr,
    stride_value_row,
    stride_output_row,
    stride_gate_row,
    num_rows,
    feature_dim: tl.constexpr,
    eps,
    BLOCK_FEATURE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    features = tl.arange(0, BLOCK_FEATURE)
    row_mask = rows[:, None] < num_rows
    feature_mask = features[None, :] < feature_dim
    mask = row_mask & feature_mask

    value_offsets = rows[:, None] * stride_value_row + features[None, :]
    values = tl.load(value_ptr + value_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(tl.where(mask, values * values, 0.0), axis=1) / feature_dim
    inverse_rms = tl.rsqrt(variance + eps)
    tl.store(inverse_rms_ptr + rows, inverse_rms, mask=rows < num_rows)

    weights = tl.load(
        weight_ptr + features,
        mask=features < feature_dim,
        other=0.0,
    ).to(tl.float32)
    normalized = values * inverse_rms[:, None] * weights[None, :]
    gate_offsets = rows[:, None] * stride_gate_row + features[None, :]
    gates = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    normalized *= gates * tl.sigmoid(gates)

    output_offsets = rows[:, None] * stride_output_row + features[None, :]
    tl.store(output_ptr + output_offsets, normalized, mask=mask)


def _rows_per_block(num_rows: int, device: torch.device) -> int:
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    blocks_per_sm_pair = triton.cdiv(num_rows, 2 * multiprocessors)
    return min(triton.next_power_of_2(blocks_per_sm_pair), 4)


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute ``RMSNorm(value) * SiLU(gate)`` using vLLM's layout."""
    if value.shape != gate.shape:
        raise ValueError("value and gate must have identical shapes")
    original_shape = value.shape
    value_2d = value.reshape(-1, value.shape[-1])
    gate_2d = gate.reshape(-1, gate.shape[-1])
    if value_2d.stride(-1) != 1:
        value_2d = value_2d.contiguous()
    if gate_2d.stride(-1) != 1:
        gate_2d = gate_2d.contiguous()
    weight = weight.contiguous()

    num_rows, feature_dim = value_2d.shape
    if weight.shape != (feature_dim,):
        raise ValueError("weight must match the final input dimension")
    max_fused_size = 65536 // value.element_size()
    block_feature = min(max_fused_size, triton.next_power_of_2(feature_dim))
    if feature_dim > block_feature:
        raise RuntimeError("gated RMSNorm does not support dimensions >= 64 KiB")

    output = torch.empty_like(value_2d)
    inverse_rms = torch.empty(num_rows, dtype=torch.float32, device=value.device)
    rows_per_block = _rows_per_block(num_rows, value.device)
    grid = (triton.cdiv(num_rows, rows_per_block),)
    _rms_norm_gated_kernel[grid](
        value_2d,
        output,
        weight,
        gate_2d,
        inverse_rms,
        value_2d.stride(0),
        output.stride(0),
        gate_2d.stride(0),
        num_rows,
        feature_dim,
        eps,
        BLOCK_FEATURE=block_feature,
        ROWS_PER_BLOCK=rows_per_block,
        num_warps=min(max(block_feature // 256, 1), 8),
    )
    return output.reshape(original_shape)

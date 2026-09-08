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
"""DCP packed AllToAll: pack output+LSE into one HCCL payload, then fused LSE combine.

Source: vllm-ascend ``vllm_ascend/ops/triton/sfa_cp.py`` (PR 14457).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import get_vectorcore_num


@triton.jit
def _pack_dcp_output_lse_kernel(
    output_ptr,
    lse_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
    send_stride_rank,
    send_stride_scatter,
    send_stride_replicated,
    send_stride_d,
    local_scatter_size,
    head_dim,
    num_heads,
    total_rows,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)

    for linear_idx in range(program_idx, total_rows, num_programs):
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            rank_idx = token_idx // local_scatter_size
            scatter_idx = token_idx % local_scatter_size
            replicated_idx = head_idx
        else:
            rank_idx = head_idx // local_scatter_size
            scatter_idx = head_idx % local_scatter_size
            replicated_idx = token_idx

        send_base = (
            rank_idx * send_stride_rank + scatter_idx * send_stride_scatter + replicated_idx * send_stride_replicated
        )
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        d_mask = d_offsets < head_dim
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h).to(tl.float32)
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                lse.to(send_ptr.dtype.element_ty),
            )
        else:
            finite_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            abs_lse = tl.abs(lse)
            nonzero_lse = abs_lse != 0.0
            safe_abs_lse = tl.where(finite_lse & nonzero_lse, abs_lse, 1.0)
            lse_exponent = tl.floor(tl.log2(safe_abs_lse))
            lse_exponent = tl.maximum(-126.0, tl.minimum(lse_exponent, 127.0))
            lse_exponent = tl.where(nonzero_lse, lse_exponent, 0.0)
            significand = tl.where(
                finite_lse & nonzero_lse,
                abs_lse * tl.exp2(23.0 - lse_exponent),
                0.0,
            )
            significand_hi = tl.floor(significand / 65536.0)
            significand_remainder = significand - significand_hi * 65536.0
            significand_mid = tl.floor(significand_remainder / 256.0)
            significand_lo = significand_remainder - significand_mid * 256.0
            exponent_code = lse_exponent + 128.0
            exponent_code = tl.where(lse < 0.0, -exponent_code, exponent_code)
            exponent_code = tl.where(finite_lse, exponent_code, 0.0)
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                exponent_code.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 1) * send_stride_d,
                significand_hi.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 2) * send_stride_d,
                significand_mid.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 3) * send_stride_d,
                significand_lo.to(send_ptr.dtype.element_ty),
            )


@triton.jit
def _fused_dcp_lse_combine_kernel(
    recv_ptr,
    output_ptr,
    recv_stride_rank,
    recv_stride_scatter,
    recv_stride_replicated,
    recv_stride_d,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    head_dim,
    num_heads,
    total_rows,
    DCP_SIZE: tl.constexpr,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)

    for linear_idx in range(program_idx, total_rows, num_programs):
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            scatter_idx = token_idx
            replicated_idx = head_idx
        else:
            scatter_idx = head_idx
            replicated_idx = token_idx

        lse_max = -float("inf")
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                significand_hi = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
                significand_mid = tl.load(recv_ptr + recv_base + (head_dim + 2) * recv_stride_d).to(tl.float32)
                significand_lo = tl.load(recv_ptr + recv_base + (head_dim + 3) * recv_stride_d).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))

        any_valid_lse = lse_max != -float("inf")
        safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
        weight_sum = 0.0
        merged = tl.zeros([BLOCK_D], dtype=tl.float32)
        d_mask = d_offsets < head_dim
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                significand_hi = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
                significand_mid = tl.load(recv_ptr + recv_base + (head_dim + 2) * recv_stride_d).to(tl.float32)
                significand_lo = tl.load(recv_ptr + recv_base + (head_dim + 3) * recv_stride_d).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
            partial_output = tl.load(
                recv_ptr + recv_base + d_offsets * recv_stride_d,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            partial_output = tl.where(valid_lse, partial_output, 0.0)
            merged += partial_output * weight
            weight_sum += weight

        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        merged /= denominator
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + output_offsets, merged, mask=d_mask)


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    if output_dtype in (torch.bfloat16, torch.float16):
        return 4
    if output_dtype == torch.float32:
        return 1
    raise TypeError(f"DCP packed A2A supports bfloat16, float16, or float32 output, got {output_dtype}.")


def packed_send_shape(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dcp_size: int,
    scatter_dim: int,
    dtype: torch.dtype,
) -> tuple[int, int, int, int]:
    local_scatter_size = (num_tokens if scatter_dim == 0 else num_heads) // dcp_size
    replicated_size = num_heads if scatter_dim == 0 else num_tokens
    return (dcp_size, local_scatter_size, replicated_size, head_dim + _lse_pack_dim(dtype))


def _lse_token_head_strides(softmax_lse: torch.Tensor, num_tokens: int, num_heads: int) -> tuple[int, int]:
    if softmax_lse.ndim == 2:
        if tuple(softmax_lse.shape) != (num_tokens, num_heads):
            raise RuntimeError(
                f"DCP packed A2A expects LSE [tokens, heads] matching output, got {tuple(softmax_lse.shape)}"
            )
        return int(softmax_lse.stride(0)), int(softmax_lse.stride(1))
    if softmax_lse.shape != (num_tokens, num_heads, 1):
        raise RuntimeError(
            f"DCP packed A2A expects LSE [tokens, heads, 1] matching output, got {tuple(softmax_lse.shape)}"
        )
    return int(softmax_lse.stride(0)), int(softmax_lse.stride(1))


def _validate_dcp_packed_a2a_inputs(
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
) -> tuple[int, int, int, int, int]:
    if output.ndim != 3:
        raise RuntimeError(f"DCP packed A2A expects output [tokens, heads, head_dim], got {tuple(output.shape)}.")
    if softmax_lse.dtype != torch.float32:
        raise TypeError(f"DCP packed A2A requires float32 LSE, got {softmax_lse.dtype}.")
    if output.device != softmax_lse.device:
        raise RuntimeError(
            f"DCP packed A2A requires output and LSE on the same device, got {output.device} and {softmax_lse.device}."
        )
    if output.device.type != "npu":
        raise RuntimeError(f"DCP packed A2A requires an NPU tensor, got {output.device}.")
    if not isinstance(dcp_size, int) or isinstance(dcp_size, bool) or dcp_size <= 0:
        raise ValueError(f"DCP packed A2A requires a positive integer dcp_size, got {dcp_size}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"DCP packed A2A scatter_dim must be 0 or 1, got {scatter_dim}.")

    num_tokens, num_heads, head_dim = (int(x) for x in output.shape)
    _lse_token_head_strides(softmax_lse, num_tokens, num_heads)
    if num_tokens <= 0 or num_heads <= 0 or head_dim <= 0:
        raise RuntimeError(f"DCP packed A2A requires non-empty dimensions, got {tuple(output.shape)}.")
    scatter_size = int(output.shape[scatter_dim])
    if scatter_size % dcp_size != 0:
        raise RuntimeError(
            "DCP packed A2A requires the scatter dimension to be divisible "
            f"by dcp_size, got shape={tuple(output.shape)}, "
            f"scatter_dim={scatter_dim}, and dcp_size={dcp_size}."
        )
    local_scatter_size = scatter_size // dcp_size
    replicated_size = num_heads if scatter_dim == 0 else num_tokens
    return num_tokens, num_heads, head_dim, local_scatter_size, replicated_size


def _grid_size(total_rows: int) -> int:
    vector_cores = get_vectorcore_num()
    if vector_cores <= 0:
        return max(total_rows, 1)
    return min(total_rows, vector_cores)


def pack_dcp_output_lse(
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    send: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack strided attention output and FP32 LSE into one HCCL payload."""
    num_tokens, num_heads, head_dim, local_scatter_size, replicated_size = _validate_dcp_packed_a2a_inputs(
        output, softmax_lse, dcp_size, scatter_dim
    )
    lse_pack_dim = _lse_pack_dim(output.dtype)
    expected = (dcp_size, local_scatter_size, replicated_size, head_dim + lse_pack_dim)
    if send is None:
        send = torch.empty(expected, dtype=output.dtype, device=output.device)
    elif tuple(send.shape) != expected or send.dtype != output.dtype or not send.is_contiguous():
        raise RuntimeError(
            f"DCP packed A2A send buffer must be contiguous {expected} {output.dtype}, "
            f"got {tuple(send.shape)} {send.dtype} contiguous={send.is_contiguous()}."
        )
    lse_stride_t, lse_stride_h = _lse_token_head_strides(softmax_lse, num_tokens, num_heads)
    total_rows = num_tokens * num_heads
    _pack_dcp_output_lse_kernel[(_grid_size(total_rows),)](
        output,
        softmax_lse,
        send,
        *output.stride(),
        lse_stride_t,
        lse_stride_h,
        *send.stride(),
        local_scatter_size,
        head_dim,
        num_heads,
        total_rows,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return send


def fused_dcp_lse_combine(
    recv: torch.Tensor,
    head_dim: int,
    scatter_dim: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack one HCCL payload and merge rank outputs using their LSE."""
    if recv.ndim != 4:
        raise RuntimeError(f"DCP packed A2A combine expects a 4D receive buffer, got {tuple(recv.shape)}.")
    if not recv.is_contiguous():
        raise RuntimeError("DCP packed A2A combine requires a contiguous HCCL receive buffer.")
    if recv.device.type != "npu":
        raise RuntimeError(f"DCP packed A2A combine requires an NPU tensor, got {recv.device}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"DCP packed A2A combine scatter_dim must be 0 or 1, got {scatter_dim}.")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError(f"DCP packed A2A combine requires a positive integer head_dim, got {head_dim}.")

    dcp_size, local_scatter_size, replicated_size, packed_dim = (int(x) for x in recv.shape)
    lse_pack_dim = _lse_pack_dim(recv.dtype)
    if packed_dim != head_dim + lse_pack_dim:
        raise RuntimeError(
            "DCP packed A2A combine received an invalid packed dimension: "
            f"expected {head_dim + lse_pack_dim}, got {packed_dim}."
        )
    num_tokens, num_heads = (
        (local_scatter_size, replicated_size) if scatter_dim == 0 else (replicated_size, local_scatter_size)
    )
    expected = (num_tokens, num_heads, head_dim)
    if output is None:
        output = torch.empty(expected, dtype=recv.dtype, device=recv.device)
    elif tuple(output.shape) != expected or output.dtype != recv.dtype:
        raise RuntimeError(
            f"DCP packed A2A combine output must be {expected} {recv.dtype}, got {tuple(output.shape)} {output.dtype}."
        )
    total_rows = num_tokens * num_heads
    _fused_dcp_lse_combine_kernel[(_grid_size(total_rows),)](
        recv,
        output,
        *recv.stride(),
        *output.stride(),
        head_dim,
        num_heads,
        total_rows,
        DCP_SIZE=dcp_size,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return output

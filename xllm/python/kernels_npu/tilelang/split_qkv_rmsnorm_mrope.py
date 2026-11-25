#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
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

"""Source-tree reference checks and AOT settings for fused mRoPE."""

import tilelang
from xllm_kernel.ops.positional_encoding._tilelang._mrope_geometry import (
    ALL_HEAD_CONFIGS,
    COMPILE_MAX_TOKENS,
    SUPPORTED_HEAD_SPECS,
    VEC_NUM,
    _validate_mrope_section,
    build_mrope_gather_pattern_merged,
)
from xllm_kernel.ops.positional_encoding._tilelang._mrope_tilelang import (
    KERNEL_PASS_CONFIGS,
    _validate_head_spec,
    build_split_qkv_rmsnorm_mrope_kernel,
)

from scripts.logger import logger

from .utils import detect_vec_core_num

COS_SIN_MERGED_LAYOUT = "token_3rope"
DEFAULT_DTYPE = "bf16"
MAX_VEC_CORE_NUM = detect_vec_core_num()
MAX_LAUNCH_NUM_TOKENS = (MAX_VEC_CORE_NUM // VEC_NUM) * VEC_NUM
NUM_TOKEN_SPECIALIZATIONS = tuple(range(VEC_NUM, MAX_LAUNCH_NUM_TOKENS + 1, VEC_NUM))
REF_CHECK_NUM_TOKENS = 16
REF_CHECK_EPS = 1e-6
DEFAULT_MROPE_SECTION = (11, 11, 10)
DEFAULT_NUM_Q_HEADS = 16
DEFAULT_NUM_KV_HEADS = 4

REF_CHECK_HEAD_CONFIGS = tuple(ALL_HEAD_CONFIGS)


@tilelang.jit(pass_configs=KERNEL_PASS_CONFIGS)
def split_qkv_rmsnorm_mrope_kernel_jit(
    head_size: int,
    rope_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    vec_core_num: int,
    max_num_tokens: int = COMPILE_MAX_TOKENS,
):
    return build_split_qkv_rmsnorm_mrope_kernel(
        head_size=head_size,
        rope_dim=rope_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vec_core_num=vec_core_num,
        max_num_tokens=max_num_tokens,
    )


def _validate_specialization_num_tokens(num_tokens: int) -> None:
    if num_tokens not in NUM_TOKEN_SPECIALIZATIONS:
        raise ValueError(
            "split_qkv_rmsnorm_mrope only supports specialization num_tokens in "
            f"{NUM_TOKEN_SPECIALIZATIONS}, got {num_tokens}"
        )


def _validate_runtime_config(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    del is_interleaved
    _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens({num_tokens}) must be > 0")
    if num_tokens > COMPILE_MAX_TOKENS:
        raise ValueError(f"num_tokens({num_tokens}) must be <= COMPILE_MAX_TOKENS({COMPILE_MAX_TOKENS})")
    if num_q_heads <= 0:
        raise ValueError(f"num_q_heads({num_q_heads}) must be > 0")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads({num_kv_heads}) must be > 0")
    _validate_mrope_section(rope_dim=rope_dim, mrope_section=mrope_section)


def _torch_rms_norm(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    eps: float,
) -> "torch.Tensor":
    import torch

    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    reciprocal_std = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x_fp32 * reciprocal_std * weight_fp32


def _assemble_non_interleaved_mrope_rows(
    cos_sin: "torch.Tensor",
    mrope_section: tuple[int, int, int],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    half_rope_dim = cos_sin.shape[-1] // 2
    t_len, h_len, w_len = mrope_section
    h_end = t_len + h_len
    w_end = h_end + w_len

    cos_axes = cos_sin[:, :, :half_rope_dim].to(torch.float32)
    sin_axes = cos_sin[:, :, half_rope_dim:].to(torch.float32)
    num_tokens = cos_sin.shape[1]
    cos_rows = torch.zeros((num_tokens, half_rope_dim), device=cos_sin.device, dtype=torch.float32)
    sin_rows = torch.zeros((num_tokens, half_rope_dim), device=cos_sin.device, dtype=torch.float32)

    if t_len > 0:
        cos_rows[:, :t_len] = cos_axes[0, :, :t_len]
        sin_rows[:, :t_len] = sin_axes[0, :, :t_len]
    if h_len > 0:
        cos_rows[:, t_len:h_end] = cos_axes[1, :, t_len:h_end]
        sin_rows[:, t_len:h_end] = sin_axes[1, :, t_len:h_end]
    if w_len > 0:
        cos_rows[:, h_end:w_end] = cos_axes[2, :, h_end:w_end]
        sin_rows[:, h_end:w_end] = sin_axes[2, :, h_end:w_end]
    return cos_rows, sin_rows


def _assemble_interleaved_mrope_rows(
    cos_sin: "torch.Tensor",
    mrope_section: tuple[int, int, int],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    half_rope_dim = cos_sin.shape[-1] // 2
    _, h_len, w_len = mrope_section

    cos_axes = cos_sin[:, :, :half_rope_dim].to(torch.float32)
    sin_axes = cos_sin[:, :, half_rope_dim:].to(torch.float32)

    cos_rows = cos_axes[0].clone()
    sin_rows = sin_axes[0].clone()

    cos_rows[:, 1 : h_len * 3 : 3] = cos_axes[1, :, 1 : h_len * 3 : 3]
    sin_rows[:, 1 : h_len * 3 : 3] = sin_axes[1, :, 1 : h_len * 3 : 3]

    cos_rows[:, 2 : w_len * 3 : 3] = cos_axes[2, :, 2 : w_len * 3 : 3]
    sin_rows[:, 2 : w_len * 3 : 3] = sin_axes[2, :, 2 : w_len * 3 : 3]

    return cos_rows, sin_rows


def _apply_partial_mrope(
    x: "torch.Tensor",
    cos_rows: "torch.Tensor",
    sin_rows: "torch.Tensor",
    rope_dim: int,
) -> "torch.Tensor":
    half_rope_dim = rope_dim // 2

    out = x.clone()
    x1 = x[:, :, :half_rope_dim]
    x2 = x[:, :, half_rope_dim:rope_dim]
    cos = cos_rows.unsqueeze(1)
    sin = sin_rows.unsqueeze(1)
    out[:, :, :half_rope_dim] = x1 * cos - x2 * sin
    out[:, :, half_rope_dim:rope_dim] = x2 * cos + x1 * sin
    return out


def _torch_split_qkv_rmsnorm_mrope(
    *,
    qkv: "torch.Tensor",
    q_weight: "torch.Tensor",
    k_weight: "torch.Tensor",
    cos_sin: "torch.Tensor",
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    _validate_runtime_config(
        num_tokens=qkv.shape[0],
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size
    q = qkv[:, :q_width].reshape(qkv.shape[0], num_q_heads, head_size)
    gate = qkv[:, q_width : q_width * 2]
    k = qkv[:, q_width * 2 : q_width * 2 + kv_width].reshape(qkv.shape[0], num_kv_heads, head_size)
    v = qkv[:, q_width * 2 + kv_width : q_width * 2 + kv_width * 2]

    q_norm = _torch_rms_norm(q, q_weight[0], eps)
    k_norm = _torch_rms_norm(k, k_weight[0], eps)
    if is_interleaved:
        cos_rows, sin_rows = _assemble_interleaved_mrope_rows(cos_sin, mrope_section)
    else:
        cos_rows, sin_rows = _assemble_non_interleaved_mrope_rows(cos_sin, mrope_section)

    q_out = _apply_partial_mrope(q_norm, cos_rows, sin_rows, rope_dim)
    k_out = _apply_partial_mrope(k_norm, cos_rows, sin_rows, rope_dim)
    return (
        q_out.reshape(qkv.shape[0], q_width).to(torch.bfloat16),
        k_out.reshape(qkv.shape[0], kv_width).to(torch.bfloat16),
        v.to(torch.bfloat16),
        gate.to(torch.bfloat16),
    )


def _run_ref_check(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning("Skip split_qkv_rmsnorm_mrope reference check: NPU is not available")
        return

    _validate_runtime_config(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    torch.manual_seed(42)
    device = torch.device("npu")
    half_rope_dim = rope_dim // 2
    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size
    qkv_stride = q_width * 2 + kv_width * 2

    qkv = torch.randn((num_tokens, qkv_stride), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    phase = torch.randn((3, num_tokens, half_rope_dim), device=device)
    cos_sin = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1).to(torch.bfloat16)

    q_out = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)
    k_out = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
    v_out = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
    gate_out = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)

    kernel = split_qkv_rmsnorm_mrope_kernel_jit(
        head_size=head_size,
        rope_dim=rope_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vec_core_num=detect_vec_core_num(),
        max_num_tokens=num_tokens,
    )
    cos_sin_reshaped = cos_sin.permute(1, 0, 2).contiguous().view(num_tokens, 3 * rope_dim)
    gather_pad_dim = ((3 * rope_dim + 127) // 128) * 128
    gather_pat = build_mrope_gather_pattern_merged(
        half_rope_dim=half_rope_dim,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        gather_pad_dim=gather_pad_dim,
        device=str(device),
    )
    kernel(
        qkv,
        q_weight,
        k_weight,
        cos_sin_reshaped,
        gather_pat,
        q_out,
        k_out,
        v_out,
        gate_out,
        num_tokens,
        eps,
    )
    torch.npu.synchronize()

    q_ref, k_ref, v_ref, gate_ref = _torch_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    torch.testing.assert_close(
        q_out.to(torch.float32),
        q_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        k_out.to(torch.float32),
        k_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(v_out, v_ref, rtol=0, atol=0)
    torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    il_tag = "interleaved" if is_interleaved else "non-interleaved"
    logger.info(
        "split_qkv_rmsnorm_mrope output matches torch reference for num_tokens=%s, num_q_heads=%s, num_kv_heads=%s, %s",
        num_tokens,
        num_q_heads,
        num_kv_heads,
        il_tag,
    )


def _run_ref_suite(
    *,
    num_tokens: int,
    head_configs: list[tuple[int, int]],
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    for num_q_heads, num_kv_heads in head_configs:
        _run_ref_check(
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            rope_dim=rope_dim,
            eps=eps,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
        )

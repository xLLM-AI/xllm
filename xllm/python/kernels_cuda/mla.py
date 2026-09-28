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

"""CUDA contracts for DeepSeek MLA preprocessing kernels."""

from __future__ import annotations

import torch


def deepseek_mla_preprocess_decode(
    hidden: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Report that fused decode preprocessing is NPU-only."""
    del (
        hidden,
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        w_uk,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        q_lora_rank,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        q_norm_epsilon,
        kv_norm_epsilon,
    )
    raise NotImplementedError("deepseek_mla_preprocess_decode is available only on NPU")


def has_mla_preprocess_v2() -> bool:
    """CUDA does not provide the NPU MLAPO v2 operator."""
    return False


def prepare_mla_preprocess_v2_qkv(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Report that MLAPO-v2 weight packing is NPU-only."""
    del weight, descale, bias, kv_lora_rank, qk_rope_head_dim
    raise NotImplementedError("prepare_mla_preprocess_v2_qkv is available only on NPU")


def prepare_mla_preprocess_v2_q_b(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Report that MLAPO-v2 weight packing is NPU-only."""
    del weight, descale, bias, num_heads, qk_nope_head_dim, qk_rope_head_dim
    raise NotImplementedError("prepare_mla_preprocess_v2_q_b is available only on NPU")


def deepseek_mla_preprocess_decode_v2(
    hidden: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    w_uk: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Report that MLAPO-v2 decode preprocessing is NPU-only."""
    del (
        hidden,
        input_norm_weight,
        input_norm_bias,
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_norm_bias,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        w_uk,
        kv_cache,
        rope_cache,
        slot_mapping,
        kv_lora_rank,
        q_lora_rank,
        qk_rope_head_dim,
        norm_epsilon,
    )
    raise NotImplementedError("deepseek_mla_preprocess_decode_v2 is available only on NPU")


__all__ = [
    "deepseek_mla_preprocess_decode",
    "deepseek_mla_preprocess_decode_v2",
    "has_mla_preprocess_v2",
    "prepare_mla_preprocess_v2_q_b",
    "prepare_mla_preprocess_v2_qkv",
]

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

"""Capturable DeepSeek MLA preprocessing regions for Ascend NPU."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu

from .normalization import rms_norm
from .quantization import quant_matmul, quantize_per_tensor

_FRACTAL_NZ_FORMAT = 29
_KROPE_CTKV_CACHE_MODE = 1
_NZ_CACHE_MODE = 3
_PER_TENSOR_QUANT_ASYMM_MODE = 0
_INT8_NZ_ROW_BLOCK_SIZE = 16
_INT8_NZ_COLUMN_BLOCK_SIZE = 32
_KV_RMSNORM_ROPE_CACHE = getattr(
    torch_npu,
    "npu_kv_rmsnorm_rope_cache",
    None,
)


def has_mla_preprocess_v2() -> bool:
    """Return whether the xLLM MLAPO v2 operator is registered."""
    return hasattr(torch.ops.xllm_ops, "mla_preprocess_v2")


def _reorder_rope_axis(
    tensor: torch.Tensor,
    rope_dim: int,
    dim: int,
) -> torch.Tensor:
    """Move even RoPE channels before odd channels on one tensor axis."""
    axis_size = tensor.shape[dim]
    rope_start = axis_size - rope_dim
    prefix = tensor.narrow(dim, 0, rope_start)
    rope = tensor.narrow(dim, rope_start, rope_dim)
    even = rope.index_select(
        dim,
        torch.arange(0, rope_dim, 2, device=tensor.device),
    )
    odd = rope.index_select(
        dim,
        torch.arange(1, rope_dim, 2, device=tensor.device),
    )
    return torch.cat((prefix, even, odd), dim=dim).contiguous()


def _pack_mla_int8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack a two-dimensional int8 weight into MLAPO's NZ block order."""
    rows, columns = weight.shape
    padded_rows = (rows + _INT8_NZ_ROW_BLOCK_SIZE - 1) // _INT8_NZ_ROW_BLOCK_SIZE * _INT8_NZ_ROW_BLOCK_SIZE
    padded_columns = (
        (columns + _INT8_NZ_COLUMN_BLOCK_SIZE - 1) // _INT8_NZ_COLUMN_BLOCK_SIZE * _INT8_NZ_COLUMN_BLOCK_SIZE
    )
    padded = F.pad(
        weight,
        (0, padded_columns - columns, 0, padded_rows - rows),
    )
    packed = padded.reshape(
        padded_rows // _INT8_NZ_ROW_BLOCK_SIZE,
        _INT8_NZ_ROW_BLOCK_SIZE,
        padded_columns // _INT8_NZ_COLUMN_BLOCK_SIZE,
        _INT8_NZ_COLUMN_BLOCK_SIZE,
    ).permute(2, 0, 1, 3)
    return (
        packed.reshape(
            packed.shape[0],
            packed.shape[1] * packed.shape[2],
            packed.shape[3],
        )
        .unsqueeze(0)
        .contiguous()
    )


def prepare_mla_preprocess_v2_qkv(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare fused KV-A/Q-A tensors for ``MlaPreprocessV2``."""
    kv_dim = kv_lora_rank + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        kv = _reorder_rope_axis(tensor.narrow(0, 0, kv_dim), qk_rope_head_dim, 0)
        q = tensor.narrow(0, kv_dim, tensor.shape[0] - kv_dim)
        return torch.cat((kv, q), dim=0).contiguous()

    prepared_weight = _prepare(weight)
    if prepared_weight.device.type != "cpu":
        prepared_weight = torch_npu.npu_format_cast(prepared_weight, _FRACTAL_NZ_FORMAT)
    return prepared_weight, _prepare(descale), _prepare(bias)


def prepare_mla_preprocess_v2_q_b(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q-B tensors for ``MlaPreprocessV2``."""
    head_dim = qk_nope_head_dim + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        viewed = tensor.view(num_heads, head_dim, *shape[1:])
        reordered = _reorder_rope_axis(viewed, qk_rope_head_dim, 1)
        return reordered.reshape(shape).contiguous()

    prepared_weight = _pack_mla_int8_weight(_prepare(weight))
    if prepared_weight.device.type != "cpu":
        prepared_weight = torch_npu.npu_format_cast(prepared_weight, _FRACTAL_NZ_FORMAT)
    return prepared_weight, _prepare(descale), _prepare(bias)


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
    """Run the fused xLLM MLAPO v2 decode path."""
    cache_mode = _NZ_CACHE_MODE if _mla_cache_mode(kv_cache) == "PA_NZ" else _KROPE_CTKV_CACHE_MODE
    num_tokens = hidden.shape[0]
    q_latent, _, q_pe, _, q_c = torch.ops.xllm_ops.mla_preprocess_v2(
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
        rope_cos.view(num_tokens, -1),
        rope_sin.view(num_tokens, -1),
        w_uk,
        kv_cache,
        rope_cache,
        slot_mapping,
        qkv_input_scale,
        q_b_input_scale,
        q_lora_rank,
        qk_rope_head_dim,
        qk_rope_head_dim,
        norm_epsilon,
        2,
        2,
        True,
        True,
        True,
        cache_mode,
        _PER_TENSOR_QUANT_ASYMM_MODE,
        False,
        1,
        True,
    )
    return q_c, q_latent, q_pe


def _mla_cache_mode(kv_cache: torch.Tensor) -> str:
    """Return the cache layout name expected by the fused KV operator."""
    get_npu_format = getattr(torch_npu, "get_npu_format", None)
    if get_npu_format is not None and get_npu_format(kv_cache) == _FRACTAL_NZ_FORMAT:
        return "PA_NZ"
    return "PA"


def _write_mla_kv_cache(
    kv: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_norm_epsilon: float,
) -> None:
    """Normalize and rotate MLA KV, then write it into the paged cache."""
    num_tokens = kv.shape[0]
    cache_mode = _mla_cache_mode(kv_cache)
    # torch-npu does not accept xLLM's internal FRACTAL_NZ cache as PA_NZ.
    if _KV_RMSNORM_ROPE_CACHE is not None and cache_mode == "PA":
        kv_no_split = kv.view(
            num_tokens,
            1,
            1,
            kv_lora_rank + qk_rope_head_dim,
        )
        _KV_RMSNORM_ROPE_CACHE(
            kv_no_split,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            slot_mapping.to(torch.int64),
            rope_cache,
            kv_cache,
            epsilon=kv_norm_epsilon,
            cache_mode=cache_mode,
        )
        return

    k_latent_raw, k_rope_raw = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_latent = rms_norm(k_latent_raw, kv_norm_weight, kv_norm_epsilon).view(num_tokens, 1, kv_lora_rank)
    k_pe = torch_npu.npu_interleave_rope(
        k_rope_raw.view(num_tokens, 1, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, 1, qk_rope_head_dim)
    torch.ops.xllm_ops.reshape_paged_cache(
        slot_mapping,
        k_latent,
        k_pe,
        kv_cache,
        rope_cache,
    )


@torch.library.custom_op(
    "xllm_python::deepseek_mla_preprocess_decode",
    mutates_args={"kv_cache", "rope_cache"},
)
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
    """Run the complete capturable DeepSeek decode MLA preprocessing path."""
    hidden_int8 = quantize_per_tensor(
        hidden,
        qkv_input_scale,
        qkv_input_offset,
        torch.qint8,
        -1,
    )
    qkv_a = quant_matmul(
        hidden_int8,
        qkv_weight,
        False,
        qkv_deq_scale,
        None,
        None,
        qkv_quant_bias,
        torch.bfloat16,
    )
    kv_dim = kv_lora_rank + qk_rope_head_dim
    kv, q_a = qkv_a.split([kv_dim, q_lora_rank], dim=-1)
    q_c = rms_norm(q_a, q_norm_weight, q_norm_epsilon)
    q_c_quant = quantize_per_tensor(
        q_c,
        q_b_input_scale,
        q_b_input_offset,
        torch.qint8,
        -1,
    )
    q = quant_matmul(
        q_c_quant,
        q_b_weight,
        False,
        q_b_deq_scale,
        None,
        None,
        q_b_quant_bias,
        torch.bfloat16,
    ).view(
        hidden.shape[0],
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
    )
    q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_latent = torch.bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)
    num_tokens = hidden.shape[0]
    q_pe = torch_npu.npu_interleave_rope(
        q_rope.view(num_tokens, num_heads, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, num_heads, qk_rope_head_dim)
    _write_mla_kv_cache(
        kv,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_norm_epsilon,
    )
    return q_c, q_latent, q_pe


@deepseek_mla_preprocess_decode.register_fake
def _deepseek_mla_preprocess_decode_fake(
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
    del (
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
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_nope_head_dim,
        q_norm_epsilon,
        kv_norm_epsilon,
    )
    num_tokens = hidden.shape[0]
    q_c = hidden.new_empty((num_tokens, q_lora_rank))
    q_latent = hidden.new_empty((num_tokens, num_heads, w_uk.shape[-1]))
    q_pe = hidden.new_empty((num_tokens, num_heads, qk_rope_head_dim))
    return q_c, q_latent, q_pe


__all__ = [
    "deepseek_mla_preprocess_decode",
    "deepseek_mla_preprocess_decode_v2",
    "has_mla_preprocess_v2",
    "prepare_mla_preprocess_v2_q_b",
    "prepare_mla_preprocess_v2_qkv",
]

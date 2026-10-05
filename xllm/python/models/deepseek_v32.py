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

"""Shared DeepSeek-V3.2/GLM target, decoder, MLA and indexer computation.

GLM configures checkpoint projections, index sharing and parallel scheduling
through local methods; mathematical forward paths are owned here.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    MlaIndexContext,
    MlaPreprocessContext,
)
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.model_executor.cp_utils import (
    CpContext,
    cp_gather_kv,
    cp_merge_rows,
    cp_shard_positions,
    cp_shard_rows,
)
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.model_loader import (
    W8A8WeightLoader,
    mla_head_split,
    moe_shard,
)
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase

_SHARED_EXPERT_STREAMS: dict[tuple[str, int | None], torch.npu.Stream] = {}


def _shared_expert_stream(device: torch.device) -> torch.npu.Stream:
    key = (device.type, device.index)
    stream = _SHARED_EXPERT_STREAMS.get(key)
    if stream is None:
        stream = torch.npu.Stream(device=device)
        _SHARED_EXPERT_STREAMS[key] = stream
    return stream


_GATE_STREAMS: dict[tuple[str, int | None], torch.npu.Stream] = {}


def _gate_stream(device: torch.device) -> torch.npu.Stream:
    key = (device.type, device.index)
    stream = _GATE_STREAMS.get(key)
    if stream is None:
        stream = torch.npu.Stream(device=device)
        _GATE_STREAMS[key] = stream
    return stream


def _tp_rank_from_device(device: object) -> int:
    """Local device index from the worker device string ("npu:3" -> 3)."""
    s = str(device)
    if ":" in s:
        try:
            return int(s.rsplit(":", 1)[-1])
        except ValueError:
            return 0
    return 0


def _create_hadamard_matrix(
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.size(0) < dim:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix.contiguous()


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    """YaRN magnitude scaling factor."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = math.floor(low)
    high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity.
    linear = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def _gather_half_rope_cos_sin(
    cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = cos_sin_cache[positions]
    half = cos_sin.size(-1) // 2
    return cos_sin[..., :half], cos_sin[..., half:]


def _expand_interleave_rope_cos_sin(
    half_cos: torch.Tensor, half_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.cat([half_cos, half_cos], dim=-1).unsqueeze(1).unsqueeze(1)
    sin = torch.cat([half_sin, half_sin], dim=-1).unsqueeze(1).unsqueeze(1)
    return cos, sin


def _gather_interleave_cos_sin(
    cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-token cos/sin and double for ``npu_interleave_rope``."""
    return _expand_interleave_rope_cos_sin(*_gather_half_rope_cos_sin(cos_sin_cache, positions))


def _interleave_rope_with(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE to ``[T, H, D]`` with precomputed cos/sin."""
    return kernels.interleaved_rotary_embedding(x, cos, sin)


def _apply_half_rope(cos_sin_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Half-rotate RoPE (NeoX style) for ``[T, H, D]`` tensors."""
    half_cos, half_sin = _gather_half_rope_cos_sin(cos_sin_cache, positions)
    return _apply_half_rope_with_cos_sin(x, half_cos, half_sin)


def _apply_half_rope_with_cos_sin(x: torch.Tensor, half_cos: torch.Tensor, half_sin: torch.Tensor) -> torch.Tensor:
    c = half_cos.unsqueeze(1)
    s = half_sin.unsqueeze(1)
    half = half_cos.size(-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _select_indexer_query_cos_sin(
    interleaved: bool,
    half_rope_cos: torch.Tensor,
    half_rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cp_context: CpContext | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select compact CP query rows once; cache K keeps the padded local rows."""
    cos, sin = (rope_cos, rope_sin) if interleaved else (half_rope_cos, half_rope_sin)
    if cp_context is not None:
        return cos.index_select(0, cp_context.query_index), sin.index_select(0, cp_context.query_index)
    return cos, sin


def _validate_rope_cos_sin(
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    value: torch.Tensor,
    rope_dim: int,
    interleaved: bool,
    consumer: str,
    dtype: torch.dtype | None = None,
) -> None:
    """Check tensor metadata without synchronizing captured device values."""
    expected = (value.shape[0], 1, 1, rope_dim) if interleaved else (value.shape[0], rope_dim // 2)
    expected_dtype = value.dtype if dtype is None else dtype
    for name, coefficient in zip(("cos", "sin"), cos_sin):
        if coefficient.shape != expected or coefficient.dtype != expected_dtype or coefficient.device != value.device:
            raise ValueError(
                f"{consumer} {name}: expected shape={expected}, dtype={expected_dtype}, device={value.device}; "
                f"got shape={tuple(coefficient.shape)}, dtype={coefficient.dtype}, device={coefficient.device}"
            )


class DeepseekYarnRotaryEmbedding(RotaryEmbedding):
    """YaRN-scaled RoPE for DeepSeek-V3.2."""

    def __init__(
        self,
        head_dim: int,
        original_max_position_embeddings: int,
        scaling_factor: float,
        base: float,
        beta_fast: int,
        beta_slow: int,
        mscale: float,
        mscale_all_dim: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.head_dim = head_dim
        inv_freq = self._yarn_inv_freq(
            scaling_factor,
            head_dim,
            base,
            beta_fast,
            beta_slow,
            original_max_position_embeddings,
            device,
        )
        t = torch.arange(
            int(original_max_position_embeddings * scaling_factor),
            dtype=torch.float32,
            device=device,
        )
        freqs = torch.outer(t, inv_freq)
        rope_mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
        cos = freqs.cos() * rope_mscale
        sin = freqs.sin() * rope_mscale
        cache = torch.cat([cos, sin], dim=-1)
        if dtype is not None:
            cache = cache.to(dtype)
        self.register_buffer("cos_sin_cache", cache.contiguous(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_cos, half_sin = _gather_half_rope_cos_sin(self.cos_sin_cache, positions)
        cos, sin = _expand_interleave_rope_cos_sin(half_cos, half_sin)
        return half_cos, half_sin, cos, sin

    @staticmethod
    def _yarn_inv_freq(
        scaling_factor: float,
        rotary_dim: int,
        base: float,
        beta_fast: int,
        beta_slow: int,
        max_position_embeddings: int,
        device: torch.device,
    ) -> torch.Tensor:
        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast,
            beta_slow,
            rotary_dim,
            base,
            max_position_embeddings,
        )
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float32, device)
        return inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask


@dataclass
class DeepseekV3Config:
    """DeepSeek-V3.2 architecture parameters."""

    hidden_size: int = 2048
    n_layers: int = 61
    n_heads: int = 128
    head_dim: int = 0
    intermediate_size: int = 10240
    vocab_size: int = 129280
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1.0e6
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    rope_scaling_factor: float = 40.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    tie_word_embeddings: bool = False
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 2048
    tp_size: int = 1  # TP for dense layers (attn, embed, shared expert, lm_head)
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    moe_tp_size: int = 1  # TP for routed MoE experts (equals tp_size when ep_size == 1)
    moe_tp_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV3Config:
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        rs_raw = d.get("rope_scaling")
        rs = rs_raw if isinstance(rs_raw, dict) else {}

        def rpick(*keys, default=None):
            for k in keys:
                if isinstance(rs, dict) and k in rs and rs[k] is not None:
                    return rs[k]
                fk = f"rope_scaling_{k}"
                if fk in d and d[fk] is not None:
                    return d[fk]
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=2048))
        n_heads = int(pick("n_heads", "num_attention_heads", default=128))
        return cls(
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=61)),
            n_heads=n_heads,
            head_dim=int(pick("head_dim", default=hidden // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=10240)),
            vocab_size=int(pick("vocab_size", default=129280)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1.0e6)),
            max_position_embeddings=int(pick("max_position_embeddings", default=4096)),
            original_max_position_embeddings=int(rpick("original_max_position_embeddings", default=4096)),
            rope_scaling_factor=float(rpick("factor", "rope_scaling_factor", default=40.0)),
            rope_beta_fast=int(rpick("beta_fast", default=32)),
            rope_beta_slow=int(rpick("beta_slow", default=1)),
            rope_mscale=float(rpick("mscale", default=1.0)),
            rope_mscale_all_dim=float(rpick("mscale_all_dim", default=1.0)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            index_n_heads=int(pick("index_n_heads", default=64)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=128)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            v_head_dim=int(pick("v_head_dim", default=128)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=3)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            n_routed_experts=int(pick("n_routed_experts", default=256)),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=8)),
            topk_group=int(pick("topk_group", default=4)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=2048)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            ep_size=int(pick("ep_size", default=1)),
            ep_rank=int(pick("ep_rank", default=0)),
            dp_size=int(pick("dp_size", default=1)),
            dp_rank=int(pick("dp_rank", default=0)),
            moe_tp_size=int(pick("moe_tp_size", default=1)),
            moe_tp_rank=int(pick("moe_tp_rank", default=0)),
            world_size=int(pick("world_size", default=1)),
        )

    def head_split(self) -> tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1) — MLA has one latent KV head per rank."""
        return mla_head_split(self.n_heads, self.tp_size)

    def validate(self) -> None:
        if self.ep_size not in (1, self.world_size):
            raise ValueError(f"ep_size must be 1 or world_size ({self.world_size}), got {self.ep_size}")
        if self.ep_size > 1 and self.n_routed_experts % self.ep_size:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        if self.ep_size > 1 and self.moe_tp_size * self.ep_size != self.world_size:
            raise ValueError(
                f"world_size ({self.world_size}) must equal moe_tp_size ({self.moe_tp_size}) * ep_size ({self.ep_size})"
            )


class W8A8AttentionLinear(nn.Module):
    """Attention linear compatible with static and dynamic W8A8 checkpoints."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        row_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.row_parallel = row_parallel
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer("deq_scale", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("quant_bias", torch.empty(0, dtype=torch.int32, device=device))
        self.register_buffer("input_scale", torch.empty(0, dtype=torch.bfloat16, device=device))
        self.register_buffer("input_offset", torch.empty(0, dtype=torch.bfloat16, device=device))
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("weight_offset", torch.empty(0, dtype=torch.float32, device=device))
        self._dynamic_activation: bool | None = None

    def _set_dynamic_activation(self, enabled: bool) -> None:
        self._dynamic_activation = enabled
        device = self.weight.device
        if enabled:
            self.weight_scale.data = torch.empty(self.out_features, 1, dtype=torch.float32, device=device)
            self.weight_offset.data = torch.empty(self.out_features, 1, dtype=torch.float32, device=device)
            self.deq_scale.data = torch.empty(0, dtype=torch.float32, device=device)
            self.quant_bias.data = torch.empty(0, dtype=torch.int32, device=device)
            self.input_scale.data = torch.empty(0, dtype=torch.bfloat16, device=device)
            self.input_offset.data = torch.empty(0, dtype=torch.bfloat16, device=device)
            return
        self.deq_scale.data = torch.empty(self.out_features, dtype=torch.float32, device=device)
        self.quant_bias.data = torch.empty(self.out_features, dtype=torch.int32, device=device)
        self.input_scale.data = torch.empty(1, dtype=torch.bfloat16, device=device)
        self.input_offset.data = torch.empty(1, dtype=torch.bfloat16, device=device)
        self.weight_scale.data = torch.empty(0, dtype=torch.float32, device=device)
        self.weight_offset.data = torch.empty(0, dtype=torch.float32, device=device)

    def process_weights_after_loading(self) -> None:
        if self._dynamic_activation is None:
            raise RuntimeError("W8A8 attention quantization format must be selected before processing weights")
        if not self._dynamic_activation:
            self.weight.data = kernels.prepare_quant_weight(self.weight.data)
            return
        if not bool(torch.all(self.weight_offset == 0)):
            raise ValueError("dynamic W8A8 attention requires symmetric INT8 weights with zero weight_offset")
        self.weight.data = kernels.prepare_quant_weight(self.weight.data)
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._dynamic_activation is None:
            raise RuntimeError("W8A8 attention quantization format must be selected before execution")
        if self._dynamic_activation:
            x_int8, pertoken = kernels.dynamic_quant(x)
            return self.forward_quantized(x_int8, pertoken)
        x_int8 = kernels.quantize_per_tensor(x, self.input_scale, self.input_offset, torch.qint8, -1)
        return self.forward_quantized(x_int8)

    def forward_quantized(self, x_int8: torch.Tensor, pertoken: torch.Tensor | None = None) -> torch.Tensor:
        if self._dynamic_activation is None:
            raise RuntimeError("W8A8 attention quantization format must be selected before execution")
        if self._dynamic_activation != (pertoken is not None):
            raise ValueError("W8A8 activation scale does not match the loaded quantization format")
        bias = None
        if not self._dynamic_activation and not (self.row_parallel and distributed.tp_rank(x_int8.device) != 0):
            bias = self.quant_bias
        return kernels.quant_matmul(
            x_int8,
            self.weight,
            False,
            self.weight_scale if self._dynamic_activation else self.deq_scale,
            None,
            pertoken,
            bias,
            torch.bfloat16,
        )

    @classmethod
    def combine(cls, kv: W8A8AttentionLinear, q: W8A8AttentionLinear) -> W8A8AttentionLinear | None:
        """Choose the legal A projection once, before preparing either weight.

        The derived buffers are nonpersistent: checkpoint keys remain on the
        original KV/Q modules. Static projections with different activation
        scales or offsets, and mixed static/dynamic formats, stay separate.
        """
        if kv._dynamic_activation is None or q._dynamic_activation is None:
            raise RuntimeError("Select both A projection formats before combining weights")
        if kv.in_features != q.in_features or kv.row_parallel or q.row_parallel:
            raise ValueError("Combined Q/KV A projections require the same input width and replicated inputs")
        if kv._dynamic_activation != q._dynamic_activation:
            return None
        if not kv._dynamic_activation and not (
            torch.equal(kv.input_scale, q.input_scale) and torch.equal(kv.input_offset, q.input_offset)
        ):
            return None
        combined = cls.__new__(cls)
        nn.Module.__init__(combined)
        combined.in_features = kv.in_features
        combined.out_features = kv.out_features + q.out_features
        combined.row_parallel = False
        combined._dynamic_activation = kv._dynamic_activation
        combined.register_buffer("weight", torch.cat((kv.weight, q.weight), dim=0), persistent=False)
        for name in ("deq_scale", "quant_bias", "weight_scale", "weight_offset"):
            combined.register_buffer(name, torch.cat((getattr(kv, name), getattr(q, name)), dim=0), persistent=False)
        for name in ("input_scale", "input_offset"):
            combined.register_buffer(name, getattr(kv, name), persistent=False)
        return combined


class W8A8StaticLinear(W8A8AttentionLinear):
    """Static checkpoint adapter for the shared attention projection."""

    def __init__(self, in_features: int, out_features: int, device: torch.device, row_parallel: bool = False) -> None:
        super().__init__(in_features, out_features, device, row_parallel)
        self._set_dynamic_activation(False)
        # These fields are unused by static checkpoints.
        self._non_persistent_buffers_set.update(("weight_scale", "weight_offset"))


class RouterGate(nn.Module):
    """MoE router gate: bf16 [T, H] x bf16 [H, E] -> fp32 [T, E].

    Keep the checkpoint-facing [E, H] weight as a view of contiguous [H, E]
    storage, so addmm reads it without a second weight copy.
    """

    def __init__(self, hidden_size: int, num_experts: int, device: torch.device) -> None:
        super().__init__()
        # The loader sees [E, H], while weight.t() is contiguous [H, E].
        self.weight = nn.Parameter(
            torch.empty(hidden_size, num_experts, dtype=torch.bfloat16, device=device).t(),
            requires_grad=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.empty(
            (hidden.shape[0], self.weight.shape[0]),
            dtype=torch.float32,
            device=hidden.device,
        )
        return torch.addmm(logits, hidden, self.weight.t(), beta=0, alpha=1, out=logits)


class W8A8DynamicLinear(nn.Module):
    """Dynamic-activation W8A8 linear (MLP / experts)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        transpose_weight_after_loading: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transpose_weight_after_loading = transpose_weight_after_loading
        self._weight_is_transposed = False
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        self.register_buffer("weight_offset", torch.empty(out_features, 1, dtype=torch.float32, device=device))

    def process_weights_after_loading(self) -> None:
        if not bool(torch.all(self.weight_offset == 0)):
            raise ValueError("W8A8DynamicLinear requires symmetric INT8 weights with zero weight_offset")
        if self.transpose_weight_after_loading:
            self.weight.data = kernels.prepare_quant_weight(self.weight.data)
            self._weight_is_transposed = True
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        return self.forward_quantized(x_int8, pertoken)

    def forward_quantized(
        self,
        x_int8: torch.Tensor,
        pertoken: torch.Tensor,
    ) -> torch.Tensor:
        return kernels.quant_matmul(
            x_int8,
            self.weight,
            not self._weight_is_transposed,
            self.weight_scale,
            None,
            pertoken.view(-1) if pertoken is not None else None,
            None,
            torch.bfloat16,
        )

    def forward_quantized_out(
        self,
        x_int8: torch.Tensor,
        pertoken: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return kernels.quant_matmul_out(
            x_int8,
            self.weight,
            transpose2=not self._weight_is_transposed,
            scale=self.weight_scale,
            offset=None,
            pertoken_scale=pertoken,
            bias=None,
            output_dtype=torch.bfloat16,
            out=output,
        )

    def forward_accumulated(self, x_int8: torch.Tensor) -> torch.Tensor:
        return kernels.quant_matmul(
            x_int8,
            self.weight,
            not self._weight_is_transposed,
            self.weight_scale,
            None,
            None,
            None,
            torch.int32,
        )


def _swiglu_with_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if 0.0 < limit < 1_000_000.0:
        gate = gate.float().clamp_max(limit)
        up = up.float().clamp(min=-limit, max=limit)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)
    return kernels.silu_and_mul(x)


class DeepseekV3MLP(nn.Module):
    """Dense gated-SiLU FFN (layers < first_k_dense_replace)."""

    def __init__(
        self,
        cfg: DeepseekV3Config,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
        skip_tp_reduce: bool = False,
        tp_override: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        tp = tp_override if tp_override is not None else cfg.tp_size
        assert intermediate_size % tp == 0, f"intermediate_size {intermediate_size} not divisible by tp {tp}"
        inter_local = intermediate_size // tp
        self.tp = tp
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = W8A8DynamicLinear(cfg.hidden_size, 2 * inter_local, device)
        self.down_proj = W8A8DynamicLinear(
            inter_local,
            cfg.hidden_size,
            device,
        )

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def load_from_checkpoint(
        self,
        loader: W8A8WeightLoader,
        mlp_prefix: str,
        *,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Load a dense W8A8 MLP; ``world``/``rank`` = ``None`` means use the loader's TP."""
        loader.load_w8a8_mlp(mlp_prefix, world=world, rank=rank)
        self.process_weights_after_loading()

    def forward(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self._forward_gate_up(gate_up, tp_reduce_add)

    def forward_quantized(
        self,
        x_int8: torch.Tensor,
        pertoken: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward_quantized(x_int8, pertoken)
        return self._forward_gate_up(gate_up, tp_reduce_add)

    def forward_dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        act_int8, act_scale = kernels.dequant_swiglu_quant(
            gate_up,
            self.gate_up_proj.weight_scale,
            pertoken,
        )
        if output is None:
            out = self.down_proj.forward_quantized(act_int8, act_scale)
        else:
            out = self.down_proj.forward_quantized_out(act_int8, act_scale, output)
        return self._reduce_output(out, tp_reduce_add)

    def quantize_and_project_gate_up(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_int8, pertoken = kernels.dynamic_quant(x)
        return self.gate_up_proj.forward_accumulated(x_int8), pertoken

    def activate_and_quantize(self, gate_up: torch.Tensor, pertoken: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernels.dequant_swiglu_quant(gate_up, self.gate_up_proj.weight_scale, pertoken)

    def project_down(self, act_int8: torch.Tensor, act_scale: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward_quantized(act_int8, act_scale)

    def _forward_gate_up(
        self,
        gate_up: torch.Tensor,
        tp_reduce_add: torch.Tensor | None,
    ) -> torch.Tensor:
        act = _swiglu_with_clamp(gate_up, self.swiglu_limit)
        out = self.down_proj(act)
        return self._reduce_output(out, tp_reduce_add)

    def _reduce_output(self, out: torch.Tensor, tp_reduce_add: torch.Tensor | None) -> torch.Tensor:
        if tp_reduce_add is not None:
            out = out + tp_reduce_add
        if self.tp > 1 and (not self.skip_tp_reduce or tp_reduce_add is not None):
            distributed.tp_all_reduce(out)
        return out


class DeepseekV3MLAAttention(Attention):
    """Absorbed-MLA attention. KV cache stores latent (kv_lora) + rope."""

    _linear_type = W8A8StaticLinear

    def __init__(
        self,
        cfg: DeepseekV3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        num_heads, _ = cfg.head_split()
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v_head = cfg.v_head_dim
        scale = (qk_nope + qk_rope) ** -0.5
        attn_mscale = _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
        scale = scale * attn_mscale * attn_mscale
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=kv_lora,
            scale=scale,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.qk_nope_head_dim = qk_nope
        self.qk_rope_head_dim = qk_rope
        self.v_head_dim = v_head
        self.kv_lora_rank = kv_lora
        self.num_heads_local = num_heads
        self.q_lora_rank = cfg.q_lora_rank
        self._use_fused_mla_decode = device.type in (
            "npu",
            "privateuseone",
        )
        self._use_mlapo_v2 = self._mlapo_enabled(cfg, device)

        self._fused_mla_ready = False
        self._dynamic_mla_ready = False

        self._init_a_projections(cfg, device)
        self.q_b_proj = self._linear_type(cfg.q_lora_rank, num_heads * (qk_nope + qk_rope), device)
        self.o_proj = self._linear_type(
            num_heads * v_head,
            cfg.hidden_size,
            device,
            row_parallel=True,
        )
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(kv_lora, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora,
            num_heads * (qk_nope + v_head),
            tp,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "W_UK",
            torch.empty(num_heads, qk_nope, kv_lora, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(num_heads, kv_lora, v_head, dtype=dtype, device=device),
            persistent=False,
        )
        for name in (
            "_mlapo_input_norm_weight",
            "_mlapo_input_norm_bias",
            "_mlapo_q_norm_bias",
            "_mlapo_qkv_input_offset",
            "_mlapo_qkv_weight",
            "_mlapo_qkv_deq_scale",
            "_mlapo_qkv_quant_bias",
            "_mlapo_q_b_input_offset",
            "_mlapo_q_b_weight",
            "_mlapo_q_b_deq_scale",
            "_mlapo_q_b_quant_bias",
        ):
            self.register_buffer(
                name,
                torch.empty(0, dtype=dtype, device=device),
                persistent=False,
            )
        self.indexer = self._make_indexer(cfg, layer_id, dtype, device)

    def _mlapo_enabled(self, cfg: DeepseekV3Config, device: torch.device) -> bool:
        return (
            self._use_fused_mla_decode
            and os.environ.get("XLLM_ENABLE_MLAPO_V2") == "1"
            and kernels.has_mla_preprocess_v2()
        )

    def _init_a_projections(self, cfg: DeepseekV3Config, device: torch.device) -> None:
        self.qkv_a_proj = W8A8StaticLinear(
            cfg.hidden_size, cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim, device
        )

    def _make_indexer(
        self, cfg: DeepseekV3Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> DeepseekV3Indexer | None:
        return DeepseekV3Indexer(cfg, dtype, device, layer_id) if cfg.index_topk > 0 else None

    def process_weights_after_loading(self) -> None:
        projection = self._prepare_a_projection()
        self._fused_mla_ready = projection is not None and (
            projection._dynamic_activation == self.q_b_proj._dynamic_activation
        )
        self._dynamic_mla_ready = self._fused_mla_ready and projection._dynamic_activation
        self._use_mlapo_v2 = self._use_mlapo_v2 and self._fused_mla_ready and not self._dynamic_mla_ready
        if self._use_mlapo_v2:
            (
                self._mlapo_qkv_weight,
                self._mlapo_qkv_deq_scale,
                self._mlapo_qkv_quant_bias,
            ) = kernels.prepare_mla_preprocess_v2_qkv(
                projection.weight.data,
                projection.deq_scale,
                projection.quant_bias,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            (
                self._mlapo_q_b_weight,
                self._mlapo_q_b_deq_scale,
                self._mlapo_q_b_quant_bias,
            ) = kernels.prepare_mla_preprocess_v2_q_b(
                self.q_b_proj.weight.data,
                self.q_b_proj.deq_scale,
                self.q_b_proj.quant_bias,
                self.num_heads_local,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            )
            self._mlapo_input_norm_weight = torch.ones(
                self.cfg.hidden_size,
                dtype=self.q_a_layernorm.weight.dtype,
                device=self.q_a_layernorm.weight.device,
            )
            self._mlapo_input_norm_bias = torch.zeros_like(self._mlapo_input_norm_weight)
            self._mlapo_q_norm_bias = torch.zeros_like(self.q_a_layernorm.weight)
            self._mlapo_qkv_input_offset = projection.input_offset.to(torch.int8)
            self._mlapo_q_b_input_offset = self.q_b_proj.input_offset.to(torch.int8)
        if projection is not None:
            projection.process_weights_after_loading()
        self._prepare_separate_a_projections()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())

        self._prepare_indexer_weights()

    def _prepare_a_projection(self) -> W8A8AttentionLinear | None:
        return self.qkv_a_proj

    def _prepare_separate_a_projections(self) -> None:
        pass

    def _prepare_indexer_weights(self) -> None:
        pass

    def _a_projection(self) -> W8A8AttentionLinear | None:
        return self.qkv_a_proj

    def _can_fuse_input_norm_quant(self) -> bool:
        return False

    def _validate_hidden_scale(self, hidden: torch.Tensor, hidden_scale: torch.Tensor | None) -> None:
        if hidden_scale is None:
            if not hidden.is_floating_point():
                raise ValueError("floating attention input is required when hidden_scale is absent")
            return
        if not self._can_fuse_input_norm_quant():
            raise ValueError("this attention configuration cannot consume quantized normalized hidden")
        if hidden.dtype != torch.int8 or hidden.ndim != 2 or hidden.shape[1] != self.cfg.hidden_size:
            raise ValueError("quantized attention input must be INT8 [tokens, hidden_size]")
        if (
            hidden_scale.dtype != torch.float32
            or hidden_scale.device != hidden.device
            or hidden_scale.shape not in ((hidden.shape[0],), (hidden.shape[0], 1))
        ):
            raise ValueError("hidden_scale must be FP32 with one scale per input row on the same device")

    def _project_qkv_a(
        self, hidden: torch.Tensor, hidden_scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self._a_projection()
        if hidden_scale is not None:
            self._validate_hidden_scale(hidden, hidden_scale)
            projected = projection.forward_quantized(hidden, hidden_scale.reshape(-1))
        elif projection is None:
            return self._project_separate_a(hidden)
        else:
            projected = projection(hidden)
        kv, q_a = projected.split([self.kv_lora_rank + self.qk_rope_head_dim, self.q_lora_rank], dim=-1)
        return q_a, kv

    def _project_separate_a(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("The loaded model requires a combined Q/KV A projection")

    def _can_fuse_q_norm_quant(self) -> bool:
        # Every consumer must accept the same dynamic INT8 activation. In
        # particular, DeepSeek's BF16 indexer still needs normalized Q-A.
        return (
            isinstance(self.q_b_proj, W8A8AttentionLinear)
            and self.q_b_proj._dynamic_activation is True
            and (
                self.indexer is None
                or (
                    isinstance(self.indexer.wq_b, W8A8AttentionLinear) and self.indexer.wq_b._dynamic_activation is True
                )
            )
        )

    def _normalize_and_project_query(
        self, q_a: torch.Tensor
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self._can_fuse_q_norm_quant():
            quantized_q = kernels.rms_norm_dynamic_quant(q_a, self.q_a_layernorm.weight, self.q_a_layernorm.eps)
            return quantized_q, self.q_b_proj.forward_quantized(*quantized_q)
        q_c = self.q_a_layernorm(q_a)
        return q_c, self.q_b_proj(q_c)

    def _preprocess_decode(
        self,
        hidden: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: MlaPreprocessContext,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        self._validate_hidden_scale(hidden, hidden_scale)
        projection = self._a_projection()
        assert projection is not None
        if projection._dynamic_activation:
            fuse_q_norm_quant = self._can_fuse_q_norm_quant()
            q_c, q_c_scale, q_latent, q_pe = kernels.deepseek_mla_preprocess_decode_dynamic(
                hidden,
                projection.weight,
                projection.weight_scale,
                self.q_a_layernorm.weight,
                self.q_b_proj.weight,
                self.q_b_proj.weight_scale,
                self.W_UK,
                self.kv_a_layernorm.weight,
                rope_cos,
                rope_sin,
                context.slot_mapping[: hidden.shape[0]],
                context.kv_cache,
                context.rope_cache,
                self.kv_lora_rank,
                self.q_lora_rank,
                self.num_heads_local,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.q_a_layernorm.eps,
                self.kv_a_layernorm.eps,
                fuse_q_norm_quant,
                hidden_scale,
                slot_mapping_int64[: hidden.shape[0]] if slot_mapping_int64 is not None else None,
            )
            if fuse_q_norm_quant:
                q_c = (q_c, q_c_scale)
        elif self._use_mlapo_v2 and hidden.shape[0] <= kernels.MLA_PREPROCESS_V2_MAX_TOKENS:
            q_c, q_latent, q_pe = kernels.deepseek_mla_preprocess_decode_v2(
                hidden,
                self._mlapo_input_norm_weight,
                self._mlapo_input_norm_bias,
                projection.input_scale,
                self._mlapo_qkv_input_offset,
                self._mlapo_qkv_weight,
                self._mlapo_qkv_deq_scale,
                self._mlapo_qkv_quant_bias,
                self.q_a_layernorm.weight,
                self._mlapo_q_norm_bias,
                self.q_b_proj.input_scale,
                self._mlapo_q_b_input_offset,
                self._mlapo_q_b_weight,
                self._mlapo_q_b_deq_scale,
                self._mlapo_q_b_quant_bias,
                self.kv_a_layernorm.weight,
                rope_cos,
                rope_sin,
                self.W_UK,
                context.kv_cache,
                context.rope_cache,
                context.slot_mapping[: hidden.shape[0]],
                self.kv_lora_rank,
                self.q_lora_rank,
                self.qk_rope_head_dim,
                self.q_a_layernorm.eps,
            )
        else:
            q_c, q_latent, q_pe = kernels.deepseek_mla_preprocess_decode(
                hidden,
                projection.input_scale,
                projection.input_offset,
                projection.weight,
                projection.deq_scale,
                projection.quant_bias,
                self.q_a_layernorm.weight,
                self.q_b_proj.input_scale,
                self.q_b_proj.input_offset,
                self.q_b_proj.weight,
                self.q_b_proj.deq_scale,
                self.q_b_proj.quant_bias,
                self.W_UK,
                self.kv_a_layernorm.weight,
                rope_cos,
                rope_sin,
                context.slot_mapping[: hidden.shape[0]],
                context.kv_cache,
                context.rope_cache,
                self.kv_lora_rank,
                self.q_lora_rank,
                self.num_heads_local,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.q_a_layernorm.eps,
                self.kv_a_layernorm.eps,
            )
        return q_c, q_latent, q_pe

    def _can_fuse_decode(self) -> bool:
        return self._use_fused_mla_decode and self._fused_mla_ready

    def _select_topk(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        backend: AttentionBackend,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        prev_topk: torch.Tensor | None,
        reuse_topk: bool,
    ) -> torch.Tensor | None:
        if self.indexer is None:
            return None
        return self.indexer.select_qli(hidden, q_c, backend.mla_index_context(self), half_rope_cos, half_rope_sin)

    def _execute_attention(
        self,
        backend: AttentionBackend,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor,
        k_pe: torch.Tensor,
        topk: torch.Tensor | None,
    ) -> torch.Tensor:
        return backend.execute_mla(q_latent, q_pe, k_latent, k_pe, self, topk=topk)

    def _reduce_attention_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(output)
        return output

    def _project_attention_output(self, attn_out: torch.Tensor) -> torch.Tensor:
        v_full = kernels.atb_matmul_ein_sum(attn_out, self.W_UV)
        v_full = v_full.reshape(attn_out.shape[0], self.num_heads_local * self.v_head_dim)
        return self._reduce_attention_output(self.o_proj(v_full))

    def _forward_with_topk(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk: torch.Tensor | None = None,
        reuse_topk: bool = False,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_hidden_scale(hidden, hidden_scale)
        num_tokens = hidden.shape[0]
        backend = get_forward_context().attention_backend
        preprocess = backend.mla_preprocess_context(self) if self._can_fuse_decode() else None
        if preprocess is not None:
            q_c, q_latent, q_pe = self._preprocess_decode(
                hidden, rope_cos, rope_sin, preprocess, hidden_scale, slot_mapping_int64
            )
            topk = self._select_topk(
                hidden,
                q_c,
                backend,
                half_rope_cos,
                half_rope_sin,
                rope_cos,
                rope_sin,
                query_cos_sin,
                prev_topk,
                reuse_topk,
            )
            attn_out = backend.execute_mla(q_latent, q_pe, None, None, self, topk=topk, cache_is_preprocessed=True)
            return self._project_attention_output(attn_out), topk
        q_a, kv = self._project_qkv_a(hidden, hidden_scale)
        q_c, q = self._normalize_and_project_query(q_a)
        q = q.view(
            num_tokens,
            self.num_heads_local,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = kernels.atb_matmul_ein_sum(q_nope, self.W_UK)
        q_pe = _interleave_rope_with(q_rope, rope_cos, rope_sin)
        k_latent_raw, k_rope_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_pe = _interleave_rope_with(k_rope_raw.unsqueeze(1), rope_cos, rope_sin)
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe_3d = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)

        topk = self._select_topk(
            hidden,
            q_c,
            backend,
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
            query_cos_sin,
            prev_topk,
            reuse_topk,
        )
        attn_out = self._execute_attention(backend, q_latent, q_pe, k_latent_3d, k_pe_3d, topk)
        return self._project_attention_output(attn_out), topk

    def forward(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self._forward_with_topk(
            hidden,
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
            hidden_scale=hidden_scale,
            slot_mapping_int64=slot_mapping_int64,
        )
        return output


class DeepseekV3Indexer(nn.Module):
    """DeepSeek-V3.2 LightningIndexer with optional INT8 Q/K cache."""

    def __init__(self, cfg: DeepseekV3Config, dtype: torch.dtype, device: torch.device, layer_id: int = 0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.indexer_rope_interleave = self._uses_interleaved_rope(cfg)
        self._init_streams(cfg, device)
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self._init_projections(cfg, dtype, device)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.register_buffer(
            "hadamard",
            _create_hadamard_matrix(self.head_dim, dtype, device),
            persistent=False,
        )

    def _uses_interleaved_rope(self, cfg: DeepseekV3Config) -> bool:
        return False

    def _init_streams(self, cfg: DeepseekV3Config, device: torch.device) -> None:
        self._q_stream = None
        self._weights_stream = None

    def _init_projections(self, cfg: DeepseekV3Config, dtype: torch.dtype, device: torch.device) -> None:
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_head * self.head_dim, bias=False, dtype=dtype, device=device)
        self.wk_weights_proj = nn.Linear(
            cfg.hidden_size,
            self.head_dim + self.n_head,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def _project_k_and_weights(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wk_weights_proj(hidden).split([self.head_dim, self.n_head], dim=-1)

    def _project_key(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.wk_weights_proj.weight[: self.head_dim])

    def _project_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.wk_weights_proj.weight[self.head_dim :])

    def _project_index_inputs(
        self,
        hidden: torch.Tensor,
        cache_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_hidden is hidden:
            return self._project_k_and_weights(hidden)
        return self._project_key(cache_hidden), self._project_weights(hidden)

    def _pad_q_heads_to_kernel_gsize(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        required_q_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # aclnnQuantLightningIndexer tiling hard-requires n_heads_q/n_heads_k == 64
        # (G_SIZE_LIMIT in quant_lightning_indexer_tiling.h; xllm_ops kernel hard-codes
        # matmul M=256 tile, head_dim=128 and kv_head=1 for DSV3.2/DSV4 shapes). GLM-5.2
        # is index_n_heads=32 / kv_head=1, so pad Q from 32 to 64 heads to satisfy the
        # kernel. The score sum_h(w_h * q_h . k) is mathematically unchanged: padded Q
        # rows are zero, so q_h . k = 0 for h >= n_head; padded weights are zero so those
        # zero terms cannot contribute even if the kernel processed them differently.
        pad_heads = required_q_heads - self.n_head
        if pad_heads == 0:
            return q, q_scale, weights
        if pad_heads < 0:
            raise RuntimeError(f"Indexer expected index_n_heads<={required_q_heads}, got {self.n_head}")
        q = torch.cat(
            [q, torch.zeros((q.size(0), pad_heads, q.size(2)), dtype=q.dtype, device=q.device)],
            dim=1,
        )
        q_scale = torch.cat(
            [q_scale, torch.zeros((q_scale.size(0), pad_heads), dtype=q_scale.dtype, device=q_scale.device)],
            dim=1,
        )
        weights = torch.cat(
            [weights, torch.zeros((weights.size(0), pad_heads), dtype=weights.dtype, device=weights.device)],
            dim=1,
        )
        return q, q_scale, weights

    def _apply_interleaved_rope(
        self,
        value: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply indexer RoPE in place without materializing split tensors."""
        _validate_rope_cos_sin(cos_sin, value, self.rope_dim, True, "interleaved indexer")
        cos, sin = cos_sin
        cos = cos.view(-1, self.rope_dim)
        sin = sin.view(-1, self.rope_dim)
        if value.dim() == 2:
            value = value.view(-1, 1, self.head_dim)
            kernels.npu_inplace_partial_rotary_mul(value, cos, sin, 0, self.rope_dim)
            return value.view(-1, self.head_dim)
        kernels.npu_inplace_partial_rotary_mul(value, cos, sin, 0, self.rope_dim)
        return value

    def _update_index_cache(
        self,
        cache_hidden: torch.Tensor,
        ctx: MlaIndexContext,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        projected_k: torch.Tensor | None = None,
    ) -> None:
        k = self._project_key(cache_hidden) if projected_k is None else projected_k
        k = self.k_norm(k)
        if self.indexer_rope_interleave:
            k = self._apply_interleaved_rope(k, cos_sin)
        else:
            _validate_rope_cos_sin(cos_sin, k, self.rope_dim, False, "indexer cache K")
            k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            k_pe = _apply_half_rope_with_cos_sin(k_pe.unsqueeze(1), *cos_sin).squeeze(1)
            k = torch.cat([k_pe, k_nope], dim=-1)
        if ctx.cp_context is not None:
            # Only the padded K rows obey the equal-size CP gather contract.
            k = cp_gather_kv(k, ctx.cp_context).contiguous()

        index_cache = ctx.index_cache
        index_cache_scale = ctx.index_cache_scale
        k_scale = None
        use_quant_indexer = index_cache.dtype == torch.int8 and index_cache_scale is not None
        if use_quant_indexer:
            rotation_scale = self.head_dim**-0.5
            k = torch.matmul(k, self.hadamard) * rotation_scale
            k, k_scale = kernels.dynamic_quant(k)
            assert k_scale is not None
            k_scale = k_scale.unsqueeze(-1).to(torch.float16)
        ctx.update_index_cache(k, k_scale)

    def _project_query(
        self,
        qr: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(qr, tuple):
            assert isinstance(self.wq_b, W8A8AttentionLinear)
            q = self.wq_b.forward_quantized(*qr)
        else:
            q = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        if self.indexer_rope_interleave:
            return self._apply_interleaved_rope(q, cos_sin)
        _validate_rope_cos_sin(cos_sin, q, self.rope_dim, False, "indexer query")
        q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_pe = _apply_half_rope_with_cos_sin(q_pe, *cos_sin)
        return torch.cat([q_pe, q_nope], dim=-1)

    def _select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ctx: MlaIndexContext,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        key_cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actual_seq_q = ctx.actual_seq_q
        actual_seq_kv = ctx.actual_seq_kv
        cache_hidden = hidden if cache_hidden is None else cache_hidden
        # Empty CP ranks still update/gather K without launching empty Q or
        # weights projections.
        has_queries = ctx.cp_context is None or ctx.cp_context.query_index.numel() != 0
        if has_queries:
            if self._weights_stream is not None:
                self._weights_stream.wait_for_current()
            with self._weights_stream.activate() if self._weights_stream is not None else nullcontext():
                k, weights = self._project_index_inputs(hidden, cache_hidden)
            if self._q_stream is not None:
                self._q_stream.wait_for_current()
                with self._q_stream.activate():
                    q = self._project_query(qr, query_cos_sin)
            elif self._weights_stream is not None:
                q = self._project_query(qr, query_cos_sin)
            if self._weights_stream is not None:
                # The fused projection also produces K; join before cache
                # preparation consumes it, keeping the main-path fusion.
                self._weights_stream.join()
                self._weights_stream.record_on_current(k)
                self._weights_stream.record_on_current(weights)
        else:
            k = self._project_key(cache_hidden)
        self._update_index_cache(
            cache_hidden,
            ctx,
            key_cos_sin,
            projected_k=k,
        )
        index_cache = ctx.index_cache
        index_cache_scale = ctx.index_cache_scale
        use_quant_indexer = index_cache.dtype == torch.int8 and index_cache_scale is not None
        index_cache, index_cache_scale, block_table = ctx.materialize_index_cache()
        if ctx.cp_context is not None and ctx.cp_context.query_index.numel() == 0:
            # Other ranks still need this rank's keys/cache materialization.
            # Complete those collectives before skipping empty Q kernels.
            return torch.full(
                (ctx.cp_context.total_local, index_cache.size(2), self.topk),
                -1,
                dtype=torch.int32,
                device=hidden.device,
            )

        if self._q_stream is not None:
            self._q_stream.join()
            self._q_stream.record_on_current(q)
        elif self._weights_stream is None:
            q = self._project_query(qr, query_cos_sin)
        if use_quant_indexer:
            rotation_scale = self.head_dim**-0.5
            q = torch.matmul(q, self.hadamard) * rotation_scale
            q, q_scale = kernels.dynamic_quant(q)
            assert q_scale is not None
            q_scale = q_scale.to(torch.float16)
            assert index_cache_scale is not None
            weight_scale = self.head_dim**-0.5 * self.n_head**-0.5
            # xLLM stores one index key per source token.
            cmp_ratio = 1

            required_q_heads = index_cache.size(2) * 64
            q, q_scale, weights_padded = self._pad_q_heads_to_kernel_gsize(q, q_scale, weights, required_q_heads)

            qli_metadata = ctx.get_quant_indexer_metadata(required_q_heads, self.head_dim, self.topk, cmp_ratio)
            topk = kernels.quant_lightning_indexer(
                q,
                index_cache,
                (weights_padded * weight_scale).to(torch.float16),
                q_scale,
                index_cache_scale,
                qli_metadata,
                actual_seq_q,
                actual_seq_kv,
                block_table,
                self.topk,
                cmp_ratio,
            )
        else:
            topk = self._select_unquantized(q, index_cache, weights, ctx, block_table)
        if ctx.cp_context is not None:
            local_topk = topk.new_full((ctx.cp_context.total_local, *topk.shape[1:]), -1)
            local_topk.index_copy_(0, ctx.cp_context.query_index, topk)
            topk = local_topk
        return topk

    def _select_unquantized(
        self,
        q: torch.Tensor,
        index_cache: torch.Tensor,
        weights: torch.Tensor,
        ctx: MlaIndexContext,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        key_heads = index_cache.size(2) if index_cache.dim() >= 3 else 1
        shape = (q.size(0), key_heads, self.topk)
        indices = torch.empty(shape, dtype=torch.int32, device=q.device)
        values = torch.empty(shape, dtype=torch.bfloat16, device=q.device)
        return kernels.lightning_indexer_out(
            q,
            index_cache,
            weights,
            ctx.actual_seq_q,
            ctx.actual_seq_kv,
            block_table,
            "TND",
            "PA_BSND",
            self.topk,
            3,
            9223372036854775807,
            9223372036854775807,
            False,
            indices,
            values,
        )

    def select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ctx: MlaIndexContext,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        cos_sin = (half_rope_cos, half_rope_sin)
        return self._select_qli(hidden, qr, ctx, cos_sin, cos_sin)


class DeepseekV3MoE(nn.Module):
    """EP-aware MoE: experts split across EP ranks, intermediate TP-sharded."""

    def __init__(
        self,
        cfg: DeepseekV3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling = cfg.routed_scaling_factor
        self.moe_inter = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        self.ep_size = cfg.ep_size
        self.ep_rank = cfg.ep_rank
        self.dp_size = cfg.dp_size
        self.dp_rank = cfg.dp_rank
        self.moe_tp_size = cfg.moe_tp_size

        tp = moe_shard(cfg)[0]
        assert self.moe_inter % tp == 0
        self.inter_local = self.moe_inter // tp

        num_local_experts = self.num_experts // max(self.ep_size, 1)
        self.num_local_experts = num_local_experts
        self.local_expert_start = self.ep_rank * num_local_experts
        self.local_expert_end = self.local_expert_start + num_local_experts

        # bf16 operands into an fp32 accumulator: same precision as the ATB
        # router's fp32 gate, without upcasting the activation or the weight.
        self.gate = RouterGate(cfg.hidden_size, self.num_experts, device)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device),
            persistent=False,
        )
        # Materialize and format expert weights one at a time to limit loading peak memory.
        self.experts_w13 = nn.Parameter(
            torch.empty(0, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "experts_w13_scale",
            torch.empty(
                num_local_experts,
                2 * self.inter_local,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(0, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "experts_w2_scale_compute",
            torch.empty(
                num_local_experts,
                self.hidden,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_experts = DeepseekV3MLP(
            cfg,
            shared_inter,
            dtype,
            device,
            skip_tp_reduce=True,
            tp_override=tp,
        )
        self._expert_parallel_enabled = hasattr(torch, "npu") and device.type in ("npu", "privateuseone")
        self._gate_overlap_enabled = self._expert_parallel_enabled and os.environ.get("XLLM_MOE_GATE_OVERLAP") == "1"
        self._fine_overlap_enabled = self._gate_overlap_enabled and os.environ.get("XLLM_MOE_FINE_OVERLAP") == "1"
        self._fuse_shared_expert = self._expert_parallel_enabled
        self._shared_expert_start_event: torch.npu.Event | None = None
        self._gate_done_event: torch.npu.Event | None = None
        self._shared_expert_done_event: torch.npu.Event | None = None
        self._before_dispatch_event: torch.npu.Event | None = None
        self._before_gmm2_event: torch.npu.Event | None = None

    def allocate_experts_w13_for_loading(self) -> None:
        assert self.experts_w13.numel() == 0, "experts_w13 is already loaded"
        self.experts_w13.data = torch.empty(
            self.num_local_experts,
            2 * self.inter_local,
            self.hidden,
            dtype=torch.int8,
            device=self.experts_w13.device,
        )

    def allocate_experts_w2_for_loading(self) -> None:
        assert self.experts_w2.numel() == 0, "experts_w2 is already loaded"
        self.experts_w2.data = torch.empty(
            self.num_local_experts,
            self.hidden,
            self.inter_local,
            dtype=torch.int8,
            device=self.experts_w2.device,
        )

    @staticmethod
    def _format_and_release_expert_weight(parameter: nn.Parameter) -> None:
        assert parameter.numel() > 0, "expert weight was not loaded"
        transposed = parameter.data.transpose(1, 2).contiguous()
        parameter.data = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
        parameter.data = kernels.format_cast_nz(transposed)
        del transposed

    def process_experts_w13_after_loading(self) -> None:
        self._format_and_release_expert_weight(self.experts_w13)

    def process_experts_w2_after_loading(self) -> None:
        self._format_and_release_expert_weight(self.experts_w2)

    def load_experts(
        self,
        loader: W8A8WeightLoader,
        src_prefix: str,
        *,
        world: int,
        rank: int,
    ) -> None:
        """Allocate, fill, and format the stacked W8A8 experts, staged w13 then w2.

        Staging keeps peak load memory at one raw int8 expert buffer, not two.
        The grouped-MoE path keeps no per-expert offset buffer, so each expert's
        symmetric-int8 (zero ``weight_offset``) invariant is asserted here while
        its shard is warm.
        """
        self.allocate_experts_w13_for_loading()
        for idx, j in enumerate(range(self.local_expert_start, self.local_expert_end)):
            src = f"{src_prefix}{j}."
            loader.assert_symmetric_int8(src, ("gate_proj", "up_proj"))
            w = loader.pack_gate_up(src, "weight", world=world, rank=rank)
            s = loader.pack_gate_up(src, "weight_scale", world=world, rank=rank)
            self.experts_w13.data[idx].copy_(w)
            self.experts_w13_scale.data[idx].copy_(s.reshape(-1))
        self.process_experts_w13_after_loading()
        self.allocate_experts_w2_for_loading()
        for idx, j in enumerate(range(self.local_expert_start, self.local_expert_end)):
            src = f"{src_prefix}{j}."
            loader.assert_symmetric_int8(src, ("down_proj",))
            w, s = loader.load_w8a8_down(src, world=world, rank=rank)
            self.experts_w2.data[idx].copy_(w)
            # GMM2 needs bf16 scales; compute buffer holds the cast.
            self.experts_w2_scale_compute.data[idx].copy_(s.reshape(-1))
        self.process_experts_w2_after_loading()

    def load_from_checkpoint(self, loader: W8A8WeightLoader, mlp_prefix: str) -> None:
        """Load this routed-MoE layer: experts, router gate, and shared experts."""
        world, rank = moe_shard(self.cfg)
        self.load_experts(loader, mlp_prefix + "experts.", world=world, rank=rank)
        loader.copy_replicated(mlp_prefix + "gate.weight")
        loader.copy_in(
            mlp_prefix + "e_score_correction_bias",
            loader.get_tensor(mlp_prefix + "gate.e_score_correction_bias"),
        )
        self.shared_experts.load_from_checkpoint(loader, mlp_prefix + "shared_experts.", world=world, rank=rank)

    def _run_routed_experts(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden)
        return kernels.grouped_moe(
            hidden,
            logits,
            self.experts_w13,
            self.experts_w2,
            self.experts_w13_scale,
            self.experts_w2_scale_compute,
            self.e_score_correction_bias,
            self.topk,
            self.topk_group,
            self.n_group,
            self.cfg.norm_topk_prob,
            self.routed_scaling,
            [self.local_expert_start, self.local_expert_end],
        )

    def _run_shared_experts(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._fuse_shared_expert:
            output = None
            context = get_forward_context()
            if context.execution_state is not None:
                output_shape = (hidden.shape[0], self.hidden)
                output = torch.empty(output_shape, dtype=torch.bfloat16, device=hidden.device)
            return self.shared_experts.forward_dequant_swiglu_quant(hidden, output=output)
        return self.shared_experts(hidden)

    def _combine_expert_outputs(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_size > 1:
            distributed.all_reduce_(routed, "moe_ep")

        final = routed + shared
        if self.moe_tp_size > 1:
            distributed.all_reduce_(final, "moe_tp")
        elif self.cfg.tp_size > 1 and self.ep_size == 1:
            distributed.all_reduce_(final)
        return final

    def _ensure_expert_parallel_resources(self) -> None:
        if self._shared_expert_start_event is not None:
            return
        self._shared_expert_start_event = torch.npu.Event()
        self._shared_expert_done_event = torch.npu.Event()
        if self._gate_overlap_enabled:
            self._gate_done_event = torch.npu.Event()
        if self._fine_overlap_enabled:
            self._before_dispatch_event = torch.npu.Event()
            self._before_gmm2_event = torch.npu.Event()

    def _forward_parallel(self, hidden: torch.Tensor) -> torch.Tensor:
        self._ensure_expert_parallel_resources()
        shared_stream = _shared_expert_stream(hidden.device)
        start_event = self._shared_expert_start_event
        shared_done_event = self._shared_expert_done_event
        assert start_event is not None
        assert shared_done_event is not None

        current_stream = torch.npu.current_stream()
        start_event.record(current_stream)

        if self._gate_overlap_enabled:
            gate_stream = _gate_stream(hidden.device)
            gate_done_event = self._gate_done_event
            assert gate_done_event is not None

            gate_stream.wait_event(start_event)
            with torch.npu.stream(gate_stream):
                logits = self.gate(hidden)
                topk_weights, topk_ids = kernels.moe_gate_routing(
                    logits,
                    self.e_score_correction_bias,
                    self.topk,
                    self.topk_group,
                    self.n_group,
                    self.cfg.norm_topk_prob,
                    self.routed_scaling,
                )
                gate_done_event.record(gate_stream)

            shared_stream.wait_event(start_event)
            with torch.npu.stream(shared_stream):
                shared = self._run_shared_experts(hidden)
                shared_done_event.record(shared_stream)

            current_stream.wait_event(gate_done_event)
            routed = kernels.moe_expert_compute(
                hidden,
                topk_weights,
                topk_ids,
                self.experts_w13,
                self.experts_w2,
                self.experts_w13_scale,
                self.experts_w2_scale_compute,
                self.topk,
            )
            current_stream.wait_event(shared_done_event)
        else:
            shared_stream.wait_event(start_event)
            with torch.npu.stream(shared_stream):
                shared = self._run_shared_experts(hidden)
                shared_done_event.record(shared_stream)
            routed = self._run_routed_experts(hidden)
            current_stream.wait_event(shared_done_event)

        return self._combine_expert_outputs(routed, shared)

    def _forward_fine_grained_parallel(self, hidden: torch.Tensor) -> torch.Tensor:
        self._ensure_expert_parallel_resources()
        shared_stream = _shared_expert_stream(hidden.device)
        gate_stream = _gate_stream(hidden.device)
        current_stream = torch.npu.current_stream()

        before_dispatch_event = self._before_dispatch_event
        before_gmm2_event = self._before_gmm2_event
        assert before_dispatch_event is not None
        assert before_gmm2_event is not None

        # Auxiliary streams must fork from and rejoin the ACL Graph capture stream.
        gate_stream.wait_stream(current_stream)
        shared_stream.wait_stream(current_stream)

        with torch.npu.stream(gate_stream):
            logits = self.gate(hidden)
            topk_weights, topk_ids = kernels.moe_gate_routing(
                logits,
                self.e_score_correction_bias,
                self.topk,
                self.topk_group,
                self.n_group,
                self.cfg.norm_topk_prob,
                self.routed_scaling,
            )

        with torch.npu.stream(shared_stream):
            gate_up, pertoken = self.shared_experts.quantize_and_project_gate_up(hidden)

        current_stream.wait_stream(gate_stream)
        before_dispatch_event.record(current_stream)
        sorted_hidden_i8, expanded_row_idx, group_list, pt_scale = kernels.moe_token_dispatch(
            hidden, topk_ids, self.topk, self.num_experts
        )
        act_i8, act_pt = kernels.moe_gmm1(
            sorted_hidden_i8,
            self.experts_w13,
            self.experts_w13_scale,
            pt_scale,
            group_list,
        )
        before_gmm2_event.record(current_stream)
        routed = kernels.moe_gmm2_combine(
            act_i8,
            act_pt,
            self.experts_w2,
            self.experts_w2_scale_compute,
            group_list,
            expanded_row_idx,
            topk_weights,
        )

        # Record events before the auxiliary stream waits on them.
        with torch.npu.stream(shared_stream):
            shared_stream.wait_event(before_dispatch_event)
            act_int8, act_scale = self.shared_experts.activate_and_quantize(gate_up, pertoken)
            shared_stream.wait_event(before_gmm2_event)
            shared = self.shared_experts.project_down(act_int8, act_scale)

        current_stream.wait_stream(shared_stream)
        return self._combine_expert_outputs(routed, shared)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        local_tokens: int = 0
        padded_tokens: int = 0
        use_compact_gather: bool = False
        if self.dp_size > 1:
            ctx = get_forward_context()
            execution_token_counts = list(ctx.metadata.dp_execution_token_counts)
            local_tokens = hidden.shape[0]
            is_graph = ctx.execution_state is not None
            is_prefill = ctx.metadata.is_prefill or ctx.metadata.is_chunked_prefill
            dp_is_decode = getattr(ctx.metadata, "dp_is_decode", None)
            all_decode = dp_is_decode is not None and all(dp_is_decode)
            if is_graph or is_prefill or not all_decode:
                padded_tokens = max(execution_token_counts)
                pad_size = padded_tokens - local_tokens
                if pad_size > 0:
                    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_size))
                hidden = distributed.all_gather(hidden, dim=0, world_size=self.dp_size, group_name="dp")
            else:
                use_compact_gather = True
                hidden = distributed.all_gather_variable(
                    hidden,
                    execution_token_counts,
                    self.dp_rank,
                    "dp",
                )

        if self._fine_overlap_enabled:
            final = self._forward_fine_grained_parallel(hidden)
        elif self._expert_parallel_enabled:
            final = self._forward_parallel(hidden)
        else:
            routed = self._run_routed_experts(hidden)
            shared = self._run_shared_experts(hidden)
            final = self._combine_expert_outputs(routed, shared)

        if use_compact_gather:
            offset = sum(execution_token_counts[: self.dp_rank])
            final = final.narrow(0, offset, local_tokens)
        elif padded_tokens > 0:
            start = self.dp_rank * padded_tokens
            final = final.narrow(0, start, local_tokens)
        return final


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = self._make_attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = self._make_mlp(cfg, layer_id, dtype, device)
        self._fuse_dense_norm_quant = isinstance(self.mlp, DeepseekV3MLP) and device.type in ("npu", "privateuseone")

    def _make_attention(
        self, cfg: DeepseekV3Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> DeepseekV3MLAAttention:
        return DeepseekV3MLAAttention(cfg, layer_id, dtype, device)

    def _make_mlp(
        self, cfg: DeepseekV3Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> DeepseekV3MLP | DeepseekV3MoE:
        if layer_id < cfg.first_k_dense_replace:
            return DeepseekV3MLP(cfg, cfg.intermediate_size, dtype, device)
        return DeepseekV3MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        indexer_query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        reuse_topk_indices: bool = False,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_scale: torch.Tensor | None = None
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        elif self.self_attn._can_fuse_input_norm_quant():
            hidden, hidden_scale, residual = kernels.fused_add_rms_norm_dynamic_quant(
                hidden, residual, self.input_layernorm.weight, self.input_layernorm.eps
            )
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, topk = self._attention(
            hidden,
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
            indexer_query_cos_sin,
            prev_topk_indices,
            reuse_topk_indices,
            hidden_scale,
            slot_mapping_int64,
        )
        fused_dense_input: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._fuse_dense_norm_quant:
            hidden_quant, hidden_scale, residual = kernels.fused_add_rms_norm_dynamic_quant(
                hidden,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
            fused_dense_input = (hidden_quant, hidden_scale)
        else:
            hidden, residual = self.post_attention_layernorm(hidden, residual)
        if fused_dense_input is not None:
            assert isinstance(self.mlp, DeepseekV3MLP)
            hidden = self.mlp.forward_quantized(*fused_dense_input)
        else:
            hidden = self.mlp(hidden)
        return hidden, residual, topk

    def _attention(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        prev_topk: torch.Tensor | None,
        reuse_topk: bool,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.self_attn(
            hidden, half_rope_cos, half_rope_sin, rope_cos, rope_sin, hidden_scale, slot_mapping_int64
        ), None


class DeepseekV3Model(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV3Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([self._make_decoder(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = DeepseekYarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )

        self.aux_hidden_capture = AuxHiddenCapture(())

    def _make_decoder(
        self, cfg: DeepseekV3Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> DeepseekV3DecoderLayer:
        return DeepseekV3DecoderLayer(cfg, layer_id, dtype, device)

    def _record_layer_event(self, layer_id: int) -> None:
        pass

    def _cp_context(self) -> CpContext | None:
        return None

    def _indexer_interleaved(self) -> bool:
        return False

    def _prepare_mla_slots(self) -> torch.Tensor | None:
        return None

    def _prepare_layer_inputs(
        self, hidden: torch.Tensor, positions: torch.Tensor, cp_context: CpContext | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        positions = positions.to(torch.int64).contiguous()
        if cp_context is not None:
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()
        rope = self.rotary(positions)
        query_cos_sin = _select_indexer_query_cos_sin(self._indexer_interleaved(), *rope, cp_context)
        return hidden, rope, query_cos_sin

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        cp_context = self._cp_context()
        hidden, rope, query_cos_sin = self._prepare_layer_inputs(hidden, positions, cp_context)
        slot_mapping_int64 = self._prepare_mla_slots()
        residual: torch.Tensor | None = None
        topk: torch.Tensor | None = None
        aux = self.aux_hidden_capture.create_buffer(hidden)
        for layer_id, layer in enumerate(self.layers):
            hidden, residual, topk = layer(
                hidden, residual, *rope, query_cos_sin, topk, slot_mapping_int64=slot_mapping_int64
            )
            self.aux_hidden_capture.capture_layer(layer_id, hidden, residual, aux)
            self._record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)
        return self.aux_hidden_capture.finalize(hidden, aux)


class DeepseekV3ForCausalLM(PyModelBase):
    """DeepSeek-V3.2 causal LM. Registered under ``model_type='deepseek_v32'``."""

    def __init__(self, config: dict, build_model: bool = True) -> None:
        super().__init__()
        self.cfg = DeepseekV3Config.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get("tp_rank", _tp_rank_from_device(config.get("device", "npu:0"))))
        self.cfg.ep_size = int(config.get("ep_size", 1))
        self.cfg.ep_rank = int(config.get("ep_rank", 0))
        self.cfg.dp_size = int(config.get("dp_size", 1))
        self.cfg.dp_rank = int(config.get("dp_rank", 0))
        self.cfg.moe_tp_size = int(config.get("moe_tp_size", 1))
        self.cfg.moe_tp_rank = int(config.get("moe_tp_rank", 0))
        self.cfg.world_size = int(config.get("world_size", self.cfg.tp_size))
        # C++ computes moe_tp_size = world_size / ep_size, which conflates DP
        # replicas with TP shards when dp > 1.  With ep=1 the expert weights
        # are sharded by tp_size (not moe_tp_size), and the C++-created
        # "moe_tp" group still spans all world_size ranks including DP peers.
        # Force moe_tp_size=1 so the all_reduce falls through to the "tp"
        # group that is correctly scoped to one DP replica.
        if self.cfg.dp_size > 1 and self.cfg.moe_tp_size > 1:
            if self.cfg.ep_size == 1:
                self.cfg.moe_tp_size = 1
            else:
                # TODO: fix C++ to compute moe_tp_size excluding DP ranks
                if self.cfg.moe_tp_size % self.cfg.dp_size != 0:
                    raise ValueError(
                        f"moe_tp_size ({self.cfg.moe_tp_size}) must be divisible by "
                        f"dp_size ({self.cfg.dp_size}) when ep_size > 1"
                    )
                self.cfg.moe_tp_size = self.cfg.moe_tp_size // self.cfg.dp_size
        if hasattr(self.cfg, "validate"):
            self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        if build_model:
            self._build_model()

    def _build_model(self) -> None:
        tp = self.cfg.tp_size
        self.model = DeepseekV3Model(
            self.cfg,
            self.dtype,
            self.device,
        )
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
        load_lm_head: bool = True,
        load_embedding: bool = True,
        loader: Optional[W8A8WeightLoader] = None,
    ) -> None:
        cfg = self.cfg
        if loader is None:
            loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        if load_embedding:
            loader.copy_shard("model.embed_tokens.weight", dim=1)

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            loader.copy_replicated(p + "input_layernorm.weight")
            loader.copy_replicated(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            loader.load_fused_w8a8_projection(
                attn,
                "qkv_a_proj",
                ("kv_a_proj_with_mqa", "q_a_proj"),
            )
            loader.copy_replicated(attn + "q_a_layernorm.weight")
            loader.load_compatible_w8a8_projection(
                attn, "q_b_proj", {"weight": 0, "deq_scale": 0, "quant_bias": 0}, dynamic_activation=False
            )
            loader.copy_replicated(attn + "kv_a_layernorm.weight")
            loader.copy_shard(attn + "kv_b_proj.weight", dim=0)
            loader.load_compatible_w8a8_projection(attn, "o_proj", {"weight": 1}, dynamic_activation=False)
            if cfg.index_topk > 0:
                idx = attn + "indexer."
                loader.copy_replicated(idx + "wq_b.weight")
                loader.copy_in(
                    idx + "wk_weights_proj.weight",
                    torch.cat(
                        [
                            loader.get_tensor(idx + "wk.weight"),
                            loader.get_tensor(idx + "weights_proj.weight"),
                        ],
                        dim=0,
                    ),
                )
                loader.copy_replicated(idx + "k_norm.weight")
                loader.copy_replicated(idx + "k_norm.bias")
            self.model.layers[i].self_attn.process_weights_after_loading()

            self.model.layers[i].mlp.load_from_checkpoint(loader, p + "mlp.")

        loader.copy_replicated("model.norm.weight")
        if load_lm_head:
            loader.copy_shard("lm_head.weight", dim=0)

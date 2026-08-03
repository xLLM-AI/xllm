# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V3.2 causal LM (Python model executor target)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch_npu  # noqa: F401

from xllm.python import ops
from xllm.python.attention.backend import MlaIndexContext
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.models.base import PyModelBase


def _tp_rank_from_device(device: object) -> int:
    """Local device index from the worker device string ("npu:3" -> 3)."""
    s = str(device)
    if ":" in s:
        try:
            return int(s.rsplit(":", 1)[-1])
        except ValueError:
            return 0
    return 0


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
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = math.floor(low)
    high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity.
    linear = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


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
        rope_mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        cos = freqs.cos() * rope_mscale
        sin = freqs.sin() * rope_mscale
        cache = torch.cat([cos, sin], dim=-1)
        if dtype is not None:
            cache = cache.to(dtype)
        self.register_buffer(
            "cos_sin_cache", cache.contiguous(), persistent=False
        )

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
        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast,
            beta_slow,
            rotary_dim,
            base,
            max_position_embeddings,
        )
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(
                low, high, rotary_dim // 2, torch.float32, device
            )
        )
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
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "DeepseekV3Config":
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
            max_position_embeddings=int(
                pick("max_position_embeddings", default=4096)
            ),
            original_max_position_embeddings=int(
                rpick("original_max_position_embeddings", default=4096)
            ),
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
            moe_intermediate_size=int(
                pick("moe_intermediate_size", default=2048)
            ),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
        )

    def head_split(self) -> Tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1)."""
        num_heads_local = self.n_heads // self.tp_size
        return num_heads_local, 1


class W8A8StaticLinear(nn.Module):
    """Static-activation W8A8 linear (attention projections)."""

    def __init__(self, in_features: int, out_features: int, device: torch.device,
                 row_parallel: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.row_parallel = row_parallel
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "deq_scale", torch.empty(out_features, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "quant_bias", torch.empty(out_features, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "input_scale", torch.empty(1, dtype=torch.bfloat16, device=device)
        )
        self.register_buffer(
            "input_offset", torch.empty(1, dtype=torch.bfloat16, device=device)
        )

    def process_weights_after_loading(self) -> None:
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.input_scale_recip = (1.0 / self.input_scale).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mult = self.input_scale_recip
        x_int8 = torch.clamp(
            torch.round(
                x.to(torch.float32) * mult + self.input_offset.to(torch.float32)
            ),
            -128, 127,
        ).to(torch.int8)
        return ops.quant_matmul(
            x_int8, self.weight, False, self.deq_scale, None, None,
            self.quant_bias if not (self.row_parallel and ops.tp_rank(x.device) != 0) else None,
            torch.bfloat16,
        )


class W8A8DynamicLinear(nn.Module):
    """Dynamic-activation W8A8 linear (MLP / experts)."""

    def __init__(self, in_features: int, out_features: int, device: torch.device) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "weight_offset", torch.empty(out_features, 1, dtype=torch.float32, device=device)
        )

    def process_weights_after_loading(self) -> None:
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()
        if not bool(torch.all(self.weight_offset == 0)):
            import logging
            logging.getLogger(__name__).warning(
                "W8A8DynamicLinear loaded with non-zero weight_offset; the "
                "int8 matmul path drops the antiquant offset -- output may be "
                "wrong. Expected symmetric int8 (offset == 0)."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int8, pertoken = torch.ops.npu.npu_dynamic_quant(x)
        return ops.quant_matmul(
            x_int8, self.weight, False, self.weight_scale, None,
            pertoken, None, torch.bfloat16,
        )


class DeepseekV3MLP(nn.Module):
    """Dense gated-SiLU FFN (layers < first_k_dense_replace)."""

    def __init__(
        self,
        cfg: DeepseekV3Config,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
        skip_tp_reduce: bool = False,
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert intermediate_size % tp == 0, (
            f"intermediate_size {intermediate_size} not divisible by tp {tp}"
        )
        inter_local = intermediate_size // tp
        self.tp = tp
        self.skip_tp_reduce = skip_tp_reduce
        self.gate_up_proj = W8A8DynamicLinear(cfg.hidden_size, 2 * inter_local, device)
        self.down_proj = W8A8DynamicLinear(inter_local, cfg.hidden_size, device)

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        act = ops.silu_and_mul(gate_up)
        out = self.down_proj(act)
        if self.tp > 1 and not self.skip_tp_reduce:
            ops.all_reduce_(out)
        return out


class DeepseekV3MLAAttention(Attention):
    """Absorbed-MLA attention. KV cache stores latent (kv_lora) + rope."""

    def __init__(
        self,
        cfg: DeepseekV3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        assert cfg.n_heads % tp == 0
        num_heads = cfg.n_heads // tp
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v_head = cfg.v_head_dim
        scale = (qk_nope + qk_rope) ** -0.5
        attn_mscale = _yarn_get_mscale(
            cfg.rope_scaling_factor, cfg.rope_mscale_all_dim
        )
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

        self.q_a_proj = W8A8StaticLinear(cfg.hidden_size, cfg.q_lora_rank, device)
        self.kv_a_proj_with_mqa = W8A8StaticLinear(cfg.hidden_size, kv_lora + qk_rope, device)
        self.q_a_layernorm = RMSNorm(
            cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.q_b_proj = W8A8StaticLinear(
            cfg.q_lora_rank, num_heads * (qk_nope + qk_rope), device
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora,
            num_heads * (qk_nope + v_head),
            tp,
            dtype=dtype,
            device=device,
        )
        self.o_proj = W8A8StaticLinear(num_heads * v_head, cfg.hidden_size, device,
                                       row_parallel=True)
        self.rotary = DeepseekYarnRotaryEmbedding(
            qk_rope,
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
        self.indexer: DeepseekV3Indexer | None = (
            DeepseekV3Indexer(cfg, dtype, device) if cfg.index_topk > 0 else None
        )
        if self.indexer is not None:
            self.indexer.rotary = self.rotary

    def process_weights_after_loading(self) -> None:
        self.q_a_proj.process_weights_after_loading()
        self.kv_a_proj_with_mqa.process_weights_after_loading()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())

    def _interleaved_rope(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        cos_sin = self.rotary.cos_sin_cache[positions]
        half = cos_sin.size(-1) // 2
        cos32 = cos_sin[..., :half]
        sin32 = cos_sin[..., half:]
        cos = torch.cat([cos32, cos32], dim=-1).unsqueeze(1).unsqueeze(1)
        sin = torch.cat([sin32, sin32], dim=-1).unsqueeze(1).unsqueeze(1)
        T, H, D = x.shape
        return torch_npu.npu_interleave_rope(
            x.view(T, H, 1, D), cos, sin
        ).view(T, H, D)

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        q_a = self.q_a_proj(hidden)
        q_c = self.q_a_layernorm(q_a)
        backend = get_forward_context().attention_backend
        topk = None
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            topk = self.indexer.select_qli(hidden, q_c, positions, ctx)
        q = self.q_b_proj(q_c)
        q = q.view(
            num_tokens,
            self.num_heads_local,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_latent = torch.bmm(
            q_nope.transpose(0, 1), self.W_UK
        ).transpose(0, 1)
        q_pe = self._interleaved_rope(q_rope, positions)
        kv = self.kv_a_proj_with_mqa(hidden)
        k_latent_raw, k_rope_raw = kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_pe = self._interleaved_rope(
            k_rope_raw.unsqueeze(1), positions
        )
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe_3d = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)

        attn_out = backend.execute_mla(
            q_latent, q_pe, k_latent_3d, k_pe_3d, self, topk=topk
        )
        v_full = torch.bmm(
            attn_out.transpose(0, 1), self.W_UV
        ).transpose(0, 1)
        v_full = v_full.reshape(
            num_tokens, self.num_heads_local * self.v_head_dim
        )
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            ops.all_reduce_(o)
        return o


class DeepseekV3Indexer(nn.Module):
    """DeepSeek-V3.2 lightning indexer (bf16 weights, non-quant aclnnLightningIndexer)."""

    def __init__(self, cfg: "DeepseekV3Config", dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_head * self.head_dim,
                             bias=False, dtype=dtype, device=device)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim,
                            bias=False, dtype=dtype, device=device)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head,
                                      bias=False, dtype=dtype, device=device)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6,
                                   dtype=dtype, device=device)
        self.rotary = None

    def select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: MlaIndexContext,
    ) -> torch.Tensor:
        q = self.wq_b(qr).view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k = self.wk(hidden)
        weights = self.weights_proj(hidden)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        cos_sin = self.rotary.cos_sin_cache[positions]
        half = cos_sin.size(-1) // 2
        c = cos_sin[:, :half]
        s = cos_sin[:, half:]
        q1 = q_pe[..., :half]; q2 = q_pe[..., half:]
        o1 = q1 * c.unsqueeze(1) - q2 * s.unsqueeze(1)
        o2 = q2 * c.unsqueeze(1) + q1 * s.unsqueeze(1)
        q_pe = torch.cat([o1, o2], dim=-1)
        k1 = k_pe[..., :half]; k2 = k_pe[..., half:]
        ko1 = k1 * c - k2 * s
        ko2 = k2 * c + k1 * s
        k_pe = torch.cat([ko1, ko2], dim=-1)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        if ctx.index_cache is not None and ctx.slot_mapping is not None:
            k_view = ctx.index_cache.view(-1, ctx.index_cache.size(-1))
            ops.scatter_nd_update(
                k_view, ctx.slot_mapping.reshape(-1, 1).clamp_min(0), k
            )
        topk = ops.lightning_indexer(
            q, ctx.index_cache, weights,
            ctx.actual_seq_q, ctx.actual_seq_kv, ctx.block_table,
            "TND", "PA_BSND", self.topk, 3,
            9223372036854775807, 9223372036854775807,
            False,
        )
        return topk


class DeepseekV3MoE(nn.Module):
    """Pure-TP MoE: 256 routed experts replicated, expert intermediate TP-sharded."""

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
        tp = cfg.tp_size
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling = cfg.routed_scaling_factor
        self.moe_inter = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        assert self.moe_inter % tp == 0
        self.inter_local = self.moe_inter // tp

        self.gate = nn.Linear(
            cfg.hidden_size, self.num_experts, bias=False, dtype=dtype, device=device
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.experts_w13 = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * self.inter_local, self.hidden,
                dtype=torch.int8, device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "experts_w13_scale",
            torch.empty(
                self.num_experts, 2 * self.inter_local, 1,
                dtype=torch.float32, device=device,
            ),
        )
        self.register_buffer(
            "experts_w13_offset",
            torch.empty(
                self.num_experts, 2 * self.inter_local, 1,
                dtype=torch.float32, device=device,
            ),
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden, self.inter_local,
                dtype=torch.int8, device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "experts_w2_scale",
            torch.empty(
                self.num_experts, self.hidden, 1,
                dtype=torch.float32, device=device,
            ),
        )
        self.register_buffer(
            "experts_w2_offset",
            torch.empty(
                self.num_experts, self.hidden, 1,
                dtype=torch.float32, device=device,
            ),
        )
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_experts = DeepseekV3MLP(cfg, shared_inter, dtype, device,
                                            skip_tp_reduce=True)

    def process_weights_after_loading(self) -> None:
        assert torch.all(self.experts_w13_offset == 0), (
            "DeepseekV3MoE int8-grouped path needs symmetric int8 experts "
            "(experts_w13_offset == 0)")
        assert torch.all(self.experts_w2_offset == 0), (
            "DeepseekV3MoE int8-grouped path needs symmetric int8 experts "
            "(experts_w2_offset == 0)")
        self.experts_w13.data = self.experts_w13.data.transpose(1, 2).contiguous()
        self.experts_w2.data = self.experts_w2.data.transpose(1, 2).contiguous()
        self.experts_w13.data = torch_npu.npu_format_cast(
            self.experts_w13.data, 29)  # ACL_FORMAT_FRACTAL_NZ
        self.experts_w2.data = torch_npu.npu_format_cast(
            self.experts_w2.data, 29)  # ACL_FORMAT_FRACTAL_NZ
        self.experts_w13_scale.data = self.experts_w13_scale.data.view(
            self.num_experts, -1
        ).contiguous()
        self.experts_w13_offset.data = self.experts_w13_offset.data.view(
            self.num_experts, -1
        ).contiguous()
        self.experts_w2_scale.data = self.experts_w2_scale.data.view(
            self.num_experts, -1
        ).contiguous()
        self.experts_w2_offset.data = self.experts_w2_offset.data.view(
            self.num_experts, -1
        ).contiguous()
        self.shared_experts.process_weights_after_loading()

    def _grouped_topk(
        self, gating_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """noaux_tc groupwise top-k via ``npu_moe_gating_top_k``."""
        bias = self.e_score_correction_bias
        if bias is not None and bias.dtype != gating_output.dtype:
            bias = bias.to(gating_output.dtype)
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            gating_output,
            k=self.topk,
            bias=bias,
            k_group=self.topk_group,
            group_count=self.n_group,
            group_select_mode=1,
            renorm=1 if self.cfg.norm_topk_prob else 0,
            norm_type=1,
            routed_scaling_factor=1.0,
            eps=1e-20,
        )
        return topk_weights, topk_ids

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        logits = self.gate(hidden)
        topk_w, topk_idx = self._grouped_topk(logits)

        sorted_hidden_i8, expanded_row_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden,
            topk_idx.to(torch.int32),
            scale=None,
            active_num=num_tokens * self.topk,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=1,
        )
        group_list = torch.cumsum(expert_tokens.to(torch.int64), 0)

        act_i8, act_pt, _ = torch.ops.npu.npu_grouped_matmul_swiglu_quant(
            x=sorted_hidden_i8,
            weight=self.experts_w13,
            group_list=group_list,
            weight_scale=self.experts_w13_scale,
            x_scale=pertoken_scale,
        )

        out = torch.ops.npu.npu_grouped_matmul(
            x=[act_i8], weight=[self.experts_w2],
            scale=[self.experts_w2_scale.to(torch.bfloat16)],
            per_token_scale=[act_pt],
            split_item=2, group_list_type=0, group_type=0,
            group_list=group_list, output_dtype=torch.bfloat16)[0]

        routed = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=out,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_w.to(out.dtype),
        )
        routed = routed * self.routed_scaling
        shared_out = self.shared_experts(hidden)
        final = routed + shared_out
        if self.cfg.tp_size > 1:
            ops.all_reduce_(final)
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
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.self_attn = DeepseekV3MLAAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        if layer_id < cfg.first_k_dense_replace:
            self.mlp = DeepseekV3MLP(
                cfg, cfg.intermediate_size, dtype, device
            )
        else:
            self.mlp = DeepseekV3MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(hidden, positions)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class DeepseekV3Model(nn.Module):
    def __init__(
        self, cfg: DeepseekV3Config, dtype: torch.dtype, device: torch.device
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
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(cfg, i, dtype, device)
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        hidden, last_hidden = self.norm(hidden, residual)
        return hidden


class DeepseekV3ForCausalLM(PyModelBase):
    """DeepSeek-V3.2 causal LM. Registered under ``model_type='deepseek_v32'``."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV3Config.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get(
            "tp_rank", _tp_rank_from_device(config.get("device", "npu:0"))))
        dtype = self.resolve_dtype(
            config.get("dtype") or config.get("torch_dtype")
        )
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model = DeepseekV3Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg
        tp_size = cfg.tp_size
        tp_rank = cfg.tp_rank

        params_by_name = dict(self.named_parameters())
        buffers_by_name = dict(self.named_buffers())

        def find(name: str):
            for sd in state_dicts:
                if sd.has(name):
                    return sd
            return None

        def load_tensor(name: str) -> torch.Tensor:
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            return sd.get_tensor(name)

        def shard(t: torch.Tensor, dim: int, world: int = tp_size, rank: int = tp_rank) -> torch.Tensor:
            if world <= 1:
                return t
            cs = t.size(dim) // world
            return t.narrow(dim, rank * cs, cs).contiguous()

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            p = params_by_name.get(param_name)
            if p is None:
                p = buffers_by_name.get(param_name)
            assert p is not None, f"no parameter/buffer named {param_name}"
            p.data.copy_(tensor.to(dtype=p.dtype, device=p.device))

        def load_w8a8_a(prefix: str, proj: str, shard_dims=None) -> None:
            for suffix in ("weight", "deq_scale", "quant_bias",
                           "input_scale", "input_offset"):
                t = load_tensor(prefix + proj + "." + suffix)
                dim = (shard_dims or {}).get(suffix)
                if dim is not None:
                    t = shard(t, dim=dim)
                copy_in(prefix + proj + "." + suffix, t)

        def load_w8a8_b(mlp_pfx: str) -> None:
            gw = load_tensor(mlp_pfx + "gate_proj.weight")
            gs = load_tensor(mlp_pfx + "gate_proj.weight_scale")
            go = load_tensor(mlp_pfx + "gate_proj.weight_offset")
            uw = load_tensor(mlp_pfx + "up_proj.weight")
            us = load_tensor(mlp_pfx + "up_proj.weight_scale")
            uo = load_tensor(mlp_pfx + "up_proj.weight_offset")
            copy_in(mlp_pfx + "gate_up_proj.weight",
                    torch.cat([shard(gw, 0), shard(uw, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "gate_up_proj.weight_scale",
                    torch.cat([shard(gs, 0), shard(us, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "gate_up_proj.weight_offset",
                    torch.cat([shard(go, 0), shard(uo, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "down_proj.weight",
                    shard(load_tensor(mlp_pfx + "down_proj.weight"), dim=1))
            copy_in(mlp_pfx + "down_proj.weight_scale",
                    load_tensor(mlp_pfx + "down_proj.weight_scale"))
            copy_in(mlp_pfx + "down_proj.weight_offset",
                    load_tensor(mlp_pfx + "down_proj.weight_offset"))

        copy_in("model.embed_tokens.weight",
                shard(load_tensor("model.embed_tokens.weight"), dim=1))

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            copy_in(p + "input_layernorm.weight",
                    load_tensor(p + "input_layernorm.weight"))
            copy_in(p + "post_attention_layernorm.weight",
                    load_tensor(p + "post_attention_layernorm.weight"))
            attn = p + "self_attn."
            load_w8a8_a(attn, "q_a_proj")
            copy_in(attn + "q_a_layernorm.weight",
                    load_tensor(attn + "q_a_layernorm.weight"))
            load_w8a8_a(attn, "q_b_proj",
                        {"weight": 0, "deq_scale": 0, "quant_bias": 0})
            load_w8a8_a(attn, "kv_a_proj_with_mqa")
            copy_in(attn + "kv_a_layernorm.weight",
                    load_tensor(attn + "kv_a_layernorm.weight"))
            copy_in(attn + "kv_b_proj.weight",
                    shard(load_tensor(attn + "kv_b_proj.weight"), dim=0))
            load_w8a8_a(attn, "o_proj", {"weight": 1})
            if cfg.index_topk > 0:
                idx = attn + "indexer."
                copy_in(idx + "wq_b.weight", load_tensor(idx + "wq_b.weight"))
                copy_in(idx + "wk.weight", load_tensor(idx + "wk.weight"))
                copy_in(idx + "weights_proj.weight",
                        load_tensor(idx + "weights_proj.weight"))
                copy_in(idx + "k_norm.weight", load_tensor(idx + "k_norm.weight"))
                copy_in(idx + "k_norm.bias", load_tensor(idx + "k_norm.bias"))
            self.model.layers[i].self_attn.process_weights_after_loading()

            if i < cfg.first_k_dense_replace:
                load_w8a8_b(p + "mlp.")
                self.model.layers[i].mlp.process_weights_after_loading()
            else:
                se = p + "mlp.experts."
                w13_param = self.get_parameter(p + "mlp.experts_w13")
                w2_param = self.get_parameter(p + "mlp.experts_w2")
                w13_scale = self.get_buffer(p + "mlp.experts_w13_scale")
                w13_offset = self.get_buffer(p + "mlp.experts_w13_offset")
                w2_scale = self.get_buffer(p + "mlp.experts_w2_scale")
                w2_offset = self.get_buffer(p + "mlp.experts_w2_offset")
                for j in range(cfg.n_routed_experts):
                    gw = load_tensor(se + f"{j}.gate_proj.weight")
                    gs = load_tensor(se + f"{j}.gate_proj.weight_scale")
                    go = load_tensor(se + f"{j}.gate_proj.weight_offset")
                    uw = load_tensor(se + f"{j}.up_proj.weight")
                    us = load_tensor(se + f"{j}.up_proj.weight_scale")
                    uo = load_tensor(se + f"{j}.up_proj.weight_offset")
                    dw = load_tensor(se + f"{j}.down_proj.weight")
                    ds = load_tensor(se + f"{j}.down_proj.weight_scale")
                    do = load_tensor(se + f"{j}.down_proj.weight_offset")
                    w13_param.data[j].copy_(
                        torch.cat([shard(gw, 0), shard(uw, 0)], dim=0).contiguous())
                    w13_scale.data[j].copy_(
                        torch.cat([shard(gs, 0), shard(us, 0)], dim=0).contiguous())
                    w13_offset.data[j].copy_(
                        torch.cat([shard(go, 0), shard(uo, 0)], dim=0).contiguous())
                    w2_param.data[j].copy_(shard(dw, 1).contiguous())
                    w2_scale.data[j].copy_(ds.contiguous())
                    w2_offset.data[j].copy_(do.contiguous())
                copy_in(p + "mlp.gate.weight", load_tensor(p + "mlp.gate.weight"))
                copy_in(p + "mlp.e_score_correction_bias",
                        load_tensor(p + "mlp.gate.e_score_correction_bias"))
                load_w8a8_b(p + "mlp.shared_experts.")
                self.model.layers[i].mlp.process_weights_after_loading()

        copy_in("model.norm.weight", load_tensor("model.norm.weight"))
        copy_in("lm_head.weight", shard(load_tensor("lm_head.weight"), dim=0))

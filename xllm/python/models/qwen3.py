# Copyright 2025-2026 The xLLM Authors.
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

"""Qwen3 dense causal LM (Python model executor target).

Architecture: fused add + RMSNorm carrying (hidden, residual) between layers,
QK-norm before RoPE, gated-SiLU MLP. Tensor parallelism when tp_size>1.

Attention is delegated to the FlashInferBackend via the scoped ForwardContext.
The model does not import FlashInfer, own wrappers, or call plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from xllm.python import ops
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


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    sliding_window: int = 0
    attention_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3Config":
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=1024))
        n_heads = int(pick("n_heads", "num_attention_heads", default=16))
        return cls(
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=28)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=int(pick("head_dim", default=hidden // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=3072)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1e6)),
            max_position_embeddings=int(
                pick("max_position_embeddings", default=40960)
            ),
            vocab_size=int(pick("vocab_size", default=151936)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=True)),
            sliding_window=int(pick("sliding_window", default=0)),
            attention_bias=bool(pick("attention_bias", default=False)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
        )

    def head_split(self) -> Tuple[int, int, int]:
        """Per-rank ``(num_heads, num_kv_heads, num_kv_head_replicas)``."""
        tp = self.tp_size
        assert self.n_heads % tp == 0, (
            f"n_heads {self.n_heads} not divisible by tp_size {tp}"
        )
        num_heads = self.n_heads // tp
        if self.n_kv_heads >= tp:
            assert self.n_kv_heads % tp == 0, (
                f"n_kv_heads {self.n_kv_heads} not divisible by tp_size {tp}"
            )
            num_kv_heads = self.n_kv_heads // tp
            replicas = 1
        else:
            assert tp % self.n_kv_heads == 0, (
                f"tp_size {tp} not divisible by n_kv_heads {self.n_kv_heads}"
            )
            num_kv_heads = 1
            replicas = tp // self.n_kv_heads
        return num_heads, num_kv_heads, replicas


class Qwen3MLP(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.intermediate_size % tp == 0
        inter_per_rank = cfg.intermediate_size // tp
        self.gate_up_proj = ColumnParallelLinear(
            cfg.hidden_size,
            2 * inter_per_rank,
            tp,
            dtype=dtype,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            inter_per_rank,
            cfg.hidden_size,
            tp,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        act = ops.silu_and_mul(gate_up)
        return self.down_proj(act)


class Qwen3Attention(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        num_heads, num_kv_heads, replicas = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = RMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.k_norm = RMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)

        q, k, v = ops.fused_qk_norm_rope(
            qkv,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            num_heads_v=self.num_kv_heads,
            head_dim=self.head_dim,
            eps=self.q_norm.eps,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin_cache=cos_sin_cache,
            position_ids=positions,
            cos=cos,
            sin=sin,
        )

        attn_out = self.attn(q, k, v)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.self_attn = Qwen3Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.mlp = Qwen3MLP(cfg, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(
            positions, hidden, cos_sin_cache, cos, sin
        )

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen3Model(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(cfg, i, dtype, device)
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        # The fused QK-norm+RoPE kernel requires int64 position ids, but C++
        # passes them as int32. Cast once here instead of once per layer. In the
        # captured decode graph this single cast is recorded inside the graph
        # (its output lives in the graph memory pool), so replay re-casts the
        # updated static_positions correctly.
        positions = positions.to(torch.int64).contiguous()
        cos, sin = None, None
        if get_forward_context().device.type in ("npu", "privateuseone"):
            cos, sin = self.rotary(positions)
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden, residual = layer(
                hidden, residual, positions, self.rotary.cos_sin_cache, cos, sin
            )
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3ForCausalLM(PyModelBase):
    """Top-level entry the C++ PyCausalLM drives."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3Config.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model = Qwen3Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # -- weight loading ---------------------------------------------------
    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg

        total_kv_heads = cfg.n_kv_heads
        kv_replicas = tp_size // total_kv_heads if total_kv_heads < tp_size else 1
        kv_rank = tp_rank // kv_replicas if kv_replicas > 1 else tp_rank
        kv_world = tp_size // kv_replicas if kv_replicas > 1 else tp_size

        def find(name: str):
            for sd in state_dicts:
                if sd.has(name):
                    return sd
            return None

        def load_tensor(name: str) -> "torch.Tensor":
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            return sd.get_tensor(name)

        def shard(name: str, dim: int, kv: bool = False) -> "torch.Tensor":
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            r = kv_rank if kv else tp_rank
            w = kv_world if kv else tp_size
            t = sd.get_tensor(name)
            if w <= 1:
                return t
            chunk_size = t.size(dim) // w
            return t.narrow(dim, r * chunk_size, chunk_size).contiguous()

        def copy_in(param_name: str, tensor: "torch.Tensor") -> None:
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        embed_name = "model.embed_tokens.weight"
        if not find(embed_name):
            embed_name = "embed_tokens.weight"
        copy_in("model.embed_tokens.weight", shard(embed_name, dim=1))

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."

            copy_in(p + "input_layernorm.weight",
                    load_tensor(p + "input_layernorm.weight"))
            copy_in(p + "post_attention_layernorm.weight",
                    load_tensor(p + "post_attention_layernorm.weight"))
            copy_in(p + "self_attn.q_norm.weight",
                    load_tensor(p + "self_attn.q_norm.weight"))
            copy_in(p + "self_attn.k_norm.weight",
                    load_tensor(p + "self_attn.k_norm.weight"))

            q = shard(p + "self_attn.q_proj.weight", dim=0)
            k = shard(p + "self_attn.k_proj.weight", dim=0, kv=True)
            v = shard(p + "self_attn.v_proj.weight", dim=0, kv=True)
            copy_in(p + "self_attn.qkv_proj.weight", torch.cat([q, k, v], dim=0))

            copy_in(p + "self_attn.o_proj.weight",
                    shard(p + "self_attn.o_proj.weight", dim=1))

            if cfg.attention_bias:
                qb = shard(p + "self_attn.q_proj.bias", dim=0)
                kb = shard(p + "self_attn.k_proj.bias", dim=0, kv=True)
                vb = shard(p + "self_attn.v_proj.bias", dim=0, kv=True)
                copy_in(p + "self_attn.qkv_proj.bias",
                        torch.cat([qb, kb, vb], dim=0))
                # o_proj bias is replicated and added after the all-reduce, so
                # every rank loads the full (unsharded) bias.
                copy_in(p + "self_attn.o_proj.bias",
                        load_tensor(p + "self_attn.o_proj.bias"))

            gate = shard(p + "mlp.gate_proj.weight", dim=0)
            up = shard(p + "mlp.up_proj.weight", dim=0)
            copy_in(p + "mlp.gate_up_proj.weight", torch.cat([gate, up], dim=0))

            copy_in(p + "mlp.down_proj.weight",
                    shard(p + "mlp.down_proj.weight", dim=1))

        norm_name = "model.norm.weight"
        if not find(norm_name):
            norm_name = "norm.weight"
        copy_in("model.norm.weight", load_tensor(norm_name))

        if cfg.tie_word_embeddings:
            lm_name = embed_name
        else:
            lm_name = "lm_head.weight"
        copy_in("lm_head.weight", shard(lm_name, dim=0))

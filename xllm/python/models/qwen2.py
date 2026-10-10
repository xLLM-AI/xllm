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

"""Qwen2 dense decoder tower for the Python model executor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import cast

import torch
import torch.nn as nn

from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    apply_rotary_half,
    gather_half_rope_cos_sin,
)
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    gqa_head_split,
    load_gqa_fused_attention,
)


@dataclass
class Qwen2Config:
    hidden_size: int = 3584
    n_layers: int = 28
    n_heads: int = 28
    n_kv_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    vocab_size: int = 152064
    tie_word_embeddings: bool = False
    sliding_window: int = 0
    attention_bias: bool = True
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> Qwen2Config:
        kwargs = {field.name: d[field.name] for field in fields(cls) if d.get(field.name) is not None}
        for field, key in (
            ("n_layers", "num_hidden_layers"),
            ("n_heads", "num_attention_heads"),
            ("n_kv_heads", "num_key_value_heads"),
        ):
            if field not in kwargs and d.get(key) is not None:
                kwargs[field] = d[key]
        cfg = cls(**kwargs)
        # HF fallbacks: absent kv heads means MHA, absent head_dim is derived.
        if "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if d.get("head_dim") is None:
            cfg.head_dim = cfg.hidden_size // cfg.n_heads
        return cfg

    def head_split(self) -> tuple[int, int]:
        """Per-rank ``(num_heads, num_kv_heads)``."""
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        num_heads, num_kv_heads = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # Unsharded kv head count (the fused loader derives the split from it).
        self.n_kv_heads_full = cfg.n_kv_heads
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
            bias=False,  # HF Qwen2 never has o_proj bias
            dtype=dtype,
            device=device,
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
        hidden: torch.Tensor,
        half_cos: torch.Tensor,
        half_sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        num_tokens = qkv.size(0)
        q = qkv[:, : self.q_size].view(num_tokens, self.num_heads, self.head_dim)
        k = qkv[:, self.q_size : self.q_size + self.kv_size].view(num_tokens, self.num_kv_heads, self.head_dim)
        v = qkv[:, self.q_size + self.kv_size :]

        q = apply_rotary_half(q, half_cos, half_sin)
        k = apply_rotary_half(k, half_cos, half_sin)

        attn_out = self.attn(q.reshape(num_tokens, self.q_size), k.reshape(num_tokens, self.kv_size), v)
        return self.o_proj(attn_out)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        # HF Qwen2 has no per-head q/k RMSNorm.
        load_gqa_fused_attention(self, weights, context, self.n_kv_heads_full, qk_norm=False)
        self.o_proj.process_weights_after_loading()


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen2Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen2Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.tp_size,
            dtype,
            device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        half_cos: torch.Tensor,
        half_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(hidden, half_cos, half_sin)

        residual_out: torch.Tensor
        hidden, residual_out = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual_out

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        if tp <= 0 or cfg.hidden_size % tp != 0:
            raise ValueError("Qwen2 hidden_size must be divisible by a positive tp_size")
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
        self.layers = cast(
            "Sequence[Qwen2DecoderLayer]",
            nn.ModuleList(Qwen2DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)),
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids) if input_embedding is None else input_embedding

        positions = positions.to(torch.int64).contiguous()
        half_cos, half_sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, half_cos, half_sin)
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        return hidden

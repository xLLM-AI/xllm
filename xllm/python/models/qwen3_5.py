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

"""Qwen3.5 hybrid-attention causal LM for the Python executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, GemmaRMSNorm, HiddenParallelEmbedding
from xllm.python.layers.qwen3_5_decoder_layer import (
    PartialRotaryEmbedding,
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.weight_utils import gqa_head_split


@dataclass
class Qwen3_5Config:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    vocab_size: int
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    attention_bias: bool
    attn_output_gate: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: list[int]
    norm_topk_prob: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    world_size: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3_5Config:
        def pick(*keys, default=None):
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return default

        n_layers = int(pick("n_layers", "num_hidden_layers", default=0))
        interval = int(pick("full_attention_interval", default=4))
        layer_types = list(pick("layer_types", default=[]))
        if not layer_types:
            layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)]
        if len(layer_types) != n_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")

        hidden_size = int(pick("hidden_size", default=0))
        n_heads = int(pick("n_heads", "num_attention_heads", default=0))
        tp_size = int(pick("tp_size", default=1))
        dp_size = int(pick("dp_size", default=1))
        world_size = int(pick("world_size", default=tp_size * dp_size))
        ep_size = int(pick("ep_size", default=1))
        return cls(
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=0)),
            head_dim=int(pick("head_dim", default=hidden_size // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=0)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1e7)),
            partial_rotary_factor=float(pick("partial_rotary_factor", default=0.25)),
            max_position_embeddings=int(pick("max_position_embeddings", default=262144)),
            vocab_size=int(pick("vocab_size", default=248320)),
            layer_types=layer_types,
            linear_conv_kernel_dim=int(pick("linear_conv_kernel_dim", default=4)),
            linear_key_head_dim=int(pick("linear_key_head_dim", default=128)),
            linear_value_head_dim=int(pick("linear_value_head_dim", default=128)),
            linear_num_key_heads=int(pick("linear_num_key_heads", default=16)),
            linear_num_value_heads=int(pick("linear_num_value_heads", default=32)),
            attention_bias=bool(pick("attention_bias", default=False)),
            attn_output_gate=bool(pick("attn_output_gate", default=True)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            num_experts=int(pick("num_experts", "n_routed_experts", default=0)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=0)),
            decoder_sparse_step=int(pick("decoder_sparse_step", default=1)),
            mlp_only_layers=[int(layer_id) for layer_id in pick("mlp_only_layers", default=[])],
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=0)),
            shared_expert_intermediate_size=int(pick("shared_expert_intermediate_size", default=0)),
            tp_size=tp_size,
            tp_rank=int(pick("tp_rank", default=0)),
            dp_size=dp_size,
            dp_rank=int(pick("dp_rank", default=0)),
            world_size=world_size,
            moe_tp_size=int(pick("moe_tp_size", default=world_size // max(ep_size, 1))),
            moe_tp_rank=int(pick("moe_tp_rank", default=0)),
            ep_size=ep_size,
            ep_rank=int(pick("ep_rank", default=0)),
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0 or self.n_layers <= 0:
            raise ValueError("invalid Qwen3.5 model dimensions")
        if min(self.tp_size, self.dp_size, self.moe_tp_size, self.ep_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size")
        if self.moe_tp_size * self.ep_size != self.world_size:
            raise ValueError("world_size must equal moe_tp_size * ep_size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if not 0 <= self.moe_tp_rank < self.moe_tp_size:
            raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        for name, count in (
            ("hidden size", self.hidden_size),
            ("attention heads", self.n_heads),
            ("linear key heads", self.linear_num_key_heads),
            ("linear value heads", self.linear_num_value_heads),
            ("dense intermediate size", self.intermediate_size),
            ("vocabulary size", self.vocab_size),
        ):
            if count % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size")
        if self.n_kv_heads >= self.tp_size:
            if self.n_kv_heads % self.tp_size:
                raise ValueError("KV heads must be divisible by tp_size when KV heads are sharded")
        elif self.tp_size % self.n_kv_heads:
            raise ValueError("tp_size must be divisible by KV heads when KV heads are replicated")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if self.num_experts:
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok must be positive for MoE")
            if self.num_experts % self.ep_size:
                raise ValueError("num_experts must be divisible by ep_size")
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive for MoE")
            if self.moe_intermediate_size % self.moe_tp_size:
                raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
            if self.shared_expert_intermediate_size <= 0:
                raise ValueError("shared_expert_intermediate_size must be positive for Qwen3.5 MoE")
            if self.shared_expert_intermediate_size % self.tp_size:
                raise ValueError("shared_expert_intermediate_size must be divisible by tp_size")

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
            and layer_id not in self.mlp_only_layers
        )

    def head_split(self) -> tuple[int, int]:
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rotary = PartialRotaryEmbedding(
            cfg.head_dim,
            rotary_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
        decoder_layer_cls = get_qwen3_5_decoder_layer_class(device)
        self.layers = nn.ModuleList(decoder_layer_cls(cfg, i, dtype, device, self.rotary) for i in range(cfg.n_layers))
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3_5ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3_5Config.from_dict(config)
        self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        if self.cfg.vocab_size % self.cfg.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")
        self.model = Qwen3_5Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        all_weights = ScopedWeightLoader(state_dicts)
        model_weights = all_weights.find_root(
            ("model.language_model.", "model.", ""),
            "embed_tokens.weight",
        )
        context = ParallelLoadContext(
            tp_rank=tp_rank,
            tp_size=tp_size,
            dp_rank=self.cfg.dp_rank,
            dp_size=self.cfg.dp_size,
            moe_tp_rank=self.cfg.moe_tp_rank,
            moe_tp_size=self.cfg.moe_tp_size,
            ep_rank=self.cfg.ep_rank,
            ep_size=self.cfg.ep_size,
        )
        copy_parameter(
            self.model.embed_tokens.weight,
            model_weights.shard(
                "embed_tokens.weight",
                1,
                tp_rank,
                tp_size,
            ),
            model_weights.prefix + "embed_tokens.weight",
        )
        for layer_id, layer in enumerate(self.model.layers):
            layer.load_weights(
                model_weights.with_prefix(f"layers.{layer_id}."),
                context,
            )
        copy_parameter(
            self.model.norm.weight,
            model_weights.tensor("norm.weight"),
            model_weights.prefix + "norm.weight",
        )
        if self.cfg.tie_word_embeddings or not all_weights.has("lm_head.weight"):
            lm_head_weights = model_weights
            lm_head_name = "embed_tokens.weight"
        else:
            lm_head_weights = all_weights
            lm_head_name = "lm_head.weight"
        copy_parameter(
            self.lm_head.weight,
            lm_head_weights.shard(lm_head_name, 0, tp_rank, tp_size),
            lm_head_weights.prefix + lm_head_name,
        )

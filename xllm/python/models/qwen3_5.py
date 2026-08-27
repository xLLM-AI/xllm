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

from xllm.python import distributed
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    FusedMoE,
    GatedMLP,
    GemmaRMSNorm,
    HiddenParallelEmbedding,
    RowParallelLinear,
)
from xllm.python.layers.gated_delta_net import Qwen3_5GatedDeltaNet
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import load_qwen3_attention
from xllm.python.models.weight_utils import WeightLoader, gqa_head_split, kv_replica_shard


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
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("invalid Qwen3.5 model dimensions")
        if min(self.tp_size, self.dp_size, self.moe_tp_size, self.ep_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size")
        if self.ep_size not in (1, self.world_size):
            raise ValueError("Qwen3.5 Python supports only ep_size=1 or world_size")
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
            ("attention heads", self.n_heads),
            ("linear key heads", self.linear_num_key_heads),
            ("linear value heads", self.linear_num_value_heads),
        ):
            if count % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size")
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

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
            and layer_id not in self.mlp_only_layers
        )

    def head_split(self) -> tuple[int, int]:
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen3_5SparseMoEBlock(nn.Module):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        # The routed and shared branches each produce a partial sum over the same
        # set of ranks whenever every group spans the whole world, and the gate is
        # computed from replicated hidden states so it is bit-identical on every
        # rank. Summing first then reducing once is therefore exact, and removes
        # one of the three all-reduces this block would otherwise issue per layer.
        # With dp_size > 1 the groups no longer coincide, so each branch keeps its
        # own reduction.
        self.fuse_reductions = cfg.dp_size == 1 and cfg.tp_size > 1
        self.experts = FusedMoE(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            renormalize=cfg.norm_topk_prob,
            moe_tp_size=cfg.moe_tp_size,
            moe_tp_rank=cfg.moe_tp_rank,
            ep_size=cfg.ep_size,
            ep_rank=cfg.ep_rank,
            dp_size=cfg.dp_size,
            dp_rank=cfg.dp_rank,
            dtype=dtype,
            device=device,
            reduce_results=not self.fuse_reductions,
        )
        self.shared_expert = GatedMLP(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            cfg.tp_size,
            dtype,
            device,
            reduce_results=not self.fuse_reductions,
        )
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        routed = self.experts(hidden)
        shared = self.shared_expert(hidden)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))
        output = routed + shared * shared_gate
        if self.fuse_reductions:
            distributed.all_reduce_(output)
        return output


class PartialRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        rope_theta: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be positive and even")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float32, device=device), inv_freq)
        self.register_buffer("cos", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin", freqs.sin().to(dtype), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = x.split([self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
        pos = positions.to(torch.long)
        cos = torch.cat((self.cos[pos], self.cos[pos]), dim=-1).unsqueeze(1)
        sin = torch.cat((self.sin[pos], self.sin[pos]), dim=-1).unsqueeze(1)
        rotary = rotary * cos + self._rotate_half(rotary) * sin
        return torch.cat((rotary, passthrough), dim=-1)


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads, self.num_kv_heads = cfg.head_split()
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        q_multiplier = 2 if cfg.attn_output_gate else 1
        self.attn_output_gate = cfg.attn_output_gate
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            q_multiplier * self.q_size + 2 * self.kv_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = GemmaRMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = rotary
        self.attn = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim**-0.5,
            0,
            layer_id,
        )

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
            q, gate = q_gate.chunk(2, dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rotary(positions, q).reshape(-1, self.q_size)
        k = self.rotary(positions, k).reshape(-1, self.kv_size)
        output = self.attn(q, k, v)
        if gate is not None:
            output = output * torch.sigmoid(gate.reshape(-1, self.q_size))
        return self.o_proj(output)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_type = cfg.layer_types[layer_id]
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(cfg, layer_id, dtype, device, rotary)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(cfg, layer_id, dtype, device)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        if cfg.is_moe_layer(layer_id):
            self.mlp = Qwen3_5SparseMoEBlock(cfg, dtype, device)
        else:
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
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(positions, hidden)
        else:
            hidden = self.linear_attn(hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


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
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(cfg, i, dtype, device, self.rotary) for i in range(cfg.n_layers)
        )
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # TODO: The current tilelang-ascend version has a cache bug that prevents kernels with
        # dynamic symbols from being cached, causing the service to crash. This is a temporary
        # workaround; we will resubmit once tilelang-ascend fixes the issue.
        import tilelang

        tilelang.disable_cache()
        tilelang.cache.clear_cache()
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
        cfg = self.cfg
        loader = WeightLoader(
            self,
            state_dicts,
            tp_size,
            tp_rank,
            src_prefixes=("model.language_model.", "model.", ""),
        )

        kv_world, kv_rank = kv_replica_shard(cfg.n_kv_heads, tp_rank, tp_size)

        # linear_attention branch only: split fused in_proj_qkv (and conv1d)
        # into (key, key, value) global chunks, then TP-shard each chunk.
        global_key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        global_value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        qkv_sizes = (global_key, global_key, global_value)

        def split_shard_cat(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([loader.shard(p, 0, contiguous=False) for p in t.split(qkv_sizes, dim=0)])

        # MoE experts are sharded on the expert axis by EP, then on the
        # intermediate axis by MoE-TP; both tuples are load-time constants.
        ep = (cfg.ep_size, cfg.ep_rank)
        mtp = (cfg.moe_tp_size, cfg.moe_tp_rank)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.load_shard("embed_tokens.weight", 1),
        )
        for layer_id, layer_type in enumerate(cfg.layer_types):
            source = f"layers.{layer_id}."
            target = f"model.layers.{layer_id}."
            for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                loader.copy_in(target + norm, loader.load_tensor(source + norm))

            if layer_type == "full_attention":
                load_qwen3_attention(
                    loader,
                    source,
                    target,
                    kv_world=kv_world,
                    kv_rank=kv_rank,
                    attention_bias=cfg.attention_bias,
                )
            else:
                linear = source + "linear_attn."
                loader.copy_in(
                    target + "linear_attn.in_proj_qkv.weight",
                    split_shard_cat(loader.load_tensor(linear + "in_proj_qkv.weight")),
                )
                for projection in ("in_proj_z", "in_proj_b", "in_proj_a"):
                    loader.copy_in(
                        target + f"linear_attn.{projection}.weight",
                        loader.load_shard(linear + f"{projection}.weight", 0),
                    )
                loader.copy_in(
                    target + "linear_attn.conv1d_weight",
                    split_shard_cat(loader.load_tensor(linear + "conv1d.weight").squeeze(1)),
                )
                for name in ("A_log", "dt_bias"):
                    loader.copy_in(
                        target + f"linear_attn.{name}",
                        loader.load_shard(linear + name, 0),
                    )
                loader.copy_in(
                    target + "linear_attn.norm_weight",
                    loader.load_tensor(linear + "norm.weight"),
                )
                loader.copy_in(
                    target + "linear_attn.out_proj.weight",
                    loader.load_shard(linear + "out_proj.weight", 1),
                )

            if cfg.is_moe_layer(layer_id):
                moe = source + "mlp."
                loader.copy_in(target + "mlp.experts.gate.weight", loader.load_tensor(moe + "gate.weight"))
                loader.copy_in(
                    target + "mlp.shared_expert_gate.weight",
                    loader.load_tensor(moe + "shared_expert_gate.weight"),
                )

                # contiguous=False: one materialization at the final cat/copy, not per split.
                gate_up = loader.shard(
                    loader.load_tensor(moe + "experts.gate_up_proj"), 0, *ep, contiguous=False
                )
                gate, up = gate_up.chunk(2, dim=1)
                gate = loader.shard(gate, 1, *mtp, contiguous=False)
                up = loader.shard(up, 1, *mtp, contiguous=False)
                # The checkpoint is [gate, up]; xLLM CUTLASS SwiGLU
                # consumes [linear/up, gate].
                loader.copy_in(target + "mlp.experts.w13", torch.cat((up, gate), dim=1))

                down = loader.shard(
                    loader.load_tensor(moe + "experts.down_proj"), 0, *ep, contiguous=False
                )
                loader.copy_in(target + "mlp.experts.w2", loader.shard(down, 2, *mtp, contiguous=False))

                shared = moe + "shared_expert."
                loader.load_gated_mlp(target + "mlp.shared_expert.", shared)
            else:
                loader.load_gated_mlp(target + "mlp.", source + "mlp.")

        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        lm_head_name = "lm_head.weight"
        if cfg.tie_word_embeddings or not loader.has(lm_head_name):
            lm_head_name = "embed_tokens.weight"
        loader.copy_in("lm_head.weight", loader.load_shard(lm_head_name, 0))

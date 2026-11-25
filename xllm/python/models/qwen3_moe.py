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

"""Qwen3-MoE on the Python model path.

Qwen3-MoE is the dense Qwen3 stack with its feed-forward block replaced by
routed experts: a replicated router selects ``num_experts_per_tok`` of
``num_experts`` and their outputs are summed, with no shared expert. Attention
is unchanged, so this module reuses the dense model's attention and its
forward, and adds the three things that differ -- the routing config, the
sparse block, and the checkpoint layout, which stores one tensor per expert
rather than one stacked tensor per layer.

Experts are sharded by their intermediate dimension across the MoE
tensor-parallel group, so every rank sees every expert and holds a slice of
each. Each rank therefore computes a partial sum and the group reduces once
per layer, which keeps the routing decision identical on every rank without
exchanging tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import ColumnParallelLinear, GatedMLP, RMSNorm
from xllm.python.layers.moe_parallel import dp_gather_tokens, reduce_and_scatter
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_causal_lm_weights,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import Qwen3Attention, Qwen3Config, Qwen3Model


@dataclass
class Qwen3MoeConfig(Qwen3Config):
    """The dense Qwen3 config plus the fields that describe expert routing."""

    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3MoeConfig:
        base = cast(Qwen3MoeConfig, super().from_dict(d))

        def pick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        return replace(
            base,
            num_experts=int(pick("num_experts", "n_routed_experts", default=0)),
            num_experts_per_tok=int(pick("num_experts_per_tok", "num_experts_per_token", default=0)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=0)),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            decoder_sparse_step=int(pick("decoder_sparse_step", default=1)),
            mlp_only_layers=tuple(int(layer_id) for layer_id in pick("mlp_only_layers", default=[])),
            # Experts are sharded with the attention by default, which is the
            # only topology this implementation serves.
            moe_tp_size=int(pick("moe_tp_size", default=base.tp_size)),
            moe_tp_rank=int(pick("moe_tp_rank", default=base.tp_rank)),
            ep_size=int(pick("ep_size", default=1)),
            ep_rank=int(pick("ep_rank", default=0)),
        )

    def validate(self) -> None:
        if self.num_experts <= 0 or self.num_experts_per_tok <= 0:
            raise ValueError("Qwen3-MoE requires a positive num_experts and num_experts_per_tok")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
        if self.moe_intermediate_size % self.moe_tp_size:
            raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
        if self.moe_tp_size != self.tp_size:
            raise ValueError("the Python Qwen3-MoE shards experts with the attention: moe_tp_size must equal tp_size")
        if self.ep_size != 1:
            raise NotImplementedError("the Python Qwen3-MoE does not implement expert parallelism")

    def is_moe_layer(self, layer_id: int) -> bool:
        """Whether this layer routes, mirroring the reference implementation."""
        if layer_id in self.mlp_only_layers:
            return False
        return self.decoder_sparse_step > 0 and (layer_id + 1) % self.decoder_sparse_step == 0


class Qwen3MoeSparseBlock(nn.Module):
    """One layer's routed experts, in the NPU grouped-matmul layout.

    The router is replicated and each rank holds its slice of every expert, so
    the only collective is the reduction of the rank-local partial sums.
    """

    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if dtype != torch.bfloat16:
            raise NotImplementedError("the Python Qwen3-MoE experts support BF16 only")
        local_intermediate = cfg.moe_intermediate_size // cfg.moe_tp_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.renormalize = cfg.norm_topk_prob
        self.moe_tp_size = cfg.moe_tp_size
        self.dp_size = cfg.dp_size
        self.dp_rank = cfg.dp_rank

        self.gate = nn.Linear(
            cfg.hidden_size,
            cfg.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # The grouped matmul reads gate and up fused on the output dimension.
        self.w13 = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                cfg.hidden_size,
                2 * local_intermediate,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                local_intermediate,
                cfg.hidden_size,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden, scatter_state = dp_gather_tokens(hidden, self.dp_size, self.dp_rank)
        topk_weights, topk_ids = kernels.moe_fused_topk(
            self.gate(hidden),
            self.top_k,
            self.renormalize,
            "softmax",
        )
        # Every rank owns a slice of all experts, so the whole expert range is
        # active here and the reduction below completes the sum.
        output = kernels.grouped_moe_bf16(
            hidden,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            self.num_experts,
            0,
            self.num_experts,
        )
        return reduce_and_scatter(
            output,
            scatter_state,
            reduce_results=True,
            moe_tp_size=self.moe_tp_size,
            ep_size=1,
        )

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        """Stack the per-expert checkpoint tensors into the grouped layout.

        The checkpoint keeps ``experts.{i}.{gate,up,down}_proj.weight`` apart,
        one tensor per expert. Each is sharded on its intermediate dimension
        and written straight into its slice of the stacked parameter, so the
        48 layers never hold more than one expert's staging copy at a time.
        """
        weights.load_tensor(self.gate.weight, "gate.weight")
        rank, world = context.moe_tp_rank, context.moe_tp_size
        for expert in range(self.num_experts):
            prefix = f"experts.{expert}."
            gate_name, up_name, down_name = (
                prefix + "gate_proj.weight",
                prefix + "up_proj.weight",
                prefix + "down_proj.weight",
            )
            gate = weights.shard_value(weights.get_tensor(gate_name), gate_name, 0, rank, world)
            up = weights.shard_value(weights.get_tensor(up_name), up_name, 0, rank, world)
            down = weights.shard_value(weights.get_tensor(down_name), down_name, 1, rank, world)
            weights.copy(self.w13[expert], torch.cat((gate, up), dim=0).t(), gate_name)
            weights.copy(self.w2[expert], down.t(), down_name)


class Qwen3MoeDecoderLayer(nn.Module):
    """Qwen3's decoder layer with routed experts in place of the dense FFN."""

    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen3Attention(cfg, layer_id, dtype, device, causal=causal)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.mlp: nn.Module = (
            Qwen3MoeSparseBlock(cfg, dtype, device)
            if self.is_moe
            else GatedMLP(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        if not self.is_moe:
            self.mlp.process_weights_after_loading()


class Qwen3MoeForCausalLM(PyModelBase):
    """Top-level entry the C++ PyCausalLM drives."""

    model: Qwen3Model
    lm_head: ColumnParallelLinear | None

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3MoeConfig.from_dict(config)
        self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu"))
        self.dtype = dtype
        self.device = device

        tp = self.cfg.tp_size
        dp = self.cfg.dp_size
        if tp * dp != int(config.get("world_size", tp * dp)):
            raise ValueError("world_size must equal tp_size * dp_size")
        if not 0 <= self.cfg.dp_rank < dp:
            raise ValueError("dp_rank must be in [0, dp_size)")
        self.model = Qwen3Model(  # pyright: ignore[reportIncompatibleVariableOverride]
            self.cfg,
            dtype,
            device,
            decoder_layer_type=cast(Any, Qwen3MoeDecoderLayer),
        )
        self.lm_head = None
        if not config.get("embedding_mode", False):
            if self.cfg.vocab_size % tp:
                raise ValueError("vocab_size must be divisible by tp_size")
            self.lm_head = ColumnParallelLinear(  # pyright: ignore[reportIncompatibleVariableOverride]
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
        all_weights = ScopedWeightLoader(state_dicts, src_prefixes=("model.", ""))
        ctx = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        load_causal_lm_weights(
            self.model,
            None if self.lm_head is None else self.lm_head.weight,
            all_weights,
            ctx,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
        )


__all__: Sequence[str] = (
    "Qwen3MoeConfig",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeSparseBlock",
)

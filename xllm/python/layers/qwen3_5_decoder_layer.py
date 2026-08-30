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

"""Shared semantics and construction-time dispatch for Qwen3.5 decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.model_loader import ScopedWeightLoader


class Qwen3_5LayerConfig(Protocol):
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    layer_types: list[str]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    attention_bias: bool
    attn_output_gate: bool
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    tp_size: int
    dp_size: int
    dp_rank: int
    world_size: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int

    def is_moe_layer(self, layer_id: int) -> bool: ...

    def head_split(self) -> tuple[int, int]: ...


@dataclass(frozen=True, slots=True)
class Qwen3_5LoadContext:
    tp_rank: int
    tp_size: int


class Qwen3_5SparseMoEBlock(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
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
        self.shared_expert_gate = nn.Linear(
            cfg.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )

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
            rope_theta
            ** (
                torch.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / rotary_dim
            )
        )
        positions = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin", freqs.sin().to(dtype), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = x.split(
            [self.rotary_dim, self.head_dim - self.rotary_dim],
            dim=-1,
        )
        pos = positions.to(torch.long)
        cos = torch.cat((self.cos[pos], self.cos[pos]), dim=-1).unsqueeze(1)
        sin = torch.cat((self.sin[pos], self.sin[pos]), dim=-1).unsqueeze(1)
        rotary = rotary * cos + self._rotate_half(rotary) * sin
        return torch.cat((rotary, passthrough), dim=-1)


class Qwen3_5DecoderLayerProtocol(Protocol):
    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None: ...

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: Qwen3_5LoadContext,
    ) -> None: ...


def get_qwen3_5_decoder_layer_class(
    device: torch.device | str,
) -> type[nn.Module]:
    device_type = torch.device(device).type
    if device_type == "cuda":
        from xllm.python.layers.cuda.qwen3_5.decoder_layer import (
            CudaQwen3_5DecoderLayer,
        )

        return CudaQwen3_5DecoderLayer
    if device_type in ("npu", "privateuseone"):
        from xllm.python.layers.npu.qwen3_5.decoder_layer import (
            NpuQwen3_5DecoderLayer,
        )

        return NpuQwen3_5DecoderLayer
    raise ValueError(f"Qwen3.5 Python has no decoder implementation for device {device_type!r}")

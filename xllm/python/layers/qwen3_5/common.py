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

"""Backend-neutral Qwen3.5 contracts and value objects."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class Qwen3_5AttentionConfig(Protocol):
    hidden_size: int
    n_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    attn_output_gate: bool
    tp_size: int

    def head_split(self) -> tuple[int, int]: ...


class Qwen3_5GatedDeltaNetConfig(Protocol):
    hidden_size: int
    rms_norm_eps: float
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    tp_size: int


class Qwen3_5MoEConfig(Protocol):
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    tp_size: int
    dp_size: int
    dp_rank: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int


class Qwen3_5DecoderConfig(
    Qwen3_5AttentionConfig,
    Qwen3_5GatedDeltaNetConfig,
    Qwen3_5MoEConfig,
    Protocol,
):
    intermediate_size: int
    layer_types: list[str]

    def is_moe_layer(self, layer_id: int) -> bool: ...


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

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

from typing import Protocol

import torch
import torch.nn as nn

from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


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
        context: ParallelLoadContext,
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

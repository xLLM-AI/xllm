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
from xllm_kernel.ops import partial_rotary_embedding


class Qwen3_5AttentionConfig(Protocol):
    hidden_size: int
    n_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    partial_rotary_factor: float
    rope_scaling_mrope_section: list[int]
    rope_scaling_mrope_interleaved: bool
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
    tp_rank: int
    dp_size: int
    dp_rank: int
    moe_tp_size: int
    moe_tp_rank: int
    ep_size: int
    ep_rank: int
    enable_mega_moe: bool
    mega_moe_context: torch.Tensor | None
    mega_moe_ccl_buffer_size: int
    mega_moe_num_max_tokens_per_rank: int


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
        *,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be positive and even")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.mrope_section = tuple(mrope_section or ())
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section and (
            len(self.mrope_section) != 3 or min(self.mrope_section) < 0 or sum(self.mrope_section) != rotary_dim // 2
        ):
            raise ValueError("mrope_section must partition the half rotary dimension into three axes")
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
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype),
            persistent=False,
        )

    def build_mrope_cos_sin(self, positions: torch.Tensor) -> torch.Tensor:
        """Gather the three-axis cosine/sine rows consumed by fused mRoPE."""
        rows = positions.shape[-1]
        cos_sin = self.cos_sin_cache.new_empty((rows, 3 * self.rotary_dim))
        if positions.dim() == 1:
            gathered = self.cos_sin_cache.index_select(0, positions)
            cos_sin.view(rows, 3, self.rotary_dim).copy_(gathered.unsqueeze(1))
        else:
            indices = positions.transpose(0, 1).contiguous().view(-1)
            torch.index_select(self.cos_sin_cache, 0, indices, out=cos_sin.view(-1, self.rotary_dim))
        return cos_sin

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pos = positions.to(torch.long)
        cached = self.cos_sin_cache[pos]
        half_rotary_dim = self.rotary_dim // 2
        cos_half = cached[..., :half_rotary_dim]
        sin_half = cached[..., half_rotary_dim:]
        if positions.dim() == 2:
            if positions.shape[0] != 3 or not self.mrope_section:
                raise ValueError("multimodal positions require three axes and mrope_section")
            cos_half = self._merge_mrope_axes(cos_half)
            sin_half = self._merge_mrope_axes(sin_half)
        return partial_rotary_embedding(x, cos_half, sin_half)

    def _merge_mrope_axes(self, values: torch.Tensor) -> torch.Tensor:
        if self.mrope_interleaved:
            merged = values[0].clone()
            _, height, width = self.mrope_section
            merged[:, 1 : height * 3 : 3] = values[1, :, 1 : height * 3 : 3]
            merged[:, 2 : width * 3 : 3] = values[2, :, 2 : width * 3 : 3]
            return merged
        return torch.cat(
            [part[axis] for axis, part in enumerate(values.split(self.mrope_section, dim=-1))],
            dim=-1,
        )

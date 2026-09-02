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

"""CUDA-owned Qwen3.5 decoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.cuda.qwen3_5.attention import CudaQwen3_5Attention
from xllm.python.layers.cuda.qwen3_5.gated_delta_net import (
    CudaQwen3_5GatedDeltaNet,
)
from xllm.python.layers.cuda.qwen3_5.moe import CudaQwen3_5SparseMoEBlock
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.qwen3_5_decoder_layer import (
    PartialRotaryEmbedding,
    Qwen3_5LayerConfig,
)
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)


class CudaQwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.layer_type = cfg.layer_types[layer_id]
        self.input_layernorm = GemmaRMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if self.layer_type == "full_attention":
            self.self_attn = CudaQwen3_5Attention(
                cfg,
                layer_id,
                dtype,
                device,
                rotary,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = CudaQwen3_5GatedDeltaNet(
                cfg,
                layer_id,
                dtype,
                device,
            )
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.post_attention_layernorm = GemmaRMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if cfg.is_moe_layer(layer_id):
            self.mlp = CudaQwen3_5SparseMoEBlock(cfg, dtype, device)
        else:
            self.mlp = GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.tp_size,
                dtype,
                device,
            )

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        copy_parameter(
            self.input_layernorm.weight,
            state.tensor("input_layernorm.weight"),
            state.prefix + "input_layernorm.weight",
        )
        copy_parameter(
            self.post_attention_layernorm.weight,
            state.tensor("post_attention_layernorm.weight"),
            state.prefix + "post_attention_layernorm.weight",
        )
        if self.layer_type == "full_attention":
            self.self_attn.load_weights(
                state.with_prefix("self_attn."),
                context,
            )
        else:
            self.linear_attn.load_weights(
                state.with_prefix("linear_attn."),
                context,
            )
        self.mlp.load_weights(state.with_prefix("mlp."), context)

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

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

"""NPU-owned Qwen3.5 decoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.qwen3_5_decoder_layer import (
    PartialRotaryEmbedding,
    Qwen3_5LayerConfig,
    Qwen3_5LoadContext,
    Qwen3_5SparseMoEBlock,
)
from xllm.python.model_loader import (
    ScopedWeightLoader,
    copy_parameter,
)

from .attention import NpuQwen3_5Attention
from .gated_delta_net import NpuQwen3_5GatedDeltaNet


class NpuQwen3_5DecoderLayer(nn.Module):
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
            self.self_attn = NpuQwen3_5Attention(
                cfg,
                layer_id,
                dtype,
                device,
                rotary,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = NpuQwen3_5GatedDeltaNet(
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
            self.mlp = Qwen3_5SparseMoEBlock(cfg, dtype, device)
        else:
            self.mlp = GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.tp_size,
                dtype,
                device,
            )

    def _load_mlp_weights(
        self,
        state: ScopedWeightLoader,
        context: Qwen3_5LoadContext,
    ) -> None:
        if self.cfg.is_moe_layer(self.layer_id):
            copy_parameter(
                self.mlp.experts.gate.weight,
                state.tensor("gate.weight"),
                state.prefix + "gate.weight",
            )
            copy_parameter(
                self.mlp.shared_expert_gate.weight,
                state.tensor("shared_expert_gate.weight"),
                state.prefix + "shared_expert_gate.weight",
            )

            gate_up = state.tensor("experts.gate_up_proj")
            local_experts = self.cfg.num_experts // self.cfg.ep_size
            start_expert = self.cfg.ep_rank * local_experts
            gate_up = gate_up.narrow(0, start_expert, local_experts)
            gate, up = gate_up.chunk(2, dim=1)
            gate = gate.chunk(self.cfg.moe_tp_size, dim=1)[self.cfg.moe_tp_rank]
            up = up.chunk(self.cfg.moe_tp_size, dim=1)[self.cfg.moe_tp_rank]
            copy_parameter(
                self.mlp.experts.w13,
                torch.cat((up, gate), dim=1),
                state.prefix + "experts.gate_up_proj",
            )

            down = state.tensor("experts.down_proj").narrow(
                0,
                start_expert,
                local_experts,
            )
            copy_parameter(
                self.mlp.experts.w2,
                down.chunk(self.cfg.moe_tp_size, dim=2)[self.cfg.moe_tp_rank],
                state.prefix + "experts.down_proj",
            )

            shared = state.with_prefix("shared_expert.")
            shared_gate = shared.shard(
                "gate_proj.weight",
                0,
                context.tp_rank,
                context.tp_size,
            )
            shared_up = shared.shard(
                "up_proj.weight",
                0,
                context.tp_rank,
                context.tp_size,
            )
            copy_parameter(
                self.mlp.shared_expert.gate_up_proj.weight,
                torch.cat((shared_gate, shared_up)),
                shared.prefix + "{gate,up}_proj.weight",
            )
            copy_parameter(
                self.mlp.shared_expert.down_proj.weight,
                shared.shard(
                    "down_proj.weight",
                    1,
                    context.tp_rank,
                    context.tp_size,
                ),
                shared.prefix + "down_proj.weight",
            )
        else:
            gate = state.shard(
                "gate_proj.weight",
                0,
                context.tp_rank,
                context.tp_size,
            )
            up = state.shard(
                "up_proj.weight",
                0,
                context.tp_rank,
                context.tp_size,
            )
            copy_parameter(
                self.mlp.gate_up_proj.weight,
                torch.cat((gate, up)),
                state.prefix + "{gate,up}_proj.weight",
            )
            copy_parameter(
                self.mlp.down_proj.weight,
                state.shard(
                    "down_proj.weight",
                    1,
                    context.tp_rank,
                    context.tp_size,
                ),
                state.prefix + "down_proj.weight",
            )

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: Qwen3_5LoadContext,
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
        self._load_mlp_weights(state.with_prefix("mlp."), context)

    @staticmethod
    def _prepare_tilelang_forward() -> None:
        # TODO: Remove this backend-local workaround once TileLang's dynamic
        # symbol cache can safely persist between service forwards.
        import tilelang

        tilelang.disable_cache()
        tilelang.cache.clear_cache()

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_id == 0:
            self._prepare_tilelang_forward()
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

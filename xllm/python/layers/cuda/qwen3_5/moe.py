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

"""CUDA-owned Qwen3.5 sparse MoE composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.qwen3_5_decoder_layer import Qwen3_5LayerConfig
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)


class CudaQwen3_5SparseMoEBlock(nn.Module):
    """CUDA expert graph with CUTLASS/Triton-native weight ordering."""

    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.fuse_reductions = (
            cfg.dp_size == 1 and cfg.ep_size == 1 and cfg.tp_size == cfg.moe_tp_size and cfg.tp_size > 1
        )
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

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        copy_parameter(
            self.experts.gate.weight,
            state.tensor("gate.weight"),
            state.prefix + "gate.weight",
        )
        copy_parameter(
            self.shared_expert_gate.weight,
            state.tensor("shared_expert_gate.weight"),
            state.prefix + "shared_expert_gate.weight",
        )

        gate_up = state.tensor("experts.gate_up_proj")
        local_experts = gate_up.size(0) // context.ep_size
        start_expert = context.ep_rank * local_experts
        gate_up = gate_up.narrow(0, start_expert, local_experts)
        gate, up = gate_up.chunk(2, dim=1)
        gate = gate.chunk(context.moe_tp_size, dim=1)[context.moe_tp_rank]
        up = up.chunk(context.moe_tp_size, dim=1)[context.moe_tp_rank]
        copy_parameter(
            self.experts.w13,
            torch.cat((up, gate), dim=1),
            state.prefix + "experts.gate_up_proj",
        )

        down = state.tensor("experts.down_proj").narrow(
            0,
            start_expert,
            local_experts,
        )
        copy_parameter(
            self.experts.w2,
            down.chunk(context.moe_tp_size, dim=2)[context.moe_tp_rank],
            state.prefix + "experts.down_proj",
        )
        self.shared_expert.load_weights(
            state.with_prefix("shared_expert."),
            context,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        routed = self.experts(hidden)
        shared = self.shared_expert(hidden)
        output = routed + shared * torch.sigmoid(self.shared_expert_gate(hidden))
        if self.fuse_reductions:
            distributed.tp_all_reduce(output)
        return output

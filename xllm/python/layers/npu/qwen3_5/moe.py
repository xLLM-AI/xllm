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

"""NPU-owned Qwen3.5 sparse MoE composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.moe_dp import dp_gather_tokens, reduce_and_scatter
from xllm.python.layers.qwen3_5_common import Qwen3_5MoEConfig
from xllm.python.layers.qwen3_5_moe import Qwen3_5SparseMoEBlockBase


class _NpuQwen3_5Experts(nn.Module):
    """BF16 routed experts in NPU grouped-matmul layout."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool,
    ) -> None:
        super().__init__()
        if dtype != torch.bfloat16:
            raise NotImplementedError("NPU Qwen3.5 routed experts currently support BF16 only")
        local_experts = cfg.num_experts // cfg.ep_size
        local_intermediate = cfg.moe_intermediate_size // cfg.moe_tp_size
        self.top_k = cfg.num_experts_per_tok
        self.renormalize = cfg.norm_topk_prob
        self.num_experts = cfg.num_experts
        self.local_experts = local_experts
        self.start_expert = cfg.ep_rank * local_experts
        self.moe_tp_size = cfg.moe_tp_size
        self.ep_size = cfg.ep_size
        self.dp_size = cfg.dp_size
        self.dp_rank = cfg.dp_rank
        self.reduce_results = reduce_results

        self.gate = nn.Linear(
            cfg.hidden_size,
            cfg.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                local_experts,
                cfg.hidden_size,
                2 * local_intermediate,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                local_experts,
                local_intermediate,
                cfg.hidden_size,
                dtype=dtype,
                device=device,
            )
        )

    def _route(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kernels.moe_fused_topk(
            router_logits,
            self.top_k,
            self.renormalize,
            "softmax",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gathered_states, scatter_state = dp_gather_tokens(hidden_states, self.dp_size, self.dp_rank)
        topk_weights, topk_ids = self._route(self.gate(gathered_states))
        output = kernels.grouped_moe_bf16(
            gathered_states,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            self.num_experts,
            self.start_expert,
            self.local_experts,
        )
        return reduce_and_scatter(
            output,
            scatter_state,
            reduce_results=self.reduce_results,
            moe_tp_size=self.moe_tp_size,
            ep_size=self.ep_size,
        )


class NpuQwen3_5SparseMoEBlock(Qwen3_5SparseMoEBlockBase):
    """NPU Qwen3.5 routed and shared experts with topology-safe reductions."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, dtype, device)
        self.experts = _NpuQwen3_5Experts(
            cfg,
            dtype,
            device,
            reduce_results=not self.fuse_reductions,
        )

    def _pack_gate_up(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((gate, up), dim=1).transpose(1, 2).contiguous()

    def _pack_down(self, down: torch.Tensor) -> torch.Tensor:
        return down.transpose(1, 2).contiguous()

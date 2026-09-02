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

from xllm.python import distributed, kernels
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.qwen3_5_decoder_layer import Qwen3_5LayerConfig
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)


class _NpuQwen3_5Experts(nn.Module):
    """BF16 routed experts in NPU grouped-matmul layout."""

    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
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

    def _gather_dp_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int], int, int, bool]:
        if self.dp_size == 1:
            return hidden_states, [], 0, 0, False

        context = get_forward_context()
        token_counts = list(context.metadata.dp_token_counts)
        if len(token_counts) != self.dp_size:
            raise RuntimeError(f"expected {self.dp_size} DP token counts, got {token_counts}")
        local_tokens = hidden_states.shape[0]
        is_graph = context.execution_state is not None
        is_prefill = context.metadata.is_prefill or context.metadata.is_chunked_prefill
        dp_is_decode = getattr(context.metadata, "dp_is_decode", None)
        all_decode = dp_is_decode is not None and all(dp_is_decode)
        if not is_graph and not is_prefill and all_decode:
            gathered = distributed.all_gather_variable(
                hidden_states,
                token_counts,
                self.dp_rank,
                "dp",
            )
            return gathered, token_counts, local_tokens, 0, True

        padded_tokens = max(token_counts)
        pad_size = padded_tokens - local_tokens
        if pad_size > 0:
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, pad_size),
            )
        gathered = distributed.all_gather(
            hidden_states,
            dim=0,
            world_size=self.dp_size,
            group_name="dp",
        )
        return gathered, token_counts, local_tokens, padded_tokens, False

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
        (
            gathered_states,
            token_counts,
            local_tokens,
            padded_tokens,
            compact_gather,
        ) = self._gather_dp_inputs(hidden_states)
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
        if self.reduce_results:
            if self.moe_tp_size > 1:
                distributed.moe_tp_all_reduce(output)
            if self.ep_size > 1:
                distributed.moe_ep_all_reduce(output)

        if compact_gather:
            offset = sum(token_counts[: self.dp_rank])
            return output.narrow(0, offset, local_tokens)
        if padded_tokens > 0:
            start = self.dp_rank * padded_tokens
            return output.narrow(0, start, local_tokens)
        return output


class NpuQwen3_5SparseMoEBlock(nn.Module):
    """NPU Qwen3.5 routed and shared experts with topology-safe reductions."""

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
        self.experts = _NpuQwen3_5Experts(
            cfg,
            dtype,
            device,
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
            torch.cat((gate, up), dim=1).transpose(1, 2).contiguous(),
            state.prefix + "experts.gate_up_proj",
        )

        down = state.tensor("experts.down_proj").narrow(
            0,
            start_expert,
            local_experts,
        )
        copy_parameter(
            self.experts.w2,
            down.chunk(context.moe_tp_size, dim=2)[context.moe_tp_rank].transpose(1, 2).contiguous(),
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

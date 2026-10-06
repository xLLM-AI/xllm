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
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.layers.moe_dp import dp_gather_tokens, reduce_and_scatter
from xllm.python.layers.npu.mega_moe_metadata import MegaMoeMetadata
from xllm.python.layers.qwen3_5.common import Qwen3_5MoEConfig
from xllm.python.layers.qwen3_5.moe import Qwen3_5SparseMoEBlockBase
from xllm.python.model_executor.forward_context import get_execution_context


def _pad_token_rows(tensor: torch.Tensor, token_capacity: int) -> torch.Tensor:
    token_count = tensor.shape[0]
    if token_count > token_capacity:
        raise RuntimeError(f"MegaMoe input has {token_count} rows, exceeding token capacity {token_capacity}")
    if token_count == token_capacity:
        return tensor.contiguous()
    return F.pad(tensor, (0, 0, 0, token_capacity - token_count))


class _NpuQwen3_5Experts(nn.Module):
    """BF16 routed experts in NPU grouped-matmul layout."""

    _mega_moe_context: torch.Tensor | None

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
        self.tp_rank = cfg.tp_rank
        self.reduce_results = reduce_results
        self._mega_moe_ccl_buffer_size = cfg.mega_moe_ccl_buffer_size
        self._mega_moe_num_max_tokens_per_rank = cfg.mega_moe_num_max_tokens_per_rank
        self._enable_mega_moe = cfg.enable_mega_moe
        self.register_buffer(
            "_mega_moe_context",
            cfg.mega_moe_context,
            persistent=False,
        )
        if self._enable_mega_moe and cfg.mega_moe_context is None:
            raise ValueError("MegaMoe token ownership requires a communication context")

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

    def _forward_ep_level1(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._enable_mega_moe:
            return self._forward_ep_level1(hidden_states)
        metadata = get_execution_context(MegaMoeMetadata)
        if metadata is None:
            raise RuntimeError("MegaMoe execution metadata is unavailable")
        active_token_mask = metadata.active_token_mask
        if active_token_mask is None:
            return self._forward_ep_level1(hidden_states)
        token_capacity = active_token_mask.shape[0]
        if self.tp_rank == 0:
            topk_weights, topk_ids = self._route(self.gate(hidden_states))
            owner_input = _pad_token_rows(hidden_states, token_capacity)
            topk_weights = _pad_token_rows(topk_weights.to(torch.float32), token_capacity)
            topk_ids = _pad_token_rows(topk_ids.to(torch.int32), token_capacity)
        else:
            owner_input = metadata.dummy_input
            topk_weights = metadata.dummy_topk_weights
            topk_ids = metadata.dummy_topk_ids
            if owner_input is None or topk_weights is None or topk_ids is None:
                raise RuntimeError("MegaMoe non-owner dummy inputs are unavailable")

        context = self._mega_moe_context
        if context is None:
            raise RuntimeError("MegaMoe communication context is unavailable")
        owner_output = kernels.mega_moe(
            context,
            owner_input,
            topk_ids,
            topk_weights,
            self.w13,
            self.w2,
            None,
            None,
            self.num_experts,
            self.ep_size,
            self._mega_moe_ccl_buffer_size,
            self._mega_moe_num_max_tokens_per_rank,
            active_token_mask,
        )

        if self.tp_rank == 0:
            output = owner_output[: hidden_states.shape[0]].contiguous()
        else:
            output = torch.empty_like(hidden_states)
        distributed.broadcast_(output, 0, "tp")
        return output


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

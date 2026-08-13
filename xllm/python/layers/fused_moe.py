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

"""Fused MoE layer shared by Python model implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed, kernels
from xllm.python.model_executor.forward_context import get_forward_context


class FusedMoE(nn.Module):
    """Routed experts backed by the active platform's MoE kernels.

    This mirrors ``layers/cuda/FusedMoEImpl``: the router is replicated, the
    expert intermediate dimension is tensor-parallel, and the CUTLASS fc1
    weight layout is ``[up, gate]``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        renormalize: bool,
        moe_tp_size: int,
        moe_tp_rank: int,
        ep_size: int,
        ep_rank: int,
        dp_size: int,
        dp_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        scoring_func: str = "softmax",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if intermediate_size % moe_tp_size:
            raise ValueError("MoE intermediate_size must be divisible by moe_tp_size")
        if num_experts % ep_size:
            raise ValueError("num_experts must be divisible by ep_size")
        if not reduce_results and dp_size > 1:
            # The DP path slices the gathered output down to this rank's tokens,
            # which is only meaningful once every rank's partial sums have been
            # combined, so the reduction cannot be deferred past this layer.
            raise ValueError("a deferred reduction cannot be combined with data parallelism")

        num_experts_per_rank = num_experts // ep_size
        local_intermediate_size = intermediate_size // moe_tp_size
        self.top_k = top_k
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.moe_tp_size = moe_tp_size
        self.moe_tp_rank = moe_tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.reduce_results = reduce_results
        # Top-k routing runs on every architecture; only the native expert GEMMs
        # need Hopper/Blackwell, so the kernel package decides per device.
        self._use_cutlass = kernels.supports_cutlass_moe(device)

        self.gate = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                num_experts_per_rank,
                2 * local_intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts_per_rank,
                hidden_size,
                local_intermediate_size,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_hidden_states = hidden_states
        token_counts: list[int] | None = None
        if self.dp_size > 1:
            token_counts = list(get_forward_context().metadata.dp_token_counts)
            if len(token_counts) != self.dp_size:
                raise RuntimeError(f"expected {self.dp_size} DP token counts, got {token_counts}")
            hidden_states = distributed.all_gather_variable(
                hidden_states,
                token_counts,
                self.dp_rank,
                "dp",
            )

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = kernels.moe_fused_topk(
            router_logits,
            self.top_k,
            self.renormalize,
            self.scoring_func,
        )
        if self._use_cutlass:
            output = kernels.cutlass_fused_moe(
                hidden_states,
                topk_ids,
                topk_weights,
                self.w13,
                self.w2,
                self.moe_tp_size,
                self.moe_tp_rank,
                self.ep_size,
                self.ep_rank,
            )
        elif self.moe_tp_size == 1 and self.ep_size == 1:
            output = kernels.fused_moe(
                hidden_states,
                topk_ids,
                topk_weights,
                self.w13,
                self.w2,
            )
        else:
            raise NotImplementedError("pre-SM90 Python MoE fallback does not support TP or EP")
        if self.reduce_results:
            if self.moe_tp_size > 1:
                distributed.all_reduce_(output, "moe_tp")
            if self.ep_size > 1:
                distributed.all_reduce_(output, "moe_ep")
        if token_counts is not None:
            local_tokens = token_counts[self.dp_rank]
            if local_tokens == 0:
                return torch.zeros_like(local_hidden_states)
            start = sum(token_counts[: self.dp_rank])
            local_output = output.narrow(0, start, local_tokens)
            if local_tokens == local_hidden_states.shape[0]:
                return local_output
            padding_shape = list(local_hidden_states.shape)
            padding_shape[0] -= local_tokens
            return torch.cat([local_output, local_hidden_states.new_zeros(padding_shape)], dim=0)
        return output

# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU implementations for grouped routed experts."""

from __future__ import annotations

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29


def prepare_grouped_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch_npu.npu_format_cast(w13, _FRACTAL_NZ_FORMAT),
        torch_npu.npu_format_cast(w2, _FRACTAL_NZ_FORMAT),
    )


def grouped_moe(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    topk_group: int,
    num_expert_groups: int,
    renormalize: bool,
) -> torch.Tensor:
    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
        gating_output,
        k=topk,
        bias=correction_bias,
        k_group=topk_group,
        group_count=num_expert_groups,
        group_select_mode=1,
        renorm=1 if renormalize else 0,
        norm_type=1,
        routed_scaling_factor=1.0,
        eps=1e-20,
    )
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    sorted_hidden_i8, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids.to(torch.int32),
            scale=None,
            active_num=num_tokens * topk,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=1,
        )
    )
    group_list = torch.cumsum(expert_tokens.to(torch.int64), 0)
    act_i8, act_pt, _ = torch.ops.npu.npu_grouped_matmul_swiglu_quant(
        x=sorted_hidden_i8,
        weight=w13,
        group_list=group_list,
        weight_scale=w13_scale,
        x_scale=pertoken_scale,
    )
    output = torch.ops.npu.npu_grouped_matmul(
        x=[act_i8],
        weight=[w2],
        scale=[w2_scale.to(torch.bfloat16)],
        per_token_scale=[act_pt],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=expanded_row_idx.abs(),
        probs=topk_weights.to(output.dtype),
    )

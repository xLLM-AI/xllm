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

"""NPU mixture-of-experts kernels."""

from __future__ import annotations

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29


def _grouped_matmul_swiglu_quant_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the W8A8 GMM v2 path with vllm-ascend-compatible arguments."""
    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=x,
        weight=[weight],
        weight_scale=[weight_scale],
        x_scale=x_scale,
        group_list=group_list,
        dequant_mode=0,
        dequant_dtype=0,
        quant_mode=0,
        quant_dtype=0,
        group_list_type=0,
    )


def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor | None,
    activation_scale: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_offset: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    activate_left: bool = True,
    quant_mode: int = 1,
    swiglu_mode: int = 1,
    clamp_limit: float = 0.0,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fused dequantization, SwiGLU, and dynamic quantization."""
    return torch.ops.xllm_ops.dequant_swiglu_quant(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        activate_left,
        quant_mode,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
    )


def moe_gating_top_k_hash(
    x: torch.Tensor,
    k: int,
    bias: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select experts with the DeepSeek-V4 hash-routing gate."""
    return torch.ops.xllm_ops.moe_gating_top_k_hash(
        x,
        k,
        bias,
        input_ids,
        tid2eid,
        k_group,
        group_count,
        routed_scaling_factor,
        eps,
        group_select_mode,
        renorm,
        norm_type,
        out_flag,
    )


def supports_cutlass_moe(device: torch.device) -> bool:
    """Return whether ``device`` has the native expert GEMMs.

    Args:
        device: Device the MoE layer will run on.

    Returns:
        Always ``False``; NPU routes grouped experts through
        :func:`grouped_moe` instead.
    """
    del device
    return False


def prepare_grouped_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out grouped expert weights for the grouped-matmul kernels.

    Args:
        w13: Gate and up projections of every expert.
        w2: Down projection of every expert.

    Returns:
        The two weights in the fractal-NZ format the grouped kernels expect.
    """
    return (
        torch_npu.npu_format_cast(w13, _FRACTAL_NZ_FORMAT),
        torch_npu.npu_format_cast(w2, _FRACTAL_NZ_FORMAT),
    )


def format_cast_nz(weight: torch.Tensor) -> torch.Tensor:
    """Cast a single weight tensor to fractal-NZ format."""
    return torch_npu.npu_format_cast(weight, _FRACTAL_NZ_FORMAT)


@torch.library.custom_op("xllm_python::grouped_moe", mutates_args=())
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
    routed_scaling_factor: float,
    active_expert_range: list[int] | None = None,
) -> torch.Tensor:
    """Route and run grouped quantized experts as one fused operator.

    Args:
        hidden_states: Hidden states of shape ``[num_tokens, hidden_size]``.
        gating_output: Router logits of shape ``[num_tokens, num_experts]``.
        w13: Quantized gate and up projections of every expert.
        w2: Quantized down projection of every expert.
        w13_scale: Dequantization scales of ``w13``.
        w2_scale: Dequantization scales of ``w2``.
        correction_bias: Router bias added before group selection.
        topk: Experts selected per token.
        topk_group: Groups selected per token.
        num_expert_groups: Expert groups the router splits experts into.
        renormalize: Whether to rescale the selected weights to sum to one.
        routed_scaling_factor: Model-specific scale applied to selected routing
            weights before expert computation.
        active_expert_range: ``[start, end)`` of global expert indices handled
            by this rank.  Defaults to ``[0, num_experts]`` (all experts).

    Returns:
        Hidden states of shape ``[num_tokens, hidden_size]``.
    """
    if correction_bias is not None and correction_bias.dtype != gating_output.dtype:
        correction_bias = correction_bias.to(gating_output.dtype)
    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
        gating_output,
        k=topk,
        bias=correction_bias,
        k_group=topk_group,
        group_count=num_expert_groups,
        group_select_mode=1,
        renorm=1 if renormalize else 0,
        norm_type=1,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
    )
    num_tokens = hidden_states.shape[0]
    num_experts = gating_output.shape[1]
    expert_range = active_expert_range if active_expert_range is not None else [0, num_experts]
    sorted_hidden_i8, expanded_row_idx, group_list, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        scale=None,
        active_num=num_tokens * topk,
        expert_num=num_experts,
        # GMM v2 consumes cumulative expert-token offsets.
        expert_tokens_num_type=0,
        expert_tokens_num_flag=True,
        active_expert_range=expert_range,
        quant_mode=1,
    )
    num_local_experts = expert_range[1] - expert_range[0]
    if group_list.numel() > num_local_experts:
        group_list = group_list[:num_local_experts]
    act_i8, act_pt = _grouped_matmul_swiglu_quant_v2(
        sorted_hidden_i8,
        w13,
        w13_scale,
        pertoken_scale,
        group_list,
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
    if expert_range[0] != 0 or expert_range[1] != num_experts:
        local_mask = (topk_ids >= expert_range[0]) & (topk_ids < expert_range[1])
        topk_weights = topk_weights * local_mask
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=expanded_row_idx.abs(),
        probs=topk_weights.to(output.dtype),
    )


def _group_gemm(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    per_token_scale: torch.Tensor | None,
    group_list: torch.Tensor,
    split_item: int,
    group_type: int,
    group_list_type: int,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    outputs = torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=None if scale is None else [scale],
        per_token_scale=None if per_token_scale is None else [per_token_scale],
        group_list=group_list,
        split_item=split_item,
        group_type=group_type,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
    )
    return outputs[0]


@torch.library.custom_op("xllm_python::grouped_moe_bf16", mutates_args=())
def grouped_moe_bf16(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    num_total_experts: int,
    start_expert_id: int,
    num_experts_per_rank: int,
) -> torch.Tensor:
    """Run Qwen3.5 BF16 experts with NPU routing and grouped matmuls."""
    active_expert_range = [
        start_expert_id,
        start_expert_id + num_experts_per_rank,
    ]
    expanded_hidden, expanded_row_idx, group_list, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        scale=None,
        active_num=hidden_states.shape[0] * topk_ids.shape[1],
        expert_num=num_total_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=active_expert_range,
        quant_mode=-1,
    )
    group_list = group_list[:num_experts_per_rank].to(torch.int64)
    gate_up = _group_gemm(
        x=expanded_hidden,
        weight=w13,
        scale=None,
        per_token_scale=None,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=hidden_states.dtype,
    )
    from xllm.python import kernels as _kernels

    activated = _kernels.silu_and_mul(gate_up)
    expert_output = _group_gemm(
        x=activated,
        weight=w2,
        scale=None,
        per_token_scale=None,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=hidden_states.dtype,
    )
    local_expert_mask = (topk_ids >= active_expert_range[0]) & (topk_ids < active_expert_range[1])
    local_topk_weights = topk_weights * local_expert_mask.to(topk_weights.dtype)
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=expert_output,
        sorted_indices=expanded_row_idx.abs(),
        probs=local_topk_weights.to(expert_output.dtype),
    )


@grouped_moe_bf16.register_fake
def _grouped_moe_bf16_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    num_total_experts: int,
    start_expert_id: int,
    num_experts_per_rank: int,
) -> torch.Tensor:
    del (
        topk_weights,
        topk_ids,
        w13,
        w2,
        num_total_experts,
        start_expert_id,
        num_experts_per_rank,
    )
    return torch.empty_like(hidden_states)


def _grouped_moe_with_selected_experts_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    num_total_experts: int = -1,
    start_expert_id: int = 0,
    num_experts_per_rank: int = -1,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Run grouped quantized experts with pre-computed routing (no gate).

    The routing and W8A8 grouped-matmul sequence mirrors the native NPU
    ``FusedMoEImpl::select_experts`` and ``forward_expert`` paths.
    """
    num_tokens = hidden_states.shape[0]
    expert_num = num_total_experts if num_total_experts > 0 else w13.shape[0]
    local_expert_count = num_experts_per_rank if num_experts_per_rank > 0 else w13.shape[0]
    active_range = [start_expert_id, start_expert_id + local_expert_count]
    if start_expert_id < 0 or active_range[1] > expert_num:
        raise ValueError(f"active expert range {active_range} is outside [0, {expert_num})")
    if w13.shape[0] != local_expert_count or w2.shape[0] != local_expert_count:
        raise ValueError("local expert count must match the first dimension of w13 and w2")
    expanded_hidden, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        scale=None,
        active_num=num_tokens * topk_ids.size(-1),
        expert_num=expert_num,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=active_range,
        quant_mode=-1,
    )
    from xllm.python import kernels as _kernels

    sorted_hidden_i8, pertoken_scale = _kernels.dynamic_quant(expanded_hidden)
    if pertoken_scale is None:
        raise RuntimeError("dynamic_quant did not return a per-token scale")
    if expert_tokens.numel() < local_expert_count:
        raise RuntimeError("npu_moe_init_routing_v2 returned fewer groups than local experts")
    group_list = expert_tokens[:local_expert_count].to(torch.int64)
    gemm1_out = _group_gemm(
        x=sorted_hidden_i8,
        weight=w13,
        scale=None,
        per_token_scale=None,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.int32,
    )
    act_i8, act_pt = _kernels.dequant_swiglu_quant(
        x=gemm1_out,
        weight_scale=w13_scale,
        activation_scale=pertoken_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
        swiglu_mode=1,
        clamp_limit=swiglu_limit,
        glu_alpha=1.0,
        glu_bias=0.0,
    )
    del w13_offset, w2_offset
    output = _group_gemm(
        x=act_i8,
        weight=w2,
        scale=w2_scale.to(hidden_states.dtype),
        per_token_scale=act_pt,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=hidden_states.dtype,
    )
    local_mask = (topk_ids >= active_range[0]) & (topk_ids < active_range[1])
    local_topk_weights = topk_weights * local_mask.to(topk_weights.dtype)
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=expanded_row_idx.abs(),
        probs=local_topk_weights.to(output.dtype),
    )


@torch.library.custom_op("xllm_python::grouped_moe_with_selected_experts", mutates_args=())
def grouped_moe_with_selected_experts(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    num_total_experts: int = -1,
    start_expert_id: int = 0,
    num_experts_per_rank: int = -1,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    return _grouped_moe_with_selected_experts_impl(
        hidden_states,
        topk_weights,
        topk_ids,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_offset,
        w2_offset,
        num_total_experts,
        start_expert_id,
        num_experts_per_rank,
        swiglu_limit,
    )


@grouped_moe.register_fake
def _grouped_moe_fake(
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
    routed_scaling_factor: float,
    active_expert_range: list[int] | None = None,
) -> torch.Tensor:
    del (
        gating_output,
        w13,
        w2,
        w13_scale,
        w2_scale,
        correction_bias,
        topk,
        topk_group,
        num_expert_groups,
        renormalize,
        routed_scaling_factor,
        active_expert_range,
    )
    return torch.empty_like(hidden_states)


@grouped_moe_with_selected_experts.register_fake
def _grouped_moe_with_selected_experts_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    num_total_experts: int = -1,
    start_expert_id: int = 0,
    num_experts_per_rank: int = -1,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    del topk_weights, topk_ids, w13, w2, w13_scale, w2_scale, w13_offset, w2_offset
    del num_total_experts, start_expert_id, num_experts_per_rank, swiglu_limit
    return torch.empty_like(hidden_states)


def moe_fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the routed experts of every token.

    Args:
        gating_output: Router logits of shape ``[num_tokens, num_experts]``.
        topk: Experts selected per token.
        renormalize: Whether to rescale the selected weights to sum to one.
        scoring_func: Router scoring function, ``"softmax"`` or ``"sigmoid"``.

    Returns:
        Routing weights and expert indices, both ``[num_tokens, topk]``.
    """
    if scoring_func != "softmax":
        raise NotImplementedError("NPU moe_fused_topk currently supports softmax routing only")
    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax_v2(
        gating_output,
        k=topk,
        finished=None,
        renorm=1 if renormalize else 0,
        output_softmax=False,
    )
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """Run the routed experts through the CUTLASS grouped GEMMs.

    Args:
        input: Hidden states of shape ``[num_tokens, hidden_size]``.
        token_selected_experts: Expert index per token and slot.
        token_final_scales: Routing weight per token and slot.
        fc1_expert_weights: Gate and up projections of every expert.
        fc2_expert_weights: Down projection of every expert.
        tp_size: Tensor-parallel world size.
        tp_rank: Tensor-parallel rank.
        ep_size: Expert-parallel world size.
        ep_rank: Expert-parallel rank.

    Returns:
        Hidden states of shape ``[num_tokens, hidden_size]``.
    """
    del (
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
    )
    raise NotImplementedError("cutlass_fused_moe is a CUDA library kernel; the NPU equivalent is grouped_moe")


def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Run unquantized experts over pre-computed routing.

    Args:
        hidden_states: Hidden states of shape ``[num_tokens, hidden_size]``.
        topk_ids: Expert index per token and slot.
        topk_weights: Routing weight per token and slot.
        w13: Gate and up projections of every expert.
        w2: Down projection of every expert.

    Returns:
        Hidden states of shape ``[num_tokens, hidden_size]``.
    """
    del hidden_states, topk_ids, topk_weights, w13, w2
    raise NotImplementedError(
        "fused_moe has no NPU kernel; see kernels_cuda/triton/fused_moe.py for the reference implementation"
    )


@torch.library.custom_op("xllm_python::moe_gate_routing", mutates_args=())
def moe_gate_routing(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    topk_group: int,
    num_expert_groups: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate routing: select top-k experts per token from logits."""
    if correction_bias is not None and correction_bias.dtype != gating_output.dtype:
        correction_bias = correction_bias.to(gating_output.dtype)
    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
        gating_output,
        k=topk,
        bias=correction_bias,
        k_group=topk_group,
        group_count=num_expert_groups,
        group_select_mode=1,
        renorm=1 if renormalize else 0,
        norm_type=1,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
    )
    return topk_weights, topk_ids


@moe_gate_routing.register_fake
def _moe_gate_routing_fake(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    topk_group: int,
    num_expert_groups: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=gating_output.dtype, device=gating_output.device)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=gating_output.device)
    return topk_weights, topk_ids


@torch.library.custom_op("xllm_python::moe_expert_compute", mutates_args=())
def moe_expert_compute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Expert dispatch + grouped matmul + combine (gate-free)."""
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    sorted_hidden_i8, expanded_row_idx, group_list, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        scale=None,
        active_num=num_tokens * topk,
        expert_num=num_experts,
        expert_tokens_num_type=0,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=1,
    )
    act_i8, act_pt = _grouped_matmul_swiglu_quant_v2(
        sorted_hidden_i8,
        w13,
        w13_scale,
        pertoken_scale,
        group_list,
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


@moe_expert_compute.register_fake
def _moe_expert_compute_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@torch.library.custom_op("xllm_python::moe_token_dispatch", mutates_args=())
def moe_token_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    return torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids.to(torch.int32),
        scale=None,
        active_num=num_tokens * topk,
        expert_num=num_experts,
        expert_tokens_num_type=0,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=1,
    )


@moe_token_dispatch.register_fake
def _moe_token_dispatch_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_num = hidden_states.shape[0] * topk
    sorted_hidden_i8 = hidden_states.new_empty((active_num, hidden_states.shape[1]), dtype=torch.int8)
    expanded_row_idx = topk_ids.new_empty((active_num,), dtype=torch.int32)
    group_list = topk_ids.new_empty((num_experts,), dtype=torch.int64)
    pertoken_scale = hidden_states.new_empty((active_num, 1), dtype=torch.float32)
    return sorted_hidden_i8, expanded_row_idx, group_list, pertoken_scale


@torch.library.custom_op("xllm_python::moe_gmm1", mutates_args=())
def moe_gmm1(
    sorted_hidden_i8: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _grouped_matmul_swiglu_quant_v2(sorted_hidden_i8, w13, w13_scale, pertoken_scale, group_list)


@moe_gmm1.register_fake
def _moe_gmm1_fake(
    sorted_hidden_i8: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_num = sorted_hidden_i8.shape[0]
    inter_size = w13.shape[1] // 2
    act_i8 = sorted_hidden_i8.new_empty((active_num, inter_size), dtype=torch.int8)
    act_pt = pertoken_scale.new_empty((active_num, 1), dtype=torch.float32)
    return act_i8, act_pt


@torch.library.custom_op("xllm_python::moe_gmm2_combine", mutates_args=())
def moe_gmm2_combine(
    act_i8: torch.Tensor,
    act_pertoken_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.ops.npu.npu_grouped_matmul(
        x=[act_i8],
        weight=[w2],
        scale=[w2_scale.to(torch.bfloat16)],
        per_token_scale=[act_pertoken_scale],
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


@moe_gmm2_combine.register_fake
def _moe_gmm2_combine_fake(
    act_i8: torch.Tensor,
    act_pertoken_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return topk_weights.new_empty((topk_weights.shape[0], w2.shape[-1]), dtype=torch.bfloat16)


__all__ = [
    "dequant_swiglu_quant",
    "moe_gating_top_k_hash",
    "supports_cutlass_moe",
    "prepare_grouped_moe_weights",
    "grouped_moe",
    "moe_gate_routing",
    "moe_expert_compute",
    "grouped_moe_with_selected_experts",
    "moe_fused_topk",
    "cutlass_fused_moe",
    "fused_moe",
    "moe_token_dispatch",
    "moe_gmm1",
    "moe_gmm2_combine",
]

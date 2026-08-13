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

import os

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29


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
        routed_scaling_factor=1.0,
        eps=1e-20,
    )
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    expanded_hidden, expanded_row_idx, expert_tokens, _ = (
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
    group_list = expert_tokens.to(torch.int64)
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
    """Run grouped quantized experts with pre-computed routing (no gate).

    EP-aware: passes ``active_expert_range=[start_expert_id, start_expert_id +
    num_experts_per_rank)`` and ``expert_num=num_total_experts`` to
    ``npu_moe_init_routing_v2``, so only local experts' tokens are routed.
    Mirrors C++ FusedMoEImpl::select_experts (fused_moe.cpp:836-885).
    """
    num_tokens = hidden_states.shape[0]
    # EP: expert_num = total (global), active_expert_range = [start, start+local).
    expert_num = num_total_experts if num_total_experts > 0 else w13.shape[0]
    active_range = [start_expert_id, start_expert_id + num_experts_per_rank] if num_experts_per_rank > 0 else [0, expert_num]
    runtime_debug = os.getenv("XLLM_DSV4_RUNTIME_DEBUG", "0") == "1"
    if runtime_debug:
        import sys
        print(f"[MoE PRE] hidden={hidden_states.shape} topk_w={topk_weights.shape} "
              f"topk_ids={topk_ids.shape} expert_num={expert_num} "
              f"active_range={active_range} topk_ids_vals={topk_ids[0].tolist() if topk_ids.numel()>0 else 'empty'}",
              file=sys.stderr, flush=True)
    expanded_hidden, expanded_row_idx, expert_tokens, _ = (
        torch_npu.npu_moe_init_routing_v2(
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
    )
    # Match FusedMoEImpl::select_experts + forward_expert: routing expands
    # BF16 rows first, then xLLM dynamic_quant quantizes them as a separate
    # operation. npu_moe_init_routing_v2's fused quant_mode=1 has different
    # rounding and does not match the native eager path.
    from xllm.python import kernels as _kernels
    sorted_hidden_i8, pertoken_scale = _kernels.dynamic_quant(expanded_hidden)
    assert pertoken_scale is not None
    if runtime_debug:
        print(f"[MoE POST] expanded={expanded_hidden.shape} sorted_i8={sorted_hidden_i8.shape} row_idx={expanded_row_idx.shape} "
              f"et={expert_tokens.shape} pts={pertoken_scale.shape}",
              file=sys.stderr, flush=True)
    group_list = expert_tokens.to(torch.int64)
    # Route GEMM1/GEMM2 through the C++ xllm_ops.group_gemm wrapper, which
    # forwards to apply_npu_grouped_matmul with the same argument contract as
    # the native C++ FusedMoEImpl W8A8 path (fused_moe.cpp:1028-1073). This is
    # NOT torch.ops.npu.npu_grouped_matmul: that path coerces an empty
    # per_token_scale list into a 2-dim placeholder tensor and trips op-plugin
    # error 161002 ("PerTokenScaleOptional dim num must be 1 ... now is 2").
    # The C++ wrapper converts an omitted per_token_scale into a genuinely
    # empty TensorList (zero tensors), which is the contract the op-plugin
    # PerTokenScaleOptional check expects.
    #
    # npu_grouped_matmul computes x @ weight (not weight.T), so weight is
    # [expert, in, out] (transposed from checkpoint [expert, out, in] in
    # DeepseekV4MoE.process_weights_after_loading). Weights stay ND (no NZ) —
    # matching C++ ensure_group_gemm_weight_layout, which only transposes and
    # lets op-plugin handle format internally.
    if runtime_debug:
        print(f"[GEMM1 DEBUG] x={sorted_hidden_i8.shape} w13={w13.shape} "
              f"w13_scale={w13_scale.shape} pts={pertoken_scale.shape} "
              f"gl={group_list.shape} et={expert_tokens.shape} "
              f"gl_vals={expert_tokens.tolist()[:5]} et_sum={expert_tokens.sum().item()} "
              f"w13_fmt={torch_npu.get_npu_format(w13)} "
              f"x_fmt={torch_npu.get_npu_format(sorted_hidden_i8)} "
              f"x_dtype={sorted_hidden_i8.dtype} w_dtype={w13.dtype} "
              f"gl_dtype={group_list.dtype}",
              file=sys.stderr, flush=True)
    # Step 1: W13 GEMM (int8 x int8 -> int32). C++ omits scale_list and
    # per_token_scale_list (fused_moe.cpp:1030-1037); dequant happens in the
    # separate dequant_swiglu_quant step below. Passing None keeps them
    # nullopt so apply_npu_grouped_matmul forwards empty TensorLists.
    gemm1_out = torch.ops.xllm_ops.group_gemm(
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
    if runtime_debug:
        print(f"[GEMM1 DONE] gemm1_out={gemm1_out.shape} dtype={gemm1_out.dtype}",
              file=sys.stderr, flush=True)
    # Step 2: Fused dequant + SwiGLU + dynamic quant (matching C++ dequant_swiglu_quant).
    act_i8, act_pt = _kernels.dequant_swiglu_quant(
        x=gemm1_out,
        weight_scale=w13_scale,
        activation_scale=pertoken_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list.to(torch.int64),
        activate_left=True,
        quant_mode=1,
        swiglu_mode=1,
        clamp_limit=swiglu_limit,
        glu_alpha=1.0,
        glu_bias=0.0,
    )
    if runtime_debug:
        print(f"[DEQUANT DONE] act_i8={act_i8.shape} dtype={act_i8.dtype} "
              f"act_pt={act_pt.shape} dtype={act_pt.dtype} w2={w2.shape} "
              f"w2_scale={w2_scale.shape} dtype={w2_scale.dtype}",
              file=sys.stderr, flush=True)
    # Dump physical layout of act_i8 — dequant_swiglu_quant may return a
    # [M, K] view over [M, 2K] storage, which the grouped-matmul kernel reads
    # by storage and trips "x1 dim[-1] must match x2 dim[-2], got 2K vs K".
    if runtime_debug:
        print(f"[ACT LAYOUT] act_i8 shape={act_i8.shape} stride={act_i8.stride()} "
              f"contig={act_i8.is_contiguous()} fmt={torch_npu.get_npu_format(act_i8)} "
              f"storage_bytes={act_i8.storage().nbytes() if act_i8.storage().data_ptr() else 0} "
              f"gemm1_out_shape={gemm1_out.shape}",
              file=sys.stderr, flush=True)
    # GEMM2 contract sanity (probe-verified): x=[M,K], w2=[G,K,N],
    # scale=[G,N] bf16, per_token_scale=[M] f32. K must match w2 dim[-2].
    assert act_i8.dim() == 2 and act_i8.size(1) == w2.size(1), (
        f"GEMM2 K mismatch: act_i8={act_i8.shape} w2={w2.shape}")
    assert w2_scale.dim() == 2 and w2_scale.size(1) == w2.size(2), (
        f"GEMM2 scale N mismatch: w2_scale={w2_scale.shape} w2={w2.shape}")
    w2_scale_bf16 = w2_scale.to(torch.bfloat16)
    if runtime_debug:
        print(f"[GEMM2 CALL] x={act_i8.shape} w2={w2.shape} "
              f"scale={w2_scale_bf16.shape} pts={act_pt.shape} "
              f"gl={group_list.shape} gl_sum={group_list.sum().item()} "
              f"w2_fmt={torch_npu.get_npu_format(w2)} w2_contig={w2.is_contiguous()} "
              f"w2_stride={w2.stride()} act_fmt={torch_npu.get_npu_format(act_i8)}",
              file=sys.stderr, flush=True)
    # Step 3: W2 GEMM (int8 x int8 -> hidden dtype) with dequant via scale +
    # per_token_scale. The aclnnGroupedMatmulV5 kernel contract (verified by
    # tests/python/test_grouped_matmul_probe.py) for the per-token-quant case
    # with bf16 output is strict:
    #   * scale (weight dequant scale) MUST be bfloat16 [G, N] (float32 is
    #     rejected: "only supports scale data type bfloat16 with output bf16").
    #   * per_token_scale MUST be float32 1D [M] (2D is rejected: "dim num
    #     must be 1 ... but now is 2"; bf16 is rejected: "only supports
    #     perTokenScale with data type float32").
    # GEMM2 supplies a real per_token_scale, so call torch.ops.npu.npu_grouped_
    # matmul directly (the standalone probe proves this exact call works). The
    # xllm_ops.group_gemm wrapper is reserved for GEMM1, whose omitted
    # per_token_scale must become a genuinely empty TensorList (only the C++
    # apply_npu_grouped_matmul path does that). The wrapper's extended-signature
    # path reinterprets the per_token_scale dim and trips "scale Dim must be 2"
    # for GEMM2, so it is NOT used here.
    output = torch.ops.npu.npu_grouped_matmul(
        x=[act_i8],
        weight=[w2],
        scale=[w2_scale_bf16],
        per_token_scale=[act_pt],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=expanded_row_idx.abs(),
        probs=topk_weights.to(output.dtype),
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
    del gating_output, topk, renormalize, scoring_func
    raise NotImplementedError(
        "moe_fused_topk has no NPU kernel; NPU routes and runs experts in one "
        "step through grouped_moe"
    )


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
    raise NotImplementedError(
        "cutlass_fused_moe is a CUDA library kernel; the NPU equivalent is "
        "grouped_moe"
    )


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
        "fused_moe has no NPU kernel; see kernels_cuda/triton/fused_moe.py for "
        "the reference implementation"
    )


__all__ = [
    "supports_cutlass_moe",
    "prepare_grouped_moe_weights",
    "grouped_moe",
    "moe_fused_topk",
    "cutlass_fused_moe",
    "fused_moe",
]

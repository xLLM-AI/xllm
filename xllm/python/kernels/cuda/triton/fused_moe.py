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

"""Unquantized Triton fused MoE kernels for the Python CUDA executor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_gemm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_tokens,
    top_k: tl.constexpr,
    input_size: tl.constexpr,
    output_size: tl.constexpr,
    stride_input_token: tl.constexpr,
    stride_input_feature: tl.constexpr,
    stride_weight_expert: tl.constexpr,
    stride_weight_input: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_route: tl.constexpr,
    stride_output_feature: tl.constexpr,
    input_is_per_route: tl.constexpr,
    apply_router_weight: tl.constexpr,
    BLOCK_ROUTE: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    BLOCK_INPUT: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    output_block = tl.program_id(1)
    token = route // top_k
    choice = route % top_k
    if token >= num_tokens:
        return

    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    route_offsets = tl.arange(0, BLOCK_ROUTE)
    output_offsets = output_block * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    input_offsets = tl.arange(0, BLOCK_INPUT)
    route_mask = route_offsets == 0
    output_mask = output_offsets < output_size
    accumulator = tl.zeros((BLOCK_ROUTE, BLOCK_OUTPUT), dtype=tl.float32)
    for input_block in range(0, tl.cdiv(input_size, BLOCK_INPUT)):
        features = input_block * BLOCK_INPUT + input_offsets
        input_mask = features < input_size
        input_row = route if input_is_per_route else token
        values = tl.load(
            input_ptr
            + input_row * stride_input_token
            + route_offsets[:, None] * 0
            + features[None, :] * stride_input_feature,
            mask=route_mask[:, None] & input_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            weight_ptr
            + expert * stride_weight_expert
            + features[:, None] * stride_weight_input
            + output_offsets[None, :] * stride_weight_output,
            mask=input_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(values, weights)

    if apply_router_weight:
        router_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        accumulator *= router_weight

    tl.store(
        output_ptr
        + token * stride_output_token
        + choice * stride_output_route
        + route_offsets[:, None] * 0
        + output_offsets[None, :] * stride_output_feature,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=route_mask[:, None] & output_mask[None, :],
    )


@triton.jit
def _fused_moe_activation_kernel(
    input_ptr,
    output_ptr,
    num_routes,
    intermediate_size: tl.constexpr,
    stride_input_route: tl.constexpr,
    stride_output_route: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    if route >= num_routes:
        return

    offsets = tl.arange(0, BLOCK)
    mask = offsets < intermediate_size
    up = tl.load(
        input_ptr + route * stride_input_route + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        input_ptr
        + route * stride_input_route
        + intermediate_size
        + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    activated = activated.to(output_ptr.dtype.element_ty)
    result = activated * up
    tl.store(
        output_ptr + route * stride_output_route + offsets,
        result,
        mask=mask,
    )


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    stride_input_token: tl.constexpr,
    stride_input_route: tl.constexpr,
    stride_input_feature: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_feature: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    feature_block = tl.program_id(1)
    if token >= num_tokens:
        return

    features = feature_block * BLOCK + tl.arange(0, BLOCK)
    mask = features < hidden_size
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for choice in range(top_k):
        values = tl.load(
            input_ptr
            + token * stride_input_token
            + choice * stride_input_route
            + features * stride_input_feature,
            mask=mask,
            other=0.0,
        )
        accumulator += values.to(tl.float32)
    tl.store(
        output_ptr
        + token * stride_output_token
        + features * stride_output_feature,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def _run_moe_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_is_per_route: bool,
    apply_router_weight: bool,
) -> None:
    num_tokens, top_k = topk_ids.shape
    input_size = input.shape[-1]
    output_size = weight.shape[1]
    block_output = 64
    block_input = 128
    _fused_moe_gemm_kernel[
        (num_tokens * top_k, triton.cdiv(output_size, block_output))
    ](
        input,
        weight,
        output,
        topk_weights,
        topk_ids,
        num_tokens,
        top_k=top_k,
        input_size=input_size,
        output_size=output_size,
        stride_input_token=(
            input.stride(1) if input_is_per_route else input.stride(0)
        ),
        stride_input_feature=input.stride(-1),
        stride_weight_expert=weight.stride(0),
        stride_weight_input=weight.stride(2),
        stride_weight_output=weight.stride(1),
        stride_output_token=output.stride(0),
        stride_output_route=output.stride(1),
        stride_output_feature=output.stride(2),
        input_is_per_route=input_is_per_route,
        apply_router_weight=apply_router_weight,
        BLOCK_ROUTE=16,
        BLOCK_OUTPUT=block_output,
        BLOCK_INPUT=block_input,
        num_warps=4,
        num_stages=4,
    )


def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Run unquantized routed experts with vLLM-compatible stage rounding."""
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be a contiguous 2D tensor")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k ids and weights must have the same 2D shape")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("expert weights must be 3D tensors")
    if w13.shape[0] != w2.shape[0]:
        raise ValueError("w13 and w2 must contain the same number of experts")
    if w13.shape[1] != 2 * w2.shape[2]:
        raise ValueError("w13 must contain concatenated up and gate projections")
    if hidden_states.shape[1] != w13.shape[2]:
        raise ValueError("hidden states and w13 input dimensions do not match")
    if w2.shape[1] != hidden_states.shape[1]:
        raise ValueError("w2 output dimension must match hidden size")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("top-k tensors must contain one row per token")
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        raise ValueError("top-k ids must be int32 and weights must be float32")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused MoE supports float16 and bfloat16")
    if w13.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise ValueError("expert weights must match the hidden-state dtype")

    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    w13 = w13.contiguous()
    w2 = w2.contiguous()
    num_tokens, top_k = topk_ids.shape
    if num_tokens == 0:
        return torch.empty_like(hidden_states)
    intermediate_size = w2.shape[2]
    hidden_size = hidden_states.shape[1]

    w13_output = torch.empty(
        num_tokens,
        top_k,
        2 * intermediate_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_moe_gemm(
        hidden_states,
        w13,
        w13_output,
        topk_weights,
        topk_ids,
        input_is_per_route=False,
        apply_router_weight=False,
    )

    activated = torch.empty(
        num_tokens,
        top_k,
        intermediate_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    num_routes = num_tokens * top_k
    block = triton.next_power_of_2(intermediate_size)
    _fused_moe_activation_kernel[(num_routes,)](
        w13_output,
        activated,
        num_routes,
        intermediate_size=intermediate_size,
        stride_input_route=w13_output.stride(1),
        stride_output_route=activated.stride(1),
        BLOCK=block,
        num_warps=4,
    )

    w2_output = torch.empty(
        num_tokens,
        top_k,
        hidden_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _run_moe_gemm(
        activated,
        w2,
        w2_output,
        topk_weights,
        topk_ids,
        input_is_per_route=True,
        apply_router_weight=True,
    )

    output = torch.empty_like(hidden_states)
    block = 256
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden_size, block))](
        w2_output,
        output,
        num_tokens,
        hidden_size,
        w2_output.stride(0),
        w2_output.stride(1),
        w2_output.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK=block,
        num_warps=4,
    )
    return output

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

"""CUDA mixture-of-experts kernels."""

from __future__ import annotations

import torch

_CUTLASS_CAPABILITIES = frozenset((9, 10, 12))


def supports_cutlass_moe(device: torch.device) -> bool:
    """Return whether ``device`` has the native expert GEMMs.

    Args:
        device: Device the MoE layer will run on.

    Returns:
        ``True`` when the CUDA architecture has a CUTLASS expert GEMM.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device_index)
    return major in _CUTLASS_CAPABILITIES


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
    return torch.ops.xllm_ops.moe_fused_topk(
        gating_output, topk, renormalize, scoring_func
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
    return torch.ops.xllm_ops.cutlass_fused_moe(
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


@torch.library.custom_op("xllm_triton::fused_moe", mutates_args=())
def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Run unquantized Triton experts as one graph node."""
    from .triton.fused_moe import (
        fused_moe as triton_fused_moe,
    )

    return triton_fused_moe(hidden_states, topk_ids, topk_weights, w13, w2)


@fused_moe.register_fake
def _fused_moe_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    del topk_ids, topk_weights, w13, w2
    return torch.empty_like(hidden_states)


def prepare_grouped_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out grouped expert weights for the grouped-matmul kernels.

    Args:
        w13: Gate and up projections of every expert.
        w2: Down projection of every expert.

    Returns:
        The two weights in the layout the grouped kernels expect.
    """
    del w13, w2
    raise NotImplementedError(
        "prepare_grouped_moe_weights has no CUDA implementation; CUDA routes "
        "grouped experts through cutlass_fused_moe or fused_moe, which consume "
        "the checkpoint layout directly"
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
        active_expert_range: ``[start, end)`` of global expert indices handled
            by this rank.  Defaults to ``[0, num_experts]`` (all experts).

    Returns:
        Hidden states of shape ``[num_tokens, hidden_size]``.
    """
    del (
        hidden_states,
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
        active_expert_range,
    )
    raise NotImplementedError(
        "grouped_moe has no CUDA kernel; the equivalent CUDA path is "
        "moe_fused_topk followed by cutlass_fused_moe"
    )


__all__ = [
    "supports_cutlass_moe",
    "moe_fused_topk",
    "cutlass_fused_moe",
    "fused_moe",
    "prepare_grouped_moe_weights",
    "grouped_moe",
]

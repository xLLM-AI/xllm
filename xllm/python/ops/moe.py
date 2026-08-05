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

"""Graph interfaces for mixture-of-experts operators."""

from __future__ import annotations

from collections.abc import Callable

import torch

from xllm.python import platform


def moe_fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _moe_fused_topk_fake(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del renormalize, scoring_func
    shape = (gating_output.shape[0], topk)
    return (
        gating_output.new_empty(shape, dtype=torch.float32),
        gating_output.new_empty(shape, dtype=torch.int32),
    )


def _cutlass_fused_moe_fake(
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
    del (
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
    )
    return input.new_empty((input.shape[0], fc2_expert_weights.shape[1]))


def _register_fake_if_available(name: str, fake: Callable) -> None:
    namespace, op_name = name.split("::", 1)
    if hasattr(getattr(torch.ops, namespace), op_name):
        torch.library.register_fake(name)(fake)


_register_fake_if_available("xllm_ops::moe_fused_topk", _moe_fused_topk_fake)
_register_fake_if_available("xllm_ops::cutlass_fused_moe", _cutlass_fused_moe_fake)


def supports_cutlass_moe(device: torch.device) -> bool:
    """Return whether the selected CUDA device supports native expert GEMMs."""
    if not platform.is_gpu():
        return False
    from xllm.python.ops.cuda.moe import supports_cutlass

    return supports_cutlass(device)


def prepare_grouped_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare grouped expert weights for their selected device backend."""
    if not platform.is_npu():
        raise NotImplementedError("grouped MoE weight preparation requires NPU")
    from xllm.python.ops.npu.moe import prepare_grouped_moe_weights as prepare

    return prepare(w13, w2)


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
    """Execute grouped routed experts through the selected device backend."""
    if correction_bias is not None and correction_bias.dtype != gating_output.dtype:
        correction_bias = correction_bias.to(gating_output.dtype)
    if not platform.is_npu():
        raise NotImplementedError("grouped MoE execution requires NPU")
    from xllm.python.ops.npu.moe import grouped_moe as npu_grouped_moe

    return npu_grouped_moe(
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


@torch.library.custom_op("xllm_triton::fused_moe", mutates_args=())
def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Run unquantized CUDA Triton experts as one graph node."""
    from xllm.python.kernels.cuda.triton.fused_moe import (
        fused_moe as triton_fused_moe,
    )

    return triton_fused_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w2,
    )


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


__all__ = [
    "cutlass_fused_moe",
    "fused_moe",
    "grouped_moe",
    "moe_fused_topk",
    "prepare_grouped_moe_weights",
    "supports_cutlass_moe",
]

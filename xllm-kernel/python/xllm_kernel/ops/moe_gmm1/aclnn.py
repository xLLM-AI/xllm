# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_gmm1 implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=value,
            weight=[weight],
            weight_scale=[weight_scale],
            x_scale=activation_scale,
            group_list=groups,
            dequant_mode=0,
            dequant_dtype=0,
            quant_mode=0,
            quant_dtype=1,
            group_list_type=0,
        )
        return q, scale.reshape(-1)

    return run


def load_xllm() -> Callable[..., Any]:
    import torch

    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("moe_grouped_matmul_swiglu_quant")

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.cat((groups[:1], groups[1:] - groups[:-1]))
        pairs = torch.stack((torch.arange(groups.numel(), device=groups.device, dtype=torch.int64), counts), -1)
        return op(value, weight, weight_scale, activation_scale, pairs)

    return run

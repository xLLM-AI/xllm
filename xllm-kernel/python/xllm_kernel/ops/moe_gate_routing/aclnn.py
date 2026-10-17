# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_gate_routing implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(
        logits: torch.Tensor, bias: torch.Tensor, topk: int, groups: int, selected_groups: int, scaling: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, indices, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            k=topk,
            bias=bias,
            k_group=selected_groups,
            group_count=groups,
            group_select_mode=1,
            renorm=1,
            norm_type=1,
            routed_scaling_factor=scaling,
            eps=1e-20,
        )
        return weights, indices

    return run

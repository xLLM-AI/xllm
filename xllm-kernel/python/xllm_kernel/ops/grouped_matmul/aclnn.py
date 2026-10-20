# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""grouped_matmul implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch
    import torch_npu  # noqa: F401

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.npu.npu_grouped_matmul(
            x=[value],
            weight=[weight],
            scale=[weight_scale],
            per_token_scale=[activation_scale],
            split_item=2,
            group_type=0,
            group_list_type=0,
            group_list=groups,
            output_dtype=torch.bfloat16,
        )[0]

    return run

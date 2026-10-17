# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_token_dispatch implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, idx, groups, scale = torch_npu.npu_moe_init_routing_v2(
            value,
            expert_ids,
            active_num=expert_ids.numel(),
            expert_num=num_experts,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=1,
        )
        return q, idx, groups, scale.reshape(-1)

    return run


def load_xllm() -> Callable[..., Any]:
    import torch

    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("moe_init_routing_v3")

    def run(
        value: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, idx, pairs, scale = op(value, expert_ids, expert_ids.numel(), num_experts, [0, num_experts], 1)
        counts = torch.zeros(num_experts, dtype=torch.int64, device=value.device)
        counts.scatter_add_(0, pairs[:, 0], pairs[:, 1])
        return q, idx, counts.cumsum(0), scale.reshape(-1)

    return run

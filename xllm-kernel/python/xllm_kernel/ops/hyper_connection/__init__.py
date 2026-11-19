# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""HyperConnection contracts and startup-bound APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

HC_PRE_CONTRACT = OpContract("hc_pre", 1, (), False, "hyper_connection_pre")
HC_POST_CONTRACT = OpContract("hc_post", 1, (), False, "hyper_connection_post")

OPERATIONS = tuple(
    Operation(
        contract,
        (
            BackendModule(
                module="xllm_kernel.ops.hyper_connection.ascendc",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.ascendc.{contract.operation}",),
            ),
        ),
    )
    for contract in (HC_PRE_CONTRACT, HC_POST_CONTRACT)
)
HC_PRE_OPERATION, HC_POST_OPERATION = OPERATIONS


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse [T, HC, D] or [B, S, HC, D] and return FP32 post and comb."""
    return HC_PRE_OPERATION.get_prepared().function(
        x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps
    )


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Combine [B, S, D] output and [B, S, HC, D] residual into fresh storage."""
    return HC_POST_OPERATION.get_prepared().function(x, residual, post, comb)

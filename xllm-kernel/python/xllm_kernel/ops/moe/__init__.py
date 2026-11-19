# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""MoE routing and quantized expert contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

DEQUANT_SWIGLU_QUANT = OpContract("dequant_swiglu_quant", 1, (), False, "dequant_swiglu_quant")
MOE_GATING_TOP_K_HASH = OpContract("moe_gating_top_k_hash", 1, (), False, "moe_gating_top_k_hash")
MOE_INIT_ROUTING_V3 = OpContract("moe_init_routing_v3", 1, (), False, "moe_init_routing_v3")
MOE_GROUPED_MATMUL_SWIGLU_QUANT = OpContract(
    "moe_grouped_matmul_swiglu_quant", 1, (), False, "moe_grouped_matmul_swiglu_quant"
)

OPERATIONS = tuple(
    Operation(
        contract,
        (
            BackendModule(
                module="xllm_kernel.ops.moe.ascendc",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.ascendc.{contract.operation}",),
            ),
        ),
    )
    for contract in (DEQUANT_SWIGLU_QUANT, MOE_GATING_TOP_K_HASH, MOE_INIT_ROUTING_V3, MOE_GROUPED_MATMUL_SWIGLU_QUANT)
)
_DEQUANT_SWIGLU, _HASH_GATE, _ROUTING_V3, _GROUPED_SWIGLU = OPERATIONS
FUSED_MOE_GMM1_OPERATIONS = (MOE_INIT_ROUTING_V3.operation, MOE_GROUPED_MATMUL_SWIGLU_QUANT.operation)


def has_fused_moe_gmm1() -> bool:
    """Return whether the startup provider supports both routing V3 and fused GMM1."""
    from xllm_kernel.ops.moe.ascendc import HAS_FUSED_MOE_GMM1

    return HAS_FUSED_MOE_GMM1


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
    """Dequantize, apply SwiGLU, and return the INT8 result and its scale."""
    return _DEQUANT_SWIGLU.get_prepared().function(
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
    """Return selected weights, INT32 expert IDs, and optional normalized logits."""
    return _HASH_GATE.get_prepared().function(
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


def moe_init_routing_v3(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    active_num: int,
    expert_num: int,
    active_expert_range: list[int],
    quant_mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand input rows and emit source-to-expanded indices and expert/count pairs."""
    return _ROUTING_V3.get_prepared().function(x, expert_idx, active_num, expert_num, active_expert_range, quant_mode)


def moe_grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply INT8 grouped GEMM and SwiGLU using NZ weights and V3 group pairs."""
    return _GROUPED_SWIGLU.get_prepared().function(x, weight, weight_scale, x_scale, group_list)

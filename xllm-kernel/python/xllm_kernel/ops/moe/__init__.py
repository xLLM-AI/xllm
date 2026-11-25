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

ASCENDC_OPERATIONS = tuple(
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
_DEQUANT_SWIGLU, _HASH_GATE, _ROUTING_V3, _GROUPED_SWIGLU = ASCENDC_OPERATIONS
MOE_DISTRIBUTE_OPERATIONS = ("moe_distribute_dispatch_v2", "moe_distribute_combine_v2")
TORCH_OPERATIONS, ACLNN_OPERATIONS = (
    tuple(
        Operation(
            OpContract(name, 1, ("topk_ids",) if name == "mega_moe" else (), False, name),
            (
                BackendModule(
                    module="xllm_kernel.ops.moe.npu",
                    platform=PlatformEnum.NPU,
                    requires=("torch_npu",),
                    implementations=(f"npu.{provider}.{name}",),
                ),
            ),
        )
        for name in names
    )
    for provider, names in (
        (
            "torch",
            (
                "moe_gating_top_k_softmax",
                "moe_gating_top_k",
                "moe_init_routing_v2",
                "moe_token_unpermute",
                "moe_finalize_routing",
                *MOE_DISTRIBUTE_OPERATIONS,
            ),
        ),
        ("aclnn", ("mega_moe",)),
    )
)
_SOFTMAX_GATE, _GROUPED_GATE, _ROUTING_V2, _UNPERMUTE, _FINALIZE, _DISPATCH_V2, _COMBINE_V2 = TORCH_OPERATIONS
(_MEGA_MOE,) = ACLNN_OPERATIONS
OPERATIONS = (*ASCENDC_OPERATIONS, *TORCH_OPERATIONS, *ACLNN_OPERATIONS)
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


def moe_gating_top_k_softmax(x: torch.Tensor, topk: int, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous top-k weights and IDs, optionally renormalizing the selected weights."""
    return _SOFTMAX_GATE.get_prepared().function(x, topk, normalize)


def moe_gating_top_k(
    x: torch.Tensor,
    *,
    k: int,
    bias: torch.Tensor | None,
    k_group: int,
    group_count: int,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    routed_scaling_factor: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select experts with TorchNPU's grouped routing policy."""
    return _GROUPED_GATE.get_prepared().function(
        x,
        k=k,
        bias=bias,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )


def moe_init_routing_v2(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    scale: torch.Tensor | None,
    active_num: int,
    expert_num: int,
    expert_tokens_num_type: int,
    expert_tokens_num_flag: bool,
    active_expert_range: list[int],
    quant_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand routed token rows, optionally quantize them, and return the row map and expert counts."""
    return _ROUTING_V2.get_prepared().function(
        x,
        expert_idx,
        scale=scale,
        active_num=active_num,
        expert_num=expert_num,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag,
        active_expert_range=active_expert_range,
        quant_mode=quant_mode,
    )


def moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and weight the routed expert outputs."""
    return _UNPERMUTE.get_prepared().function(
        permuted_tokens=permuted_tokens, sorted_indices=sorted_indices, probs=probs
    )


def moe_finalize_routing(
    permuted_tokens: torch.Tensor,
    shared_input: torch.Tensor,
    probs: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Combine routed outputs and the shared expert using EP routing metadata."""
    return _FINALIZE.get_prepared().function(
        permuted_tokens, shared_input, None, None, probs, sorted_indices, None, drop_pad_mode=2
    )


def moe_distribute_dispatch_v2(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    x_active_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch unquantized rows across an EP group, retaining the metadata for combine."""
    return _DISPATCH_V2.get_prepared().function(
        x=x,
        expert_ids=expert_ids,
        scales=None,
        x_active_mask=x_active_mask,
        expert_scales=None,
        group_ep=group_ep,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        # Keep quantization in the expert path to preserve BF16 half-tie rounding.
        quant_mode=0,
        global_bs=0,
        expert_token_nums_type=1,
        comm_alg="",
    )[:7]


def moe_distribute_combine_v2(
    expand_x: torch.Tensor,
    expert_ids: torch.Tensor,
    assist_info_for_combine: torch.Tensor,
    ep_send_counts: torch.Tensor,
    expert_scales: torch.Tensor,
    *,
    tp_send_counts: torch.Tensor,
    x_active_mask: torch.Tensor | None,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
) -> torch.Tensor:
    """Return the routed outputs to the source ranks and combine them using router weights."""
    return _COMBINE_V2.get_prepared().function(
        expand_x=expand_x,
        expert_ids=expert_ids,
        assist_info_for_combine=assist_info_for_combine,
        ep_send_counts=ep_send_counts,
        expert_scales=expert_scales,
        tp_send_counts=tp_send_counts,
        x_active_mask=x_active_mask,
        expand_scales=None,
        shared_expert_x=None,
        group_ep=group_ep,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=0,
        comm_quant_mode=0,
        comm_alg="",
    )


def has_mega_moe() -> bool:
    """Return whether CANN supplies both MegaMoE entry points."""
    from xllm_kernel.ops.moe.npu import HAS_MEGA_MOE

    return HAS_MEGA_MOE


def has_moe_distribute() -> bool:
    """Return whether TorchNPU exposes both halves of EP dispatch/combine."""
    import torch_npu

    return all(hasattr(torch_npu, f"npu_{name}") for name in MOE_DISTRIBUTE_OPERATIONS)


def mega_moe(
    context: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weight1: list[torch.Tensor],
    weight2: list[torch.Tensor],
    weight_scales1: list[torch.Tensor],
    weight_scales2: list[torch.Tensor],
    moe_expert_num: int,
    ep_world_size: int,
    ccl_buffer_size: int,
    num_max_tokens_per_rank: int,
    x_active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused EP outputs and local counts. CANN may mark masked entries in topk_ids."""
    return _MEGA_MOE.get_prepared().function(
        context,
        x,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        weight_scales1,
        weight_scales2,
        moe_expert_num,
        ep_world_size,
        ccl_buffer_size,
        num_max_tokens_per_rank,
        x_active_mask,
    )

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free GDN contracts and startup-bound semantic APIs.

The non-Mega kernels use FP32 [slot, value_head, key_dim, value_dim] state.
The host still owns state allocation, checkpoint indices and graph lifetimes.
Mega GDN's separate FLA layout and packed projections are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

FUSED_RECURRENT_GATED_DELTA_RULE_CONTRACT = OpContract(
    "fused_recurrent_gated_delta_rule", 1, ("initial_state",), True, "fp32_recurrence_bf16_output"
)
FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_CONTRACT = OpContract(
    "fused_sigmoid_gating_delta_rule_update", 1, ("initial_state_source",), False, "fp32_gating_recurrence_bf16_output"
)
FUSED_QKVZBA_SPLIT_RESHAPE_CAT_CONTRACT = OpContract(
    "fused_qkvzba_split_reshape_cat", 1, (), False, "exact_bf16_layout_copy"
)


def _operation(contract: OpContract) -> Operation:
    return Operation(
        contract,
        (
            BackendModule(
                module="xllm_kernel.ops.attention.gdn.triton",
                platform=PlatformEnum.NPU,
                requires=("torch_npu", "triton"),
                implementations=(f"npu.triton.{contract.operation}",),
            ),
        ),
    )


FUSED_RECURRENT_GATED_DELTA_RULE_OPERATION = _operation(FUSED_RECURRENT_GATED_DELTA_RULE_CONTRACT)
FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_OPERATION = _operation(FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_CONTRACT)
FUSED_QKVZBA_SPLIT_RESHAPE_CAT_OPERATION = _operation(FUSED_QKVZBA_SPLIT_RESHAPE_CAT_CONTRACT)
OPERATIONS = (
    FUSED_RECURRENT_GATED_DELTA_RULE_OPERATION,
    FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_OPERATION,
    FUSED_QKVZBA_SPLIT_RESHAPE_CAT_OPERATION,
)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    *,
    num_accepted_tokens: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update packed BF16 Q/K/V [1, tokens, heads, 128] from FP32 KV state.

    ``g`` is FP32 [1, tokens, value_heads], ``beta`` has that shape in BF16.
    ``cu_seqlens`` is contiguous int64 [sequences + 1]. Ordinary decode uses
    one token per sequence and int32 state slots [sequences]. With
    ``inplace_final_state=False``, arbitrary positive packed lengths are supported:
    read the named slots, return fresh [sequences, value_heads, 128, 128]
    final state, and leave the pool unchanged.

    Spec verify requires in-place state and int32 checkpoint indices
    [sequences, checkpoint_capacity], including strided views. Read checkpoint
    ``num_accepted_tokens[sequence] - 1`` (int32 accepted counts, at least 1),
    then write every verify-token state to that sequence's indexed checkpoints.
    The returned state aliases the supplied pool only in the in-place modes.
    Slots must be valid and sequences must not write overlapping state rows.
    Metadata values and their lifetimes are owned by the host batch builder.
    Output is fresh contiguous BF16 with the shape of ``v``. Both execution
    modes support eager and ACLGraph with dynamic device-resident metadata.
    """
    return FUSED_RECURRENT_GATED_DELTA_RULE_OPERATION.get_prepared().function(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        inplace_final_state=inplace_final_state,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_sigmoid_gating_delta_rule_update(
    a_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """Fuse FP32 softplus decay, sigmoid beta and recurrent KV-state update.

    Contiguous BF16 Q/K/V are [B, T, heads, 128]; ``a``/``b`` are BF16
    [B, T, value_heads], and ``a_log``/``dt_bias`` are FP32 [value_heads].
    Int32 cumulative lengths partition flattened B*T tokens; int32 slots
    [sequences] select the FP32 [slot, value_heads, 128, 128] KV-state pool.
    Update each nonnegative slot in place after its last token. A negative
    slot starts from zero state and never writes the pool. Slots written by
    different sequences must be distinct. Empty sequences leave state intact.
    Return fresh contiguous BF16 output with the shape of ``v``; no output
    aliases inputs. Eager and ACLGraph use the same startup-prepared kernels.
    """
    return FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_OPERATION.get_prepared().function(
        a_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
    )


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int = 128,
    head_v: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split per-QK-head packed BF16 [Q,K,V,Z] and [B,A] projections.

    Each QK head has 1–4 value heads; both head dimensions are 128. Inputs are
    contiguous matrices with the same token count. Return fresh contiguous
    ``mixed_qkv=[Q|K|V]``, ``z=[tokens,value_heads,128]`` and ``b/a`` matrices
    [tokens,value_heads], without input mutation or alias. GQA ratios are
    internal specializations of one semantic operation, prepared at startup.
    This layout differs from separately projected QKV/Z and B/A used by Mega
    GDN; callers retain the existing path for those projections.
    """
    return FUSED_QKVZBA_SPLIT_RESHAPE_CAT_OPERATION.get_prepared().function(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
    )

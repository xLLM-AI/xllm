# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Startup-bound MegaGDN and shared state-index operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

MEGA_GDN_PREFILL_CONTRACT = OpContract("mega_gdn_prefill", 1, ("conv_state", "ssm_cache"), False, "mega_gdn_prefill")
MEGA_GDN_DECODE_CONTRACT = OpContract("mega_gdn_decode", 1, ("conv_state", "ssm_state"), False, "mega_gdn_decode")
MEGA_GDN_MTP_DECODE_CONTRACT = OpContract(
    "mega_gdn_mtp_decode", 1, ("conv_state", "ssm_state"), False, "mega_gdn_mtp_decode"
)
MEGA_GDN_MTP_DECODE_VARLEN_CONTRACT = OpContract(
    "mega_gdn_mtp_decode_varlen", 1, ("conv_state", "ssm_state", "out"), False, "mega_gdn_mtp_decode_varlen"
)
SHARED_LINEAR_STATE_INDICES_OUT_CONTRACT = OpContract(
    "shared_linear_state_indices_out", 1, ("out",), False, "shared_linear_state_indices_out"
)

OPERATIONS = tuple(
    Operation(
        contract,
        (
            BackendModule(
                module=f"xllm_kernel.ops.attention.gdn.mega_{solution}",
                platform=PlatformEnum.NPU,
                requires=requires,
                implementations=(f"npu.{solution}.{contract.operation}",),
            ),
        ),
    )
    for contract, solution, requires in (
        (MEGA_GDN_PREFILL_CONTRACT, "ascendc", ("torch_npu",)),
        (MEGA_GDN_DECODE_CONTRACT, "ascendc", ("torch_npu",)),
        (MEGA_GDN_MTP_DECODE_CONTRACT, "ascendc", ("torch_npu",)),
        (MEGA_GDN_MTP_DECODE_VARLEN_CONTRACT, "ascendc", ("torch_npu",)),
        (SHARED_LINEAR_STATE_INDICES_OUT_CONTRACT, "triton", ("torch_npu", "triton")),
    )
)
_PREFILL, _DECODE, _MTP_DECODE, _MTP_DECODE_VARLEN, _SHARED_INDICES_OUT = OPERATIONS


def has_mega_gdn_mtp_decode_varlen() -> bool:
    """Return whether the installed NPU package provides packed verification."""
    from xllm_kernel.ops.attention.gdn.mega_ascendc import HAS_VARLEN

    return HAS_VARLEN


def mega_gdn_prefill(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state_read_indices: torch.Tensor,
    conv_state_write_indices: torch.Tensor,
    ssm_state_read_indices: torch.Tensor,
    ssm_state_write_indices: torch.Tensor,
    ssm_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    norm_weight: torch.Tensor,
    num_matrices: int,
) -> torch.Tensor:
    """Run the fused Qwen3.5 prefill GDN and update both state caches."""
    return _PREFILL.get_prepared().function(
        mixed_qkv,
        b,
        a,
        z,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        conv_state_read_indices,
        conv_state_write_indices,
        ssm_state_read_indices,
        ssm_state_write_indices,
        ssm_cache,
        cu_seqlens,
        norm_weight,
        num_matrices,
    )


def mega_gdn_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> torch.Tensor:
    """Run fused single-token Qwen3.5 decode and update both state caches.

    State indices are either legacy common slots ``[batch]`` or independent
    ``[batch, 2]`` Conv slots and SSM rows. A negative pair disables a row.
    """
    return _DECODE.get_prepared().function(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_state_indices,
        write_state_indices,
        norm_weight,
        fla_ssm_state_layout,
    )


def mega_gdn_mtp_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> torch.Tensor:
    """Restore the accepted checkpoint and record each verify-token state.

    Paired ``[batch, 2]`` indices name the Conv slot and the first SSM
    checkpoint row independently; legacy ``[batch]`` slots retain their stride.
    """
    return _MTP_DECODE.get_prepared().function(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_state_indices,
        write_state_indices,
        num_accepted_tokens,
        norm_weight,
        fla_ssm_state_layout,
    )


def mega_gdn_mtp_decode_varlen(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> None:
    """Verify packed prefixes using common slots or independent role rows."""
    _MTP_DECODE_VARLEN.get_prepared().function(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        state_indices,
        num_accepted_tokens,
        norm_weight,
        cu_seqlens,
        out,
        fla_ssm_state_layout,
    )


def shared_linear_state_indices_out(ids: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    """Resolve positive int32 state IDs into caller-owned [groups, batch, 2].

    The table is [2 * groups, capacity] with unit inner stride. Row pairs
    resolve independent Conv slots and SSM rows. Nonpositive IDs write -1
    in both roles. The output is contiguous and retains its storage.
    """
    _SHARED_INDICES_OUT.get_prepared().function(ids, rows, out)

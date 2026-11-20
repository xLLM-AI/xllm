# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the package's MegaGDN ACLNN entry points."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.attention.gdn.ascendc import _chunk_masks
from xllm_kernel.ops.attention.gdn.mega import (
    MEGA_GDN_DECODE_CONTRACT,
    MEGA_GDN_MTP_DECODE_CONTRACT,
    MEGA_GDN_MTP_DECODE_VARLEN_CONTRACT,
    MEGA_GDN_PREFILL_CONTRACT,
)
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract, register_kernel

HAS_VARLEN = torch.ops.xllm_kernel.prepare_ascendc_mega_gdn()


@register_kernel(
    name="npu.ascendc.mega_gdn_prefill",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Packed BF16 Qwen GDN prefill with device cumulative lengths and per-role state indices.",
    solution="ascendc",
    binding="aclnn",
    contract=MEGA_GDN_PREFILL_CONTRACT,
)
def _prefill(
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
    return torch.ops.xllm_kernel.mega_gdn_prefill(
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
        *_chunk_masks(mixed_qkv.device, torch.bfloat16),
        cu_seqlens,
        norm_weight,
        num_matrices,
    )


def _register(contract: OpContract, domain: str) -> None:
    name = contract.operation
    register_kernel(
        name=f"npu.ascendc.{name}",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain=domain,
        solution="ascendc",
        binding="aclnn",
        contract=contract,
    )(getattr(torch.ops.xllm_kernel, name).default)


_register(MEGA_GDN_DECODE_CONTRACT, "BF16 single-token Qwen GDN decode with common slots or paired role rows.")
_register(MEGA_GDN_MTP_DECODE_CONTRACT, "BF16 fixed-width Qwen GDN verify with accepted checkpoint counts.")
if HAS_VARLEN:
    _register(MEGA_GDN_MTP_DECODE_VARLEN_CONTRACT, "Packed BF16 Qwen GDN verify with a caller-owned output buffer.")

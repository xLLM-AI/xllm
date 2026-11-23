# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the packaged causal convolution and chunk GDN ACLNN bindings."""

from functools import cache

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.attention.gdn import (
    CAUSAL_CONV1D_PREFILL_CONTRACT,
    CAUSAL_CONV1D_QKV_PREFILL_CONTRACT,
    CHUNK_GATED_DELTA_RULE_CONTRACT,
)
from xllm_kernel.ops.attention.kda import CAUSAL_CONV1D_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

_CHUNK_SIZE = 128

torch.ops.xllm_kernel.prepare_ascendc_gdn()

register_kernel(
    name="npu.ascendc.causal_conv1d",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Contiguous channel-last FP16 or BF16 convolution and staged state with host query lengths.",
    solution="ascendc",
    binding="aclnn",
    contract=CAUSAL_CONV1D_CONTRACT,
)(torch.ops.xllm_kernel.causal_conv1d.default)

register_kernel(
    name="npu.ascendc.causal_conv1d_qkv_prefill",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Packed BF16 prefill with 128-wide Q/K/V heads and host state metadata.",
    solution="ascendc",
    binding="aclnn",
    contract=CAUSAL_CONV1D_QKV_PREFILL_CONTRACT,
)(torch.ops.xllm_kernel.causal_conv1d_qkv_prefill.default)


register_kernel(
    name="npu.ascendc.causal_conv1d_prefill",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Packed FP16/BF16 convolution with host query lengths and indexed state.",
    solution="ascendc",
    binding="aclnn",
    contract=CAUSAL_CONV1D_PREFILL_CONTRACT,
)(torch.ops.xllm_kernel.causal_conv1d_prefill.default)


@cache
def _chunk_masks(
    device: torch.device, dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones((_CHUNK_SIZE, _CHUNK_SIZE), device=device, dtype=torch.float32)
    return ones.tril(-1), ones.tril(0), -torch.eye(_CHUNK_SIZE, device=device, dtype=dtype)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    value_fp32 = value.float()
    norm = torch.rsqrt(value_fp32.square().sum(dim=-1, keepdim=True) + 1e-6)
    return (value_fp32 * norm).to(value.dtype).half().unsqueeze(0)


@register_kernel(
    name="npu.ascendc.chunk_gated_delta_rule",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Packed Q/K/V with head width 128 and graph-stable host sequence lengths.",
    solution="ascendc",
    binding="aclnn",
    contract=CHUNK_GATED_DELTA_RULE_CONTRACT,
)
def _chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    sequence_lengths: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_lengths is None:
        offsets = cu_seqlens.tolist()
        sequence_lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if (
        len(sequence_lengths) != cu_seqlens.numel() - 1
        or any(length < 0 for length in sequence_lengths)
        or sum(sequence_lengths) != q.shape[0]
    ):
        raise ValueError("chunk GDN sequence lengths must partition the packed tokens")
    num_chunks = sum((length + _CHUNK_SIZE - 1) // _CHUNK_SIZE for length in sequence_lengths)
    output, state = torch.ops.xllm_kernel.mega_chunk_gdn(
        _normalize(q),
        _normalize(k),
        v.half().unsqueeze(0),
        g.to(v.dtype).float().unsqueeze(0),
        beta.to(v.dtype).half().unsqueeze(0),
        *_chunk_masks(q.device),
        cu_seqlens.to(dtype=torch.int32),
        initial_state.half(),
        num_chunks,
    )
    return (output.squeeze(0) * q.shape[-1] ** -0.5).to(q.dtype), state.float()

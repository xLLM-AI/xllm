# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Package-owned xlite rms_norm registration and fake contract."""

from __future__ import annotations

import math

import torch

from xllm_kernel.backends.xlite import load_extension
from xllm_kernel.custom_ops import register_custom_op
from xllm_kernel.ops.rms_norm import CONTRACT
from xllm_kernel.registry import register_kernel


def _fake(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import torch

    if input.ndim < 1 or weight.ndim != 1 or input.shape[-1] != weight.numel():
        raise ValueError("xlite RMSNorm requires a weight matching the input's last dimension")
    if input.device.type != "npu" or input.device != weight.device:
        raise ValueError("xlite RMSNorm requires tensors on the same NPU")
    if input.dtype not in (torch.float16, torch.bfloat16) or input.dtype != weight.dtype:
        raise ValueError("xlite RMSNorm requires matching FP16/BF16 tensors")
    if weight.numel() <= 0 or weight.numel() > 8192 or weight.numel() % 64:
        raise ValueError("xlite RMSNorm requires a width divisible by 64 up to 8192")
    if input.numel() > 2**32 - 1:
        raise ValueError("xlite RMSNorm requires element offsets to fit uint32")
    if not math.isfinite(eps) or eps < 0 or eps > torch.finfo(torch.float32).max:
        raise ValueError("xlite RMSNorm requires a finite nonnegative FP32 epsilon")
    return torch.empty(input.shape, dtype=input.dtype, device=input.device)


load_extension()
register_kernel(
    name="npu.xlite.rms_norm",
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="FP16/BF16, same NPU/dtype, ND, width divisible by 64 and <=8192, numel<=2**32-1",
    solution="xlite",
    variant="",
    binding="xllm_kernel._xlite",
    priority=0,
    contract=CONTRACT,
)(register_custom_op("xlite_rms_norm", fake_impl=_fake))

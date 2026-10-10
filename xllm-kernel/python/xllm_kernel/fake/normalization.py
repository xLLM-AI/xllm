# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Fake registration for the optional package-owned xlite bridge."""

import math

import torch


def _xlite_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
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


def register_fake() -> None:
    """Called once, after the package bridge has registered its schema."""
    torch.library.register_fake("xllm_kernel::xlite_rms_norm")(_xlite_rms_norm)

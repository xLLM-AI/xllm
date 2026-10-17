# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical metadata checks shared by package-owned fake implementations."""

import math

import torch


def tensor(value: torch.Tensor, ndim: int, dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)) -> None:
    torch._check(value.ndim == ndim)
    torch._check(value.device.type in ("npu", "privateuseone"))
    torch._check(value.dtype in dtypes)
    torch._check(value.numel() <= 2**31 - 1)


def pair(value: torch.Tensor, other: torch.Tensor, ndim: int) -> None:
    tensor(other, ndim, (value.dtype,))
    torch._check(value.device == other.device)


def width(size: int, alignment: int, maximum: int) -> None:
    torch._check(0 < size <= maximum and size % alignment == 0)


def epsilon(value: float) -> None:
    torch._check(math.isfinite(value) and 0 <= value <= torch.finfo(torch.float32).max)

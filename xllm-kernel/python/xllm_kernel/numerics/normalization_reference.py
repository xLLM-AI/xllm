# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Device-independent reference arithmetic shared by compound accuracy cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def rms(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Unrounded FP32 RMS normalization, including the affine transform."""
    import torch

    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Device-independent reference arithmetic shared by compound accuracy cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def symmetric_int8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 per-row symmetric quantization; zero rows have zero scale and codes."""
    import torch

    value = value.float()
    scale = value.abs().amax(-1) / 127.0
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (value / divisor.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8), scale

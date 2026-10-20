# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""CPU-seeded input construction; no implicit device RNG or backend discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Inputs:
    def __init__(self, dtype: str, seed: int) -> None:
        import torch

        self.dtype = getattr(torch, dtype)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def normal(self, *shape: int, dtype: torch.dtype | None = None, scale: float = 1.0) -> torch.Tensor:
        import torch

        return (torch.randn(shape, generator=self.generator) * scale).to(dtype or self.dtype)

    def integer(self, low: int, high: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        import torch

        return torch.randint(low, high, shape, generator=self.generator, dtype=dtype)

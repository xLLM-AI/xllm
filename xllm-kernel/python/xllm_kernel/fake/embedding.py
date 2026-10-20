# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned embedding bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tensor(weight, 2)
    tensor(indices, 1, (torch.int32,))
    torch._check(indices.device == weight.device and weight.shape[0] > 0)
    width(weight.shape[1], 64, 8192)
    torch._check(indices.numel() * weight.shape[1] <= 2**31 - 1)
    return weight.new_empty((indices.numel(), weight.shape[1]))


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_embedding")(_fake)

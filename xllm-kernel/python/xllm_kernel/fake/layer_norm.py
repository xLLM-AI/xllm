# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned layer_norm bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    tensor(value, 2)
    pair(value, weight, 1)
    pair(value, bias, 1)
    torch._check(value.shape[-1] == weight.numel() and bias.shape == weight.shape)
    width(value.shape[1], 64, 8192)
    epsilon(eps)
    return value.new_empty(value.shape)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_layer_norm")(_fake)

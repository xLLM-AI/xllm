# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned fused_add_rms_norm bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(
    value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor(value, 2)
    pair(value, residual, 2)
    pair(value, weight, 1)
    torch._check(value.shape == residual.shape and value.shape[1] == weight.numel())
    width(value.shape[1], 64, 8192)
    epsilon(eps)
    return value.new_empty(value.shape), value.new_empty(value.shape)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_fused_add_rms_norm")(_fake)

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned add bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    tensor(value, 2)
    pair(value, residual, 2)
    torch._check(value.shape == residual.shape)
    width(value.shape[1], 64, 8192)
    return value.new_empty(value.shape)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_add")(_fake)

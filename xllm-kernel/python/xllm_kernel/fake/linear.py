# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned linear bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tensor(value, 2)
    pair(value, weight, 2)
    torch._check(value.shape[1] == weight.shape[1])
    width(weight.shape[0], 16, 2**31 - 1)
    width(weight.shape[1], 16, 2**31 - 1)
    torch._check(value.shape[0] * weight.shape[0] <= 2**31 - 1)
    return value.new_empty((value.shape[0], weight.shape[0]))


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_linear")(_fake)

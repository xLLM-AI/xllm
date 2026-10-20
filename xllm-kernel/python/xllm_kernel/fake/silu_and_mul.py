# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned silu_and_mul bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor) -> torch.Tensor:
    tensor(value, 2)
    width(value.shape[1], 128, 16384)
    return value.new_empty((value.shape[0], value.shape[1] // 2))


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_silu_and_mul")(_fake)

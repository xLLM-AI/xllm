# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned dynamic_quant bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor(value, 2, (torch.bfloat16,))
    width(value.shape[1], 64, 8192)
    return value.new_empty(value.shape, dtype=torch.int8), value.new_empty((value.shape[0],), dtype=torch.float32)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_dynamic_quant")(_fake)

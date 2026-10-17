# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned interleaved_rotary_embedding bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    tensor(value, 3)
    pair(value, cosine, 4)
    pair(value, sine, 4)
    torch._check(cosine.shape == sine.shape and cosine.shape == (value.shape[0], 1, 1, value.shape[2]))
    torch._check(value.shape[2] in (64, 128) and 0 < value.shape[1] * value.shape[2] <= 4096)
    return value.new_empty(value.shape)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_interleaved_rotary_embedding")(_fake)

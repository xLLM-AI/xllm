# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Logical contract of the package-owned moe_gate_routing bridge."""

import torch

from xllm_kernel.fake.validation import epsilon, pair, tensor, width


def _fake(
    logits: torch.Tensor, bias: torch.Tensor, topk: int, groups: int, selected_groups: int, scaling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    tensor(logits, 2, (torch.float32,))
    pair(logits, bias, 1)
    torch._check(logits.shape[1] == 256 and bias.shape == (256,))
    torch._check((groups == 1 and selected_groups == 1) or (groups == 8 and selected_groups == 4))
    torch._check(0 < topk <= 32 and math.isfinite(scaling) and 0 < scaling <= torch.finfo(torch.float32).max)
    return logits.new_empty((logits.shape[0], topk)), logits.new_empty((logits.shape[0], topk), dtype=torch.int32)


def register_fake() -> None:
    torch.library.register_fake("xllm_kernel::xlite_moe_gate_routing")(_fake)

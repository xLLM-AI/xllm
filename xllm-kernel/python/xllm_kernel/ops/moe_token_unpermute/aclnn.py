# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_token_unpermute implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_moe_token_unpermute(permuted_tokens=value, sorted_indices=indices, probs=probs)

    return run

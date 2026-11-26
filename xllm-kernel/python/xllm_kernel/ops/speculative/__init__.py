# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Device primitives for verification; scheduling remains with the caller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

GREEDY_PREFIX_VERIFY = Operation(
    OpContract("greedy_prefix_verify", 1, (), False, "greedy_prefix_verify"),
    (
        BackendModule(
            module="xllm_kernel.ops.speculative.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "tilelang", "triton"),
            implementations=("npu.tilelang.greedy_prefix_verify",),
        ),
    ),
)
OPERATIONS = (GREEDY_PREFIX_VERIFY,)


def greedy_prefix_verify(draft: torch.Tensor, target: torch.Tensor, bonus: torch.Tensor) -> torch.Tensor:
    """Return accepted target tokens through the first rejection, padded with -1."""
    return GREEDY_PREFIX_VERIFY.get_prepared().function(draft, target, bonus)

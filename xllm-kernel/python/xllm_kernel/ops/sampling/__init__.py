# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Sampling device primitives shared with native logits processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

TOP_K_TOP_P = Operation(
    OpContract("top_k_top_p", 1, ("logits",), False, "inplace_top_k_top_p"),
    (
        BackendModule(
            module="xllm_kernel.ops.sampling.npu",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.aclnn.top_k_top_p",),
        ),
    ),
)
OPERATIONS = (TOP_K_TOP_P,)


def top_k_top_p(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> None:
    """Filter logits in place, comparing top-p cutoffs in logits dtype.

    The caller supplies per-row INT32 top-k values normalized into [1, vocab].
    Sampling policy and disabled-threshold normalization stay with the caller.
    """
    TOP_K_TOP_P.get_prepared().function(logits, top_k, top_p)

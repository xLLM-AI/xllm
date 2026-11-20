# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the DSA RoPE sparse attention ACLNN entry points."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.attention.dsa import OPERATIONS
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_ascendc_sparse_attention()

for operation, domain in zip(
    OPERATIONS,
    (
        "Sparse RoPE attention with optional softmax max and sum.",
        "Sparse RoPE attention with a caller-owned output.",
    ),
    strict=True,
):
    name = operation.contract.operation
    register_kernel(
        name=operation.implementations[0],
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain=domain,
        solution="ascendc",
        binding="aclnn",
        contract=operation.contract,
    )(getattr(torch.ops.xllm_kernel, name).default)

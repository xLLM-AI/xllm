# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the compressed KV and sparse attention ACLNN entry points."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.attention.sparse_attention import OPERATIONS
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_ascendc_sparse_attention()

for operation, domain in zip(
    OPERATIONS,
    (
        "Project and compress KV while updating paged compressor state.",
        "Attend jointly over original and compressed KV.",
        "Device tiling metadata for original and compressed KV.",
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

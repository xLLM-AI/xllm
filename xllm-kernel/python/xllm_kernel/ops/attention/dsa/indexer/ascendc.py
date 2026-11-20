# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the sparse indexing operations shipped in the package OPP."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.attention.dsa.indexer import OPERATIONS
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_ascendc_sparse_indexing()

for operation, domain in zip(
    OPERATIONS,
    (
        "Packed quantized queries with per-token scales and paged key rows.",
        "Device int32 query and key lengths, and packed-paged indexer geometry.",
        "In-place cache updates with int32 or int64 indices.",
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

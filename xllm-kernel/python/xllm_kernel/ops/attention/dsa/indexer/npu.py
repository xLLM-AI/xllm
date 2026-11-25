# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the floating-point LightningIndexer CANN operator."""

import torch

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.attention.dsa.indexer import LIGHTNING_INDEXER_OPERATIONS
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

for operation in LIGHTNING_INDEXER_OPERATIONS:
    contract = operation.contract
    register_kernel(
        name=f"npu.aclnn.{contract.operation}",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Floating-point LightningIndexer with packed or batched queries and optional paged key metadata.",
        solution="aclnn",
        binding="aclnn",
        contract=contract,
    )(getattr(torch.ops.xllm_kernel, contract.operation).default)

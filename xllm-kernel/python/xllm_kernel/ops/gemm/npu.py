# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind allocating and caller-owned NPU grouped products."""

import torch

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.gemm import GROUPED_MATMUL_CONTRACT, GROUPED_MATMUL_OUT_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

HAS_GROUPED_MATMUL_OUT = torch.ops.xllm_kernel.has_grouped_matmul_out()

register_kernel(
    name="npu.torch.grouped_matmul",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="TorchNPU grouped GEMM with tensor group metadata or separate projections.",
    solution="torch",
    binding="torch_npu",
    contract=GROUPED_MATMUL_CONTRACT,
)(torch.ops.npu.npu_grouped_matmul)

if HAS_GROUPED_MATMUL_OUT:
    register_kernel(
        name="npu.aclnn.grouped_matmul_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Quantized grouped GEMM into caller-owned contiguous NPU storage.",
        solution="aclnn",
        binding="aclnn",
        contract=GROUPED_MATMUL_OUT_CONTRACT,
    )(torch.ops.xllm_kernel.grouped_matmul_out.default)

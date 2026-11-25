# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind current-stream HCCL operations and fused matmul-reduce-scatter."""

import torch
import torch_npu

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.collectives import OPERATIONS, has_matmul_reduce_scatter
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

for operation in OPERATIONS[:3]:
    name = operation.contract.operation
    register_kernel(
        name=f"npu.hccl.{name}",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Contiguous NPU storage and a borrowed HCCL communicator.",
        solution="hccl",
        binding="hccl",
        contract=operation.contract,
    )(getattr(torch.ops.xllm_kernel, f"npu_{name}").default)

if has_matmul_reduce_scatter():
    register_kernel(
        name="npu.torch.matmul_reduce_scatter",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="ND FP16/BF16 operands and an HCCL communicator name.",
        solution="torch",
        binding="torch_npu",
        contract=OPERATIONS[-1].contract,
    )(torch_npu.npu_mm_reduce_scatter_base)

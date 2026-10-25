# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""ATB GEMM bindings supplied by the matching xLLM native library."""

from __future__ import annotations

import torch

from xllm_kernel.ops.gemm import EINSUM_CONTRACT, EINSUM_OUT_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    import torch_npu  # noqa: F401
    import xllm_export  # noqa: F401

    register_kernel(
        name="npu.atb.einsum",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 ND [T,H,D] x contiguous [H,D,O] -> [T,H,O]; strided input is packed by ATB.",
        solution="atb",
        binding="torch.ops.xllm_ops",
        priority=10,
        contract=EINSUM_CONTRACT,
    )(torch.ops.xllm_ops.atb_matmul_ein_sum)

    register_kernel(
        name="npu.atb.einsum_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 ND [T,H,D] x contiguous [H,D,O] into non-aliasing contiguous output[T,H,O].",
        solution="atb",
        binding="torch.ops.xllm_ops",
        priority=10,
        contract=EINSUM_OUT_CONTRACT,
    )(torch.ops.xllm_ops.atb_matmul_ein_sum_out)

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Package-owned ATB MLA bindings, loaded before model import and capture."""

from __future__ import annotations

import torch

from xllm_kernel.ops.gemm import MATMUL_EIN_SUM_CONTRACT, MATMUL_EIN_SUM_OUT_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel import _npu_binding  # noqa: F401

    for _contract in (MATMUL_EIN_SUM_CONTRACT, MATMUL_EIN_SUM_OUT_CONTRACT):
        register_kernel(
            name="npu.atb." + _contract.operation,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="BF16 ND NPU [T,H,D] x contiguous [H,D,O]; optional contiguous nonoverlapping out.",
            solution="atb",
            binding="atb",
            contract=_contract,
        )(getattr(torch.ops.xllm_kernel, _contract.operation).default)

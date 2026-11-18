# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Select the paged KV provider for the worker's Ascend generation."""

from __future__ import annotations

import torch

from xllm_kernel.ops.kv_cache import RESHAPE_PAGED_CACHE_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel import _npu_binding  # noqa: F401

    if current_platform.get_ascend_soc_generation() == "a5":
        _solution = "aclnn"
        _binding = torch.ops.xllm_kernel.reshape_paged_cache_a5.default
        _domain = "Contiguous ND FP16/BF16; matching K and V dimensions; int32 slot mapping."
    else:
        _solution = "atb"
        _binding = torch.ops.xllm_kernel.reshape_paged_cache_atb.default
        _domain = "ND or FRACTAL_NZ KV, including separate K/V widths and shared MLA cache; int32 slot mapping."
    register_kernel(
        name=f"npu.{_solution}.reshape_paged_cache",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain=_domain,
        solution=_solution,
        binding=_solution,
        contract=RESHAPE_PAGED_CACHE_CONTRACT,
    )(_binding)

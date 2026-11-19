# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the package-owned AscendC in-place partial rotary operator."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.positional_encoding import INPLACE_PARTIAL_ROTARY_MUL_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_inplace_partial_rotary_mul()
register_kernel(
    name="npu.ascendc.inplace_partial_rotary_mul",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="4D FP16, BF16 or FP32 NPU ND tensors with an even-width rotary slice.",
    solution="ascendc",
    binding="aclnn",
    contract=INPLACE_PARTIAL_ROTARY_MUL_CONTRACT,
)(torch.ops.xllm_kernel.inplace_partial_rotary_mul.default)

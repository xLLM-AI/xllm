# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the packaged MoE providers at worker startup."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.moe import (
    DEQUANT_SWIGLU_QUANT,
    MOE_GATING_TOP_K_HASH,
    MOE_GROUPED_MATMUL_SWIGLU_QUANT,
    MOE_INIT_ROUTING_V3,
)
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract, register_kernel

HAS_FUSED_MOE_GMM1 = torch.ops.xllm_kernel.prepare_ascendc_moe()


def _register(contract: OpContract, input_domain: str) -> None:
    name = contract.operation
    register_kernel(
        name=f"npu.ascendc.{name}",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain=input_domain,
        solution="ascendc",
        binding="aclnn",
        contract=contract,
    )(getattr(torch.ops.xllm_kernel, name).default)


_register(DEQUANT_SWIGLU_QUANT, "NPU gate/up tensors with quantization scales for the selected SwiGLU mode.")
_register(MOE_GATING_TOP_K_HASH, "NPU floating-point logits and optional bias or token-to-expert lookup.")
if HAS_FUSED_MOE_GMM1:
    _register(MOE_INIT_ROUTING_V3, "NPU token rows and INT32 expert IDs, with a half-open active expert range.")
    _register(MOE_GROUPED_MATMUL_SWIGLU_QUANT, "NPU W8A8 expert inputs, NZ weights and V3 expert/count pairs.")

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind CANN and TorchNPU router and expert operations."""

import torch
import torch_npu

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.moe import ACLNN_OPERATIONS, TORCH_OPERATIONS
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

HAS_MEGA_MOE = torch.ops.xllm_kernel.has_mega_moe()


def _moe_gating_top_k_softmax(x: torch.Tensor, topk: int, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(x, k=topk)
    if normalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights.contiguous(), ids.contiguous()


for operation in TORCH_OPERATIONS:
    name = operation.contract.operation
    function = (
        _moe_gating_top_k_softmax if name == "moe_gating_top_k_softmax" else getattr(torch_npu, f"npu_{name}", None)
    )
    if function is not None:
        register_kernel(
            name=f"npu.torch.{name}",
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="NPU router logits, expert IDs, token rows, or EP communication buffers.",
            solution="torch",
            binding="torch_npu",
            contract=operation.contract,
        )(function)

if HAS_MEGA_MOE:
    register_kernel(
        name="npu.aclnn.mega_moe",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="MegaMoE EP context, BF16 token rows and BF16 or packed W8A8 weights with optional active mask.",
        solution="aclnn",
        binding="aclnn",
        contract=ACLNN_OPERATIONS[0].contract,
    )(torch.ops.xllm_kernel.mega_moe.default)

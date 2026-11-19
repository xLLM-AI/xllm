# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the package-owned HyperConnection implementations."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.hyper_connection import HC_POST_CONTRACT, HC_PRE_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_ascendc_hyper_connection()
register_kernel(
    name="npu.ascendc.hc_pre",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Nonempty NPU [T, HC, D] or [B, S, HC, D] residual with matching projection and gating parameters.",
    solution="ascendc",
    binding="aclnn",
    contract=HC_PRE_CONTRACT,
)(torch.ops.xllm_kernel.hc_pre.default)
register_kernel(
    name="npu.ascendc.hc_post",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Nonempty NPU [B, S, D] output and [B, S, HC, D] residual with matching post and comb.",
    solution="ascendc",
    binding="aclnn",
    contract=HC_POST_CONTRACT,
)(torch.ops.xllm_kernel.hc_post.default)

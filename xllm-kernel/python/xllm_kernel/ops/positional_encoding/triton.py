# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Register the Triton split, RMSNorm and RoPE implementation."""

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel.ops.positional_encoding import FUSED_QK_NORM_ROPE_CONTRACT
    from xllm_kernel.ops.positional_encoding._triton._rope import split_qkv_rmsnorm_rope

    register_kernel(
        name="npu.triton.fused_qk_norm_rope",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Packed NPU QKV and half-style rotary cache for per-head RMSNorm.",
        solution="triton",
        binding="python",
        contract=FUSED_QK_NORM_ROPE_CONTRACT,
    )(split_qkv_rmsnorm_rope)

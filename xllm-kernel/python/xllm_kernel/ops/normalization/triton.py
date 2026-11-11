# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Triton normalization implementations registered during startup."""

from __future__ import annotations

from importlib.util import find_spec

import torch

from xllm_kernel.ops.normalization import (
    FUSED_EH_NORM_CONTRACT,
    FUSED_QK_NORM_ROPE_CONTRACT,
    L2_NORM_CONTRACT,
    RMS_NORM_SIGMOID_GATED_CONTRACT,
    TRITON_JIT_REQUIRES,
)
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import register_kernel

if Platform.is_npu():
    # JIT-only preparation does not require the host's compiled L2 schema;
    # compiled L2 preparation does not require the optional Python JIT SDKs.
    if hasattr(torch.ops.xllm_ops, "l2_norm"):
        _l2_norm = torch.ops.xllm_ops.l2_norm

        @register_kernel(
            name="npu.triton.l2_norm.aot",
            contract=L2_NORM_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Nonempty BF16 NPU tensor; L2 normalization over the last dimension with epsilon added before rsqrt.",
            solution="triton",
            variant="aot",
            binding="xllm_ops",
            priority=10,
        )
        def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Normalize the last dimension of ``value`` to unit L2 norm.

            Args:
                value: Tensor whose last dimension is normalized.
                eps: Added to the squared norm before the reciprocal square root.

            Returns:
                A tensor with the shape and dtype of ``value``.
            """
            return _l2_norm(value, eps)

    if all(find_spec(dependency) is not None for dependency in TRITON_JIT_REQUIRES):
        from xllm_kernel.ops.normalization._triton.eh_norm import fused_eh_norm as _fused_eh_norm
        from xllm_kernel.ops.normalization._triton.rms_norm import rms_norm_sigmoid_gated as _rms_norm_sigmoid_gated
        from xllm_kernel.ops.normalization._triton.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope as _split_qkv_rmsnorm_rope,
        )

        @register_kernel(
            name="npu.triton.rms_norm_sigmoid_gated",
            contract=RMS_NORM_SIGMOID_GATED_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Matching NPU value/gate shape and dtype, matching 1D weight, feature row below 64 KiB.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def rms_norm_sigmoid_gated(
            value: torch.Tensor,
            gate: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            return _rms_norm_sigmoid_gated(value, gate, weight, eps)

        @register_kernel(
            name="npu.triton.fused_eh_norm",
            contract=FUSED_EH_NORM_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Matching NPU embedding/carried shape and dtype; final feature row below 64 KiB; concatenate normalized inputs.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def fused_eh_norm(
            embed: torch.Tensor,
            carried: torch.Tensor,
            enorm_weight: torch.Tensor,
            hnorm_weight: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            return _fused_eh_norm(embed, carried, enorm_weight, hnorm_weight, eps)

        @register_kernel(
            name="npu.triton.fused_qk_norm_rope",
            contract=FUSED_QK_NORM_ROPE_CONTRACT,
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="Packed NPU QKV, per-head Q/K weights and split-layout position-indexed cosine/sine table.",
            solution="triton",
            binding="triton.jit",
            priority=10,
        )
        def fused_qk_norm_rope(
            qkv: torch.Tensor,
            *,
            num_heads_q: int,
            num_heads_k: int,
            num_heads_v: int,
            head_dim: int,
            eps: float,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            position_ids: torch.Tensor,
            cos: torch.Tensor | None = None,
            sin: torch.Tensor | None = None,
            interleaved: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply per-head QK-RMSNorm and RoPE to a packed QKV projection.

            Args:
                qkv: Packed projection of shape ``[num_tokens, q_size + 2 * kv_size]``.
                num_heads_q: Query heads on this rank.
                num_heads_k: Key heads on this rank.
                num_heads_v: Value heads on this rank; derived from ``qkv`` here.
                head_dim: Size of one head.
                eps: RMSNorm epsilon shared by the query and key norms.
                q_weight: Query RMSNorm weight of shape ``[head_dim]``.
                k_weight: Key RMSNorm weight of shape ``[head_dim]``.
                cos_sin_cache: Interleaved cosine/sine table indexed by position.
                position_ids: Position of every token.
                cos: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
                sin: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
                interleaved: Unused on NPU; the kernel always uses the split layout.

            Returns:
                Query, key and value, each of shape ``[num_tokens, heads * head_dim]``.
                They may be views into one fused buffer, so a caller must not write
                into them in place.
            """
            del num_heads_v, cos, sin, interleaved
            return _split_qkv_rmsnorm_rope(
                qkv,
                cos_sin_cache,
                position_ids,
                q_weight,
                k_weight,
                num_heads_q * head_dim,
                num_heads_k * head_dim,
                head_dim,
                eps,
            )

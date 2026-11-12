# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Startup-registered bindings for compiled TileLang normalization kernels."""

from __future__ import annotations

import torch

from xllm_kernel.ops.normalization import (
    BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_CONTRACT,
    HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_CONTRACT,
    SPLIT_QKV_RMSNORM_MROPE_CONTRACT,
)
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    _split_qkv_rmsnorm_mrope = torch.ops.xllm_ops.split_qkv_rmsnorm_mrope
    _has_split_qkv_rmsnorm_mrope_specialization = torch.ops.xllm_ops.has_split_qkv_rmsnorm_mrope_specialization
    _build_split_qkv_rmsnorm_mrope_gather_pattern = torch.ops.xllm_ops.build_split_qkv_rmsnorm_mrope_gather_pattern

    @register_kernel(
        name="npu.tilelang.split_qkv_rmsnorm_mrope",
        contract=SPLIT_QKV_RMSNORM_MROPE_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 NPU Q/G/K/V packing; compiled Q/K-head and head-size specializations; merged mRoPE table.",
        solution="tilelang",
        binding="xllm_ops",
        priority=10,
    )
    def split_qkv_rmsnorm_mrope(
        qkvg: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        gather_pattern: torch.Tensor,
        eps: float,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split Q/G/K/V and apply fused Q/K RMSNorm plus mRoPE."""
        return _split_qkv_rmsnorm_mrope(
            qkvg,
            q_weight,
            k_weight,
            cos_sin,
            gather_pattern,
            eps,
            num_q_heads,
            num_kv_heads,
            head_size,
        )

    @register_kernel(
        name="npu.tilelang.has_split_qkv_rmsnorm_mrope_specialization",
        contract=HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Query compiled Q/K-head and head-size specializations without executing a device kernel.",
        solution="tilelang",
        binding="xllm_ops",
        priority=10,
    )
    def has_split_qkv_rmsnorm_mrope_specialization(
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> bool:
        """Return whether the AOT fused QKV specialization is available."""
        return _has_split_qkv_rmsnorm_mrope_specialization(
            num_q_heads,
            num_kv_heads,
            head_size,
        )

    @register_kernel(
        name="npu.tilelang.build_split_qkv_rmsnorm_mrope_gather_pattern",
        contract=BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Construct persistent mRoPE byte offsets for the compiled fused-QKV kernel.",
        solution="tilelang",
        binding="xllm_ops",
        priority=10,
    )
    def build_split_qkv_rmsnorm_mrope_gather_pattern(
        rope_dim: int,
        mrope_section: list[int],
        is_interleaved: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the fused kernel's persistent mRoPE byte-offset table."""
        return _build_split_qkv_rmsnorm_mrope_gather_pattern(
            rope_dim,
            mrope_section,
            is_interleaved,
            device,
        )

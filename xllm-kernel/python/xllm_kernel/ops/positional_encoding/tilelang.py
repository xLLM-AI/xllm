# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""TileLang runtime binding for the fused Q/K mRoPE projection."""

from functools import cache
from typing import TYPE_CHECKING

import tilelang
import torch

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    from tilelang.jit import JITKernel

if current_platform.is_npu():
    from xllm_kernel.ops._tilelang import VECTOR_CORES
    from xllm_kernel.ops.positional_encoding import SPLIT_QKV_RMSNORM_MROPE_CONTRACT
    from xllm_kernel.ops.positional_encoding._tilelang._mrope_geometry import COMPILE_MAX_TOKENS
    from xllm_kernel.ops.positional_encoding._tilelang._mrope_tilelang import (
        KERNEL_PASS_CONFIGS,
        build_split_qkv_rmsnorm_mrope_kernel,
    )

    @cache
    def prepare_mrope(num_q_heads: int, num_kv_heads: int, head_size: int) -> "JITKernel":
        return tilelang.compile(
            build_split_qkv_rmsnorm_mrope_kernel(
                head_size=head_size,
                rope_dim=64,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                vec_core_num=VECTOR_CORES,
                dynamic_launch=True,
            ),
            pass_configs=KERNEL_PASS_CONFIGS,
        )

    @register_kernel(
        name="npu.tilelang.split_qkv_rmsnorm_mrope",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 Q|G|K|V with 256-wide heads and 64 rotary channels.",
        solution="tilelang",
        binding="python",
        contract=SPLIT_QKV_RMSNORM_MROPE_CONTRACT,
    )
    def _split_qkv_rmsnorm_mrope(
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
        device = qkvg.device
        if device.type != "npu" or any(
            tensor.device != device or tensor.dtype != torch.bfloat16 for tensor in (qkvg, q_weight, k_weight, cos_sin)
        ):
            raise ValueError("fused QKV mRoPE requires BF16 tensors on one NPU")
        if gather_pattern.device != device or gather_pattern.dtype != torch.uint32 or gather_pattern.shape != (256,):
            raise ValueError("fused QKV mRoPE requires the 256-entry uint32 gather pattern on the same NPU")
        if qkvg.ndim != 2:
            raise ValueError("fused QKV mRoPE expects a Q|G|K|V matrix")
        num_tokens, width = qkvg.shape
        if not 0 < num_tokens <= COMPILE_MAX_TOKENS or eps <= 0:
            raise ValueError("fused QKV mRoPE requires positive epsilon and a supported token count")
        if width != 2 * (num_q_heads + num_kv_heads) * head_size:
            raise ValueError("fused QKV mRoPE input width does not match its head geometry")
        if q_weight.shape != (head_size,) or k_weight.shape != (head_size,) or cos_sin.shape != (num_tokens, 192):
            raise ValueError("fused QKV mRoPE expects per-head weights and three packed rotary tables")

        with torch.npu.device(device):
            kernel = prepare_mrope(num_q_heads, num_kv_heads, head_size)
            q = qkvg.new_empty((num_tokens, num_q_heads, head_size))
            k = qkvg.new_empty((num_tokens, num_kv_heads, head_size))
            v = torch.empty_like(k)
            gate = torch.empty_like(q)
            kernel(
                qkvg.contiguous(),
                q_weight.contiguous(),
                k_weight.contiguous(),
                cos_sin.contiguous(),
                gather_pattern.contiguous(),
                q,
                k,
                v,
                gate,
                num_tokens,
                eps,
            )
        return q, k, v, gate

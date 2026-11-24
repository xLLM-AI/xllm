# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""TileLang adapters for GDN gating and indexed single-token convolution."""

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
    from xllm_kernel.ops.attention.gdn import CAUSAL_CONV1D_DECODE_CONTRACT, FUSED_GDN_GATING_CONTRACT
    from xllm_kernel.ops.attention.gdn._tilelang._causal_conv1d_decode import DIM_PER_CORE, _build_decode_kernel_jit
    from xllm_kernel.ops.attention.gdn._tilelang._fused_gdn_gating import (
        GDN_GATING_PASS_CONFIGS,
        build_fused_gdn_gating_kernel,
    )

    @cache
    def _prepare_gdn_gating(num_batches: int, num_heads: int) -> "JITKernel":
        return tilelang.compile(
            build_fused_gdn_gating_kernel(
                batch_size=num_batches,
                compile_max_batch=num_batches,
                num_heads=num_heads,
                vec_core_num=VECTOR_CORES,
            ),
            pass_configs=GDN_GATING_PASS_CONFIGS,
        )

    @register_kernel(
        name="npu.tilelang.fused_gdn_gating",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="BF16 gate projections with per-head FP32 decay and bias.",
        solution="tilelang",
        binding="python",
        contract=FUSED_GDN_GATING_CONTRACT,
    )
    def _fused_gdn_gating(
        a_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_batches, num_heads = a.shape
        g_out = torch.empty((1, num_batches, num_heads), dtype=torch.float32, device=a.device)
        beta_out = a.new_empty((1, num_batches, num_heads))
        with torch.npu.device(a.device):
            kernel = _prepare_gdn_gating(num_batches, num_heads)
            kernel(
                a_log.to(torch.float32).contiguous(),
                a.contiguous(),
                b.contiguous(),
                dt_bias.to(torch.float32).contiguous(),
                g_out.squeeze(0),
                beta_out.squeeze(0),
                num_batches,
                1.0,  # softplus_beta
                20.0,  # softplus_threshold
            )
        return g_out, beta_out

    @register_kernel(
        name="npu.tilelang.causal_conv1d_decode",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Channel-last 1-token convolution with device state-slot indices.",
        solution="tilelang",
        binding="python",
        contract=CAUSAL_CONV1D_DECODE_CONTRACT,
    )
    def _causal_conv1d_decode(
        value: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        state_indices: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        _, dim = value.shape
        dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
        with torch.npu.device(value.device):
            kernel = _build_decode_kernel_jit(
                width=weight.size(0),
                dim_chunks=dim_chunks,
                dim_per_core=DIM_PER_CORE,
                dtype_str=str(value.dtype).removeprefix("torch."),
                has_silu=True,
            )
            indices = state_indices.to(dtype=torch.int32).contiguous()
            initial_state_mode = torch.ones_like(indices)
            if bias is None:
                bias = value.new_zeros(dim)
            return kernel(value.contiguous(), weight, conv_state, indices, indices, initial_state_mode, bias)

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Launch the DCP remap with the package-owned TileLang kernel."""

import tilelang
import torch

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel.ops._tilelang import VECTOR_CORES
    from xllm_kernel.ops.attention.dsa import (
        SFA_DCP_REMAP_MAX_TOKENS,
        SFA_DCP_REMAP_OUT_CONTRACT,
        SFA_DCP_REMAP_TOPK,
    )
    from xllm_kernel.ops.attention.dsa._sfa_dcp_remap import REMAP_PASS_CONFIGS, build_remap_topk_kernel

    _KERNEL = tilelang.compile(
        build_remap_topk_kernel(vec_core_num=VECTOR_CORES, topk=SFA_DCP_REMAP_TOPK),
        pass_configs=REMAP_PASS_CONFIGS,
    )

    @register_kernel(
        name="npu.tilelang.sfa_dcp_remap_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Contiguous int32 top-k rows, caller-owned output and index scratch.",
        solution="tilelang",
        binding="python",
        contract=SFA_DCP_REMAP_OUT_CONTRACT,
    )
    def _sfa_dcp_remap_out(
        topk_indices: torch.Tensor,
        physical_block_size: int,
        shard_size: int,
        shard_rank: int,
        out: torch.Tensor,
        idx_scratch: torch.Tensor,
    ) -> torch.Tensor:
        device = topk_indices.device
        if device.type != "npu" or any(
            tensor.device != device or tensor.dtype != torch.int32 or not tensor.is_contiguous()
            for tensor in (topk_indices, out, idx_scratch)
        ):
            raise ValueError("DCP remap requires contiguous int32 buffers on one NPU")
        if topk_indices.dim() == 0 or topk_indices.size(-1) != SFA_DCP_REMAP_TOPK:
            raise ValueError(f"DCP remap requires top-k width {SFA_DCP_REMAP_TOPK}")
        num_tokens = topk_indices.numel() // SFA_DCP_REMAP_TOPK
        if not 0 < num_tokens <= SFA_DCP_REMAP_MAX_TOKENS:
            raise ValueError(f"DCP remap requires 1 to {SFA_DCP_REMAP_MAX_TOKENS} tokens")
        if physical_block_size <= 0 or shard_size <= 0 or not 0 <= shard_rank < shard_size:
            raise ValueError("invalid DCP page size or shard")
        if out.shape != topk_indices.shape or idx_scratch.numel() < topk_indices.numel():
            raise ValueError("DCP remap output and scratch must cover every top-k index")

        scratch = idx_scratch.view(-1)
        with torch.npu.device(device):
            _KERNEL(
                topk_indices.view(-1),
                out.view(-1),
                scratch,
                scratch.view(torch.uint32),
                num_tokens,
                physical_block_size,
                shard_size,
                shard_rank,
            )
        return out

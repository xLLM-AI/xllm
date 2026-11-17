# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Triton Ascend normalization kernels and Python launchers."""

import torch

from xllm_kernel.platform import PlatformEnum, current_platform

if current_platform.is_npu():
    import triton
    import triton.language as tl

    from xllm_kernel.ops.normalization import L2_NORM_CONTRACT
    from xllm_kernel.registry import register_kernel

    _BLOCK_N = 128
    _MBLOCK = 69
    _DEVICE = torch.npu.current_device()
    _NUM_VECTOR_CORES = triton.runtime.driver.active.utils.get_device_properties(_DEVICE)["num_vectorcore"]
    if _NUM_VECTOR_CORES <= 0:
        raise RuntimeError("Failed to detect NPU vector cores")

    @triton.jit(do_not_specialize=["X", "Y", "eps", "M", "N", "NUM_ROW_CHUNKS", "NUM_COL_CHUNKS"])
    def _l2_norm_kernel(
        X: tl.tensor,
        Y: tl.tensor,
        eps: tl.tensor,
        M: tl.tensor,
        N: tl.tensor,
        NUM_ROW_CHUNKS: tl.tensor,
        NUM_COL_CHUNKS: tl.tensor,
        BLOCK_N: tl.constexpr,
        MBLOCK: tl.constexpr,
    ) -> None:
        row_offsets = tl.arange(0, MBLOCK)[:, None]
        col_offsets = tl.arange(0, BLOCK_N)[None, :]
        base_row = tl.program_id(0) * (NUM_ROW_CHUNKS * MBLOCK)

        for row_chunk in tl.range(0, NUM_ROW_CHUNKS):
            row_idx = base_row + row_chunk * MBLOCK + row_offsets
            row_mask = row_idx < M
            square_sum = tl.zeros((MBLOCK, 1), dtype=tl.float32)

            for col_chunk in tl.range(0, NUM_COL_CHUNKS):
                col_idx = col_chunk * BLOCK_N + col_offsets
                mask = row_mask & (col_idx < N)
                xs = tl.load(X + row_idx * N + col_idx, mask=mask, other=0.0).to(tl.float32)
                square_sum += tl.sum(xs * xs, 1)[:, None]

            rsqrt = tl.rsqrt(square_sum + eps)
            for col_chunk in tl.range(0, NUM_COL_CHUNKS):
                col_idx = col_chunk * BLOCK_N + col_offsets
                mask = row_mask & (col_idx < N)
                xs = tl.load(X + row_idx * N + col_idx, mask=mask, other=0.0).to(tl.float32)
                tl.store(Y + row_idx * N + col_idx, xs * rsqrt, mask=mask)

    # Compile and load one generic BF16 kernel before ACLGraph capture.
    _l2_norm_kernel.warmup(
        torch.bfloat16,
        torch.bfloat16,
        1e-6,
        1,
        1,
        1,
        1,
        BLOCK_N=_BLOCK_N,
        MBLOCK=_MBLOCK,
        grid=(1,),
    )._init_handles()

    @register_kernel(
        name="npu.triton.l2_norm",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Nonempty BF16 NPU tensor of rank >= 1; arbitrary strides; FP32 reduction and epsilon.",
        solution="triton",
        variant="jit",
        binding="python",
        contract=L2_NORM_CONTRACT,
    )
    def l2_norm(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if input.device.type != "npu" or input.device.index != torch.npu.current_device():
            raise ValueError("L2Norm input must be on the current NPU device")
        if input.dtype != torch.bfloat16:
            raise ValueError("L2Norm requires BF16 input")
        if input.ndim == 0 or input.numel() == 0:
            raise ValueError("L2Norm requires nonempty input of rank >= 1")
        cols = input.shape[-1]
        rows = input.numel() // cols
        if max(rows, cols) > 2**31 - 1:
            raise ValueError("L2Norm dimensions must fit in int32")
        contiguous = input.contiguous()
        output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        rows_per_core = triton.cdiv(rows, _NUM_VECTOR_CORES)
        _l2_norm_kernel[(_NUM_VECTOR_CORES,)](
            contiguous,
            output,
            float(eps),
            rows,
            cols,
            triton.cdiv(rows_per_core, _MBLOCK),
            triton.cdiv(cols, _BLOCK_N),
            BLOCK_N=_BLOCK_N,
            MBLOCK=_MBLOCK,
        )
        return output

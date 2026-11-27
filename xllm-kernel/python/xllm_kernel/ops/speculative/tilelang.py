# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Launch shared verification kernels without loading the xLLM engine."""

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
    from xllm_kernel.ops.speculative import GREEDY_PREFIX_VERIFY
    from xllm_kernel.ops.speculative._tilelang._greedy_prefix_verify import (
        GREEDY_PREFIX_VERIFY_PASS_CONFIGS,
        build_greedy_prefix_verify_kernel,
        select_greedy_prefix_verify_task_count,
    )

    @cache
    def _greedy_kernel(tasks: int, draft_bits: int, target_bits: int, bonus_bits: int) -> "JITKernel":
        return tilelang.compile(
            build_greedy_prefix_verify_kernel(
                task_count=tasks, draft_bits=draft_bits, target_bits=target_bits, bonus_bits=bonus_bits
            ),
            pass_configs=GREEDY_PREFIX_VERIFY_PASS_CONFIGS,
        )

    def _greedy(draft: torch.Tensor, target: torch.Tensor, bonus: torch.Tensor) -> torch.Tensor:
        int32_max = (1 << 31) - 1
        buffers = []
        for tensor in (draft, target, bonus):
            if (
                tensor.device.type != "npu"
                or tensor.device != draft.device
                or tensor.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("greedy verification requires int32/int64 tensors on one NPU")
            if tensor.dim() != 2 or any(size > int32_max for size in tensor.shape):
                raise ValueError("greedy verification requires rank-two int32-indexable inputs")
            if any(not 0 <= stride <= int32_max for stride in tensor.stride()):
                raise ValueError("greedy verification requires nonnegative int32 strides")
            span = (
                1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
                if tensor.numel()
                else 0
            )
            if span > int32_max:
                raise ValueError("greedy verification storage span exceeds int32")
            # The JIT binding requires contiguous handles; the DSL applies the
            # original strides within these storage views without copying.
            buffers.append(tensor.as_strided((span,), (1,)))
        batch, width = draft.shape
        if (
            target.shape != draft.shape
            or bonus.shape != (batch, 1)
            or width >= int32_max
            or batch * (width + 1) > int32_max
        ):
            raise ValueError("invalid greedy verification shapes")
        output = torch.full((batch, width + 1), -1, dtype=torch.int32, device=draft.device)
        if batch:
            tasks = select_greedy_prefix_verify_task_count(batch, VECTOR_CORES)
            with torch.npu.device(draft.device):
                _greedy_kernel(tasks, draft.element_size() * 8, target.element_size() * 8, bonus.element_size() * 8)(
                    *buffers, output, batch, width, *draft.stride(), *target.stride(), bonus.stride(0)
                )
        return output

    register_kernel(
        name="npu.tilelang.greedy_prefix_verify",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Strided int32/int64 token tensors on one NPU.",
        solution="tilelang",
        binding="python",
        contract=GREEDY_PREFIX_VERIFY.contract,
    )(_greedy)

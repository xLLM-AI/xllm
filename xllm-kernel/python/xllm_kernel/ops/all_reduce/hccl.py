# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_reduce implementations supplied by hccl; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_current_stream() -> Callable[..., Any]:
    from xllm_kernel.backends.collective import prepare_hccl
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("npu_all_reduce")
    group, comm = prepare_hccl()

    def run(value: torch.Tensor) -> torch.Tensor:
        op(value, comm)
        return value

    return run


def load_process_group() -> Callable[..., Any]:
    import torch.distributed as dist

    from xllm_kernel.backends.collective import prepare_hccl

    group, _ = prepare_hccl()

    def run(value: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(value, group=group)
        return value

    return run

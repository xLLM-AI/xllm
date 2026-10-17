# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_to_all_single implementations supplied by hccl; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_process_group() -> Callable[..., Any]:
    import torch.distributed as dist

    from xllm_kernel.backends.collective import prepare_hccl

    group, _ = prepare_hccl()

    def run(value: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        dist.all_to_all_single(output, value, group=group)
        return output

    return run

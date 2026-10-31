# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit torchrun lifecycle and cross-rank validation consensus."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def session(enabled: bool) -> Iterator[dict[str, Any]]:
    if not enabled:
        yield {}
        return
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    if "LOCAL_RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("--distributed requires torchrun with at least two NPU ranks")
    if dist.is_initialized():
        raise RuntimeError("the CLI requires ownership of a new process group")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        yield {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
        }
    finally:
        dist.destroy_process_group()


def agree(passed: bool) -> bool:
    """All ranks either benchmark a case or skip it after a numerical failure."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return passed
    flag = torch.tensor(int(passed), dtype=torch.int32, device="npu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def rank_max_samples(timing: dict[str, Any]) -> list[dict[str, float]] | None:
    """Preserve local timings and additionally report per-repeat maxima across ranks."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    names = tuple(timing["samples"][0])
    samples = torch.tensor(
        [[sample[name] for name in names] for sample in timing["samples"]], dtype=torch.float32, device="npu"
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    return [dict(zip(names, row, strict=True)) for row in samples.cpu().tolist()]

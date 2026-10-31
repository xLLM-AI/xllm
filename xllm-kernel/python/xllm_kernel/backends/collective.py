# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Resolve the default HCCL group before capture; never initialize in a hot call."""

from __future__ import annotations

from typing import Any


def prepare_hccl() -> tuple[Any, int]:
    import torch
    import torch.distributed as dist

    from xllm.python.distributed.npu.hccl import get_hccl_comm

    if not dist.is_initialized() or dist.get_backend() != "hccl":
        raise RuntimeError("initialize a HCCL process group before preparing a collective")
    group = dist.distributed_c10d._get_default_group()
    comm = get_hccl_comm(group, torch.device("npu", torch.npu.current_device()), initialize=True)
    return group, comm

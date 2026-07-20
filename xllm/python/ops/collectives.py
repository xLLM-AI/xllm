from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

_tp_groups = {}
_tp_stores = {}


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
):
    device_key = str(torch.device(device))
    group = _tp_groups.get(device_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"TP group for {device_key} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        return group

    store = dist.TCPStore(
        host,
        port,
        world_size,
        rank == 0,
        timedelta(minutes=5),
        wait_for_workers=False,
    )
    group = dist.ProcessGroupNCCL(
        store,
        rank,
        world_size,
        timedelta(minutes=5),
    )
    _tp_stores[device_key] = store
    _tp_groups[device_key] = group
    return group


def _require_tp_group(x: torch.Tensor):
    group = _tp_groups.get(str(x.device))
    if group is None:
        raise RuntimeError(
            "tensor-parallel collective called before the TP process group "
            f"was initialized for {x.device}"
        )
    return group


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor) -> None:
    group = _require_tp_group(x)
    dist.all_reduce(x, group=group)


@all_reduce_.register_fake
def _(x: torch.Tensor) -> None:
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    group = _require_tp_group(x)
    if group.size() != world_size:
        raise RuntimeError(
            f"TP world-size mismatch: expected {world_size}, "
            f"got {group.size()}"
        )
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)

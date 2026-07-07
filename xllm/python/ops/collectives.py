from __future__ import annotations

import torch

_tp_group = None


def set_tp_group(group) -> None:
    global _tp_group
    _tp_group = group


@torch.library.custom_op("xllm_ops::all_reduce", mutates_args=())
def all_reduce(x: torch.Tensor) -> torch.Tensor:
    if _tp_group is None:
        return x
    out = x.clone()
    torch.distributed.all_reduce(out, group=_tp_group)
    return out


@all_reduce.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    if _tp_group is None:
        return x
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, x, group=_tp_group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)

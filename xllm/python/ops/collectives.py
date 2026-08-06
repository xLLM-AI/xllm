from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

_tp_groups = {}
_tp_stores = {}
_cp_groups = {}
_cp_stores = {}


def _create_process_group(
    host: str, port: int, rank: int, world_size: int, device: str,
    group_id: str,
):
    """Create HCCL or NCCL ProcessGroup depending on device type.

    ``group_id`` names the underlying HCCL communicator. It must be distinct per
    logical group (TP vs CP): torch_npu keys each device's HCCL comm by this id,
    and two comms sharing the default id collide in hcclCommInitRootInfoConfig
    when an orthogonal TP x CP launch brings up both on the same device.
    """
    store = dist.TCPStore(
        host,
        port,
        world_size,
        rank == 0,
        timedelta(minutes=5),
        wait_for_workers=False,
    )
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        group = dist.ProcessGroupNCCL(store, rank, world_size, timedelta(minutes=5))
    else:
        import torch_npu  # noqa: F401
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        options = ProcessGroupHCCL.Options()
        options.group_id = group_id
        group = ProcessGroupHCCL(store, rank, world_size, options)
    return store, group


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

    store, group = _create_process_group(
        host, port, rank, world_size, device, "python_tp_group"
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


def init_cp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
):
    """Initialize the context-parallel (CP) process group for ``device``.

    Mirrors ``init_tp_group`` but keeps a separate group so CP collectives do
    not collide with TP ones. CP is orthogonal to TP: a CP group gathers the
    ranks that hold different sequence shards of the same request.
    """
    device_key = str(torch.device(device))
    group = _cp_groups.get(device_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"CP group for {device_key} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        return group

    store, group = _create_process_group(
        host, port, rank, world_size, device, "python_cp_group"
    )
    _cp_stores[device_key] = store
    _cp_groups[device_key] = group
    return group


def _require_cp_group(x: torch.Tensor):
    group = _cp_groups.get(str(x.device))
    if group is None:
        raise RuntimeError(
            "context-parallel collective called before the CP process group "
            f"was initialized for {x.device}"
        )
    return group


def tp_rank(device) -> int:
    """Rank in the TP group for ``device`` (0 when no TP group exists)."""
    group = _tp_groups.get(str(torch.device(device)))
    return group.rank() if group is not None else 0


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


def cp_rank(device) -> int:
    """Rank in the CP group for ``device`` (0 when no CP group exists)."""
    group = _cp_groups.get(str(torch.device(device)))
    return group.rank() if group is not None else 0


def cp_world_size(device) -> int:
    """Size of the CP group for ``device`` (1 when no CP group exists)."""
    group = _cp_groups.get(str(torch.device(device)))
    return group.size() if group is not None else 1


@torch.library.custom_op("xllm_ops::cp_all_gather", mutates_args=())
def cp_all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    group = _require_cp_group(x)
    if group.size() != world_size:
        raise RuntimeError(
            f"CP world-size mismatch: expected {world_size}, "
            f"got {group.size()}"
        )
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@cp_all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)

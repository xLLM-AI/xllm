# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel process groups and graph-visible collectives.

All-reduce selects its platform implementation once at import time. The graph
operators keep communication nodes in a compiled graph rather than splitting
around them; process-group rendezvous and topology remain shared.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from xllm.python.platform import current_platform

if current_platform.is_npu():
    from xllm.python.distributed.npu import all_reduce_on_current_stream as _all_reduce

    _cuda_collectives = None
elif current_platform.is_cuda() or current_platform.is_dcu() or current_platform.is_ilu():
    from xllm.python.distributed import cuda as _cuda_collectives

    _all_reduce = _cuda_collectives.all_reduce
else:
    _cuda_collectives = None
    _all_reduce = dist.all_reduce

_GROUP_NAMES = frozenset(("tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"))
# ``tp`` and ``moe_tp`` own a contiguous block of global ranks, while ``dp``,
# ``moe_ep``, ``cp`` and ``dcp`` stride across those blocks. Both layouts follow
# from how the caller derives a rank within each group, so a group's full
# membership is determined by its own size and needs no extra information from
# the caller.
# ``cp`` strides by the attention TP size: ranks sharing a (dp, tp) slot but
# holding different sequence shards form one CP group, matching the C++
# compute_cp_group_ranks layout (rank = dp*cp*tp + cp_rank*tp + tp_rank).
# ``dcp`` matches ParallelArgs::kv_split_rank when cp_size == 1:
# rank_in_group = global_rank / (world / dcp_size).
_CONTIGUOUS_GROUPS = frozenset(("tp", "moe_tp", "layerwise"))

_groups = {}
_group_ranks = {}
_stores = {}
_world_topology = None
_world_initialized = False


def _backend_for(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    import torch_npu  # noqa: F401

    return "hccl"


def _shared_store(host: str, port: int, global_rank: int, global_world_size: int) -> dist.Store:
    store_key = (host, port)
    store = _stores.get(store_key)
    if store is None:
        store = dist.TCPStore(
            host,
            port,
            global_world_size,
            global_rank == 0,
            timedelta(minutes=5),
            wait_for_workers=False,
            multi_tenant=True,
        )
        _stores[store_key] = store
    return store


def _exchange_world_topology(
    store: dist.Store,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> list[dict[str, object]]:
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    local = {"hostname": socket.gethostname(), "device_index": device_index}
    key_prefix = "xllm/python_collectives/topology/v1"
    store.set(f"{key_prefix}/{global_rank}", json.dumps(local))
    return [json.loads(store.get(f"{key_prefix}/{rank}").decode("utf-8")) for rank in range(global_world_size)]


def _ensure_world(
    host: str,
    port: int,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> None:
    """Initialize the default process group covering every rank.

    The parallel groups are subgroups of it. Building them with ``new_group``
    instead of constructing communicators directly registers them with c10d,
    which is what lets a group back a symmetric-memory allocation.
    """
    global _world_initialized, _world_topology
    if _world_initialized:
        return
    store = _shared_store(host, port, global_rank, global_world_size)
    dist.init_process_group(
        backend=_backend_for(device),
        store=store,
        rank=global_rank,
        world_size=global_world_size,
        timeout=timedelta(minutes=5),
    )
    _world_topology = _exchange_world_topology(store, device, global_rank, global_world_size)
    _world_initialized = True


def _group_memberships(group_name: str, world_size: int, global_world_size: int) -> list[list[int]]:
    """Every group of this kind, in an order all ranks agree on.

    ``new_group`` is collective over the whole world, so each rank has to create
    all groups of a kind in the same order, not only the one it belongs to.
    """
    if world_size <= 0 or global_world_size % world_size:
        raise ValueError(f"{group_name} size {world_size} does not divide the world size {global_world_size}")
    count = global_world_size // world_size
    if group_name in _CONTIGUOUS_GROUPS:
        return [[index * world_size + offset for offset in range(world_size)] for index in range(count)]
    return [[index + offset * count for offset in range(world_size)] for index in range(count)]


def init_process_group(
    group_name: str,
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
) -> ProcessGroup:
    if group_name not in _GROUP_NAMES:
        raise ValueError(f"unsupported parallel group: {group_name}")
    device_obj = torch.device(device)
    group_key = (group_name, str(device_obj))
    group = _groups.get(group_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"{group_name} group for {group_key[1]} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        return group

    _ensure_world(host, port, device_obj, global_rank, global_world_size)
    backend = _backend_for(device_obj)

    own = None
    own_ranks = None
    memberships = _group_memberships(group_name, world_size, global_world_size)
    for index, ranks in enumerate(memberships):
        candidate = dist.new_group(ranks=ranks, timeout=timedelta(minutes=5), backend=backend)
        if global_rank in ranks:
            own = candidate
            own_index = index
            own_ranks = ranks
    if own is None or own.rank() != rank:
        raise RuntimeError(
            f"derived {group_name} membership disagrees with the caller: global "
            f"rank {global_rank} of {global_world_size} expected rank {rank} of "
            f"{world_size}, got {None if own is None else own.rank()}"
        )
    if own_index != group_index:
        raise RuntimeError(f"derived {group_name} group index {own_index} does not match the caller's {group_index}")

    assert own_ranks is not None
    _groups[group_key] = own
    _group_ranks[group_key] = tuple(own_ranks)
    if _cuda_collectives is not None:
        _cuda_collectives.init_group(own, device_obj, own_ranks, _world_topology)
    return own


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
) -> ProcessGroup:
    return init_process_group(
        "tp",
        host,
        port,
        rank,
        world_size,
        device,
        global_rank,
        global_world_size,
        group_index,
    )


def _require_group(x: torch.Tensor, group_name: str) -> ProcessGroup:
    group = _groups.get((group_name, str(x.device)))
    if group is None:
        raise RuntimeError(f"{group_name} collective called before its process group was initialized for {x.device}")
    return group


def tp_rank(device: torch.device | str) -> int:
    """Rank in the TP group for ``device`` (0 when no TP group exists)."""
    group = _groups.get(("tp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_rank(device: torch.device | str) -> int:
    """Rank in the CP group for ``device`` (0 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_world_size(device: torch.device | str) -> int:
    """Size of the CP group for ``device`` (1 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.size() if group is not None else 1


def layerwise_rank(device: torch.device | str) -> int:
    """Rank in the layerwise split group, or zero when disabled."""
    group = _groups.get(("layerwise", str(torch.device(device))))
    return group.rank() if group is not None else 0


def _native_runtime_op(name: str) -> object | None:
    """Return an embedded C++ collective when running under PyExecutorImpl."""
    try:
        import xllm_runtime
    except ImportError:
        return None
    return getattr(xllm_runtime, name, None)


def tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "tp")


def tp_all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    op = _native_runtime_op("tp_all_gather")
    if op is not None:
        return op(x, dim)
    return all_gather(x, dim, world_size, "tp")


def moe_tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_tp")


def moe_ep_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_ep_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_ep")


def dcp_group(device: torch.device | str) -> ProcessGroup | None:
    return _groups.get(("dcp", str(torch.device(device))))


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    _all_reduce(x, group=group)


@torch.library.custom_op("xllm_ops::broadcast_", mutates_args={"x"})
def broadcast_(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    if not 0 <= src < group.size():
        raise RuntimeError(f"invalid {group_name} source rank {src}")
    ranks = _group_ranks.get((group_name, str(x.device)))
    if ranks is None:
        raise RuntimeError(f"{group_name} group rank map is unavailable")
    dist.broadcast(x, src=ranks[src], group=group)


@broadcast_.register_fake
def _(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    del src, group_name
    return None


@all_reduce_.register_fake
def _(x: torch.Tensor, group_name: str = "tp") -> None:
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)


@torch.library.custom_op("xllm_ops::all_gather_variable", mutates_args=())
def all_gather_variable(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != len(token_counts):
        raise RuntimeError(
            f"{group_name} size mismatch: group has {group.size()} ranks, "
            f"but token_counts has {len(token_counts)} entries"
        )
    if not 0 <= rank < len(token_counts):
        raise RuntimeError(f"invalid {group_name} rank {rank}")

    local_tokens = token_counts[rank]
    if local_tokens < 0 or local_tokens > x.shape[0]:
        raise RuntimeError(f"invalid local token count {local_tokens} for input with {x.shape[0]} rows")
    padded_tokens = max(max(token_counts, default=0), 1)
    padded_shape = list(x.shape)
    padded_shape[0] = padded_tokens
    padded = x.new_zeros(padded_shape)
    if local_tokens:
        padded[:local_tokens].copy_(x[:local_tokens])

    chunks = [torch.empty_like(padded) for _ in token_counts]
    dist.all_gather(chunks, padded, group=group)
    valid_chunks = [chunk[:count] for chunk, count in zip(chunks, token_counts) if count]
    if not valid_chunks:
        empty_shape = list(x.shape)
        empty_shape[0] = 0
        return x.new_empty(empty_shape)
    return torch.cat(valid_chunks, dim=0)


@all_gather_variable.register_fake
def _(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    del rank, group_name
    shape = list(x.shape)
    shape[0] = sum(token_counts)
    return x.new_empty(shape)


def gather_dp_execution_tokens(
    x: torch.Tensor,
    execution_token_counts: Sequence[int],
    rank: int,
) -> tuple[torch.Tensor, int]:
    """Gather the execution rows materialized by every DP rank."""
    counts = [int(count) for count in execution_token_counts]
    if not counts:
        return x, 0
    if any(count <= 0 for count in counts):
        raise RuntimeError(f"DP execution token counts must be positive, got {counts}")
    if not 0 <= rank < len(counts):
        raise RuntimeError(f"invalid DP rank {rank} for token counts {counts}")
    if counts[rank] != x.shape[0]:
        raise RuntimeError(
            "DP execution token count does not match the local tensor: "
            f"rank={rank}, rows={x.shape[0]}, token_counts={counts}"
        )

    if all(count == counts[0] for count in counts):
        gathered = all_gather(
            x,
            dim=0,
            world_size=len(counts),
            group_name="dp",
        )
        return gathered, rank * counts[0]

    gathered = all_gather_variable(x, counts, rank, "dp")
    return gathered, sum(counts[:rank])


__all__ = [
    "init_process_group",
    "init_tp_group",
    "tp_rank",
    "cp_rank",
    "cp_world_size",
    "layerwise_rank",
    "dcp_group",
    "all_reduce_",
    "broadcast_",
    "all_gather",
    "all_gather_variable",
    "gather_dp_execution_tokens",
]

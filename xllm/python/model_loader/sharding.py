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

"""Architecture-neutral tensor-parallel / expert-parallel shard math.

Pure per-rank ``(rank, world_size)`` and head-count arithmetic shared by every
model's ``load_weights``, independent of how the weights are copied in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _validate_gqa_kv_split(n_kv_heads: int, tp_size: int) -> None:
    """Assert GQA K/V heads tile evenly against ``tp_size`` (shared by the KV shard/head helpers)."""
    if n_kv_heads >= tp_size:
        if n_kv_heads % tp_size:
            raise ValueError(f"n_kv_heads {n_kv_heads} not divisible by tp_size {tp_size}")
    elif tp_size % n_kv_heads:
        raise ValueError(f"tp_size {tp_size} not divisible by n_kv_heads {n_kv_heads}")


def _kv_head_replica_shard(n_kv_heads: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(kv_rank, kv_world)`` for GQA K/V projections under TP head replication."""
    _validate_gqa_kv_split(n_kv_heads, tp_size)
    if n_kv_heads >= tp_size:
        return tp_rank, tp_size
    return tp_rank // (tp_size // n_kv_heads), n_kv_heads


def gqa_qkv_shards(
    suffix: str,
    n_kv_heads: int,
    tp_rank: int,
    tp_size: int,
) -> list[tuple[str, int, int]]:
    """``(name, rank, world_size)`` fuse triple for a GQA q/k/v projection.

    ``q`` rides the attention TP split; ``k``/``v`` ride the replicated-kv split
    (:func:`_kv_head_replica_shard`) so the fused qkv tensor is packed per rank. Callers
    pass ``suffix`` (e.g. ``".weight"`` or ``".bias"``) and own the fused report name.
    """
    kv_rank, kv_world = _kv_head_replica_shard(n_kv_heads, tp_rank, tp_size)
    return [
        ("q_proj" + suffix, tp_rank, tp_size),
        ("k_proj" + suffix, kv_rank, kv_world),
        ("v_proj" + suffix, kv_rank, kv_world),
    ]


def gqa_head_split(n_heads: int, n_kv_heads: int, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(num_heads, num_kv_heads)`` for GQA under tensor parallelism.

    ``n_kv_heads < tp_size`` -> K/V heads replicated (one KV head per rank).
    """
    if n_heads % tp_size:
        raise ValueError(f"n_heads {n_heads} not divisible by tp_size {tp_size}")
    _validate_gqa_kv_split(n_kv_heads, tp_size)
    num_heads = n_heads // tp_size
    return num_heads, n_kv_heads // tp_size if n_kv_heads >= tp_size else 1


def shard_tensor(
    t: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
    *,
    name: str = "tensor",
    contiguous: bool = True,
) -> torch.Tensor:
    """Narrow ``t`` to this rank's shard along ``dim`` for a ``world_size``-way split.

    ``contiguous=False`` returns the narrow view; callers whose shard flows
    straight into a ``cat``/``transpose``/pack that re-materializes it should
    pass it to avoid an extra full copy of the checkpoint tensor.
    """
    if world_size <= 0:
        raise ValueError(f"cannot shard {name}: world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"cannot shard {name}: rank {rank} out of range for world_size {world_size}")
    if world_size == 1:
        return t
    if t.size(dim) % world_size:
        raise ValueError(f"cannot shard {name} (dim {dim}, size {t.size(dim)}) across {world_size} ranks")
    chunk_size = t.size(dim) // world_size
    shard = t.narrow(dim, rank * chunk_size, chunk_size)
    return shard.contiguous() if contiguous else shard

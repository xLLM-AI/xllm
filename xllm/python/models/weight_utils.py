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

"""Architecture-neutral checkpoint weight loader shared by model load_weights.

Owns the value-preserving state-dict lookup, TP sharding, and param/buffer copy
mechanics common to every model's ``load_weights``. Quantization- or
architecture-specific packing (e.g. W8A8) extends this base with extra methods;
the model-specific per-layer loop stays in each model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Optional, Protocol, Sequence

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict

    from xllm.python.models.base import PyModelBase


class MoeParallelConfig(Protocol):
    """Structural config contract read by the MoE sharding helpers."""

    ep_size: int
    tp_size: int
    tp_rank: int
    moe_tp_size: int
    moe_tp_rank: int


def kv_replica_shard(n_kv_heads: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(kv_world, kv_rank)`` for GQA K/V projections under TP head replication."""
    if n_kv_heads >= tp_size:
        return tp_size, tp_rank
    if tp_size % n_kv_heads:
        raise ValueError(f"tp_size {tp_size} not divisible by n_kv_heads {n_kv_heads}")
    return n_kv_heads, tp_rank // (tp_size // n_kv_heads)


def gqa_head_split(n_heads: int, n_kv_heads: int, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(num_heads, num_kv_heads)`` for GQA under tensor parallelism.

    ``n_kv_heads < tp_size`` -> K/V heads replicated (one KV head per rank).
    """
    if n_heads % tp_size:
        raise ValueError(f"n_heads {n_heads} not divisible by tp_size {tp_size}")
    num_heads = n_heads // tp_size
    if n_kv_heads >= tp_size:
        if n_kv_heads % tp_size:
            raise ValueError(f"n_kv_heads {n_kv_heads} not divisible by tp_size {tp_size}")
        return num_heads, n_kv_heads // tp_size
    if tp_size % n_kv_heads:
        raise ValueError(f"tp_size {tp_size} not divisible by n_kv_heads {n_kv_heads}")
    return num_heads, 1


def mla_head_split(n_heads: int, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(num_heads, 1)`` for MLA attention (single latent KV head per rank)."""
    if n_heads % tp_size:
        raise ValueError(f"n_heads {n_heads} not divisible by tp_size {tp_size}")
    return n_heads // tp_size, 1


def effective_moe_tp(cfg: MoeParallelConfig) -> int:
    """TP degree for MoE weights: ``moe_tp_size`` under EP, otherwise attention ``tp_size``."""
    return cfg.moe_tp_size if cfg.ep_size > 1 else cfg.tp_size


def moe_shard(cfg: MoeParallelConfig) -> tuple[int, int]:
    """(world, rank) for sharding routed-expert / shared-expert weights."""
    rank = cfg.moe_tp_rank if cfg.ep_size > 1 else cfg.tp_rank
    return effective_moe_tp(cfg), rank


class WeightLoader:
    """Generic checkpoint tensor lookup / TP sharding / param copy."""

    def __init__(
        self,
        model: nn.Module,
        state_dicts: Sequence[StateDict],
        tp_size: int,
        tp_rank: int,
        src_prefixes: Sequence[str] = ("",),
        name_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        # Snapshot params/buffers by name; callers must mutate ``.data`` in
        # place rather than reassigning parameters (e.g. via ``setattr``), or
        # this cache goes stale.
        self._tensors_by_name: dict[str, torch.Tensor] = dict(model.named_parameters())
        self._tensors_by_name.update(model.named_buffers())
        self._state_dicts = state_dicts
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self._src_prefixes = tuple(src_prefixes)
        # Lookup order per requested name: alias list (if mapped) → src prefixes → state dicts.
        self._name_aliases: Mapping[str, Sequence[str]] = name_aliases or {}

    def _resolve(self, name: str) -> Optional[tuple[StateDict, str]]:
        """First present ``(state_dict, resolved_name)``: tries each ``src_prefix``
        prepended, then stripped (bidirectional prefix match)."""
        for alias in self._name_aliases.get(name, (name,)):
            for prefix in self._src_prefixes:
                full = prefix + alias
                for sd in self._state_dicts:
                    if sd.has(full):
                        return sd, full
            for prefix in self._src_prefixes:
                if prefix and alias.startswith(prefix):
                    stripped = alias[len(prefix) :]
                    for sd in self._state_dicts:
                        if sd.has(stripped):
                            return sd, stripped
        return None

    def has(self, name: str) -> bool:
        return self._resolve(name) is not None

    @staticmethod
    def state_dict_has(
        state_dicts: Sequence[StateDict],
        name: str,
        src_prefixes: Sequence[str] = ("",),
    ) -> bool:
        """True if any state dict holds ``name`` under one of ``src_prefixes``.
        Snapshot-free presence probe (no model param walk), for deciding whether
        to create an optional param before constructing the loader."""
        return any(sd.has(prefix + name) for prefix in src_prefixes for sd in state_dicts)

    def load_tensor(self, name: str) -> torch.Tensor:
        resolved = self._resolve(name)
        if resolved is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        sd, full = resolved
        return sd.get_tensor(full)

    def load_shard(self, name: str, dim: int, world: Optional[int] = None, rank: Optional[int] = None) -> torch.Tensor:
        return self.shard(self.load_tensor(name), dim=dim, world=world, rank=rank)

    def shard(
        self,
        t: torch.Tensor,
        dim: int,
        world: Optional[int] = None,
        rank: Optional[int] = None,
        contiguous: bool = True,
    ) -> torch.Tensor:
        world = self.tp_size if world is None else world
        rank = self.tp_rank if rank is None else rank
        if world <= 1:
            return t
        if t.size(dim) % world:
            raise ValueError(f"cannot shard dim {dim} of size {t.size(dim)} across {world} ranks")
        chunk_size = t.size(dim) // world
        piece = t.narrow(dim, rank * chunk_size, chunk_size)
        return piece.contiguous() if contiguous else piece

    def copy_in(self, param_name: str, tensor: torch.Tensor) -> None:
        param = self._tensors_by_name.get(param_name)
        if param is None:
            raise KeyError(f"no parameter/buffer named {param_name}")
        # copy_ streams the H2D transfer and dtype cast in one shot;
        # a prior tensor.to(device=...) would materialize a full device
        # temporary and spike peak HBM on stacked MoE experts.
        param.data.copy_(tensor)

    def copy_replicated(self, name: str) -> None:
        """Load ``name`` from checkpoint (no shard) and copy into the same-named param."""
        self.copy_in(name, self.load_tensor(name))

    def copy_shard(self, name: str, dim: int) -> None:
        """Load ``name``, shard on ``dim`` by the loader's TP, and copy into the same-named param."""
        self.copy_in(name, self.load_shard(name, dim))

    def pack_gate_up(
        self,
        prefix: str,
        suffix: str = "weight",
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        gate = self.load_shard(prefix + "gate_proj." + suffix, 0, world=world, rank=rank)
        up = self.load_shard(prefix + "up_proj." + suffix, 0, world=world, rank=rank)
        return torch.cat([gate, up], dim=0)

    def load_gated_mlp(self, dst_prefix: str, src_prefix: str) -> None:
        self.copy_in(dst_prefix + "gate_up_proj.weight", self.pack_gate_up(src_prefix))
        self.copy_in(dst_prefix + "down_proj.weight", self.load_shard(src_prefix + "down_proj.weight", 1))


class W8A8WeightLoader(WeightLoader):
    """W8A8 projection/MLP weight packing on top of the generic WeightLoader."""

    def load_w8a8_projection(self, prefix: str, proj: str, shard_dims: Optional[dict[str, int]] = None) -> None:
        """Load one W8A8 projection (weight + 4 quant tensors), sharding suffixes named in ``shard_dims``."""
        dims = shard_dims or {}
        for suffix in ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset"):
            t = self.load_tensor(prefix + proj + "." + suffix)
            dim = dims.get(suffix)
            if dim is not None:
                t = self.shard(t, dim=dim)
            self.copy_in(prefix + proj + "." + suffix, t)

    def load_fused_w8a8_projection(
        self,
        prefix: str,
        target_proj: str,
        source_projs: tuple[str, ...],
    ) -> None:
        """Concat W8A8 ``source_projs`` into one fused ``target_proj``; asserts shared input scale/offset."""
        for suffix in ("weight", "deq_scale", "quant_bias"):
            tensors = [self.load_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            self.copy_in(prefix + target_proj + "." + suffix, torch.cat(tensors, dim=0))

        for suffix in ("input_scale", "input_offset"):
            tensors = [self.load_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            reference = tensors[0]
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                names = ", ".join(source_projs)
                raise ValueError(f"{prefix}{names} must share {suffix} for fused W8A8")
            self.copy_in(prefix + target_proj + "." + suffix, reference)

    def assert_symmetric_int8(self, prefix: str, projs: Sequence[str]) -> None:
        """Assert ``weight_offset`` is all-zero for each ``proj`` under ``prefix``."""
        for proj in projs:
            if self.load_tensor(prefix + proj + ".weight_offset").any():
                raise ValueError(f"{prefix}{proj} requires zero offset (symmetric int8)")

    def load_w8a8_down(
        self,
        prefix: str,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """down_proj (weight sharded on dim 1, replicated weight_scale)."""
        return (
            self.load_shard(prefix + "down_proj.weight", 1, world=world, rank=rank),
            self.load_tensor(prefix + "down_proj.weight_scale"),
        )

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        """Fill fused gate_up + down projections of one dense/shared W8A8 MLP block."""
        for suffix in ("weight", "weight_scale", "weight_offset"):
            self.copy_in(
                prefix + "gate_up_proj." + suffix,
                self.pack_gate_up(prefix, suffix, world=world, rank=rank),
            )
        dw, ds = self.load_w8a8_down(prefix, world=world, rank=rank)
        self.copy_in(prefix + "down_proj.weight", dw)
        self.copy_in(prefix + "down_proj.weight_scale", ds)
        # W8A8DynamicLinear.process_weights_after_loading asserts
        # weight_offset==0 on the down_proj buffer, which stays torch.empty
        # (garbage) unless we copy the checkpoint's offset in. Dense/shared
        # MLPs carry the offset just like gate_up_proj.
        self.copy_in(
            prefix + "down_proj.weight_offset",
            self.load_tensor(prefix + "down_proj.weight_offset"),
        )


def load_own_weight(
    model: PyModelBase,
    state_dicts: list,
    tp_rank: int,
    tp_size: int,
    weight_name: str,
    attr: str,
    make_layer: Callable[[], nn.Module],
    shard_dim: int,
) -> WeightLoader:
    """Load a draft-owned weight when the checkpoint ships one, else leave the
    attribute None so the C++ bridge shares the target's. Returns a loader the
    caller can keep loading from."""
    present = WeightLoader.state_dict_has(state_dicts, weight_name, ("", "model."))
    if present:
        setattr(model, attr, make_layer())
    # Built after the optional setattr so the new param is in its snapshot.
    loader = WeightLoader(model, state_dicts, tp_size, tp_rank, src_prefixes=("", "model."))
    if present:
        loader.copy_shard(weight_name, dim=shard_dim)
    return loader


def maybe_load_own_lm_head(
    model: PyModelBase,
    state_dicts: list,
    tp_rank: int,
    tp_size: int,
) -> WeightLoader:
    """Load the draft's own lm_head when the checkpoint ships one; otherwise
    leave it None so the C++ bridge shares the target's. Returns a loader the
    caller can keep loading from."""
    # Imported lazily so this module stays torch-only at import time (the layers
    # package needs the C++ runtime bootstrap; load_weights runs after it).
    from xllm.python.layers import ColumnParallelLinear

    cfg = model.cfg
    return load_own_weight(
        model,
        state_dicts,
        tp_rank,
        tp_size,
        "lm_head.weight",
        "lm_head",
        lambda: ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size // tp_size,
            tp_size,
            gather_output=True,
            dtype=model.dtype,
            device=model.device,
        ),
        shard_dim=0,
    )

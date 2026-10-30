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

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import (
        AttentionBackend,
        AttentionMetadata,
        ExecutionMetadataSource,
        LayerCache,
    )
    from xllm.python.attention.npu_paged_attention import PagedAttentionGraphState
    from xllm.python.model_executor.cp_utils import CpContext
    from xllm.python.model_executor.input_batch import InputBatch


class ExecutionMetadataBuilder(Protocol):
    metadata_type: type[object]

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None: ...

    def build(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> object: ...

    def allocate_persistent(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> object: ...

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: ExecutionMetadataSource
    ) -> None: ...


class LayerSynchronizer(Protocol):
    """Records a per-layer completion event for the PD KV-cache transfer thread.

    Implemented in C++ (``NPULayerSynchronizerImpl``) and passed in from the
    executor; the model forward calls ``record_event`` after each layer so the
    transfer thread can push that layer's KV cache without waiting for the whole
    forward to finish.
    """

    def record_event(self, layer_id: int) -> bool:
        """Return ``False`` only when recording the completion event fails."""
        ...


class KVLoadSynchronizer(Protocol):
    """Orders the current compute stream after a Host KV copy group."""

    def wait_group(self, group_id: int) -> bool: ...

    def size(self) -> int: ...


@dataclass(slots=True)
class KVLoadContext:
    """Per-forward load dependencies; never retained by a captured graph."""

    synchronizer: KVLoadSynchronizer
    layers_per_event: int
    num_layers: int
    _next_group: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.layers_per_event <= 0 or self.num_layers <= 0:
            raise ValueError("Host KV load requires positive layer and group sizes")
        groups = (self.num_layers + self.layers_per_event - 1) // self.layers_per_event
        # The composite native handle may include one final draft-cache event.
        if self.synchronizer.size() not in (groups, groups + 1):
            raise ValueError("Host KV load event count does not match the model layers")

    def wait_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"Host KV load layer index is out of range: {layer_id}")
        if layer_id % self.layers_per_event:
            return
        group_id = layer_id // self.layers_per_event
        if group_id != self._next_group:
            raise RuntimeError("Host KV load groups must be consumed once in layer order")
        if not self.synchronizer.wait_group(group_id):
            raise RuntimeError(f"Host KV reload failed for layer group {group_id}")
        self._next_group += 1

    def check_consumed(self) -> None:
        groups = (self.num_layers + self.layers_per_event - 1) // self.layers_per_event
        if self._next_group != groups:
            raise RuntimeError("Python model did not consume every Host KV load group")


@dataclass(frozen=True, slots=True)
class AclGraphTask:
    event: object
    handle: object
    update: Callable[[], None]


@dataclass(slots=True)
class AclGraphExecutionState:
    """Persistent resources owned by one model-execution graph entry."""

    persistent_buffers: dict[tuple[object, ...], object]
    paged_attention: dict[int, PagedAttentionGraphState] = field(default_factory=dict)
    # Serial MTP draft steps may share temporary attention storage.  Keep the
    # per-step map above for outputs and metadata whose addresses are part of
    # the captured graph, while allowing explicitly shared workspaces to use a
    # role-owned pool.  The pool is only used on the single graph stream, so a
    # later step cannot overwrite a live workspace from an earlier step.
    shared_persistent_buffers: dict[tuple[object, ...], object] | None = None
    # Only runners that serialize graph replay may supply this pool. Growing
    # it never replaces an allocation already referenced by a captured graph.
    serial_workspace_pool: list[torch.Tensor] | None = None


@dataclass(slots=True)
class AclGraphCaptureContext:
    stream: object
    tasks: list[AclGraphTask]


@dataclass(slots=True)
class EplbRuntimeState:
    """Per-forward EPLB tensors owned by the C++ worker runtime."""

    expert_load_data: torch.Tensor
    decode_token_mask: torch.Tensor | None
    is_graph_warmup: bool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    attention_backend: AttentionBackend
    device: torch.device
    metadata: AttentionMetadata
    layer_caches: list[LayerCache]
    acl_graph: AclGraphCaptureContext | None = None
    layer_synchronizer: LayerSynchronizer | None = None
    execution_state: AclGraphExecutionState | None = None
    # Context-Parallel sharding plan for this forward, or None when CP is off
    # (cp_size <= 1) or the step is decode (CP is prefill-only).
    cp_context: CpContext | None = None
    eplb: EplbRuntimeState | None = None
    # Values derived from per-forward metadata that are shared by multiple
    # layers. A new ForwardContext gets a new cache, so entries never leak
    # across requests or graph executions.
    layer_shared_cache: dict[object, object] = field(default_factory=dict)
    kv_load: KVLoadContext | None = None


_current_context: ContextVar[ForwardContext | None] = ContextVar("_current_context", default=None)


@contextmanager
def forward_context(ctx: ForwardContext):
    token = _current_context.set(ctx)
    try:
        yield
    finally:
        _current_context.reset(token)


def get_forward_context() -> ForwardContext:
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError("forward context is not set")
    return ctx


def get_forward_context_or_none() -> ForwardContext | None:
    """Return the current forward context, or None outside a forward pass.

    Layers that must not depend on the executor (e.g. the standalone stub-loader
    align path) use this to read per-step metadata without hard-failing.
    """
    return _current_context.get()


def in_acl_graph() -> bool:
    """Whether the current forward runs under ACL graph warmup/capture.

    The decode graph runner always passes an ``execution_state`` (warmup and
    capture) and an ``acl_graph`` capture context (capture only); the eager
    runner sets neither, so eager paths stay byte-identical.
    """
    ctx = get_forward_context_or_none()
    return ctx is not None and (ctx.acl_graph is not None or ctx.execution_state is not None)


def record_layer_event(layer_id: int) -> None:
    ctx = _current_context.get()
    if ctx is not None and ctx.layer_synchronizer is not None:
        if not ctx.layer_synchronizer.record_event(layer_id):
            raise RuntimeError(f"failed to record layer completion event for layer {layer_id}")


def wait_for_kv_layer(layer_id: int) -> None:
    ctx = _current_context.get()
    if ctx is not None and ctx.kv_load is not None:
        if ctx.acl_graph is not None or ctx.execution_state is not None:
            raise RuntimeError("Per-batch Host KV load dependencies cannot be captured in a graph")
        ctx.kv_load.wait_layer(layer_id)


def get_execution_buffer(
    key: tuple[object, ...],
    factory: Callable[[], torch.Tensor],
    *,
    shared: bool = False,
) -> torch.Tensor:
    """Get a tensor owned by the active model execution graph entry."""
    state = get_forward_context().execution_state
    if state is None:
        return factory()
    # A regular single-step graph has no sibling state to share with; use its
    # private map in that case. MTP role adapters install the explicit shared
    # pool above when serial steps can reuse temporary storage.
    buffers = (
        state.shared_persistent_buffers
        if shared and state.shared_persistent_buffers is not None
        else state.persistent_buffers
    )
    buffer = buffers.get(key)
    if buffer is None:
        buffer = factory()
        buffers[key] = buffer
    if not isinstance(buffer, torch.Tensor):
        raise TypeError("execution buffer must be a torch.Tensor")
    return buffer


def get_execution_workspace(key: tuple[object, ...], factory: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Reuse scratch storage across explicitly serialized graph entries."""
    state = get_forward_context().execution_state
    pool = None if state is None else state.serial_workspace_pool
    if pool is None:
        return get_execution_buffer(key, factory, shared=True)

    def allocate() -> torch.Tensor:
        requested = factory()
        if requested.ndim != 1 or not requested.is_contiguous():
            raise ValueError("shared graph workspaces must be contiguous one-dimensional tensors")
        for buffer in pool:
            if (
                buffer.device == requested.device
                and buffer.dtype == requested.dtype
                and buffer.numel() >= requested.numel()
            ):
                return buffer[: requested.numel()]
        pool.append(requested)
        return requested

    return get_execution_buffer(key, allocate, shared=True)


def copy_into_execution_buffer(key: tuple[object, ...], source: torch.Tensor) -> torch.Tensor:
    """Copy ``source`` into a graph-owned buffer with a stable address.

    ACL graph replay does not re-run Python. Host-updated metadata must land in
    the same storage the captured kernels recorded. Eager execution has no
    ``execution_state`` and returns ``source`` unchanged.
    """
    state = get_forward_context().execution_state
    if state is None:
        return source
    buffer = get_execution_buffer(key, lambda: torch.empty_like(source))
    if buffer.shape != source.shape or buffer.dtype != source.dtype or buffer.device != source.device:
        raise RuntimeError(
            "execution buffer shape/dtype/device changed for key "
            f"{key}: got {tuple(buffer.shape)} {buffer.dtype} {buffer.device}, "
            f"expected {tuple(source.shape)} {source.dtype} {source.device}"
        )
    if buffer.data_ptr() != source.data_ptr():
        buffer.copy_(source, non_blocking=True)
    return buffer


_ExecutionMetadata = TypeVar("_ExecutionMetadata")


def get_execution_context(metadata_type: type[_ExecutionMetadata]) -> _ExecutionMetadata | None:
    return cast(_ExecutionMetadata | None, get_forward_context().layer_shared_cache.get(metadata_type))

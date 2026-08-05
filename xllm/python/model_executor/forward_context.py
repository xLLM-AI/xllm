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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

import torch

if TYPE_CHECKING:
    from xllm.python.attention.backend import (
        AttentionBackend,
        AttentionMetadata,
        LayerCache,
    )


class LayerSynchronizer(Protocol):
    """Records a per-layer completion event for the PD KV-cache transfer thread.

    Implemented in C++ (``NPULayerSynchronizerImpl``) and passed in from the
    executor; the model forward calls ``record_event`` after each layer so the
    transfer thread can push that layer's KV cache without waiting for the whole
    forward to finish.
    """

    def record_event(self, layer_id: int) -> None: ...


@dataclass(frozen=True, slots=True)
class AclGraphTask:
    event: object
    handle: object
    update: Callable[[], None]


@dataclass(slots=True)
class AclGraphExecutionState:
    """Persistent tensors owned by one model-execution graph entry."""

    persistent_buffers: dict[tuple[object, ...], object]


@dataclass(slots=True)
class AclGraphCaptureContext:
    stream: object
    tasks: list[AclGraphTask]


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
    # (cp_size <= 1) or the step is decode (CP is prefill-only in v1). Typed as
    # object to avoid a circular import with model_executor.cp_utils.CpContext.
    cp_context: object | None = None


_current_context: ContextVar[ForwardContext | None] = ContextVar(
    "_current_context", default=None
)


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


def record_layer_event(layer_id: int) -> None:
    ctx = _current_context.get()
    if ctx is not None and ctx.layer_synchronizer is not None:
        ctx.layer_synchronizer.record_event(layer_id)


def get_execution_buffer(
    key: tuple[object, ...], factory: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """Get a tensor owned by the active model execution graph entry."""
    state = get_forward_context().execution_state
    if state is None:
        return factory()
    buffer = state.persistent_buffers.get(key)
    if buffer is None:
        buffer = factory()
        state.persistent_buffers[key] = buffer
    if not isinstance(buffer, torch.Tensor):
        raise TypeError("execution buffer must be a torch.Tensor")
    return buffer

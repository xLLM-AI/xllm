# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
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

import torch

from xllm.python.attention.backend import AttentionBackend


@dataclass(frozen=True, slots=True)
class ForwardContext:
    attention_backend: AttentionBackend
    device: torch.device


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

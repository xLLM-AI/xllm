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

"""Reusable NPU auxiliary streams shared across model implementations."""

from __future__ import annotations

import torch

_SHARED_EXPERT_STREAMS: dict[tuple[str, int | None], torch.npu.Stream] = {}


def shared_expert_stream(device: torch.device) -> torch.npu.Stream:
    """Lazily return the shared-expert stream for an indexed NPU device."""
    key = (device.type, device.index)
    stream = _SHARED_EXPERT_STREAMS.get(key)
    if stream is None:
        stream = torch.npu.Stream(device=device)
        _SHARED_EXPERT_STREAMS[key] = stream
    return stream

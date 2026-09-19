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

"""Device-local stream operations for accelerator-independent model scheduling.

Backends register their runtime module with PyTorch. Stream API availability
is checked during model construction; execution and graph errors propagate.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


class DeviceStream:
    """A side stream whose fork and join target its own device's current stream."""

    def __init__(self, device: torch.device, device_module: Any) -> None:
        self._device = device
        self._device_module = device_module
        self._stream = device_module.Stream(device=device)
        if not callable(getattr(self._stream, "wait_stream", None)):
            raise RuntimeError(f"Device {device} does not support stream dependencies")

    def wait_for_current(self) -> None:
        self._stream.wait_stream(self._device_module.current_stream(self._device))

    def activate(self) -> AbstractContextManager[Any]:
        return self._device_module.stream(self._stream)

    def join(self) -> None:
        self._device_module.current_stream(self._device).wait_stream(self._stream)

    def record_on_current(self, tensor: torch.Tensor) -> None:
        """Keep a side-stream allocation alive for its current-stream consumer."""
        tensor.record_stream(self._device_module.current_stream(self._device))


_STREAMS: dict[tuple[str, int, str], DeviceStream] = {}


def get_device_stream(device: torch.device, name: str) -> DeviceStream:
    """Reuse a named stream on a concrete device, resolving unindexed devices now.

    CPU execution has no accelerator overlap. Unsupported backends fail before
    model execution rather than silently ignoring an explicitly enabled feature.
    """
    device = torch.device(device)
    if device.type == "privateuseone":
        device = torch.device(torch._C._get_privateuse1_backend_name(), device.index)
    if device.type == "cpu":
        raise RuntimeError("Device cpu does not support accelerator stream overlap")
    device_module = torch.get_device_module(device)
    required = ("Stream", "current_stream", "stream", "current_device")
    if not all(callable(getattr(device_module, method, None)) for method in required):
        raise RuntimeError(f"Device {device} does not provide the required stream API")
    index = device.index if device.index is not None else device_module.current_device()
    device = torch.device(device.type, index)
    key = (device.type, index, name)
    if key not in _STREAMS:
        _STREAMS[key] = DeviceStream(device, device_module)
    return _STREAMS[key]

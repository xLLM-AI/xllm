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

"""Stream dispatch and dependency tests independent of installed accelerators."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from xllm.python import device_stream


@dataclass(frozen=True)
class _Device:
    type: str
    index: int | None = None


def _device(value: str | _Device, index: int | None = None) -> _Device:
    if isinstance(value, _Device):
        return value
    return _Device(value, index)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    side = MagicMock()
    main = MagicMock()
    module = SimpleNamespace(
        Stream=MagicMock(return_value=side),
        current_stream=MagicMock(return_value=main),
        current_device=MagicMock(return_value=0),
        stream=MagicMock(side_effect=lambda _stream: nullcontext()),
    )
    resolver = MagicMock(return_value=module)
    monkeypatch.setattr(device_stream, "_STREAMS", {})
    monkeypatch.setattr(
        device_stream,
        "torch",
        SimpleNamespace(
            device=_device,
            get_device_module=resolver,
            _C=SimpleNamespace(_get_privateuse1_backend_name=lambda: "mlu"),
        ),
    )
    return SimpleNamespace(module=module, resolver=resolver, side=side, main=main)


@pytest.mark.parametrize("backend", ["cuda", "npu", "mlu", "musa"])
def test_dispatch_and_dependencies(runtime: SimpleNamespace, backend: str) -> None:
    device = _Device(backend, 1)
    stream = device_stream.get_device_stream(device, "indexer")
    stream.wait_for_current()
    with stream.activate():
        pass
    stream.join()
    tensor = MagicMock()
    stream.record_on_current(tensor)
    tensor.record_stream.assert_called_once_with(runtime.main)
    runtime.resolver.assert_called_with(device)
    runtime.module.Stream.assert_called_once_with(device=device)
    runtime.module.current_device.assert_not_called()
    assert all(call.args == (device,) for call in runtime.module.current_stream.call_args_list)
    runtime.side.wait_stream.assert_called_once_with(runtime.main)
    runtime.main.wait_stream.assert_called_once_with(runtime.side)
    runtime.module.stream.assert_called_once_with(runtime.side)


def test_unindexed_device_is_pinned_and_cache_is_device_local(runtime: SimpleNamespace) -> None:
    first = device_stream.get_device_stream(_Device("cuda"), "indexer")
    assert first is device_stream.get_device_stream(_Device("cuda", 0), "indexer")
    runtime.module.current_device.return_value = 1
    second = device_stream.get_device_stream(_Device("cuda"), "indexer")
    assert second is not first
    assert second is device_stream.get_device_stream(_Device("cuda", 1), "indexer")
    assert first is not device_stream.get_device_stream(_Device("cuda", 0), "copy")
    assert first is not device_stream.get_device_stream(_Device("mlu", 0), "indexer")
    first.join()
    runtime.module.current_stream.assert_called_with(_Device("cuda", 0))


def test_private_backend_alias_uses_registered_runtime(runtime: SimpleNamespace) -> None:
    aliased = device_stream.get_device_stream(_Device("privateuseone", 2), "indexer")
    assert aliased is device_stream.get_device_stream(_Device("mlu", 2), "indexer")
    runtime.module.Stream.assert_called_once_with(device=_Device("mlu", 2))


def test_cpu_is_rejected_before_stream_creation(runtime: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="cpu does not support"):
        device_stream.get_device_stream(_Device("cpu"), "indexer")
    runtime.module.Stream.assert_not_called()


@pytest.mark.parametrize("method", ["Stream", "stream", "current_stream", "current_device"])
def test_missing_stream_capability_is_rejected(runtime: SimpleNamespace, method: str) -> None:
    setattr(runtime.module, method, None)
    with pytest.raises(RuntimeError, match="required stream API"):
        device_stream.get_device_stream(_Device("mlu", 0), "indexer")


def test_creation_failure_is_propagated_and_not_cached(runtime: SimpleNamespace) -> None:
    runtime.module.Stream.side_effect = RuntimeError("device allocation failed")
    with pytest.raises(RuntimeError, match="device allocation failed"):
        device_stream.get_device_stream(_Device("cuda", 0), "indexer")
    assert not device_stream._STREAMS


def test_execution_failure_is_not_silently_ignored(runtime: SimpleNamespace) -> None:
    stream = device_stream.get_device_stream(_Device("npu", 0), "indexer")
    runtime.side.wait_stream.side_effect = RuntimeError("stream dependency failed")
    with pytest.raises(RuntimeError, match="stream dependency failed"):
        stream.wait_for_current()


@pytest.mark.parametrize("backend", ["cuda", "npu", "mlu", "musa"])
def test_stream_dependencies_on_available_device(backend: str) -> None:
    import torch

    module = getattr(torch, backend, None)
    if module is None or not module.is_available():
        pytest.skip(f"{backend} device is not available")
    device = torch.device(backend, module.current_device())
    stream = device_stream.get_device_stream(device, "test_dependencies")
    value = torch.arange(4096, device=device, dtype=torch.float32)
    for _ in range(3):
        value.add_(1)
        stream.wait_for_current()
        with stream.activate():
            indexed = value.square()
        projected = value * 2
        stream.join()
        stream.record_on_current(indexed)
        torch.testing.assert_close(indexed + projected, value.square() + value * 2)

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

"""Tests for Python process-group rendezvous ownership."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

_MODULE_PATH = Path(__file__).parents[2] / "xllm" / "python" / "distributed" / "collectives.py"
_SPEC = importlib.util.spec_from_file_location("_xllm_collectives_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collectives = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collectives)


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _clear_collective_state():
    def reset():
        collectives._groups.clear()
        collectives._stores.clear()
        collectives._symm_eligible.clear()
        collectives._symm_buffers.clear()
        collectives._world_topology = None
        collectives._world_initialized = False

    reset()
    yield
    reset()


class _FakeStore:
    def __init__(self, topology: list[dict[str, object]] | None = None) -> None:
        self.values: dict[str, bytes] = {}
        if topology is not None:
            for rank, entry in enumerate(topology):
                self.values[f"xllm/python_collectives/topology/v1/{rank}"] = json.dumps(entry).encode("utf-8")

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode("utf-8")

    def get(self, key: str) -> bytes:
        return self.values[key]


def _mock_process_groups(
    monkeypatch: pytest.MonkeyPatch,
    global_rank: int,
    topology: list[dict[str, object]] | None = None,
):
    """Stand in for c10d so the rendezvous can be inspected without a world.

    ``new_group`` reports this rank's position inside the membership it is
    handed, which is what the module checks its caller's rank against.
    """
    if topology is None:
        topology = [{"hostname": "node-0", "device_index": rank} for rank in range(16)]
    base_store = _FakeStore(topology)
    tcp_store = MagicMock(return_value=base_store)
    init_world = MagicMock()
    new_group = MagicMock(
        side_effect=lambda ranks, timeout, backend: _FakeGroup(
            ranks.index(global_rank) if global_rank in ranks else -1, len(ranks)
        )
    )
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "init_process_group", init_world)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(collectives.socket, "gethostname", lambda: "node-0")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda _a, _b: True)
    return base_store, tcp_store, init_world, new_group


def test_parallel_groups_share_one_multitenant_tcp_store(monkeypatch):
    base_store, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=0)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)
    collectives.init_process_group("moe_tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)

    tcp_store.assert_called_once()
    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 2, True)
    assert tcp_store.call_args.kwargs["wait_for_workers"] is False
    assert tcp_store.call_args.kwargs["multi_tenant"] is True

    # Every parallel group is a subgroup of one world, so the world rendezvous
    # happens once no matter how many groups the caller asks for.
    init_world.assert_called_once()
    assert init_world.call_args.kwargs["store"] is base_store
    assert init_world.call_args.kwargs["rank"] == 0
    assert init_world.call_args.kwargs["world_size"] == 2
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 1],
        [0, 1],
    ]


def test_tcp_store_master_is_global_rank_zero_not_group_rank_zero(monkeypatch):
    _, tcp_store, _, _ = _mock_process_groups(monkeypatch, global_rank=2)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 4, 1)

    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 4, False)


def test_symmetric_memory_rejects_cross_host_group(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-1", "device_index": 0},
    ]
    can_access_peer = MagicMock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access_peer)

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])
    can_access_peer.assert_not_called()


def test_symmetric_memory_rejects_incomplete_peer_domain(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-0", "device_index": 1},
    ]
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda source, destination: (source, destination) != (1, 0),
    )

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_symmetric_buffer_rejects_unsupported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    empty = MagicMock()
    rendezvous = MagicMock()
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(_FakeGroup(0, 2), group_name, tensor) is None
    empty.assert_not_called()
    rendezvous.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_symmetric_buffer_accepts_supported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    buffer = object()
    group = _FakeGroup(0, 2)
    group.group_name = "tp-group"
    empty = MagicMock(return_value=buffer)
    rendezvous = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(group, group_name, tensor) is buffer
    empty.assert_called_once_with(8, dtype=dtype, device=device)
    rendezvous.assert_called_once_with(buffer, "tp-group")

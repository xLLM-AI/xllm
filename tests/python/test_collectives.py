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
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from xllm.python.models import glm5_2

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
        collectives._group_ranks.clear()
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


def _run_glm_ep1_tp_collective(global_rank: int, rendezvous_path: str) -> None:
    world_size = 4
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        tp_groups = [
            dist.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
            dist.new_group(
                ranks=[2, 3],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
        ]
        collectives._groups[("tp", "cpu")] = tp_groups[global_rank // 2]
        # This is the topology that exposed the bug: with EP1, moe_tp spans
        # both CP cohorts and must not be used to combine expert partials.
        collectives._groups[("moe_tp", "cpu")] = dist.group.WORLD

        cp_rank = global_rank // 2
        tp_rank = global_rank % 2
        local_value = float(cp_rank * 10 + tp_rank + 1)
        routed = torch.tensor([[local_value]])
        shared = torch.tensor([[local_value * 10]])
        moe = SimpleNamespace(
            ep_size=1,
            moe_tp_size=world_size,
            cfg=SimpleNamespace(tp_size=2),
        )

        with patch.object(
            glm5_2.distributed,
            "all_reduce_",
            collectives.all_reduce_,
            create=True,
        ):
            output = glm5_2.Glm52MoE._combine_expert_outputs(moe, routed, shared)

        expected = torch.tensor([[33.0 if cp_rank == 0 else 253.0]])
        torch.testing.assert_close(output, expected)
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


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


def test_native_runtime_bridge_bypasses_python_process_groups(monkeypatch):
    calls: list[str] = []

    runtime = SimpleNamespace(
        tp_all_reduce=lambda tensor: (calls.append("tp_reduce"), tensor.add_(1)),
        tp_all_gather=lambda tensor, dim: (
            calls.append(f"tp_gather:{dim}"),
            torch.cat((tensor, tensor), dim=dim),
        )[1],
        moe_tp_all_reduce=lambda tensor: (
            calls.append("moe_tp_reduce"),
            tensor.add_(2),
        ),
        moe_ep_all_reduce=lambda tensor: (
            calls.append("moe_ep_reduce"),
            tensor.add_(4),
        ),
    )
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    python_reduce = MagicMock(side_effect=AssertionError("c10d fallback used"))
    python_gather = MagicMock(side_effect=AssertionError("c10d fallback used"))
    monkeypatch.setattr(collectives, "all_reduce_", python_reduce)
    monkeypatch.setattr(collectives, "all_gather", python_gather)

    value = torch.tensor([[1.0]])
    collectives.tp_all_reduce(value)
    gathered = collectives.tp_all_gather(value, 1, 2)
    collectives.moe_tp_all_reduce(value)
    collectives.moe_ep_all_reduce(value)

    assert calls == ["tp_reduce", "tp_gather:1", "moe_tp_reduce", "moe_ep_reduce"]
    assert gathered.tolist() == [[2.0, 2.0]]
    assert value.tolist() == [[8.0]]
    python_reduce.assert_not_called()
    python_gather.assert_not_called()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm_ep1_tp_reduce_does_not_mix_cp_cohorts(tmp_path: Path) -> None:
    rendezvous_path = tmp_path / "glm-ep1-tp-reduce"

    process_context = torch.multiprocessing.start_processes(
        _run_glm_ep1_tp_collective,
        args=(str(rendezvous_path),),
        nprocs=4,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 30.0
    try:
        while not process_context.join(
            timeout=max(0.0, deadline - time.monotonic()),
            grace_period=5.0,
        ):
            if time.monotonic() >= deadline:
                pytest.fail("Gloo CP2 x TP2 collective test timed out")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)


def test_dcp_group_is_strided_like_kv_split_rank(monkeypatch):
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)

    # world=8, dcp=2 → group_count=4; membership matches rank/(world/dcp).
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


def test_dp_execution_gather_uses_fixed_collective_for_equal_counts(monkeypatch):
    value = torch.zeros(4, 8)
    gathered = torch.zeros(8, 8)
    fixed_gather = MagicMock(return_value=gathered)
    variable_gather = MagicMock()
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (4, 4),
        rank=1,
    )

    assert output is gathered
    assert offset == 4
    fixed_gather.assert_called_once_with(
        value,
        dim=0,
        world_size=2,
        group_name="dp",
    )
    variable_gather.assert_not_called()


def test_dp_execution_gather_uses_variable_collective_for_uneven_counts(monkeypatch):
    value = torch.zeros(1, 8)
    gathered = torch.zeros(4, 8)
    fixed_gather = MagicMock()
    variable_gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (3, 1),
        rank=1,
    )

    assert output is gathered
    assert offset == 3
    variable_gather.assert_called_once_with(value, [3, 1], 1, "dp")
    fixed_gather.assert_not_called()


def test_dp_execution_gather_rejects_local_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="does not match the local tensor"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 2),
            rank=1,
        )


def test_dp_execution_gather_rejects_zero_execution_count() -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 0),
            rank=1,
        )


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

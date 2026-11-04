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

"""Fixed-geometry admission and persistent inputs for Qwen prefill graphs."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners import acl_graph
from xllm.python.model_executor.runners.prefill_acl_graph import PrefillAclGraphRunner


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SimpleNamespace, InputBatch]:
    input_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    positions = torch.tensor([[0, 1, 5], [0, 1, 6], [0, 1, 7]], dtype=torch.int32)
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([128, 129, 640], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 5], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([2, 5], dtype=torch.int32),
        block_table=torch.tensor([[1, 0, 0], [5, 0, 0]], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([2, 1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 2, 3], dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, 2, 3],
        q_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        kv_seq_lens=torch.tensor([2, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([2, 5], dtype=torch.int32),
        kv_seq_lens_host_values=[2, 5],
        kv_cu_seq_lens=torch.tensor([0, 2, 7], dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([1, 2], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([1, 2], dtype=torch.int32),
        has_initial_state=torch.tensor([False, True]),
        is_prefill=True,
        is_chunked_prefill=False,
        is_spec_verify=False,
        is_mixed=False,
        is_dummy=False,
        has_kv_shard=False,
        prepared_attention_state=None,
    )
    request_layout = SimpleNamespace(
        num_reqs=2,
        num_tokens=3,
        num_scheduled_tokens=[2, 1],
        num_computed_tokens=[0, 4],
        query_start_loc=[0, 2, 3],
        is_prefilling=[1, 1],
    )
    input_batch = InputBatch.from_runtime(input_ids, positions, request_layout)
    return input_ids, positions, embedding, metadata, input_batch


def test_prefill_graph_reuses_geometry_with_new_visual_inputs() -> None:
    input_ids, positions, embedding, metadata, input_batch = _inputs()
    runner = PrefillAclGraphRunner(nn.Identity(), SimpleNamespace(), torch.device("cpu"), 2, 16, 1)
    assert runner.can_execute(input_ids, positions, metadata, embedding, input_batch)
    entry = runner._allocate_entry(input_ids, positions, metadata, embedding)
    key = runner._graph_key(input_ids, positions, metadata, embedding, (2, 1), (2, 5))
    runner._graphs[key] = entry
    runner._fill_entry(entry, input_ids, positions, metadata, embedding)

    new_positions = positions + 10
    new_embedding = embedding + 20
    new_slots = metadata.slot_mapping + 128
    metadata.slot_mapping = new_slots
    assert runner.can_execute(input_ids, new_positions, metadata, new_embedding, input_batch)
    runner._fill_entry(entry, input_ids, new_positions, metadata, new_embedding)
    torch.testing.assert_close(entry.static_positions, new_positions)
    torch.testing.assert_close(entry.static_input_embedding, new_embedding)
    torch.testing.assert_close(entry.static_metadata.slot_mapping, new_slots)

    metadata.kv_seq_lens_host_values = [2, 6]
    assert not runner.can_execute(input_ids, new_positions, metadata, new_embedding, input_batch)


def test_prefill_graph_admits_chunked_prefix_with_matching_batch() -> None:
    input_ids, positions, embedding, metadata, input_batch = _inputs()
    metadata.is_prefill = False
    metadata.is_chunked_prefill = True
    runner = PrefillAclGraphRunner(nn.Identity(), SimpleNamespace(), torch.device("cpu"), 2, 16, 2)
    assert runner.can_execute(input_ids, positions, metadata, embedding, input_batch)
    metadata.is_spec_verify = True
    assert not runner.can_execute(input_ids, positions, metadata, embedding, input_batch)


def test_prefill_capture_restores_state_before_each_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    input_ids, positions, embedding, metadata, _ = _inputs()
    stream = Mock()
    npu = SimpleNamespace(
        NPUGraph=Mock(return_value=Mock()),
        current_stream=Mock(return_value=stream),
        stream=lambda _: nullcontext(),
        graph=lambda *args, **kwargs: nullcontext(),
        graph_pool_handle=Mock(return_value=(0, 1)),
        synchronize=Mock(),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(acl_graph, "_GRAPH_POOL", None)
    runner = PrefillAclGraphRunner(nn.Identity(), SimpleNamespace(), torch.device("cpu"), 2, 16, 1)
    entry = runner._allocate_entry(input_ids, positions, metadata, embedding)
    state = [0]
    seen = []

    def restore() -> None:
        state[0] = 0

    def forward(_: object) -> torch.Tensor:
        seen.append(state[0])
        state[0] += 1
        return torch.ones(1)

    runner._forward_static = forward
    runner._capture(entry, stream, reset_state=restore)
    assert seen == [0] * (acl_graph._CAPTURE_WARMUP_STEPS + 1)


def test_prefill_execute_restores_capture_state_after_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    input_ids, positions, embedding, metadata, input_batch = _inputs()
    events: list[str] = []

    class Snapshot:
        def restore(self) -> None:
            events.append("restore")

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_stream(self, stream: object) -> None:
            events.append(f"wait:{self.name}:{stream.name}")

    capture_stream = Stream("capture")
    current_stream = Stream("current")
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda **kwargs: capture_stream,
            current_stream=lambda: current_stream,
            stream=lambda _: nullcontext(),
        ),
        raising=False,
    )
    runner = PrefillAclGraphRunner(nn.Identity(), SimpleNamespace(), torch.device("cpu"), 2, 16, 1)
    runner._prepare_attention = Mock()
    runner._snapshot_capture_state = lambda _: (Snapshot(),)

    def fail_capture(*args: object, **kwargs: object) -> None:
        del args, kwargs
        events.append("mutate")
        raise RuntimeError("capture failed")

    runner._capture = fail_capture

    with pytest.raises(RuntimeError, match="capture failed"):
        runner.execute(input_ids, positions, metadata, embedding, input_batch=input_batch)

    assert events == ["mutate", "wait:current:capture", "restore"]
    assert runner._graphs == {}


def test_prefill_execute_does_not_cache_graph_when_first_replay_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    input_ids, positions, embedding, metadata, input_batch = _inputs()
    current_stream = Mock()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda **kwargs: Mock(),
            current_stream=lambda: current_stream,
            stream=lambda _: nullcontext(),
        ),
        raising=False,
    )
    runner = PrefillAclGraphRunner(nn.Identity(), SimpleNamespace(), torch.device("cpu"), 2, 16, 1)
    runner._prepare_attention = Mock()
    runner._snapshot_capture_state = lambda _: ()
    graph = Mock()
    graph.replay.side_effect = [RuntimeError("replay failed"), None]
    capture_count = 0

    def capture(entry: object, stream: object, **kwargs: object) -> None:
        nonlocal capture_count
        del stream, kwargs
        capture_count += 1
        entry.graph = graph
        entry.static_output = torch.ones(input_ids.numel(), 1)
        entry.graph_tasks = []

    runner._capture = capture

    with pytest.raises(RuntimeError, match="replay failed"):
        runner.execute(input_ids, positions, metadata, embedding, input_batch=input_batch)

    assert runner._graphs == {}
    output = runner.execute(input_ids, positions, metadata, embedding, input_batch=input_batch)
    assert output.shape == (input_ids.numel(), 1)
    assert capture_count == 2
    assert len(runner._graphs) == 1

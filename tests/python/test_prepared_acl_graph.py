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

"""Prepared ACL graph address binding and admission contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from xllm.python.attention.backend import AttentionBackend
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner


def _metadata(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        block_table=torch.zeros(rows, 2, dtype=torch.int32),
        q_seq_lens=torch.ones(rows, dtype=torch.int32),
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(1, rows + 1, dtype=torch.int32),
        q_cu_seq_lens_host_values=list(range(1, rows + 1)),
        kv_seq_lens_host_values=[1] * rows,
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        prepared_attention_state=object(),
    )


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> PreparedAclGraphRunner:
    stream = Mock()
    npu = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(return_value=Mock()),
        current_stream=Mock(return_value=stream),
        stream=lambda _: nullcontext(),
        memory_allocated=lambda _: 100,
        memory_reserved=lambda _: 200,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    backend = SimpleNamespace(prepare=Mock())
    backend.prepare_graph_replay = Mock(
        side_effect=lambda metadata: AttentionBackend.prepare_graph_replay(backend, metadata)
    )
    result = PreparedAclGraphRunner(torch.nn.Identity(), backend, torch.device("cpu"), 4)
    result._capture = Mock(side_effect=lambda entry, _: setattr(entry, "graph", Mock()))
    result._allocate_entry = Mock(side_effect=AssertionError("owned input allocation"))
    result._fill_entry = Mock(side_effect=AssertionError("whole input copy"))
    return result


def test_capture_and_replay_bind_each_slot_without_input_copies(runner: PreparedAclGraphRunner) -> None:
    from xllm.python.model_executor.forward_context import get_forward_context

    inputs = []
    for _ in range(2):
        tokens = torch.arange(4, dtype=torch.int32)
        positions = torch.zeros(4, dtype=torch.int32)
        metadata = _metadata(4)
        runner.warmup_prepared(tokens, positions, metadata)
        runner.warmup_prepared(tokens, positions, metadata)
        inputs.append((tokens, positions, metadata))
    entries = []
    for tokens, positions, metadata in inputs:
        entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
        assert entry.static_input_ids is tokens
        assert entry.static_positions is positions
        assert entry.static_metadata.slot_mapping is metadata.slot_mapping
        assert entry.static_metadata.block_table is metadata.block_table
        entries.append(entry)
    assert entries[0] is not entries[1]
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    runner.attention_backend.prepare.reset_mock()
    installed = []

    def install_replay(metadata: SimpleNamespace) -> None:
        installed.append((get_forward_context().execution_state, list(metadata.kv_seq_lens_host_values)))

    runner.attention_backend.prepare_graph_replay.side_effect = install_replay
    for step, index in enumerate((0, 1, 0)):
        tokens, positions, metadata = inputs[index]
        lengths = [7 + step + row for row in range(len(tokens))]
        metadata.kv_seq_lens_host_values[:] = lengths
        runner.execute(tokens, positions, metadata)
        assert installed[-1][0] is entries[index].execution_state
        assert installed[-1][1] == lengths
    runner.attention_backend.prepare.assert_not_called()
    assert runner.attention_backend.prepare_graph_replay.call_count == 3
    assert runner._capture.call_count == 2


@pytest.mark.parametrize("prepared", [True, False], ids=["pipeline", "legacy"])
@torch.inference_mode()
def test_real_paged_backend_refreshes_captured_lengths_and_isolates_entries(
    monkeypatch: pytest.MonkeyPatch, prepared: bool
) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python import kernels
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import get_forward_context

    stream = Mock()
    npu = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(side_effect=Mock),
        ExternalEvent=Mock(side_effect=Mock),
        NPUGraph=Mock(side_effect=Mock),
        current_stream=Mock(return_value=stream),
        stream=lambda _: nullcontext(),
        graph=lambda *args, **kwargs: nullcontext(),
        synchronize=Mock(),
        graph_task_group_begin=Mock(),
        graph_task_group_end=Mock(side_effect=lambda _: Mock()),
        graph_task_update_begin=Mock(),
        graph_task_update_end=Mock(),
        memory_allocated=lambda _: 0,
        memory_reserved=lambda _: 0,
    )
    monkeypatch.setattr(torch, "npu", npu)
    monkeypatch.setattr(kernels, "update_decode_graph_metadata", Mock(), raising=False)
    backend = NpuPagedAttentionBackend(8, 2, 64, 0.125, 0, False, torch.device("cpu"), torch.float16)
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    allocate_workspace = Mock(side_effect=lambda *args: torch.empty(1))
    monkeypatch.setattr(backend, "_allocate_graph_workspace", allocate_workspace)
    calls = []

    def record_fia(*args: object, **kwargs: object) -> None:
        calls.append((list(kwargs["actual_seq_kv"]), kwargs["block_table"], kwargs["output"]))
        kwargs["output"].fill_(sum(kwargs["actual_seq_kv"]))

    monkeypatch.setattr(backend, "_fia_out", record_fia)

    class PagedModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            context = get_forward_context()
            if context.acl_graph is None:
                return torch.zeros(tokens.numel(), 8 * 64)
            return backend._decode(torch.empty(tokens.numel(), 8, 64), cache, cache, context.metadata, tokens.numel())

    runner = (
        PreparedAclGraphRunner(PagedModel(), backend, torch.device("cpu"), 4)
        if prepared
        else DecodeAclGraphRunner(PagedModel(), backend, torch.device("cpu"), 4, 128)
    )
    inputs = []
    # Same-size independent slots plus a different bucket; legacy uses one
    # entry per bucket, whereas prepared execution additionally binds the slot.
    for rows in [2, 2, 4, 4] if prepared else [2, 4]:
        tokens = torch.zeros(rows, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(rows)
        metadata.kv_cu_seq_lens = None
        metadata.paged_kv_indptr = torch.arange(rows + 1, dtype=torch.int32)
        metadata.paged_kv_indices = torch.zeros(rows, dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.ones(rows, dtype=torch.int32)
        if prepared:
            metadata.prepared_attention_state = backend.prepare_metadata(metadata)
            runner.warmup_prepared(tokens, positions, metadata)
            entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
        else:
            metadata.prepared_attention_state = None
            entry = runner._prepare_graph_entry(tokens, positions, metadata, None)
        inputs.append((tokens, positions, metadata, entry))
    assert len({entry.static_output.data_ptr() for *_, entry in inputs}) == len(inputs)
    captured_lengths = [entry.execution_state.paged_attention[len(tokens)] for tokens, _, _, entry in inputs]
    assert len({id(lengths.kv) for lengths in captured_lengths}) == len(inputs)
    assert backend._paged_graph_state is captured_lengths[-1]
    workspace_count = allocate_workspace.call_count
    capture_count = npu.NPUGraph.call_count

    def reject_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("replay must reuse warmed attention buffers")

    for step, index in enumerate([0, 1, 0, len(inputs) - 1, 0]):
        tokens, positions, metadata, entry = inputs[index]
        lengths = [17 + step + row for row in range(tokens.numel())]
        metadata.kv_seq_lens_host_values[:] = lengths
        metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        if prepared:
            metadata.prepared_attention_state = backend.prepare_metadata(metadata)
            with monkeypatch.context() as replay_patch:
                replay_patch.setattr(torch, "empty", reject_allocation)
                output = runner.execute(tokens, positions, metadata)
        else:
            output = runner.execute(tokens, positions, metadata)
        assert calls[-1][0] == lengths
        assert calls[-1][1].data_ptr() == entry.static_metadata.block_table.data_ptr()
        assert calls[-1][2].data_ptr() == entry.static_output.data_ptr()
        assert torch.all(entry.static_output == sum(lengths))
        if prepared:
            assert output is entry.static_output
        else:
            # The legacy clone is queued before task update on a real device;
            # CPU mocks verify ownership without emulating that stream queue.
            assert output.data_ptr() != entry.static_output.data_ptr()
        assert captured_lengths[index].kv == lengths
        # Preparing another entry must not alter this captured task's state.
        other = (index + 1) % len(inputs)
        saved = list(captured_lengths[other].kv)
        inputs[other][3].graph_tasks[0].update()
        assert calls[-1][0] == saved
    assert allocate_workspace.call_count == workspace_count
    assert npu.NPUGraph.call_count == capture_count


def test_missing_binding_never_captures_or_mutates_backend(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.arange(4, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(4)
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions, metadata)
    runner.warmup_prepared(tokens, positions, metadata)
    runner.attention_backend.prepare.reset_mock()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens.clone(), positions, metadata)
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens[:3], positions[:3], _metadata(3))
    metadata.is_prefill = True
    with pytest.raises(ValueError, match="ordinary decode"):
        runner.execute(tokens, positions, metadata)
    runner.attention_backend.prepare.assert_not_called()
    assert runner._capture.call_count == 1


def test_prepare_keeps_previous_graph_metadata_independent(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    metadata.kv_seq_lens_host_values[:] = [7, 11]
    assert entry.static_metadata.kv_seq_lens_host_values == [1, 1]
    entry.static_output = torch.ones(2, 8)
    assert runner.execute(tokens, positions, metadata) is entry.static_output
    runner.attention_backend.prepare.assert_called_with(metadata, graph_mode=True)
    assert runner.prepared_replays == 1
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions.clone(), metadata)


def test_prepared_executor_obeys_admission_and_propagates_replay_failure(runner: PreparedAclGraphRunner) -> None:
    executor = object.__new__(ModelExecutor)
    executor._kv_bound = True
    executor._prepared_mtp = False
    executor.layerwise_split_size = 1
    executor.decode_graph_runner = None
    executor.prepared_graph_runner = runner
    executor.eager_runner = SimpleNamespace(execute=Mock(return_value="eager"))
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    # A cached address alone must not override native eager admission.
    assert executor.execute(tokens, positions, metadata) == "eager"
    entry.graph.replay.assert_not_called()
    executor.eager_runner.execute.reset_mock()
    entry.graph.replay.side_effect = RuntimeError("device replay failed")
    with pytest.raises(RuntimeError, match="device replay failed"):
        executor.execute(tokens, positions, metadata, enable_graph=True)
    executor.eager_runner.execute.assert_not_called()
    metadata.is_prefill = True
    assert executor.execute(tokens, positions, metadata) == "eager"
    assert runner._capture.call_count == 1


@pytest.mark.parametrize("rows", [0, 5])
def test_out_of_capacity_capture_is_rejected_before_work(runner: PreparedAclGraphRunner, rows: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        runner.warmup_prepared(torch.zeros(rows), torch.zeros(rows), _metadata(rows))
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


def test_prepared_mla_capture_reprepares_each_forward_and_binds_query_ends(runner: PreparedAclGraphRunner) -> None:
    runner.attention_backend.is_mla = True
    runner.model = Mock(return_value=torch.ones(2, 8))
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    assert entry.static_metadata.prepared_attention_state is metadata.prepared_attention_state
    assert entry.static_metadata.q_cu_seq_lens is metadata.q_cu_seq_lens
    runner.attention_backend.prepare.reset_mock()
    runner._forward_static(entry)
    runner._forward_static(entry)
    assert runner.attention_backend.prepare.call_count == 2
    runner.attention_backend.prepare.assert_called_with(entry.static_metadata, graph_mode=True)
    metadata.q_cu_seq_lens = metadata.q_cu_seq_lens.clone()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions, metadata)
    assert runner._capture.call_count == 1


def test_prepared_decode_rejects_mismatched_rows_before_capture(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(2)
    metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="one token per metadata row"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_prepared_dp_binds_two_slots_and_preserves_collective_shape(runner: PreparedAclGraphRunner, rows: int) -> None:
    runner.dp_size = 2
    inputs = []
    for _ in range(2):
        tokens = torch.zeros(rows, dtype=torch.int32)
        metadata = _metadata(rows)
        metadata.dp_execution_token_counts = (rows, rows)
        metadata.dp_is_decode = (1, 1)
        positions = torch.zeros_like(tokens)
        runner.warmup_prepared(tokens, positions, metadata)
        inputs.append((tokens, positions, metadata))
    entries = []
    for tokens, positions, metadata in inputs:
        entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
        assert entry.static_metadata.dp_execution_token_counts == (rows, rows)
        assert entry.static_metadata.dp_is_decode == (1, 1)
        runner.execute(tokens, positions, metadata)
        entry.graph.replay.assert_called_once()
        entries.append(entry)
    assert entries[0] is not entries[1]
    assert runner._capture.call_count == 2


def test_prepared_dp_rejects_invalid_peers_and_missing_local_binding(runner: PreparedAclGraphRunner) -> None:
    runner.dp_size = 2
    tokens = torch.ones(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 1)
    runner.warmup_prepared(tokens, positions, metadata)
    metadata.dp_execution_token_counts = (2, 7)
    with pytest.raises(ValueError, match="same decode batch size"):
        runner.execute(tokens, positions, metadata)
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 0)
    with pytest.raises(ValueError, match="same decode batch size"):
        runner.execute(tokens, positions, metadata)
    metadata.dp_is_decode = (1, 1)
    with pytest.raises(RuntimeError, match="Slot binding"):
        runner.execute(tokens.clone(), positions, metadata)
    assert runner._capture.call_count == 1


def test_prepared_dp_rejects_changed_collectives_before_replay(runner: PreparedAclGraphRunner) -> None:
    runner.dp_size = 2
    tokens = torch.ones(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 1)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    metadata.dp_execution_token_counts = (2, 4)
    with pytest.raises(ValueError, match="same decode batch size"):
        runner.execute(tokens, positions, metadata)
    entry.graph.replay.assert_not_called()


@pytest.mark.parametrize("counts,phases", [((1,), (1, 1)), ((1, 0), (1, 1)), ((1, 1), (1,)), ((2, 2), (1, 1))])
def test_prepared_dp_rejects_inconsistent_counts(
    runner: PreparedAclGraphRunner, counts: tuple[int, ...], phases: tuple[int, ...]
) -> None:
    runner.dp_size = 2
    tokens = torch.ones(1, dtype=torch.int32)
    metadata = _metadata(1)
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    with pytest.raises(ValueError, match="prepared DP"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()


@pytest.mark.parametrize("capacity", [1, 3, 4, 5])
def test_prepared_capture_accepts_native_sizes_within_capacity(runner: PreparedAclGraphRunner, capacity: int) -> None:
    runner = PreparedAclGraphRunner(runner.model, runner.attention_backend, runner.device, capacity)
    runner._capture = Mock(side_effect=lambda entry, _: setattr(entry, "graph", Mock()))
    for rows in range(1, capacity + 1):
        tokens = torch.arange(rows, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(rows)
        runner.warmup_prepared(tokens, positions, metadata)
        runner.execute(tokens, positions, metadata)
    assert runner._capture.call_count == capacity


def test_warmup_immediately_replays_and_reuses_captured_binding(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    entry.static_output = torch.ones(2, 8)
    output = runner.execute(tokens, positions, metadata)
    assert output is entry.static_output
    entry.graph.replay.assert_called_once()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens.clone(), positions, metadata)
    runner.attention_backend.prepare.reset_mock()
    runner.warmup_prepared(tokens, positions, metadata)
    assert runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)] is entry
    runner.attention_backend.prepare.assert_not_called()
    assert runner._capture.call_count == 1


def test_later_warmup_adds_buckets_without_replacing_previous_entries(runner: PreparedAclGraphRunner) -> None:
    inputs = []
    entries = []
    for rows in [4, 4, 2, 2]:
        tokens = torch.arange(rows, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(rows)
        runner.warmup_prepared(tokens, positions, metadata)
        inputs.append((tokens, positions, metadata))
        entries.append(runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)])
    for (tokens, positions, metadata), entry in zip(inputs, entries):
        runner.execute(tokens, positions, metadata)
        entry.graph.replay.assert_called_once()
    assert len({id(entry) for entry in entries}) == 4
    assert runner._capture.call_count == 4


def test_capture_failure_propagates_without_publishing_a_graph(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.ones(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner._capture.side_effect = RuntimeError("device capture failed")
    with pytest.raises(RuntimeError, match="device capture failed"):
        runner.warmup_prepared(tokens, positions, metadata)
    assert not runner._prepared_graphs
    assert runner.prepared_replays == 0


@pytest.mark.parametrize("reuse_topk", [False, True])
def test_mtp_capture_binds_hidden_and_topk_without_copying(runner: PreparedAclGraphRunner, reuse_topk: bool) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    hidden = torch.zeros(2, 4)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32) if reuse_topk else None
    runner.warmup_prepared(tokens, positions, metadata, hidden, topk)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata, hidden, topk)]
    assert entry.static_input_embedding is hidden
    assert entry.static_mtp_topk_indices is topk
    runner.model = Mock(return_value=(hidden, topk))
    runner._forward_static(entry)
    if reuse_topk:
        runner.model.assert_called_once_with(tokens, positions, hidden, topk)
    else:
        runner.model.assert_called_once_with(tokens, positions, hidden)
    runner.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk)
    runner.attention_backend.prepare.reset_mock()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions, metadata, hidden.clone(), mtp_topk_indices=topk)
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions, metadata, hidden.as_strided((2, 4), (1, 2)), mtp_topk_indices=topk)
    if reuse_topk:
        with pytest.raises(RuntimeError, match="warmed Slot binding"):
            runner.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk.clone())
        with pytest.raises(RuntimeError, match="warmed Slot binding"):
            runner.execute(tokens, positions, metadata, hidden)
    runner.attention_backend.prepare.assert_not_called()
    assert runner._capture.call_count == 1


@pytest.mark.parametrize(
    "hidden,topk",
    [
        (torch.zeros(1, 4), None),
        (torch.zeros(2), None),
        (torch.zeros(2, 4), torch.zeros(1, 1, 8, dtype=torch.int32)),
        (None, torch.zeros(2, 1, 8, dtype=torch.int32)),
    ],
)
def test_mtp_invalid_input_is_rejected_before_capture(
    runner: PreparedAclGraphRunner, hidden: torch.Tensor | None, topk: torch.Tensor | None
) -> None:
    with pytest.raises(ValueError, match="hidden state|MTP top-k"):
        runner.warmup_prepared(torch.zeros(2), torch.zeros(2), _metadata(2), hidden, topk)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


@pytest.mark.parametrize("reuse_topk", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
@torch.inference_mode()
def test_mtp_real_graph_replays_live_inputs_across_slots(reuse_topk: bool, dp_size: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.model_executor.forward_context import get_forward_context

    class MtpModel(torch.nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            positions: torch.Tensor,
            hidden: torch.Tensor,
            topk: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            kv = get_forward_context().metadata.kv_seq_lens
            indices = positions.view(-1, 1, 1) + 1 if topk is None else topk + 1
            output = hidden + (tokens + kv).to(hidden.dtype).view(-1, 1)
            return output + indices.sum(dim=(1, 2)).to(hidden.dtype).view(-1, 1), indices

    device = torch.device("npu:0")
    backend = SimpleNamespace(
        prepare=lambda *args, **kwargs: None,
        prepare_graph_replay=lambda metadata: None,
        is_mla=False,
    )
    runner = PreparedAclGraphRunner(MtpModel(), backend, device, 4 * dp_size, dp_size)
    task_stream = torch.npu.Stream(device=device)
    inputs = []
    # Two Slots, each with the first repair/current call and two subsequent
    # calls. Every invocation has its own persistent hidden/TopK input.
    for _ in range(2):
        for step, rows in enumerate((4, 2, 2)):
            tokens = torch.zeros(rows, dtype=torch.int32, device=device)
            positions = torch.zeros_like(tokens)
            hidden = torch.zeros(rows, 4, device=device)
            topk = torch.zeros(rows, 1, 2, dtype=torch.int32, device=device) if step and reuse_topk else None
            metadata = _metadata(rows)
            for name in ("slot_mapping", "block_table", "q_seq_lens", "q_cu_seq_lens", "kv_seq_lens"):
                setattr(metadata, name, getattr(metadata, name).to(device))
            metadata.dp_execution_token_counts = (rows,) * dp_size
            metadata.dp_is_decode = (1,) * dp_size
            task_stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(task_stream):
                runner.warmup_prepared(tokens, positions, metadata, hidden, topk)
            inputs.append((tokens, positions, metadata, hidden, topk))
    assert len(runner._prepared_graphs) == 6
    for round_id, index in enumerate((0, 1, 2, 3, 4, 5, 0, 4, 1)):
        tokens, positions, metadata, hidden, topk = inputs[index]
        tokens.fill_(round_id + 1)
        positions.fill_(round_id + 3)
        hidden.fill_(round_id + 5)
        # Host upper bounds stay unchanged while the accepted Device lengths
        # and previous draft outputs arrive immediately before replay.
        metadata.kv_seq_lens.fill_(round_id + 7)
        if topk is not None:
            topk.fill_(round_id + 9)
        task_stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(task_stream):
            output, indices = runner.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk)
        torch.npu.current_stream(device).wait_stream(task_stream)
        expected_indices = positions.view(-1, 1, 1) + 1 if topk is None else topk + 1
        expected = hidden + (tokens + metadata.kv_seq_lens).to(hidden.dtype).view(-1, 1)
        expected += expected_indices.sum(dim=(1, 2)).to(hidden.dtype).view(-1, 1)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(indices, expected_indices)
        assert metadata.kv_seq_lens_host_values == [1] * tokens.numel()
    assert len(runner._prepared_graphs) == 6
    assert runner.prepared_replays == 9


@pytest.mark.parametrize("dcp_rank", [0, 3])
@torch.inference_mode()
def test_dcp_real_graph_replays_device_lengths_across_pages_and_slots(dcp_rank: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
    from xllm.python.model_executor.forward_context import get_forward_context

    class DcpGroup:
        def size(self) -> int:
            return 4

        def rank(self) -> int:
            return dcp_rank

    device = torch.device("npu:0")
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=device,
        dtype=torch.bfloat16,
        dcp_group=DcpGroup(),
        index_topk=2048,
        max_num_reqs=4,
    )
    physical_page_size = 4
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(8, physical_page_size, 1, 512, device=device, dtype=torch.bfloat16),
                value=torch.empty(8, physical_page_size, 1, 64, device=device, dtype=torch.bfloat16),
                index=torch.empty(32, physical_page_size, 1, 128, device=device, dtype=torch.bfloat16),
            )
        ]
    )

    class DcpMetadataModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._bindings: dict[int, tuple[torch.Tensor, ...]] = {}

        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            context = get_forward_context()
            dcp = backend._sfa_metadata.dcp_context
            expanded = backend._expanded_indexer_block_table
            self._bindings[id(context.execution_state)] = (dcp.slot_mapping, dcp.seq_lens, expanded)
            return torch.stack((dcp.slot_mapping, dcp.seq_lens, expanded[:, 0], expanded[:, -1]), dim=1)

    model = DcpMetadataModel()
    runner = PreparedAclGraphRunner(model, backend, device, 4)
    task_stream = torch.npu.Stream(device=device)
    inputs = []
    for _ in range(2):
        tokens = torch.zeros(4, dtype=torch.int32, device=device)
        positions = torch.tensor([32, 32, 32, 0], dtype=torch.int32, device=device)
        metadata = _metadata(4)
        metadata.block_table = torch.tensor([[1, 2, 3]] * 3 + [[0, 0, 0]], dtype=torch.int32, device=device)
        metadata.slot_mapping = torch.tensor([48, 48, 48, -1], dtype=torch.int32, device=device)
        for name in ("q_seq_lens", "q_cu_seq_lens"):
            setattr(metadata, name, getattr(metadata, name).to(device))
        metadata.kv_seq_lens = torch.tensor([33, 33, 33, 1], dtype=torch.int32, device=device)
        metadata.kv_seq_lens_host_values = [33, 33, 33, 1]
        metadata.prepared_attention_state = backend.prepare_metadata(metadata)
        task_stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(task_stream):
            runner.warmup_prepared(tokens, positions, metadata)
        entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
        inputs.append((tokens, positions, metadata, entry))
    torch.npu.current_stream(device).wait_stream(task_stream)
    bindings = [model._bindings[id(entry.execution_state)] for *_, entry in inputs]
    for first, second in zip(*bindings):
        assert first.data_ptr() != second.data_ptr()

    # Native speculative invocations can advance Device lengths after Prepare
    # while their Host upper bounds stay fixed. Include both sides of a logical
    # page boundary and a padded lane, alternating the warmed Slot bindings.
    for step, index in enumerate((0, 1, 0, 1, 1, 0)):
        tokens, positions, metadata, entry = inputs[index]
        lengths = [15, 16, 17, 1] if step % 2 == 0 else [31, 32, 33, 1]
        blocks = [1 + step % 2, 3 + step % 2, 5 + step % 2]
        logical_slots = [blocks[(length - 1) // 16] * 16 + (length - 1) % 16 for length in lengths[:3]] + [-1]
        inactive = [tensor.cpu().clone() for tensor in bindings[1 - index]]
        metadata.block_table.copy_(torch.tensor([blocks] * 3 + [[0, 0, 0]], dtype=torch.int32, device=device))
        metadata.slot_mapping.copy_(torch.tensor(logical_slots, dtype=torch.int32, device=device))
        metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        positions.copy_(torch.tensor([length - 1 for length in lengths], dtype=torch.int32, device=device))
        metadata.prepared_attention_state = backend.prepare_metadata(metadata)
        task_stream.wait_stream(torch.npu.current_stream(device))
        with (
            patch.object(backend._kv_layout, "localize_slots", side_effect=AssertionError("replay derived slots")),
            patch.object(backend._kv_layout, "local_seq_lens", side_effect=AssertionError("replay derived lengths")),
            patch.object(
                backend._kv_layout, "expand_indexer_block_table", side_effect=AssertionError("replay expanded pages")
            ),
            torch.npu.stream(task_stream),
        ):
            output = runner.execute(tokens, positions, metadata)
        torch.npu.current_stream(device).wait_stream(task_stream)
        assert backend._metadata is metadata.prepared_attention_state
        expected_slots = [
            slot // 16 * 4 + slot % 4 if slot >= 0 and slot % 16 // 4 == dcp_rank else -1 for slot in logical_slots
        ]
        expected_lengths = [sum(position % 16 // 4 == dcp_rank for position in range(length)) for length in lengths]
        expected = torch.tensor(
            [
                [slot, length, blocks[0] * 4 if row < 3 else 0, blocks[-1] * 4 + 3 if row < 3 else 3]
                for row, (slot, length) in enumerate(zip(expected_slots, expected_lengths))
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(output.cpu(), expected)
        for tensor, saved in zip(bindings[1 - index], inactive):
            torch.testing.assert_close(tensor.cpu(), saved)
        assert metadata.kv_seq_lens_host_values == [33, 33, 33, 1]
        assert len(runner._prepared_graphs) == 2
    assert runner.prepared_replays == 6

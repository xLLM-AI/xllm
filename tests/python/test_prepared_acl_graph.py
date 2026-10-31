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
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from xllm.python.attention.backend import AttentionBackend, LinearVerifyGraphInputs
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.dspark_ragged import allocate_verify_prefixes, create_verify_buffers, pack_verify_inputs
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner


@pytest.mark.parametrize("limit", [-1, 0, 4096])
def test_prepared_capacity_selects_warmed_slot_owned_tables(
    runner: PreparedAclGraphRunner, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    runner._device_kv_lengths = True
    runner._block_draft_graph_warmup_max_kv_len = limit
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(
        side_effect=lambda metadata, **_: SimpleNamespace(
            device_kv_lengths=metadata.kv_seq_lens,
            kv_capacity=metadata.kv_seq_lens_host_values[0],
        )
    )
    bindings = []
    for slot in range(2):
        tokens = torch.arange(2, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(2)
        metadata.block_table = (torch.arange(130, dtype=torch.int32) + slot * 1000).view(2, 65)
        metadata.kv_seq_lens_host_values = [512, 512]
        metadata.kv_seq_lens.fill_(511)
        runner.warmup_prepared(tokens, positions, metadata, pre_capture_kv_buckets=True)
        bindings.append((tokens, positions, metadata))
    assert runner._capture.call_count == 6
    assert len(runner._prepared_graphs) == 6
    entries = list(runner._prepared_graphs.values())
    for entry in entries:
        entry.static_output = (torch.empty(2, 4), torch.empty(2, 8))
    for tokens, positions, metadata in bindings:
        outputs = runner.get_prepared_aux_hidden_states(tokens, positions, metadata)
        expected = [entry.static_output[1] for entry in entries if entry.static_input_ids is tokens]
        assert len(outputs) == 3
        assert [output.data_ptr() for output in outputs] == [output.data_ptr() for output in expected]
        with pytest.raises(RuntimeError, match="warmed Slot binding"):
            runner.get_prepared_aux_hidden_states(tokens.clone(), positions, metadata)
    assert len({entry.static_metadata.block_table.data_ptr() for entry in entries}) == 6
    for entry in entries:
        table = entry.static_metadata.block_table
        assert table.is_contiguous()
        assert table.shape == (2, entry.kv_capacity_pages)
        assert entry.static_metadata.kv_seq_lens_host_values == [entry.kv_capacity_pages * 128] * 2
        assert entry.static_metadata.q_cu_seq_lens_host_values == [1, 2]
        assert all(
            entry.static_metadata.q_cu_seq_lens_host_values is not metadata.q_cu_seq_lens_host_values
            for _, _, metadata in bindings
        )
        assert table.data_ptr() not in {metadata.block_table.data_ptr() for _, _, metadata in bindings}

    def _reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("Host tier selection must not read device KV lengths")

    selected = {}
    for slot, bound, pages in ((0, 512, 4), (1, 513, 32), (0, 4097, 65), (1, 512, 4), (0, 512, 4)):
        tokens, positions, metadata = bindings[slot]
        metadata.kv_seq_lens_host_values[:] = [bound, bound]
        metadata.kv_seq_lens.fill_(min(bound, 7))
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "cpu", _reject_readback)
            guard.setattr(torch.Tensor, "item", _reject_readback)
            guard.setattr(torch.Tensor, "tolist", _reject_readback)
            entry = runner._resolve_entry(tokens, positions, metadata, None, None)
        assert entry.kv_capacity_pages == pages
        key = (slot, pages)
        if key in selected:
            assert selected[key] is entry
        selected[key] = entry
        torch.testing.assert_close(entry.static_metadata.block_table, metadata.block_table[:, :pages])
    assert runner._capture.call_count == 6
    tokens, positions, metadata = bindings[0]
    metadata.kv_seq_lens_host_values[:] = [8321, 1]
    with pytest.raises(ValueError, match="capacity"):
        runner._resolve_entry(tokens, positions, metadata, None, None)
    assert runner._capture.call_count == 6


def test_ordinary_capacity_key_changes_without_device_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SimpleNamespace(page_size=128, logical_page_size=128, is_mla=False)
    runner = DecodeAclGraphRunner(torch.nn.Identity(), backend, torch.device("cpu"), 2, 8192, device_kv_lengths=True)
    tokens = torch.zeros(2, dtype=torch.int32)
    metadata = _metadata(2)
    metadata.block_table = torch.zeros(2, 65, dtype=torch.int32)
    # The live lengths below stay at one token. Keep their row-aligned CSR
    # valid while conservative Host bounds independently select graph tiers.
    metadata.paged_kv_indptr = torch.arange(3, dtype=torch.int32)
    metadata.paged_kv_indices = torch.zeros(2, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.ones(2, dtype=torch.int32)
    keys = {}

    def _reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("Capacity graph keys use conservative Host bounds, not device values")

    for bound, pages in ((511, 4), (512, 4), (513, 32), (4096, 32), (4097, 65), (512, 4)):
        metadata.kv_seq_lens_host_values[:] = [bound, 1]
        metadata.kv_seq_lens.fill_(1)
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "cpu", _reject_readback)
            guard.setattr(torch.Tensor, "item", _reject_readback)
            guard.setattr(torch.Tensor, "tolist", _reject_readback)
            key, _, _ = runner._resolve_graph_key(tokens, metadata, 2, None, None, None)
        assert key[-1] == pages
        if pages in keys:
            assert keys[pages] == key
        keys[pages] = key
    assert len(set(keys.values())) == 3
    runner._prepare_graph_entry = Mock(side_effect=AssertionError("runtime capture is forbidden"))
    with pytest.raises(RuntimeError, match="startup warmup"):
        runner.warmup(tokens, torch.zeros_like(tokens), metadata)
    runner._prepare_graph_entry.assert_not_called()
    metadata.kv_seq_lens_host_values[:] = [8321, 1]
    with pytest.raises(ValueError, match="capacity"):
        runner._resolve_graph_key(tokens, metadata, 2, None, None, None)
    metadata.kv_seq_lens_host_values[:] = [513, 1]
    metadata.block_table = torch.zeros(2, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="source block table"):
        runner._resolve_graph_key(tokens, metadata, 2, None, None, None)


@pytest.fixture(autouse=True)
def _isolate_graph_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # torch_npu cannot recapture a pool handle after its last graph has been
    # released. A serving runner lives for the worker lifetime; independent
    # unit tests must not inherit a retired pool from the previous test.
    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("request_count", [1, 2, 3])
@pytest.mark.parametrize("cache_pages,sliding_window", [(1, 0), (64, 0), (64, 2048), (514, 2048)])
@pytest.mark.parametrize("warmup_limit", [-1, 0], ids=["lazy", "prewarmed"])
@torch.inference_mode()
def test_real_dspark_prepared_graph_matches_eager_with_changing_device_lengths(
    request_count: int, cache_pages: int, sliding_window: int, warmup_limit: int
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.layers.embedding import HiddenParallelEmbedding
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
    from xllm.python.models.qwen3_dspark import Qwen3DSparkForCausalLM

    config = {
        "model_type": "DSparkDraftModel",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "intermediate_size": 128,
        "max_position_embeddings": cache_pages * 128,
        "sliding_window": sliding_window,
        "vocab_size": 8,
        "markov_rank": 2,
        "dtype": "bfloat16",
        "device": "npu:0",
        "num_speculative_tokens": 3,
        "python_graph_backend": "aclgraph",
        "enable_task_pipeline": True,
        "block_draft_graph_warmup_max_kv_len": warmup_limit,
    }
    torch.manual_seed(31)
    model = Qwen3DSparkForCausalLM(config)
    model.model.embed_tokens = HiddenParallelEmbedding(8, 64, 1, dtype=model.dtype, device=model.device)
    model.model.fc.load_weight(torch.randn(64, 128), tp_rank=0)
    for parameter in model.parameters():
        parameter.normal_(std=0.1)
    model.model._build_context_kv_buffers()
    caches = [
        (
            torch.randn(request_count * cache_pages, 128, 1, 64, dtype=torch.bfloat16, device="npu:0"),
            torch.randn(request_count * cache_pages, 128, 1, 64, dtype=torch.bfloat16, device="npu:0"),
        )
        for _ in range(2)
    ]
    executor = ModelExecutor(model, config, max_seqs_per_batch=request_count)
    executor.bind_kv_caches([LayerCache(*pair) for pair in caches])
    tokens = torch.tensor([1, 0, 0], dtype=torch.int32, device="npu:0").repeat(request_count)
    positions = torch.arange(3, dtype=torch.int32, device="npu:0").repeat(request_count)
    page_offsets = torch.arange(request_count, dtype=torch.int32, device="npu:0").view(-1, 1) * cache_pages * 128
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=(positions.view(request_count, 3) + page_offsets).flatten(),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        block_table=torch.arange(request_count * cache_pages, dtype=torch.int32, device="npu:0").view(
            request_count, cache_pages
        ),
        q_seq_lens=torch.full((request_count,), 3, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(3, request_count * 3 + 1, 3, dtype=torch.int32, device="npu:0"),
        q_cu_seq_lens_host_values=list(range(3, request_count * 3 + 1, 3)),
        kv_seq_lens=torch.full((request_count,), 8, dtype=torch.int32, device="npu:0"),
        kv_seq_lens_host_values=[16] * request_count,
        is_chunked_prefill=True,
    )
    stream = torch.npu.Stream(device="npu:0")
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        executor.prepare_metadata(metadata)
        executor.warmup_prepared_graph(tokens, positions, metadata)
        entries = len(executor.prepared_graph_runner._prepared_graphs)
        if cache_pages == 1:
            lengths_to_test = (8, 13, 6, 16)
        elif cache_pages == 64:
            lengths_to_test = (8, 4090, 4097, 6112, 16)
        else:
            lengths_to_test = (8, 4090, 4097, 16383, 16385, 32767, 32769, 65535, 65537, 16)
        for length in lengths_to_test:
            lengths = torch.arange(request_count, dtype=torch.int32, device="npu:0") + length
            positions.copy_((lengths.view(-1, 1) - 3 + torch.arange(3, device="npu:0")).flatten())
            metadata.slot_mapping.copy_((positions.view(request_count, 3) + page_offsets).flatten())
            metadata.kv_seq_lens.copy_(lengths)
            metadata.kv_seq_lens_host_values[:] = [length + index for index in range(request_count)]
            executor.prepare_metadata(metadata)
            reference = executor.execute(tokens, positions, metadata).clone()
            actual = executor.execute(tokens, positions, metadata, enable_graph=True)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference)
        expected_entries = entries + int(sliding_window == 0 and cache_pages > 1 and warmup_limit == -1)
        assert len(executor.prepared_graph_runner._prepared_graphs) == expected_entries
        assert executor.prepared_graph_runner._windowed_block_draft == (sliding_window > 0)
        if sliding_window > 0:
            assert entries == 1
        assert executor.prepared_graph_runner.prepared_replays == len(lengths_to_test)
        if request_count > 1:
            previous = actual.clone()
            for keys, values in caches:
                keys[0].zero_()
                values[0].zero_()
            reference = executor.execute(tokens, positions, metadata).clone()
            actual = executor.execute(tokens, positions, metadata, enable_graph=True)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(actual.view(request_count, 3, -1)[1:], previous.view(request_count, 3, -1)[1:])
            assert len(executor.prepared_graph_runner._prepared_graphs) == expected_entries


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
        linear_state_indices=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        prepared_attention_state=object(),
    )


def _block_metadata(request_count: int, query_width: int, pages: int = 64) -> SimpleNamespace:
    metadata = _metadata(request_count)
    metadata.slot_mapping = torch.arange(request_count * query_width, dtype=torch.int32)
    metadata.block_table = torch.arange(request_count * pages, dtype=torch.int32).view(request_count, pages)
    metadata.q_seq_lens = torch.full((request_count,), query_width, dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = list(range(query_width, request_count * query_width + 1, query_width))
    metadata.q_cu_seq_lens = torch.tensor(metadata.q_cu_seq_lens_host_values, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2048] * request_count
    metadata.kv_seq_lens = torch.full((request_count,), 2048, dtype=torch.int32)
    metadata.is_chunked_prefill = True
    return metadata


def _adaptive_inputs(
    device: torch.device, table_width: int = 2, graph_rows: int = 8, draft_width: int = 7
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace, DSparkAdaptiveGraphInputs]:
    capacity = 3
    metadata = _metadata(graph_rows)
    for name in ("slot_mapping", "q_seq_lens", "q_cu_seq_lens", "kv_seq_lens"):
        setattr(metadata, name, getattr(metadata, name).to(device))
    metadata.block_table = torch.zeros(graph_rows, table_width, dtype=torch.int32, device=device)
    tokens = torch.zeros(graph_rows, dtype=torch.int32, device=device)
    positions = torch.zeros_like(tokens)
    inputs = DSparkAdaptiveGraphInputs(
        arena=torch.zeros((12 * draft_width + 4) * capacity, dtype=torch.uint8, device=device),
        control=torch.tensor([1], dtype=torch.int64, device=device),
        anchors=torch.arange(100, 100 + capacity, dtype=torch.int64, device=device),
        base_positions=torch.full((capacity,), 1, dtype=torch.int32, device=device),
        base_kv_lengths=torch.full((capacity,), 2, dtype=torch.int32, device=device),
        base_block_table=torch.arange(capacity * table_width, dtype=torch.int32, device=device).view(
            capacity, table_width
        )
        + 10,
        offsets=torch.full((capacity + 1,), -99, dtype=torch.int32, device=device),
        owners=torch.full((graph_rows,), -99, dtype=torch.int32, device=device),
        columns=torch.full((graph_rows,), -99, dtype=torch.int32, device=device),
        block_size=128,
        draft_width=draft_width,
    )
    return tokens, positions, metadata, inputs


def _adaptive_tensors(
    tokens: torch.Tensor, positions: torch.Tensor, metadata: SimpleNamespace, inputs: DSparkAdaptiveGraphInputs
) -> tuple[torch.Tensor, ...]:
    return (
        tokens,
        positions,
        metadata.slot_mapping,
        metadata.block_table,
        metadata.q_seq_lens,
        metadata.q_cu_seq_lens,
        metadata.kv_seq_lens,
        inputs.arena,
        inputs.control,
        inputs.anchors,
        inputs.base_positions,
        inputs.base_kv_lengths,
        inputs.base_block_table,
        inputs.offsets,
        inputs.owners,
        inputs.columns,
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


def test_windowed_block_graph_keeps_full_source_binding_across_context_boundaries(
    runner: PreparedAclGraphRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner._block_draft = True
    runner._windowed_block_draft = True
    runner.max_batch = 16
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.is_mla = False
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    bindings = []
    for slot in range(2):
        tokens = torch.arange(14, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(2)
        metadata.slot_mapping = torch.full((14,), -1, dtype=torch.int32)
        metadata.q_seq_lens.fill_(7)
        metadata.q_cu_seq_lens = torch.tensor([7, 14], dtype=torch.int32)
        metadata.q_cu_seq_lens_host_values = [7, 14]
        metadata.block_table = (torch.arange(1026, dtype=torch.int32) + slot * 1026).view(2, 513)
        metadata.kv_seq_lens_host_values = [16, 16]
        metadata.kv_seq_lens.fill_(7)
        metadata.is_chunked_prefill = True
        runner.warmup_prepared(tokens, positions, metadata, pre_capture_kv_buckets=True)
        bindings.append((tokens, positions, metadata))
    assert runner._capture.call_count == 2

    def reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("window graph admission must not read Device lengths")

    for tokens, positions, metadata in bindings:
        for bound in (16, 4097, 16385, 32769, 65537, 16):
            metadata.kv_seq_lens_host_values[:] = [bound, bound]
            with monkeypatch.context() as guard:
                for name in ("item", "tolist", "cpu"):
                    guard.setattr(torch.Tensor, name, reject_readback)
                entry = runner._resolve_entry(tokens, positions, metadata)
            assert entry.kv_capacity_pages == 513
            assert entry.static_metadata.block_table.data_ptr() == metadata.block_table.data_ptr()
            assert entry.static_metadata.block_table.shape == (2, 513)
    assert runner._capture.call_count == 2
    assert len(runner._prepared_graphs) == 2
    metadata.kv_seq_lens_host_values[:] = [65665, 16]
    with pytest.raises(ValueError, match="capacity"):
        runner._resolve_entry(tokens, positions, metadata)


@pytest.mark.parametrize("draft_width", [5, 7])
def test_adaptive_prepared_binding_ignores_live_device_control(
    runner: PreparedAclGraphRunner, draft_width: int
) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), draft_width=draft_width)
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    binding = runner._prepared_binding(tokens, positions, metadata, dspark_adaptive=inputs)
    entry = runner._prepared_graphs[binding]
    for count, suppress in ((3, False), (2, True), (1, False)):
        inputs.control.fill_(count | (int(suppress) << 32))
        inputs.arena.fill_(count)
        inputs.anchors.add_(1)
        inputs.base_positions.add_(1)
        inputs.base_kv_lengths.add_(1)
        inputs.base_block_table.add_(1)
        runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
        assert runner._prepared_binding(tokens, positions, metadata, dspark_adaptive=inputs) == binding
        assert runner._prepared_graphs[binding] is entry
    assert runner._capture.call_count == 1
    assert runner.prepared_replays == 3
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()


@pytest.mark.parametrize("dp_rank", [0, 1])
@pytest.mark.parametrize("draft_width", [5, 7])
def test_adaptive_prepared_dp_requires_common_physical_rows(
    runner: PreparedAclGraphRunner, dp_rank: int, draft_width: int
) -> None:
    runner.max_batch = 8
    runner.dp_size = 2
    runner.dp_rank = dp_rank
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), draft_width=draft_width)
    metadata.dp_execution_token_counts = (8, 8)
    metadata.dp_is_decode = (True, True)
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    binding = runner._prepared_binding(tokens, positions, metadata, dspark_adaptive=inputs)
    for raw_counts in ((0, 1), (3, 2)):
        metadata.raw_dp_execution_token_counts = raw_counts
        inputs.control.fill_(1 | (int(raw_counts[dp_rank] == 0) << 32))
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
        assert runner._prepared_binding(tokens, positions, metadata, dspark_adaptive=inputs) == binding
    assert runner._capture.call_count == 1
    metadata.dp_execution_token_counts = (8, 4)
    with pytest.raises(ValueError, match="execution counts|same decode batch"):
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)


@pytest.mark.parametrize(
    "field",
    [
        "arena",
        "control",
        "anchors",
        "base_positions",
        "base_kv_lengths",
        "base_block_table",
        "offsets",
        "owners",
        "columns",
        "block_size",
        "missing",
    ],
)
def test_adaptive_prepared_binding_rejects_replaced_producers(runner: PreparedAclGraphRunner, field: str) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"))
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    if field == "missing":
        replacement = None
    elif field == "block_size":
        replacement = replace(inputs, block_size=64)
    else:
        replacement = replace(inputs, **{field: getattr(inputs, field).clone()})
    runner.attention_backend.prepare.reset_mock()
    runner.attention_backend.prepare_graph_replay.reset_mock()
    with pytest.raises((RuntimeError, ValueError), match="warmed Slot binding|block size"):
        runner.execute(tokens, positions, metadata, dspark_adaptive=replacement)
    runner.attention_backend.prepare.assert_not_called()
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    assert runner._capture.call_count == 1


@pytest.mark.parametrize(
    "field",
    [
        "arena",
        "control",
        "anchors",
        "base_positions",
        "base_kv_lengths",
        "base_block_table",
        "offsets",
        "owners",
        "columns",
    ],
)
def test_adaptive_prepared_rejects_invalid_capacity_layout_before_capture(
    runner: PreparedAclGraphRunner, field: str
) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"))
    malformed = replace(inputs, **{field: getattr(inputs, field).to(torch.float64)})
    with pytest.raises(ValueError, match="adaptive DSpark requires contiguous"):
        runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=malformed)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


def test_adaptive_and_ordinary_prepared_bindings_can_coexist(runner: PreparedAclGraphRunner) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"))
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    runner.warmup_prepared(tokens, positions, metadata)
    runner.execute(tokens, positions, metadata)
    runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
    assert runner._capture.call_count == 2
    assert len(runner._prepared_graphs) == 2


@pytest.mark.parametrize("live,suppress", [(0, False), (1, False), (2, False), (3, False), (2, True)])
@pytest.mark.parametrize("packed_kernel", [False, True], ids=["dense", "varlen"])
def test_recurrent_verify_maps_refresh_request_prefixes_and_zero_padding(
    live: int, suppress: bool, packed_kernel: bool
) -> None:
    _, _, _, inputs = _adaptive_inputs(torch.device("cpu"))
    inputs = replace(
        inputs,
        linear_state_indices=torch.tensor([2, 1, 3], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([4, 2, 1], dtype=torch.int32),
    )
    inputs.control.fill_(live | (int(suppress) << 32))
    widths = [2, 3, 1][:live]
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    inputs.offsets.fill_(-99)
    inputs.offsets[: live + 1].copy_(torch.tensor(offsets, dtype=torch.int32))
    inputs.owners.fill_(-1)
    inputs.columns.fill_(-1)
    for row, width in enumerate(widths):
        inputs.owners[offsets[row] : offsets[row + 1]].fill_(row)
        inputs.columns[offsets[row] : offsets[row + 1]].copy_(torch.arange(width))
    graph = LinearVerifyGraphInputs(
        torch.empty(3, dtype=torch.int32),
        torch.empty(3, dtype=torch.int32),
        8,
        None if packed_kernel else torch.empty(8, dtype=torch.int64),
        None if packed_kernel else torch.empty(24, dtype=torch.int64),
        torch.empty(4, dtype=torch.int32) if packed_kernel else None,
    )
    tensors = tuple(
        tensor
        for tensor in (
            graph.state_indices,
            graph.num_accepted_tokens,
            graph.dense_to_packed,
            graph.packed_to_dense,
            graph.cu_seqlens,
        )
        if tensor is not None
    )
    addresses = tuple(tensor.data_ptr() for tensor in tensors)
    PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph, 8)
    expected_dense = [8] * 24
    expected_packed = [24] * 8
    for row, width in enumerate(widths):
        expected_dense[row * 8 : row * 8 + width] = range(offsets[row], offsets[row + 1])
        expected_packed[offsets[row] : offsets[row + 1]] = range(row * 8, row * 8 + width)
    if not packed_kernel:
        assert graph.dense_to_packed.tolist() == expected_dense
        assert graph.packed_to_dense.tolist() == expected_packed
    assert graph.state_indices.tolist() == ([0] * 3 if suppress else [2, 1, 3][:live] + [0] * (3 - live))
    assert graph.num_accepted_tokens.tolist() == [4, 2, 1][:live] + [1] * (3 - live)
    if packed_kernel:
        assert graph.cu_seqlens.tolist() == ([0] * 4 if suppress else offsets + [offsets[-1]] * (3 - live))
    assert addresses == tuple(tensor.data_ptr() for tensor in tensors)


def test_varlen_verify_does_not_refresh_legacy_token_maps() -> None:
    _, _, _, inputs = _adaptive_inputs(torch.device("cpu"))
    inputs = replace(
        inputs,
        linear_state_indices=torch.tensor([2, 1, 3], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([4, 2, 1], dtype=torch.int32),
    )
    graph = LinearVerifyGraphInputs(
        torch.empty(3, dtype=torch.int32),
        torch.empty(3, dtype=torch.int32),
        8,
        torch.full((8,), -99, dtype=torch.int64),
        torch.full((24,), -99, dtype=torch.int64),
        torch.empty(4, dtype=torch.int32),
    )
    PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph, 8)
    assert graph.packed_to_dense.tolist() == [-99] * 8
    assert graph.dense_to_packed.tolist() == [-99] * 24


@torch.inference_mode()
def test_recurrent_prepared_forward_fuses_state_refresh_with_input_packing(
    runner: PreparedAclGraphRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = False
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), table_width=128)
    inputs = replace(
        inputs,
        linear_state_indices=torch.ones(3, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    entry = next(iter(runner._prepared_graphs.values()))
    graph = entry.static_metadata.linear_verify_graph
    fused = Mock()
    legacy_pack = Mock(side_effect=AssertionError("legacy pack must not run"))
    legacy_refresh = Mock(side_effect=AssertionError("unfused metadata must not run"))
    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_allocate_verify_prefixes_graph", Mock())
    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_pack_recurrent_verify_inputs_graph", fused)
    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_pack_verify_inputs_graph", legacy_pack)
    monkeypatch.setattr(runner, "_refresh_linear_verify_graph", legacy_refresh)
    with patch.object(AclGraphRunner, "_forward_static", return_value=torch.empty(0)):
        runner._forward_static(entry)
    fused.assert_called_once()
    arguments = fused.call_args.args
    assert arguments[15] is inputs.linear_state_indices
    assert arguments[16] is inputs.num_accepted_tokens
    assert arguments[17] is graph.state_indices
    assert arguments[18] is graph.num_accepted_tokens
    assert arguments[19] is graph.cu_seqlens
    legacy_pack.assert_not_called()
    legacy_refresh.assert_not_called()


@pytest.mark.parametrize("varlen", [False, True])
def test_recurrent_prepared_graph_binds_request_state_and_compact_device_lengths(
    runner: PreparedAclGraphRunner,
    varlen: bool,
) -> None:
    runner._varlen_recurrent_verify = varlen
    runner.max_batch = 8
    runner.attention_backend.is_mla = False
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), table_width=128)
    inputs = replace(
        inputs,
        linear_state_indices=torch.ones(3, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    entry = next(iter(runner._prepared_graphs.values()))
    graph = entry.static_metadata.linear_verify_graph
    assert graph is not None and graph.query_width == 8
    if varlen:
        assert graph.packed_to_dense is None and graph.dense_to_packed is None
        assert graph.cu_seqlens.shape == (4,)
    else:
        assert graph.cu_seqlens is None
        assert graph.packed_to_dense.shape == (8,)
        assert graph.dense_to_packed.shape == (24,)
    assert entry.source_block_table is metadata.block_table
    assert entry.static_metadata.block_table.shape == (8, 32)
    assert entry.static_metadata.kv_seq_lens_host_values == [4096] * 8
    assert entry.kv_capacity == 4096
    assert entry.static_metadata.q_cu_seq_lens_host_values == list(range(1, 9))
    runner.attention_backend.prepare_metadata.assert_called_once_with(
        entry.static_metadata, device_kv_lengths=True, workspace_query_capacity=8
    )
    for live in (1, 3, 2):
        inputs.control.fill_(live)
        metadata.linear_state_indices = inputs.linear_state_indices[:live]
        metadata.linear_state_read_indices = metadata.linear_state_indices
        metadata.linear_state_write_indices = metadata.linear_state_indices
        metadata.has_initial_state = torch.ones(live, dtype=torch.bool)
        metadata.num_accepted_tokens = inputs.num_accepted_tokens[:live]
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
    assert runner._capture.call_count == 1
    assert len(runner._prepared_graphs) == 1
    assert entry.static_metadata.linear_state_indices is None
    assert entry.static_metadata.num_accepted_tokens is None
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(
            tokens,
            positions,
            metadata,
            dspark_adaptive=replace(inputs, linear_state_indices=inputs.linear_state_indices.clone()),
        )
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(
            tokens,
            positions,
            metadata,
            dspark_adaptive=replace(inputs, num_accepted_tokens=inputs.num_accepted_tokens.clone()),
        )


@pytest.mark.parametrize("invalid", ["missing", "empty", "rows", "zero", "negative", "capacity"])
def test_recurrent_prepared_rejects_invalid_host_bounds_before_launch(
    runner: PreparedAclGraphRunner, invalid: str
) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = False
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), table_width=128)
    inputs = replace(
        inputs,
        linear_state_indices=torch.ones(3, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    entry = next(iter(runner._prepared_graphs.values()))
    original = entry.static_metadata.kv_seq_lens_host_values.copy()
    metadata.kv_seq_lens_host_values = {
        "missing": None,
        "empty": [],
        "rows": [1] * 7,
        "zero": [0] * 8,
        "negative": [-1] * 8,
        "capacity": [16385] * 8,
    }[invalid]
    runner.attention_backend.prepare_graph_replay.reset_mock()
    with pytest.raises(ValueError, match="KV bound"):
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
    entry.graph.replay.assert_not_called()
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    assert entry.static_metadata.kv_seq_lens_host_values == original
    assert runner.prepared_replays == 0


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("window", [None, 64])
@pytest.mark.parametrize("rows", [4, 8])
@torch.inference_mode()
def test_recurrent_prepared_fia_bounds_follow_two_slots_without_recapture(
    window: int | None, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import get_forward_context

    # A fully released torch_npu pool handle cannot be captured again. Keep
    # this short-lived runner independent of the suite's shared graph pool.
    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    torch.manual_seed(223)
    device = torch.device("npu:0")
    heads, head_dim, pages = 2, 64, 64
    backend = NpuPagedAttentionBackend(heads, 1, head_dim, head_dim**-0.5, 0, False, device, torch.bfloat16)
    keys = torch.randn(3 * pages + 1, 128, 1, head_dim, device=device, dtype=torch.bfloat16) * 0.1
    values = torch.randn_like(keys) * 0.1
    backend.bind_kv_caches([LayerCache(key=keys, value=values)])
    query = torch.randn(rows, heads, head_dim, device=device, dtype=keys.dtype) * 0.1
    layer = SimpleNamespace(
        causal=True,
        fia_use_attention_mask=window is not None,
        fia_sparse_mode=4 if window is not None else None,
        fia_pre_tokens=window - 1 if window is not None else 2147483647,
        fia_next_tokens=0,
    )

    class AttentionModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            metadata = get_forward_context().metadata
            return backend._prefill(query, query, query, keys, values, metadata, rows, layer)

    runner = PreparedAclGraphRunner(AttentionModel(), backend, device, 8)
    slots = []
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        for _ in range(2):
            tokens, positions, metadata, inputs = _adaptive_inputs(device, table_width=pages, graph_rows=rows)
            inputs = replace(
                inputs,
                linear_state_indices=torch.arange(1, 4, device=device, dtype=torch.int32),
                num_accepted_tokens=torch.ones(3, device=device, dtype=torch.int32),
            )
            inputs.base_block_table.copy_(
                torch.arange(1, 3 * pages + 1, device=device, dtype=torch.int32).view(3, pages)
            )
            metadata.kv_seq_lens_host_values[:] = [4097] * rows
            runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
            entry = runner._resolve_entry(tokens, positions, metadata, dspark_adaptive=inputs)
            runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
            torch.npu.synchronize()
            assert len(entry.graph_tasks) == (1 if window is None else 0)
            assert entry.kv_capacity == 8192
            assert entry.static_metadata.prepared_attention_state.workspace_query_capacity == 8
            prepared = entry.static_metadata.prepared_attention_state
            workspace_ptr = entry.execution_state.paged_attention[rows].workspace.data_ptr() if window is None else None
            slots.append((tokens, positions, metadata, inputs, entry, prepared, workspace_ptr))

        first_prepared, second_prepared = (slot[5] for slot in slots)
        assert first_prepared.actual_seq_kv is not second_prepared.actual_seq_kv
        if window is None:
            assert slots[0][6] == slots[1][6]
        assert slots[0][4].static_output.data_ptr() != slots[1][4].static_output.data_ptr()
        references = []
        for step, length in enumerate((512, 2039, 4090, 6112, 8152, 512)):
            for slot_index, (tokens, positions, metadata, inputs, entry, prepared, workspace_ptr) in enumerate(slots):
                live = 1 + (step + slot_index) % 3
                inputs.control.fill_(live)
                inputs.base_positions.fill_(length + slot_index)
                inputs.base_kv_lengths.fill_(length + slot_index + 1)
                bound = min(8192, length + 32)
                metadata.kv_seq_lens_host_values[:] = [bound] * rows
                other = slots[1 - slot_index][5]
                previous_other = other.actual_seq_kv.copy()
                actual = runner.execute(tokens, positions, metadata, dspark_adaptive=inputs).clone()
                torch.npu.synchronize()
                assert prepared.actual_seq_kv == [bound] * rows
                if window is None:
                    assert entry.execution_state.paged_attention[rows].workspace.data_ptr() == workspace_ptr
                assert entry.static_metadata.prepared_attention_state.kv_capacity == 8192
                assert entry.kv_capacity == 8192
                assert other.actual_seq_kv == previous_other
                # Readback is only the test oracle. Serving uses the Host
                # conservative bound plus the exact in-graph Device mask.
                metadata.kv_seq_lens_host_values = metadata.kv_seq_lens.cpu().tolist()
                metadata.prepared_attention_state = backend.prepare_metadata(metadata)
                backend.prepare(metadata)
                expected = backend._prefill(query, query, query, keys, values, metadata, rows, layer)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
                assert other.actual_seq_kv == previous_other
                references.append((slot_index, live, length + slot_index, bound, expected.clone()))

        # Native Launch can queue both Slots before either one completes.
        # Exercise repeated grow/shrink task updates with shared scratch and
        # delay all readback until the whole sequence has been submitted.
        pending = []
        for _ in range(2):
            for slot_index, live, position, bound, expected in reversed(references):
                tokens, positions, metadata, inputs, entry, _, workspace_ptr = slots[slot_index]
                inputs.control.fill_(live)
                inputs.base_positions.fill_(position)
                inputs.base_kv_lengths.fill_(position + 1)
                metadata.kv_seq_lens_host_values[:] = [bound] * rows
                actual = runner.execute(tokens, positions, metadata, dspark_adaptive=inputs).clone()
                pending.append((actual, expected))
                if window is None:
                    assert entry.execution_state.paged_attention[rows].workspace.data_ptr() == workspace_ptr
        torch.npu.synchronize()
        for actual, expected in pending:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
        assert len(runner._prepared_graphs) == 2
        assert runner.prepared_replays == 38
    torch.npu.current_stream(device).wait_stream(stream)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@torch.inference_mode()
def test_recurrent_prepared_fia_grows_kv_buckets_without_changing_slot_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import get_forward_context

    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    torch.manual_seed(227)
    device = torch.device("npu:0")
    rows, pages, head_dim = 8, 128, 64
    backend = NpuPagedAttentionBackend(1, 1, head_dim, head_dim**-0.5, 0, False, device, torch.bfloat16)
    keys = torch.randn(3 * pages + 1, 128, 1, head_dim, device=device, dtype=torch.bfloat16) * 0.1
    values = torch.randn_like(keys) * 0.1
    backend.bind_kv_caches([LayerCache(key=keys, value=values)])
    query = torch.randn(rows, 1, head_dim, device=device, dtype=keys.dtype) * 0.1
    layer = SimpleNamespace(
        causal=True, fia_use_attention_mask=False, fia_sparse_mode=None, fia_pre_tokens=2147483647, fia_next_tokens=0
    )

    class AttentionModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return backend._prefill(query, query, query, keys, values, get_forward_context().metadata, rows, layer)

    prepared = PreparedAclGraphRunner(AttentionModel(), backend, device, rows)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    slots = []
    with torch.npu.stream(stream):
        for _ in range(2):
            tokens, positions, metadata, inputs = _adaptive_inputs(device, table_width=pages, graph_rows=rows)
            inputs = replace(
                inputs,
                linear_state_indices=torch.arange(1, 4, dtype=torch.int32, device=device),
                num_accepted_tokens=torch.ones(3, dtype=torch.int32, device=device),
            )
            inputs.base_block_table.copy_(
                torch.arange(1, 3 * pages + 1, dtype=torch.int32, device=device).view(3, pages)
            )
            metadata.kv_seq_lens_host_values[:] = [2048] * rows
            prepared.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
            original = prepared._resolve_entry(tokens, positions, metadata, dspark_adaptive=inputs)
            slots.append((tokens, positions, metadata, inputs, original))
        for length, capacity in ((2040, 4096), (4097, 8192), (8193, 16384), (16000, 16384), (512, 4096)):
            for tokens, positions, metadata, inputs, original in slots:
                inputs.control.fill_(2)
                inputs.base_positions.fill_(length)
                inputs.base_kv_lengths.fill_(length + 1)
                metadata.kv_seq_lens_host_values[:] = [length + 16] * rows
                actual = prepared.execute(tokens, positions, metadata, dspark_adaptive=inputs).clone()
                selected = prepared._resolve_entry(tokens, positions, metadata, dspark_adaptive=inputs)
                assert selected.kv_capacity == capacity
                assert selected.static_input_ids is tokens
                assert selected.static_positions is positions
                if capacity == 4096:
                    assert selected is original
                torch.npu.synchronize()
                assert inputs.control.item() == 2
                # Readback belongs only to this exact-length test oracle.
                metadata.kv_seq_lens_host_values[:] = metadata.kv_seq_lens.cpu().tolist()
                metadata.prepared_attention_state = backend.prepare_metadata(metadata)
                backend.prepare(metadata)
                expected = backend._prefill(query, query, query, keys, values, metadata, rows, layer)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
        assert len(prepared._prepared_graphs) == 6
    torch.npu.current_stream(device).wait_stream(stream)


@pytest.mark.parametrize("case", ["recurrent_dp", "draft", "attention", "hidden", "topk"])
def test_adaptive_prepared_rejects_non_target_contract(runner: PreparedAclGraphRunner, case: str) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = True
    runner.attention_backend.logical_page_size = 128
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"))
    hidden = topk = None
    if case == "recurrent_dp":
        runner.dp_size = 2
        runner.attention_backend.is_mla = False
        inputs = replace(
            inputs,
            linear_state_indices=torch.ones(3, dtype=torch.int32),
            num_accepted_tokens=torch.ones(3, dtype=torch.int32),
        )
    elif case == "draft":
        runner._block_draft = True
    elif case == "attention":
        runner.attention_backend.is_mla = False
    elif case == "hidden":
        hidden = torch.zeros(8, 4)
    else:
        topk = torch.zeros(8, 1, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="MLA or recurrent target graph"):
        runner.warmup_prepared(tokens, positions, metadata, hidden, topk, dspark_adaptive=inputs)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


@pytest.fixture
def block_inputs(
    runner: PreparedAclGraphRunner,
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry]:
    runner._block_draft = True
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(
        side_effect=lambda metadata, **_: SimpleNamespace(
            device_kv_lengths=metadata.kv_seq_lens,
            kv_capacity=metadata.kv_seq_lens_host_values[0],
        )
    )
    tokens = torch.arange(3, dtype=torch.int32)
    positions = torch.arange(3, dtype=torch.int32)
    metadata = _metadata(1)
    metadata.slot_mapping = torch.arange(3, dtype=torch.int32)
    metadata.block_table = torch.zeros(1, 64, dtype=torch.int32)
    metadata.q_seq_lens = torch.tensor([3], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [3]
    metadata.q_cu_seq_lens = torch.tensor([3], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2048]
    metadata.is_chunked_prefill = True
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._graph_binding(tokens, positions, metadata)]
    entry.static_output = torch.ones(3, 8)
    return tokens, positions, metadata, entry


def test_block_graph_borrows_views_and_keeps_fia_capacity_constant(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
) -> None:
    tokens, positions, metadata, entry = block_inputs
    assert entry.static_input_ids is tokens
    assert entry.static_positions is positions
    assert entry.static_metadata.kv_seq_lens is metadata.kv_seq_lens
    assert entry.static_metadata.block_table.data_ptr() == metadata.block_table.data_ptr()
    assert entry.static_metadata.block_table.shape == (1, 32)
    assert entry.static_metadata.prepared_attention_state.device_kv_lengths is metadata.kv_seq_lens
    assert entry.static_metadata.kv_seq_lens_host_values == [4096]
    with (
        patch.object(runner, "_validate_prepared_input", wraps=runner._validate_prepared_input) as validate,
    ):
        for step, length in enumerate((2080, 2200, 4096, 2048)):
            metadata.kv_seq_lens_host_values[:] = [length]
            metadata.kv_seq_lens.fill_(length)
            output = runner.execute_block(tokens, positions, metadata)
            assert output is entry.static_output
            assert validate.call_count == step + 1
            if length <= 4096:
                assert runner.attention_backend.prepare_graph_replay.call_args.args[0] is entry.static_metadata
            assert entry.static_metadata.kv_seq_lens_host_values == [4096]
    assert runner.prepared_replays == 4
    assert entry.graph.replay.call_count == 4
    assert runner.attention_backend.prepare_graph_replay.call_count == 4
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()
    assert runner._capture.call_count == 1


def test_block_table_capacity_rejection_has_no_launch_side_effects(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
) -> None:
    tokens, positions, metadata, entry = block_inputs
    metadata.kv_seq_lens_host_values[:] = [8193]
    metadata.raw_dp_execution_token_counts = [3]
    entry.static_metadata.mega_moe_token_mask = torch.ones(3, dtype=torch.int8)
    runner.attention_backend.prepare.reset_mock()
    with (
        patch.object(runner, "_fill_mega_moe_token_mask") as mask,
        patch.object(runner, "_prepare_graph_eplb_state") as eplb,
        patch.object(runner, "_update_after_replay") as update,
    ):
        with pytest.raises(ValueError, match="block-table capacity"):
            runner.execute_block(tokens, positions, metadata)
        mask.assert_not_called()
        eplb.assert_not_called()
        update.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    entry.graph.replay.assert_not_called()
    assert runner.prepared_replays == 0


@pytest.mark.parametrize("query_width", [5, 7])
def test_block_graph_grows_kv_buckets_and_reuses_previous_outputs(
    runner: PreparedAclGraphRunner, query_width: int
) -> None:
    runner._block_draft = True
    runner.max_batch = query_width
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    metadata = _block_metadata(1, query_width, pages=128)
    tokens = torch.arange(query_width, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    original_slots = metadata.slot_mapping.clone()

    def capture(entry: AclGraphEntry, stream: object) -> None:
        assert (metadata.slot_mapping == -1).all()
        entry.graph = Mock()
        entry.static_output = torch.full((query_width,), len(runner._prepared_graphs))

    runner._capture.side_effect = capture
    runner.warmup_prepared(tokens, positions, metadata)
    first = next(iter(runner._prepared_graphs.values()))
    snapshot = first.static_output.clone()
    for length, pages, captures in ((4096, 32, 1), (4097, 64, 2), (8193, 128, 3), (4097, 64, 3), (2048, 32, 3)):
        metadata.kv_seq_lens_host_values[:] = [length]
        metadata.kv_seq_lens.fill_(length)
        output = runner.execute_block(tokens, positions, metadata)
        selected = runner._prepared_graphs[runner._graph_binding(tokens, positions, metadata)]
        assert output is selected.static_output
        assert selected.static_metadata.block_table.shape == (1, pages)
        assert selected.static_input_ids is tokens
        assert selected.static_positions is positions
        assert runner._capture.call_count == captures
        torch.testing.assert_close(metadata.slot_mapping, original_slots)
        torch.testing.assert_close(first.static_output, snapshot)
    assert len(runner._prepared_graphs) == 3
    assert first.graph.replay.call_count == 2


@pytest.mark.parametrize("dp_rank", [0, 1])
@pytest.mark.parametrize("pre_capture", [False, True])
def test_block_dp_empty_and_active_shards_choose_the_same_kv_bucket(
    runner: PreparedAclGraphRunner, dp_rank: int, pre_capture: bool
) -> None:
    runner._block_draft = True
    runner._block_draft_graph_warmup_max_kv_len = 0
    runner.max_batch = 7
    runner.dp_size = 2
    runner.dp_rank = dp_rank
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    metadata = _block_metadata(1, 7, pages=128)
    metadata.dp_execution_token_counts = (7, 7)
    metadata.dp_is_decode = (1, 1)
    metadata.dp_global_kv_max_seq_lens = (0, 2048)
    tokens = torch.zeros(7, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    runner.warmup_prepared(tokens, positions, metadata, pre_capture_kv_buckets=pre_capture)
    for global_kv, expected_pages in ((4090, 64), (8190, 128), (2048, 32)):
        metadata.dp_global_kv_max_seq_lens = (0, global_kv)
        metadata.kv_seq_lens_host_values[:] = [7 if dp_rank == 0 else global_kv]
        runner.execute_block(tokens, positions, metadata)
        entry = runner._prepared_graphs[runner._graph_binding(tokens, positions, metadata)]
        assert entry.static_metadata.block_table.shape[1] == expected_pages
    assert runner._capture.call_count == 3


@pytest.mark.parametrize(("limit", "expected"), [(0, [128, 64, 32]), (4096, [32]), (-1, [32])])
@pytest.mark.parametrize("query_width", [5, 7, 8])
def test_block_startup_prewarms_all_requested_kv_variants_without_changing_inputs(
    runner: PreparedAclGraphRunner, limit: int, expected: list[int], query_width: int
) -> None:
    runner._block_draft = True
    runner._block_draft_graph_warmup_max_kv_len = limit
    runner.max_batch = query_width
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    metadata = _block_metadata(1, query_width, pages=128)
    tokens = torch.arange(query_width, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    original_slots = metadata.slot_mapping.clone()
    original_bounds = metadata.kv_seq_lens_host_values.copy()

    def capture(entry: AclGraphEntry, stream: object) -> None:
        assert metadata.slot_mapping.tolist() == [-1] * query_width
        entry.graph = Mock()
        entry.static_output = (torch.ones(query_width, 4), torch.ones(query_width, 8))

    runner._capture.side_effect = capture
    runner.warmup_prepared(tokens, positions, metadata, pre_capture_kv_buckets=True)
    assert [entry.kv_capacity_pages for entry in runner._prepared_graphs.values()] == expected
    assert metadata.kv_seq_lens_host_values == original_bounds
    torch.testing.assert_close(metadata.slot_mapping, original_slots)
    outputs = [entry.static_output[1] for entry in runner._prepared_graphs.values()]
    assert len(outputs) == len(expected)
    assert len({output.data_ptr() for output in outputs}) == len(expected)
    runner.warmup_prepared(tokens, positions, metadata, pre_capture_kv_buckets=True)
    assert runner._capture.call_count == len(expected)
    for length in (2048, 4097, 8193, 2048, 4097):
        metadata.kv_seq_lens_host_values[:] = [length]
        runner.execute_block(tokens, positions, metadata)
    assert runner._capture.call_count == 3
    assert len(runner._prepared_graphs) == 3


@pytest.mark.parametrize("limit", [-1, 0, 4096])
def test_draft_warmup_limit_does_not_change_recurrent_target_capture(
    runner: PreparedAclGraphRunner, limit: int
) -> None:
    runner.max_batch = 8
    runner._block_draft_graph_warmup_max_kv_len = limit
    runner.attention_backend.is_mla = False
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), table_width=128)
    inputs = replace(
        inputs,
        linear_state_indices=torch.arange(1, 4, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    inputs.control.fill_(3)

    def capture(entry: AclGraphEntry, stream: object) -> None:
        assert inputs.control.item() == 2**32 + 3
        entry.graph = Mock()
        entry.static_output = (torch.ones(8, 4), torch.ones(8, 8))

    runner._capture.side_effect = capture
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs, pre_capture_kv_buckets=True)
    assert [entry.kv_capacity for entry in runner._prepared_graphs.values()] == [4096]
    assert inputs.control.item() == 3
    for length, captures in ((4097, 2), (8193, 3), (2048, 3)):
        metadata.kv_seq_lens_host_values[:] = [length] * 8
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
        assert runner._capture.call_count == captures


@pytest.mark.parametrize("limit", [-1, 0, 4096])
@pytest.mark.parametrize("is_mla", [False, True])
def test_draft_warmup_limit_does_not_add_plain_or_mla_mtp_graphs(
    runner: PreparedAclGraphRunner, limit: int, is_mla: bool
) -> None:
    runner._block_draft_graph_warmup_max_kv_len = limit
    runner.attention_backend.is_mla = is_mla
    tokens = torch.arange(2, dtype=torch.int32)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, tokens, metadata, pre_capture_kv_buckets=True)
    runner.warmup_prepared(tokens, tokens, metadata, pre_capture_kv_buckets=True)
    assert len(runner._prepared_graphs) == 1
    assert runner._capture.call_count == 1
    assert next(iter(runner._prepared_graphs.values())).kv_capacity_pages == 0


def test_bounded_startup_keeps_the_current_warmup_shape_even_above_the_limit(runner: PreparedAclGraphRunner) -> None:
    runner._block_draft = True
    runner._block_draft_graph_warmup_max_kv_len = 4096
    runner.max_batch = 7
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    metadata = _block_metadata(1, 7, pages=128)
    metadata.kv_seq_lens_host_values[:] = [8193]
    tokens = torch.zeros(7, dtype=torch.int32)
    runner.warmup_prepared(tokens, tokens, metadata, pre_capture_kv_buckets=True)
    assert [entry.kv_capacity_pages for entry in runner._prepared_graphs.values()] == [128, 32]
    runner.execute_block(tokens, tokens, metadata)
    assert runner._capture.call_count == 2


@pytest.mark.parametrize("pre_capture", [False, True])
def test_lazy_startup_reuses_a_fitting_larger_variant(runner: PreparedAclGraphRunner, pre_capture: bool) -> None:
    runner._block_draft = True
    runner._block_draft_graph_warmup_max_kv_len = -1
    runner.max_batch = 7
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    metadata = _block_metadata(1, 7, pages=128)
    metadata.kv_seq_lens_host_values[:] = [8193]
    tokens = torch.zeros(7, dtype=torch.int32)
    runner.warmup_prepared(tokens, tokens, metadata, pre_capture_kv_buckets=pre_capture)
    original = next(iter(runner._prepared_graphs.values()))
    assert original.kv_capacity_pages == 128

    metadata.kv_seq_lens_host_values[:] = [128]
    runner.warmup_prepared(tokens, tokens, metadata, pre_capture_kv_buckets=pre_capture)
    assert runner._resolve_entry(tokens, tokens, metadata) is original
    assert runner._capture.call_count == 1
    assert len(runner._prepared_graphs) == 1


def test_recurrent_target_grows_without_capture_writes_and_retains_old_graphs(runner: PreparedAclGraphRunner) -> None:
    runner.max_batch = 8
    runner.attention_backend.is_mla = False
    runner.attention_backend.page_size = runner.attention_backend.logical_page_size = 128
    runner.attention_backend.prepare_metadata = Mock(return_value=object())
    tokens, positions, metadata, inputs = _adaptive_inputs(torch.device("cpu"), table_width=128)
    inputs = replace(
        inputs,
        linear_state_indices=torch.arange(1, 4, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
    )
    inputs.control.fill_(3)
    original_states = inputs.linear_state_indices.clone()

    def capture(entry: AclGraphEntry, stream: object) -> None:
        assert inputs.control.item() == 2**32 + 3
        runner._refresh_linear_verify_graph(inputs, entry.static_metadata.linear_verify_graph, entry.batch_size)
        assert torch.count_nonzero(entry.static_metadata.linear_verify_graph.state_indices) == 0
        assert torch.count_nonzero(entry.static_metadata.linear_verify_graph.cu_seqlens) == 0
        entry.graph = Mock()
        entry.static_output = torch.ones(8)

    runner._capture.side_effect = capture
    runner.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    first = next(iter(runner._prepared_graphs.values()))
    for length, capacity in ((4097, 8192), (8193, 16384), (2048, 4096)):
        metadata.kv_seq_lens_host_values[:] = [length] * 8
        runner.execute(tokens, positions, metadata, dspark_adaptive=inputs)
        entry = runner._resolve_entry(tokens, positions, metadata, dspark_adaptive=inputs)
        assert entry.kv_capacity == capacity
        assert inputs.control.item() == 3
        torch.testing.assert_close(inputs.linear_state_indices, original_states)
    assert runner._capture.call_count == 3
    assert first.graph.replay.call_count == 1
    assert first.kv_capacity == 4096


def test_kv_bucket_capture_failure_restores_inputs_without_publishing_variant(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
) -> None:
    tokens, positions, metadata, entry = block_inputs
    original_slots = metadata.slot_mapping.clone()
    metadata.kv_seq_lens_host_values[:] = [4097]
    runner._capture.side_effect = RuntimeError("larger capture failed")
    with pytest.raises(RuntimeError, match="larger capture failed"):
        runner.execute_block(tokens, positions, metadata)
    assert len(runner._prepared_graphs) == 1
    torch.testing.assert_close(metadata.slot_mapping, original_slots)
    entry.graph.replay.assert_not_called()


def test_block_missing_binding_takes_priority_over_capacity(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
) -> None:
    tokens, positions, metadata, entry = block_inputs
    metadata.kv_seq_lens_host_values[:] = [4097]
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute_block(tokens.clone(), positions, metadata)
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    entry.graph.replay.assert_not_called()
    assert runner._capture.call_count == 1


@pytest.mark.parametrize("invalid", ["prefill", "decode", "verify", "layout"])
def test_block_invalid_inputs_raise_before_capacity_check(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
    invalid: str,
) -> None:
    tokens, positions, metadata, entry = block_inputs
    metadata.kv_seq_lens_host_values[:] = [4097]
    if invalid == "prefill":
        metadata.is_prefill = True
    elif invalid == "decode":
        metadata.is_chunked_prefill = False
    elif invalid == "verify":
        metadata.is_spec_verify = True
    else:
        positions = positions[:2]
    message = "equal positive query widths" if invalid == "layout" else "non-causal chunked-prefill"
    with pytest.raises(ValueError, match=message):
        runner.execute_block(tokens, positions, metadata)
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    entry.graph.replay.assert_not_called()


def test_block_replay_failure_propagates(
    runner: PreparedAclGraphRunner,
    block_inputs: tuple[torch.Tensor, torch.Tensor, SimpleNamespace, AclGraphEntry],
) -> None:
    tokens, positions, metadata, entry = block_inputs
    entry.graph.replay.side_effect = RuntimeError("device replay failed")
    with pytest.raises(RuntimeError, match="device replay failed"):
        runner.execute_block(tokens, positions, metadata)
    assert runner.prepared_replays == 0
    assert runner._capture.call_count == 1


@pytest.mark.parametrize("request_count", [2, 3, 4])
@pytest.mark.parametrize("query_width", [1, 3, 7, 8])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_block_graph_binds_isolated_request_rows_without_recapture(
    runner: PreparedAclGraphRunner, request_count: int, query_width: int, dp_size: int
) -> None:
    runner._block_draft = True
    runner.dp_size = dp_size
    rows = request_count * query_width
    runner.max_batch = rows
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(
        side_effect=lambda metadata, **_: SimpleNamespace(
            device_kv_lengths=metadata.kv_seq_lens,
            block_table=metadata.block_table,
            query_lengths=[query_width] * request_count,
        )
    )
    entries = []
    for slot_index in range(2):
        tokens = torch.arange(rows, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _block_metadata(request_count, query_width)
        metadata.dp_execution_token_counts = (rows,) * dp_size
        metadata.dp_is_decode = (1,) * dp_size
        metadata.dp_global_kv_max_seq_lens = (2048,) * dp_size
        runner.warmup_prepared(tokens, positions, metadata)
        entry = runner._prepared_graphs[runner._graph_binding(tokens, positions, metadata)]
        entries.append(entry)
        static = entry.static_metadata
        assert static.block_table.is_contiguous()
        assert static.block_table.shape == (request_count, 32)
        assert static.block_table.data_ptr() != metadata.block_table.data_ptr()
        torch.testing.assert_close(static.block_table, metadata.block_table[:, :32])
        assert static.kv_seq_lens_host_values == [4096] * request_count
        assert static.q_cu_seq_lens_host_values == [query_width * (index + 1) for index in range(request_count)]
        assert static.prepared_attention_state.device_kv_lengths is metadata.kv_seq_lens
        for base_length in (2048, 2200, 4000):
            lengths = [base_length + index + slot_index for index in range(request_count)]
            metadata.kv_seq_lens_host_values[:] = lengths
            metadata.dp_global_kv_max_seq_lens = (max(lengths),) * dp_size
            metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
            assert runner.can_execute_block(tokens, positions, metadata)
            runner.execute(tokens, positions, metadata)
            assert runner.attention_backend.prepare_graph_replay.call_args.args[0] is static
            assert static.kv_seq_lens.tolist() == lengths
            assert static.kv_seq_lens_host_values == [4096] * request_count
        table_address = static.block_table.data_ptr()
        metadata.block_table.add_(100)
        with torch.inference_mode(), patch.object(AclGraphRunner, "_forward_static", return_value=torch.empty(0)):
            runner._forward_static(entry)
        torch.testing.assert_close(static.block_table, metadata.block_table[:, :32])
        assert static.block_table.data_ptr() == table_address
        assert entry.graph.replay.call_count == 3
    assert entries[0] is not entries[1]
    assert entries[0].static_metadata.block_table.data_ptr() != entries[1].static_metadata.block_table.data_ptr()
    assert entries[0].static_metadata.kv_seq_lens.data_ptr() != entries[1].static_metadata.kv_seq_lens.data_ptr()
    assert runner._capture.call_count == 2
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()


@pytest.mark.parametrize("query_ends", [[3, 5], [3], [0, 3, 6], [0, 6], [6, 6], [3, 6, 9]])
def test_block_graph_rejects_invalid_request_query_ends_before_capture(
    runner: PreparedAclGraphRunner, query_ends: list[int]
) -> None:
    runner._block_draft = True
    runner.max_batch = 6
    runner.attention_backend.page_size = 128
    metadata = _block_metadata(2, 3)
    metadata.q_cu_seq_lens_host_values = query_ends
    tokens = torch.arange(6, dtype=torch.int32)
    with pytest.raises(ValueError, match="equal positive query widths"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


@pytest.mark.parametrize("kv_bounds", [[2048], [2048, 0], [2048, 2], [8193, 2048]])
def test_block_graph_rejects_invalid_per_request_kv_bounds_before_capture(
    runner: PreparedAclGraphRunner, kv_bounds: list[int]
) -> None:
    runner._block_draft = True
    runner.max_batch = 6
    runner.attention_backend.page_size = 128
    metadata = _block_metadata(2, 3)
    metadata.kv_seq_lens_host_values = kv_bounds
    tokens = torch.arange(6, dtype=torch.int32)
    with pytest.raises(ValueError, match="KV bound"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()


def test_block_graph_rejects_strided_request_rows_without_input_copy(runner: PreparedAclGraphRunner) -> None:
    runner._block_draft = True
    runner.max_batch = 6
    runner.attention_backend.page_size = 128
    metadata = _block_metadata(2, 3)
    metadata.block_table = metadata.block_table[:, :32]
    tokens = torch.arange(6, dtype=torch.int32)
    with pytest.raises(ValueError, match="contiguous request rows"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()


@pytest.mark.parametrize("counts,phases", [((6, 3), (1, 1)), ((6, 6), (1, 0)), ((6, 0), (1, 1))])
def test_block_graph_rejects_nonuniform_dp_geometry_before_capture(
    runner: PreparedAclGraphRunner, counts: tuple[int, ...], phases: tuple[int, ...]
) -> None:
    runner._block_draft = True
    runner.max_batch = 6
    runner.dp_size = 2
    runner.attention_backend.page_size = 128
    metadata = _block_metadata(2, 3)
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    tokens = torch.arange(6, dtype=torch.int32)
    with pytest.raises(ValueError, match="prepared DP"):
        runner.warmup_prepared(tokens, torch.zeros_like(tokens), metadata)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


def test_capture_shares_one_lazy_pool_across_runners_and_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.runners import acl_graph

    pool = (0, 7)
    stream = Mock()
    graph_context = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    npu = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(side_effect=Mock),
        NPUGraph=Mock(side_effect=Mock),
        graph_pool_handle=Mock(return_value=pool),
        current_stream=Mock(return_value=stream),
        stream=lambda _: nullcontext(),
        graph=graph_context,
        synchronize=Mock(),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(acl_graph, "_GRAPH_POOL", None)
    backend = SimpleNamespace(prepare=Mock())
    runners = [
        PreparedAclGraphRunner(torch.nn.Identity(), backend, torch.device("cpu"), 4),
        DecodeAclGraphRunner(torch.nn.Identity(), backend, torch.device("cpu"), 4, 128),
    ]
    npu.graph_pool_handle.assert_not_called()
    for graph_runner in runners:
        graph_runner._forward_static = Mock(return_value=torch.ones(1))
        for rows in (4, 2):
            entry = AclGraphEntry()
            entry.batch_size = rows
            entry.static_input_ids = torch.arange(rows, dtype=torch.int32)
            entry.static_mtp_topk_indices = None
            entry.static_metadata = _metadata(rows)
            entry.execution_state = None
            graph_runner._capture(entry, stream)
            assert entry.static_output is graph_runner._forward_static.return_value
            assert entry.graph_tasks == []
    npu.graph_pool_handle.assert_called_once_with()
    assert graph_context.call_count == 4
    assert all(call.kwargs["pool"] is pool for call in graph_context.call_args_list)
    assert all(call.kwargs["stream"] is stream for call in graph_context.call_args_list)


def test_capture_and_replay_bind_each_slot_without_input_copies(runner: PreparedAclGraphRunner) -> None:
    from xllm.python.model_executor.forward_context import get_forward_context

    inputs = []
    for rows in (4, 4, 2, 2):
        tokens = torch.arange(rows, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(rows)
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
    assert len({id(entry) for entry in entries}) == 4
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()
    runner.attention_backend.prepare_graph_replay.assert_not_called()
    runner.attention_backend.prepare.reset_mock()
    installed = []

    def install_replay(metadata: SimpleNamespace) -> None:
        installed.append((get_forward_context().execution_state, list(metadata.kv_seq_lens_host_values)))

    runner.attention_backend.prepare_graph_replay.side_effect = install_replay
    for step, index in enumerate((0, 1, 2, 3, 0)):
        tokens, positions, metadata = inputs[index]
        lengths = [7 + step + row for row in range(len(tokens))]
        metadata.kv_seq_lens_host_values[:] = lengths
        runner.execute(tokens, positions, metadata)
        assert installed[-1][0] is entries[index].execution_state
        assert installed[-1][1] == lengths
    for entry, calls in zip(entries, (2, 1, 1, 1)):
        assert entry.graph.replay.call_count == calls
    runner.attention_backend.prepare.assert_not_called()
    assert runner.attention_backend.prepare_graph_replay.call_count == 5
    assert runner._capture.call_count == 4


@pytest.mark.parametrize("explicit_read", [False, True])
@pytest.mark.parametrize("first_empty", [False, True])
def test_prepared_gdn_metadata_keeps_final_state_views_across_slots(
    runner: PreparedAclGraphRunner, explicit_read: bool, first_empty: bool
) -> None:
    from tests.python.qwen3_5_test_utils import make_linear_config
    from xllm.python.attention.backend import LayerCache
    from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnDecodeMetadata, GdnMetadata
    from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder

    builder = Qwen3_5GdnMetadataBuilder(
        make_linear_config(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
    )
    runner.execution_metadata_builders = (builder,)
    runner.bind_layer_caches(
        [
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(5, 3, 384, dtype=torch.bfloat16),
                ssm=torch.zeros(5, 1, 128, 128, dtype=torch.float32),
            )
        ]
    )

    def capture_context(entry: AclGraphEntry, _stream: object) -> None:
        entry.graph = Mock()

    runner._capture.side_effect = capture_context
    slots = []
    for _ in range(2):
        tokens = torch.arange(2, dtype=torch.int32)
        positions = torch.zeros_like(tokens)
        metadata = _metadata(2)
        metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32)
        metadata.linear_state_write_indices = metadata.linear_state_indices.view(-1)
        metadata.linear_state_read_indices = metadata.linear_state_indices.view(-1) if explicit_read else None
        metadata.has_initial_state = torch.ones(2, dtype=torch.bool)
        graph_batch = InputBatch(
            input_ids=tokens,
            positions=positions,
            num_reqs=2,
            num_tokens=2,
            num_tokens_after_padding=2,
            num_scheduled_tokens=(1, 1),
            num_computed_tokens=(0, 0),
            query_start_loc=(0, 1, 2),
            is_prefilling=(False, False),
            is_padding=None,
            is_dummy=False,
        )
        capture_batch = (
            replace(
                graph_batch,
                num_reqs=0,
                num_scheduled_tokens=(),
                num_computed_tokens=(),
                query_start_loc=(0,),
                is_prefilling=(),
                is_dummy=True,
            )
            if first_empty
            else graph_batch
        )
        with patch.object(torch, "equal", side_effect=AssertionError("Device state comparison")):
            runner.warmup_prepared(tokens, positions, metadata, input_batch=capture_batch)
        entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
        gdn = entry.layer_shared_cache[GdnMetadata]
        assert isinstance(gdn, GdnDecodeMetadata)
        assert gdn.read_state_indices is gdn.write_state_indices
        assert gdn.write_state_indices is metadata.linear_state_write_indices
        assert entry.static_metadata.has_initial_state is metadata.has_initial_state
        slots.append((tokens, positions, metadata, gdn, graph_batch))

    for step, slot_id in enumerate((0, 1, 0, 1)):
        tokens, positions, metadata, gdn, graph_batch = slots[slot_id]
        metadata.linear_state_indices.copy_(torch.tensor([3, 4] if step % 2 else [2, 1], dtype=torch.int32))
        runner.execute(tokens, positions, metadata, input_batch=graph_batch)
        torch.testing.assert_close(gdn.read_state_indices, metadata.linear_state_indices)
        inactive = slots[1 - slot_id][3]
        assert inactive.write_state_indices.data_ptr() != gdn.write_state_indices.data_ptr()
    assert runner._capture.call_count == 2
    assert runner.prepared_replays == 4


@pytest.mark.parametrize("field", ["linear_state_indices", "linear_state_write_indices", "has_initial_state"])
def test_prepared_gdn_rejects_replaced_state_views_before_replay(runner: PreparedAclGraphRunner, field: str) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    metadata = _metadata(2)
    metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32)
    metadata.linear_state_write_indices = metadata.linear_state_indices
    metadata.has_initial_state = torch.ones(2, dtype=torch.bool)
    runner.warmup_prepared(tokens, tokens, metadata)
    setattr(metadata, field, getattr(metadata, field).clone())
    runner.attention_backend.prepare_graph_replay.reset_mock()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, tokens, metadata)
    runner.attention_backend.prepare_graph_replay.assert_not_called()


@pytest.mark.parametrize("invalid", ["dtype", "rows", "stride", "separate_read"])
def test_prepared_gdn_rejects_nonfinal_state_views_before_capture(runner: PreparedAclGraphRunner, invalid: str) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    metadata = _metadata(2)
    indices = torch.tensor([1, 2], dtype=torch.int32)
    if invalid == "dtype":
        indices = indices.to(torch.int64)
    elif invalid == "rows":
        indices = indices[:1]
    elif invalid == "stride":
        indices = torch.arange(4, dtype=torch.int32)[::2]
    metadata.linear_state_indices = indices
    if invalid == "separate_read":
        metadata.linear_state_read_indices = indices.clone()
    with pytest.raises(ValueError, match="linear_state_indices|shared linear-state read/write"):
        runner.warmup_prepared(tokens, tokens, metadata)
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 5, 6, 7, 8, 16])
def test_prepared_verify_binds_logical_accepted_counts_without_host_reads(
    runner: PreparedAclGraphRunner, draft_tokens: int
) -> None:
    width = draft_tokens + 1
    runner._spec_verify_width = width
    runner._device_kv_lengths = True
    runner.attention_backend.page_size = 128
    runner.attention_backend.prepare_metadata = Mock(
        side_effect=lambda metadata, **_: SimpleNamespace(
            device_kv_lengths=metadata.kv_seq_lens,
            kv_capacity=metadata.kv_seq_lens_host_values[0],
        )
    )
    runner.max_batch = 2 * width
    tokens = torch.arange(2 * width, dtype=torch.int32)
    metadata = _metadata(2 * width)
    metadata.is_spec_verify = True
    metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32)
    metadata.num_accepted_tokens = torch.tensor([1, width], dtype=torch.int32)
    metadata.has_initial_state = torch.ones(2, dtype=torch.bool)
    runner.warmup_prepared(tokens, tokens, metadata)
    key = runner._prepared_binding(tokens, tokens, metadata)
    entry = runner._resolve_entry(tokens, tokens, metadata)
    assert entry.static_metadata.num_accepted_tokens is metadata.num_accepted_tokens
    assert entry.static_metadata.linear_state_indices.numel() == 2
    assert entry.static_metadata.prepared_attention_state.device_kv_lengths is metadata.kv_seq_lens
    assert entry.static_metadata.prepared_attention_state.kv_capacity == 256
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("accepted state readback")):
        metadata.num_accepted_tokens.fill_(width)
        runner.execute(tokens, tokens, metadata)
    assert runner._prepared_binding(tokens, tokens, metadata) == key
    metadata.num_accepted_tokens = metadata.num_accepted_tokens.clone()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, tokens, metadata)


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 8, 16])
def test_legacy_verify_graph_uses_complete_sequence_buckets(runner: PreparedAclGraphRunner, draft_tokens: int) -> None:
    width = draft_tokens + 1
    legacy = DecodeAclGraphRunner(
        torch.nn.Identity(),
        runner.attention_backend,
        torch.device("cpu"),
        8 * width,
        2048,
        num_decoding_tokens=width,
        spec_verify_width=width,
    )
    for sequences in (1, 3, 5, 8):
        bucket = 1 if sequences == 1 else 4 if sequences == 3 else 8
        assert legacy._padded_batch_size(sequences * width, _metadata(sequences * width)) == bucket * width
    with pytest.raises(ValueError, match="complete fixed-width requests"):
        legacy._padded_batch_size(width + 1, _metadata(width + 1))


@pytest.mark.parametrize("width", [1, 3, 4, 17], ids=["draft-padding", "target-k2", "target-k3", "target-k16"])
def test_legacy_device_lengths_preserve_physical_query_rows(
    runner: PreparedAclGraphRunner, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python import kernels
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    sequences, rows = 3, 3 * width
    backend = NpuPagedAttentionBackend(8, 2, 64, 0.125, 0, False, torch.device("cpu"), torch.float16)
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    monkeypatch.setattr(backend, "_allocate_graph_workspace", Mock(return_value=torch.empty(1)))
    metadata = _metadata(rows)
    metadata.prepared_attention_state = None
    metadata.kv_cu_seq_lens = None
    metadata.paged_kv_indptr = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.ones(rows, dtype=torch.int32)
    if width > 1:
        metadata.is_chunked_prefill = metadata.is_spec_verify = True
        metadata.q_seq_lens_host = torch.full((sequences,), width, dtype=torch.int32)
        metadata.q_seq_lens = metadata.q_seq_lens_host.clone()
        metadata.q_cu_seq_lens_host_values = list(range(0, rows + 1, width))
        metadata.q_cu_seq_lens = torch.tensor(metadata.q_cu_seq_lens_host_values, dtype=torch.int32)
        metadata.linear_state_indices = torch.arange(1, sequences + 1, dtype=torch.int32)
        metadata.has_initial_state = torch.ones(sequences, dtype=torch.bool)
        metadata.num_accepted_tokens = torch.ones(sequences, dtype=torch.int32)
        metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
            kv_seq_lens=metadata.kv_seq_lens,
            block_table=metadata.block_table,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
        )

    def update_lengths(*args: object) -> None:
        # Only native metadata packing is substituted. Query preparation and
        # validation run through the real paged-attention backend.
        args[11].fill_(1)
        args[11][:rows].copy_(metadata.kv_seq_lens)

    monkeypatch.setattr(kernels, "update_decode_graph_metadata", update_lengths, raising=False)
    legacy = DecodeAclGraphRunner(
        torch.nn.Identity(),
        backend,
        torch.device("cpu"),
        8 * width,
        128,
        num_decoding_tokens=width,
        spec_verify_width=width,
        device_kv_lengths=True,
    )
    legacy.bind_layer_caches([LayerCache(key=cache, value=cache)])
    legacy._capture = Mock()
    tokens = torch.zeros(rows, dtype=torch.int32)
    key = legacy.warmup(tokens, tokens, metadata, is_graph_warmup=True)
    entry = legacy._graphs[key]
    static = entry.static_metadata
    padded_rows = 4 * width
    addresses = (static.q_seq_lens.data_ptr(), static.q_cu_seq_lens.data_ptr(), static.kv_seq_lens.data_ptr())
    for value in (7, 11):
        metadata.kv_seq_lens.fill_(value)
        metadata.kv_seq_lens_host_values[:] = [value] * rows
        if width > 1:
            metadata.num_accepted_tokens.fill_(width)
        legacy._fill_entry(entry, tokens, tokens, metadata, rows, None, expanded=legacy._expanded_verify_view(metadata))
        prepared = static.prepared_attention_state
        assert prepared.query_lengths == [1] * padded_rows
        assert static.q_seq_lens.tolist() == [1] * padded_rows
        assert static.q_seq_lens_host.tolist() == ([width] * 4 if width > 1 else [1] * 4)
        assert static.q_cu_seq_lens.tolist() == static.q_cu_seq_lens_host_values == list(range(padded_rows + 1))
        assert prepared.device_kv_lengths.data_ptr() == static.kv_seq_lens.data_ptr()
        assert prepared.device_kv_lengths.tolist() == [value] * rows + [1] * (padded_rows - rows)
        assert addresses == (
            static.q_seq_lens.data_ptr(),
            static.q_cu_seq_lens.data_ptr(),
            static.kv_seq_lens.data_ptr(),
        )
        if width > 1:
            assert static.linear_state_indices.tolist() == [1, 2, 3, 0]
            assert static.num_accepted_tokens.tolist() == [width] * sequences + [1]
            assert metadata.q_seq_lens_host.tolist() == [width] * sequences
    with pytest.raises(ValueError, match="requires query lengths"):
        backend.prepare_metadata(replace(static, q_seq_lens=None), device_kv_lengths=True)


@pytest.mark.parametrize(
    ("prepared", "num_decoding_tokens"),
    [(True, 1), (False, 1), (False, 4)],
    ids=["pipeline", "legacy-plain", "legacy-speculative"],
)
@torch.inference_mode()
def test_real_paged_backend_refreshes_captured_lengths_and_isolates_entries(
    monkeypatch: pytest.MonkeyPatch, prepared: bool, num_decoding_tokens: int
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
        graph_pool_handle=Mock(return_value=(0, 1)),
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
    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    monkeypatch.setattr(kernels, "update_decode_graph_metadata", Mock(), raising=False)
    backend = NpuPagedAttentionBackend(8, 2, 64, 0.125, 0, False, torch.device("cpu"), torch.float16)
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])

    def _allocate_workspace(batch_size: int, block_table: torch.Tensor, *, actual_seq_kv: list[int]) -> torch.Tensor:
        assert block_table.shape[0] == batch_size
        assert actual_seq_kv == [1] * batch_size
        return torch.empty(1)

    allocate_workspace = Mock(side_effect=_allocate_workspace)
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
        else DecodeAclGraphRunner(
            PagedModel(), backend, torch.device("cpu"), 4, 128, num_decoding_tokens=num_decoding_tokens
        )
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
    assert workspace_count == len(inputs)
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
            # Ordinary decode borrows the replay buffer; speculative decode
            # queues a clone before task update on a real device. CPU mocks
            # verify ownership without emulating that stream queue.
            assert (output.data_ptr() != entry.static_output.data_ptr()) is (num_decoding_tokens > 1)
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


def test_missing_ordinary_graph_is_not_recaptured_as_a_kv_variant(runner: PreparedAclGraphRunner) -> None:
    tokens = torch.zeros(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    entry.static_output = (torch.zeros(2, 8), torch.ones(2, 4))
    (aux,) = runner.get_prepared_aux_hidden_states(tokens, positions, metadata)
    assert aux is entry.static_output[1]
    runner._prepared_graphs.clear()
    runner.attention_backend.prepare_graph_replay.reset_mock()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions, metadata)
    assert runner._capture.call_count == 1
    runner.attention_backend.prepare_graph_replay.assert_not_called()


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
    entry.graph.replay.assert_called_once()
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens, positions.clone(), metadata)
    with pytest.raises(RuntimeError, match="warmed Slot binding"):
        runner.execute(tokens.clone(), positions, metadata)
    runner.attention_backend.prepare.reset_mock()
    runner.warmup_prepared(tokens, positions, metadata)
    assert runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)] is entry
    runner.attention_backend.prepare.assert_not_called()
    assert runner._capture.call_count == 1


def test_prepared_executor_obeys_admission_and_propagates_replay_failure(runner: PreparedAclGraphRunner) -> None:
    executor = object.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.requires_input_batch_metadata = False
    executor._prepared_mtp = False
    executor._prepared_block_draft = False
    executor.layerwise_split_size = 1
    executor.decode_graph_runner = None
    executor.context_kv_graph_runner = None
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
    entry = runner._prepared_graphs[runner._prepared_binding(tokens, positions, metadata)]
    metadata.dp_execution_token_counts = (2, 7)
    with pytest.raises(ValueError, match="same decode batch size"):
        runner.execute(tokens, positions, metadata)
    entry.graph.replay.assert_not_called()
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 0)
    with pytest.raises(ValueError, match="same decode batch size"):
        runner.execute(tokens, positions, metadata)
    entry.graph.replay.assert_not_called()
    metadata.dp_is_decode = (1, 1)
    with pytest.raises(RuntimeError, match="Slot binding"):
        runner.execute(tokens.clone(), positions, metadata)
    entry.graph.replay.assert_not_called()
    assert runner._capture.call_count == 1


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


@pytest.mark.parametrize("table_width", [8192, 8195])
@pytest.mark.parametrize("graph_rows", [8, 16])
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_adaptive_target_graph_packs_live_control_without_recapture(
    table_width: int, graph_rows: int, draft_width: int
) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.model_executor.forward_context import get_forward_context

    class PackedInputTarget(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            metadata = get_forward_context().metadata
            # This target observes the packer-to-model dependency inside the
            # actual prepared graph; it is not a GLM/SFA numerical oracle.
            return torch.stack(
                (tokens, positions, metadata.kv_seq_lens, metadata.slot_mapping, metadata.q_seq_lens), dim=1
            )

    device = torch.device("npu:0")
    backend = SimpleNamespace(
        prepare=lambda *args, **kwargs: None,
        prepare_graph_replay=lambda metadata: None,
        is_mla=True,
        logical_page_size=128,
    )
    prepared = PreparedAclGraphRunner(PackedInputTarget(), backend, device, graph_rows)
    stream = torch.npu.Stream(device=device)
    slots = [_adaptive_inputs(device, table_width, graph_rows, draft_width) for _ in range(2)]
    for tokens, positions, metadata, inputs in slots:
        # Startup writes are suppressed by the same captured device control
        # later used for normal serving, not by a second capture variant.
        inputs.control.fill_(1 | (1 << 32))
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            prepared.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
    torch.npu.current_stream(device).wait_stream(stream)
    addresses = [tuple(tensor.data_ptr() for tensor in _adaptive_tensors(*slot)) for slot in slots]
    assert all(first != second for first, second in zip(*addresses))
    assert len(prepared._prepared_graphs) == 2
    output_addresses = tuple(entry.static_output.data_ptr() for entry in prepared._prepared_graphs.values())
    assert output_addresses[0] != output_addresses[1]
    prepared._capture = Mock(side_effect=AssertionError("runtime recapture"))

    for phase, (slot_id, batch_size, suppress) in enumerate(
        ((0, 1, False), (1, 3, False), (0, 2, True), (1, 1, True), (0, 3, False), (1, 2, False))
    ):
        tokens, positions, metadata, inputs = slots[slot_id]
        inactive = _adaptive_tensors(*slots[1 - slot_id])
        untouched = [tensor.cpu().clone() for tensor in inactive]
        arena = torch.full_like(inputs.arena, 199, device="cpu")
        proposals = (
            torch.arange(batch_size * draft_width, dtype=torch.int64).view(batch_size, draft_width) + 1000 + phase * 20
        )
        confidence = torch.full((batch_size, draft_width), 0.05)
        confidence[phase % batch_size].fill_(0.99)
        if phase == 4:
            confidence.fill_(1)
        proposal_bytes = 8 * draft_width * batch_size
        width_offset = 12 * draft_width * batch_size
        arena[:proposal_bytes].copy_(proposals.view(torch.uint8).flatten())
        arena[proposal_bytes:width_offset].copy_(confidence.view(torch.uint8).flatten())
        inputs.arena.copy_(arena)
        inputs.control.fill_(batch_size | (int(suppress) << 32))
        inputs.anchors.copy_(torch.arange(100 + phase, 103 + phase, dtype=torch.int64))
        # Exercise page 4096 and the final page, including a non-tile-aligned
        # table tail. KV attention length is deliberately not cache position.
        base_positions = torch.tensor([4096 * 128 - 4, 4096 * 128, table_width * 128 - 8], dtype=torch.int32)
        inputs.base_positions.copy_(base_positions)
        inputs.base_kv_lengths.copy_(base_positions + 1 + phase % 2)
        inputs.base_block_table.copy_(
            torch.arange(3 * table_width, dtype=torch.int32).view(3, table_width) + 10 + phase
        )

        budget = min(draft_width * batch_size, graph_rows - batch_size)
        widths = torch.empty(batch_size, dtype=torch.int32)
        offsets = torch.empty(batch_size + 1, dtype=torch.int32)
        allocate_verify_prefixes(confidence, budget, widths, offsets)
        reference = create_verify_buffers(graph_rows, table_width, "cpu")
        pack_verify_inputs(
            anchors=inputs.anchors.cpu()[:batch_size],
            proposals=proposals,
            base_positions=inputs.base_positions.cpu()[:batch_size],
            base_kv_lengths=inputs.base_kv_lengths.cpu()[:batch_size],
            base_block_table=inputs.base_block_table.cpu()[:batch_size],
            widths=widths,
            offsets=offsets,
            logical_rows=batch_size + budget,
            block_size=128,
            outputs=reference,
        )
        if suppress:
            reference.slot_mapping.fill_(-1)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            prepared.warmup_prepared(tokens, positions, metadata, dspark_adaptive=inputs)
            output = prepared.execute(tokens, positions, metadata, dspark_adaptive=inputs)
        torch.npu.current_stream(device).wait_stream(stream)
        actual_widths = inputs.arena[width_offset : width_offset + 4 * batch_size].cpu().view(torch.int32)
        torch.testing.assert_close(actual_widths, widths)
        torch.testing.assert_close(inputs.offsets[: batch_size + 1].cpu(), offsets)
        for actual, name in (
            (tokens, "tokens"),
            (positions, "positions"),
            (metadata.kv_seq_lens, "kv_seq_lens"),
            (metadata.slot_mapping, "slot_mapping"),
            (metadata.block_table, "block_table"),
            (metadata.q_seq_lens, "q_seq_lens"),
            (inputs.owners, "owners"),
            (inputs.columns, "columns"),
        ):
            torch.testing.assert_close(actual.cpu(), getattr(reference, name), rtol=0, atol=0)
        expected_output = torch.stack(
            (
                reference.tokens,
                reference.positions,
                reference.kv_seq_lens,
                reference.slot_mapping,
                reference.q_seq_lens,
            ),
            dim=1,
        )
        torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
        assert output.data_ptr() in output_addresses
        for actual, expected in zip(inactive, untouched):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert tuple(tensor.data_ptr() for tensor in _adaptive_tensors(*slots[slot_id])) == addresses[slot_id]
        assert len(prepared._prepared_graphs) == 2
    prepared._capture.assert_not_called()
    assert prepared.prepared_replays == 6


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

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

"""Per-execution metadata reuse in the ordinary ACL decode runner."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.paged_kv_metadata import build_row_aligned_paged_kv_metadata
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        expanded_decode_metadata=None,
        block_table=torch.tensor([[11, 0], [21, 22]], dtype=torch.int64),
        kv_seq_lens=torch.tensor([2, 5], dtype=torch.int64),
        kv_seq_lens_host_values=[2, 5],
        kv_cu_seq_lens=None,
        q_cu_seq_lens=None,
        # Force the ordinary runner to build token-row paging metadata.
        paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indices=torch.tensor([11, 21, 22], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        slot_mapping=torch.tensor([1, 4], dtype=torch.int32),
        linear_state_indices=torch.tensor([0, 1], dtype=torch.int32),
    )


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> DecodeAclGraphRunner:
    backend = SimpleNamespace(page_size=4, is_mla=False, create_graph_block_tables=lambda *_: ())
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )
    runner._stream = Mock()
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", Mock(return_value=Mock()), raising=False)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext(), raising=False)
    monkeypatch.setattr(runner, "_prepare_attention", Mock())

    def capture(entry: AclGraphEntry, stream: object) -> None:
        entry.graph = Mock()
        entry.static_output = torch.ones((entry.batch_size, 2))

    monkeypatch.setattr(runner, "_capture", Mock(side_effect=capture))
    monkeypatch.setattr(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", Mock()
    )
    return runner


@pytest.fixture
def executor(runner: DecodeAclGraphRunner) -> ModelExecutor:
    executor = object.__new__(ModelExecutor)
    executor.model = runner.model
    executor._kv_bound = True
    executor._serving_acl_graph_logged = False
    executor.context_kv_graph_runner = None
    executor.layerwise_split_size = 1
    executor.decode_graph_runner = runner
    executor.prepared_graph_runner = None
    executor.inductor_runner = None
    executor.eager_runner = Mock()
    return executor


@pytest.mark.parametrize("no_padding", [False, True])
@pytest.mark.parametrize("requests", [3, 23, 24, 27, 32])
@pytest.mark.parametrize("width", [3, 5])
def test_linear_verify_bucketing_preserves_complete_requests(
    runner: DecodeAclGraphRunner, no_padding: bool, requests: int, width: int
) -> None:
    runner.max_batch = 32 * width
    runner.num_decoding_tokens = width
    runner._model_owns_linear_spec_verify = True
    runner._enable_decode_no_padding = no_padding
    metadata = SimpleNamespace(is_spec_verify=True)

    bucket = runner._padded_batch_size(requests * width, metadata)

    expected_requests = requests if no_padding else (4 if requests == 3 else 32)
    assert bucket == expected_requests * width
    assert bucket % width == 0


def test_no_padding_retains_mtp_width_and_capacity_validation(runner: DecodeAclGraphRunner) -> None:
    runner.max_batch = 160
    runner.num_decoding_tokens = 5
    runner._model_owns_linear_spec_verify = True
    runner._enable_decode_no_padding = True
    metadata = SimpleNamespace(is_spec_verify=True)

    with pytest.raises(ValueError, match="complete fixed-width requests"):
        runner._padded_batch_size(119, metadata)
    with pytest.raises(ValueError, match="exceeds ACL graph capacity"):
        runner._padded_batch_size(165, metadata)


@pytest.mark.parametrize("no_padding,expected", [(False, 160), (True, 120)])
def test_no_padding_keeps_dp_ranks_on_a_shared_token_shape(
    runner: DecodeAclGraphRunner, no_padding: bool, expected: int
) -> None:
    runner.dp_size = 2
    runner.max_batch = 160
    runner.num_decoding_tokens = 5
    runner._model_owns_linear_spec_verify = True
    runner._enable_decode_no_padding = no_padding
    metadata = SimpleNamespace(is_spec_verify=True, dp_execution_token_counts=(115, 120))

    assert runner._padded_batch_size(115, metadata) == expected
    assert runner._padded_batch_size(120, metadata) == expected


def test_no_padding_graph_entries_use_exact_rows_and_reuse_matching_shapes(runner: DecodeAclGraphRunner) -> None:
    runner._enable_decode_no_padding = True
    for rows in (3, 5, 3):
        input_ids = torch.arange(rows, dtype=torch.int32)
        metadata = _metadata()
        metadata.block_table = torch.full((rows, 2), 11, dtype=torch.int64)
        metadata.kv_seq_lens = torch.full((rows,), 2, dtype=torch.int64)
        metadata.kv_seq_lens_host_values = [2] * rows
        metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
        metadata.linear_state_indices = torch.arange(rows, dtype=torch.int32)
        decoded = runner.prepare_execution(input_ids, metadata)
        assert decoded is not None
        key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
        output = runner.execute(input_ids, input_ids, metadata, graph_key=key, decode_metadata=decoded)
        entry = runner._graphs[key]
        assert entry.batch_size == rows
        assert entry.static_input_ids.numel() == rows
        assert output.shape == (rows, 2)

    assert runner._capture.call_count == 2
    assert len(runner._graphs) == 2


@pytest.mark.parametrize("no_padding", [False, True])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_capture_preserves_live_checkpoints_on_success_and_failure(
    runner: DecodeAclGraphRunner, no_padding: bool, capture_fails: bool
) -> None:
    from xllm.python.attention.backend import LayerCache

    runner._enable_decode_no_padding = no_padding
    indices = torch.tensor([1, 3], dtype=torch.int64)
    conv = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    ssm = torch.arange(80, dtype=torch.float32).reshape(20, 2, 2)
    original_conv, original_ssm = conv.clone(), ssm.clone()
    runner.bind_layer_caches([LayerCache(None, None, conv=conv, ssm=ssm)])
    metadata = _metadata()
    metadata.linear_state_indices = indices
    metadata.has_initial_state = torch.ones_like(indices, dtype=torch.bool)

    def capture(entry: AclGraphEntry, _stream: object) -> None:
        capture_indices = entry.static_metadata.linear_state_indices.long()
        assert not capture_indices.any()
        conv.index_fill_(0, capture_indices, -1)
        checkpoints = (capture_indices[:, None] * 5 + torch.arange(5)).reshape(-1)
        ssm.index_fill_(0, checkpoints, -2)
        if capture_fails:
            raise RuntimeError("capture failed")

    runner._capture.side_effect = capture
    expected = pytest.raises(RuntimeError, match="capture failed") if capture_fails else nullcontext()
    with expected:
        runner.warmup(torch.arange(2, dtype=torch.int32), torch.arange(2, dtype=torch.int32), metadata)

    torch.testing.assert_close(conv[1:], original_conv[1:])
    torch.testing.assert_close(ssm[5:], original_ssm[5:])
    torch.testing.assert_close(metadata.linear_state_indices, indices)
    assert metadata.has_initial_state.all()


@pytest.mark.parametrize("mode", ["global", "thread_local"])
def test_capture_passes_runner_error_mode_to_npu_graph(
    runner: DecodeAclGraphRunner, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runner._capture_error_mode = mode
    entry = AclGraphEntry()
    entry.batch_size = 3
    entry.static_metadata = _metadata()
    entry.static_input_ids = torch.arange(3)
    entry.static_mtp_topk_indices = None
    entry.execution_state = None
    stream = Mock()
    graph_context = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.npu, "NPUGraph", Mock(), raising=False)
    monkeypatch.setattr(torch.npu, "graph", graph_context, raising=False)
    monkeypatch.setattr(torch.npu, "synchronize", Mock(), raising=False)
    monkeypatch.setattr(runner, "_forward_static", Mock(return_value=torch.ones(3, 2)))
    with patch("xllm.python.model_executor.runners.acl_graph._get_graph_pool", return_value=(1, 2)):
        AclGraphRunner._capture(runner, entry, stream)
    assert graph_context.call_args.kwargs["capture_error_mode"] == mode
    assert graph_context.call_args.kwargs["stream"] is stream


def test_metadata_is_decoded_once_for_new_and_cached_graph_steps(runner: DecodeAclGraphRunner) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    positions = torch.tensor([1, 4], dtype=torch.int32)
    metadata = _metadata()
    with (
        patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode,
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.build_row_aligned_paged_kv_metadata",
            wraps=build_row_aligned_paged_kv_metadata,
        ) as build_paging,
    ):
        for step in range(2):
            decoded = runner.prepare_execution(input_ids, metadata)
            assert decoded is not None
            graph_key = runner.warmup(input_ids, positions, metadata, decode_metadata=decoded)
            output = runner.execute(input_ids, positions, metadata, graph_key=graph_key, decode_metadata=decoded)
            entry = runner._graphs[graph_key]

            assert output.shape == (2, 2)
            assert decode.call_count == step + 1
            assert build_paging.call_count == step + 1
            torch.testing.assert_close(entry.static_metadata.block_table[:, :2], metadata.block_table.to(torch.int32))
            assert entry.static_metadata.kv_seq_lens_host_values == metadata.kv_seq_lens_host_values
            assert decoded.kv_seq_lens.tolist() == [2 + step, 5 + step]
            assert decoded.paged_kv_indptr.tolist() == [0, 1, 3]
            assert decoded.paged_kv_indices.tolist() == [11 + step, 21 + step, 22 + step]
            assert decoded.paged_kv_last_page_len.tolist() == [2 + step, 1 + step]

            # Reuse the same metadata object with new values on the next step.
            metadata.block_table.add_(1)
            metadata.kv_seq_lens.add_(1)
            metadata.kv_seq_lens_host_values[:] = metadata.kv_seq_lens.tolist()

    runner._capture.assert_called_once()
    assert entry.graph.replay.call_count == 2


def test_executor_decodes_metadata_once_per_step_with_new_and_cached_graphs(
    runner: DecodeAclGraphRunner, executor: ModelExecutor
) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    positions = torch.tensor([1, 4], dtype=torch.int32)
    metadata = _metadata()
    with (
        patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode,
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.build_row_aligned_paged_kv_metadata",
            wraps=build_row_aligned_paged_kv_metadata,
        ) as build_paging,
    ):
        for step in range(2):
            output = executor.execute(input_ids, positions, metadata)

            assert output.shape == (2, 2)
            assert decode.call_count == step + 1
            assert build_paging.call_count == step + 1
            assert len(runner._graphs) == 1
            entry = next(iter(runner._graphs.values()))
            assert entry.static_metadata.kv_seq_lens_host_values == [2 + step, 5 + step]
            torch.testing.assert_close(entry.static_metadata.block_table[:, :2], metadata.block_table.to(torch.int32))
            metadata.block_table.add_(1)
            metadata.kv_seq_lens.add_(1)
            metadata.kv_seq_lens_host_values[:] = metadata.kv_seq_lens.tolist()

    runner._capture.assert_called_once()
    assert entry.graph.replay.call_count == 2
    executor.eager_runner.execute.assert_not_called()


def test_executor_uses_eager_when_decode_admission_returns_none(
    runner: DecodeAclGraphRunner, executor: ModelExecutor
) -> None:
    runner.max_batch = 1
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        output = executor.execute(input_ids, input_ids, metadata)

    decode.assert_not_called()
    executor.eager_runner.execute.assert_called_once_with(input_ids, input_ids, metadata, None, None, None, eplb=None)
    assert output is executor.eager_runner.execute.return_value
    runner._capture.assert_not_called()
    assert not runner._graphs


def test_executor_preserves_mtp_inputs_and_owned_outputs_with_metadata_reuse(
    runner: DecodeAclGraphRunner, executor: ModelExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def capture(entry: AclGraphEntry, stream: object) -> None:
        entry.static_output = (
            torch.empty_like(entry.static_input_embedding),
            None,
            torch.empty_like(entry.static_mtp_topk_indices),
        )

        def replay() -> None:
            entry.static_output[0].copy_(entry.static_input_embedding)
            entry.static_output[2].copy_(entry.static_mtp_topk_indices)

        entry.graph = Mock(replay=Mock(side_effect=replay))

    monkeypatch.setattr(runner, "_capture", Mock(side_effect=capture))
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    input_embedding = torch.arange(4, dtype=torch.float32).view(2, 2)
    topk = torch.arange(6, dtype=torch.int32).view(2, 1, 3)
    expected_hidden = input_embedding.clone()
    expected_topk = topk.clone()
    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        first = executor.execute(input_ids, input_ids, metadata, input_embedding, mtp_topk_indices=topk)
        assert decode.call_count == 1
        input_embedding.add_(10)
        topk.add_(20)
        metadata.kv_seq_lens.add_(1)
        metadata.kv_seq_lens_host_values[:] = metadata.kv_seq_lens.tolist()
        second = executor.execute(input_ids, input_ids, metadata, input_embedding, mtp_topk_indices=topk)
        assert decode.call_count == 2

    torch.testing.assert_close(first[0], expected_hidden)
    torch.testing.assert_close(first[2], expected_topk)
    assert first[1] is None
    torch.testing.assert_close(second[0], input_embedding)
    torch.testing.assert_close(second[2], topk)
    assert second[1] is None
    runner._capture.assert_called_once()
    entry = next(iter(runner._graphs.values()))
    assert entry.graph.replay.call_count == 2
    executor.eager_runner.execute.assert_not_called()


def test_direct_runner_calls_decode_current_metadata_without_retained_state(runner: DecodeAclGraphRunner) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        assert runner.can_execute(input_ids, metadata) is True
        graph_key = runner.warmup(input_ids, input_ids, metadata)
        entry = runner._graphs[graph_key]
        decode.reset_mock()

        metadata.block_table.add_(10)
        metadata.kv_seq_lens.add_(1)
        metadata.kv_seq_lens_host_values[:] = [3, 6]
        runner.execute(input_ids, input_ids, metadata, graph_key=graph_key)

        decode.assert_called_once_with(metadata, None)
        torch.testing.assert_close(entry.static_metadata.block_table[:, :2], metadata.block_table.to(torch.int32))
        assert entry.static_metadata.kv_seq_lens_host_values == [3, 6]


def test_new_step_still_validates_metadata_after_successful_replay(runner: DecodeAclGraphRunner) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    decoded = runner.prepare_execution(input_ids, metadata)
    assert decoded is not None
    graph_key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
    runner.execute(input_ids, input_ids, metadata, graph_key=graph_key, decode_metadata=decoded)

    metadata.kv_seq_lens_host_values = [2]
    assert runner.prepare_execution(input_ids, metadata) is None
    runner._graphs[graph_key].graph.replay.assert_called_once()


def test_admission_keeps_unsupported_and_invalid_input_order(runner: DecodeAclGraphRunner) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    metadata.block_table = None
    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        assert runner.prepare_execution(input_ids.view(1, 2), metadata) is None
        metadata.is_prefill = True
        assert runner.prepare_execution(input_ids, metadata) is None
        metadata.is_prefill = False
        with pytest.raises(ValueError, match="one leading row per decode token"):
            runner.prepare_execution(input_ids, metadata, mtp_topk_indices=torch.zeros(1, 2))
        decode.assert_not_called()
        assert runner.prepare_execution(input_ids, metadata) is None
        decode.assert_called_once_with(metadata, None)


def test_metadata_reuse_keeps_dp_admission_errors(runner: DecodeAclGraphRunner) -> None:
    runner.dp_size = 2
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    with pytest.raises(RuntimeError, match="requires valid dp_execution_token_counts"):
        runner.prepare_execution(input_ids, metadata)

    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 0)
    assert runner.prepare_execution(input_ids, metadata) is None
    metadata.dp_is_decode = (1, 1)
    metadata.dp_execution_token_counts = (3, 2)
    with pytest.raises(RuntimeError, match="does not match the local input"):
        runner.prepare_execution(input_ids, metadata)

    metadata.dp_execution_token_counts = (2, 2)
    metadata.kv_seq_lens_host_values = [2]
    with pytest.raises(RuntimeError, match="one value per metadata row|row count does not match KV lengths"):
        runner.prepare_execution(input_ids, metadata)


def test_metadata_reuse_preserves_dp_capture_and_replay_coordination(
    runner: DecodeAclGraphRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.dp_size = 2
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 1)
    monkeypatch.setattr("xllm.python.model_executor.runners.decode_acl_graph.dist.is_initialized", lambda: True)
    gather = Mock(side_effect=lambda tensor, **_: torch.cat((tensor, tensor)))
    reduce = Mock(side_effect=lambda tensor, **_: tensor.mul_(2))
    monkeypatch.setattr("xllm.python.distributed.all_gather", gather)
    monkeypatch.setattr("xllm.python.distributed.all_reduce_", reduce)

    for _ in range(2):
        decoded = runner.prepare_execution(input_ids, metadata)
        assert decoded is not None
        graph_key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
        runner.execute(input_ids, input_ids, metadata, graph_key=graph_key, decode_metadata=decoded)

    runner._capture.assert_called_once()
    assert gather.call_count == 2
    assert reduce.call_count == 4


def test_metadata_reuse_preserves_kpool_span_graph_variants(runner: DecodeAclGraphRunner) -> None:
    input_ids = torch.tensor([10, 20], dtype=torch.int32)
    metadata = _metadata()
    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        for query_lens in ((1, 1), (2,), (1, 1)):
            metadata.kpool_query_lens = query_lens
            decoded = runner.prepare_execution(input_ids, metadata)
            assert decoded is not None
            graph_key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
            runner.execute(input_ids, input_ids, metadata, graph_key=graph_key, decode_metadata=decoded)

            assert runner._graphs[graph_key].static_metadata.kpool_query_lens == query_lens

    assert decode.call_count == 3
    assert len(runner._graphs) == 2
    assert runner._capture.call_count == 2


@pytest.mark.parametrize("query_ends", ([0, 1, 2, 3, 4], [1, 2, 3, 4], [0, 2, 4], [2, 4], None))
def test_metadata_reuse_preserves_expanded_verify_rows_and_query_groups(
    runner: DecodeAclGraphRunner, query_ends: list[int] | None
) -> None:
    runner.attention_backend.supports_linear_spec_verify_graph = True
    input_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    metadata = _metadata()
    metadata.is_chunked_prefill = True
    metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
    metadata.q_seq_lens_host = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor(query_ends, dtype=torch.int32) if query_ends is not None else None
    metadata.q_cu_seq_lens_host_values = query_ends
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        block_table=torch.tensor([[11, 0], [11, 0], [21, 22], [21, 22]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([1, 2, 4, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1, 2, 4, 5], dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 4, 5],
        paged_kv_indptr=torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32),
        paged_kv_indices=torch.tensor([11, 11, 21, 21, 22], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1, 2, 4, 1], dtype=torch.int32),
        paged_attention_tiling_data=None,
    )

    with patch.object(runner, "_decode_metadata", wraps=runner._decode_metadata) as decode:
        for _ in range(2):
            decoded = runner.prepare_execution(input_ids, metadata)
            assert decoded is not None
            graph_key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
            runner.execute(input_ids, input_ids, metadata, graph_key=graph_key, decode_metadata=decoded)
            static_metadata = runner._graphs[graph_key].static_metadata

            assert static_metadata.kv_seq_lens_host_values == [1, 2, 4, 5]
            assert static_metadata.q_seq_lens_host.tolist() == [2, 2]
            assert static_metadata.q_cu_seq_lens_host_values == list(range(5))
            torch.testing.assert_close(static_metadata.q_cu_seq_lens, torch.arange(5, dtype=torch.int32))
            torch.testing.assert_close(
                static_metadata.block_table[:, :2], metadata.expanded_decode_metadata.block_table
            )

    assert decode.call_count == 2
    runner._capture.assert_called_once()


def test_expanded_query_row_offsets_survive_padding_and_bucket_reuse(runner: DecodeAclGraphRunner) -> None:
    runner.attention_backend.supports_linear_spec_verify_graph = True
    for rows in (6, 8, 6):
        groups = rows // 2
        input_ids = torch.arange(rows, dtype=torch.int32)
        metadata = _metadata()
        metadata.is_chunked_prefill = True
        metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
        metadata.linear_state_indices = torch.arange(groups, dtype=torch.int32)
        metadata.q_seq_lens_host = torch.full((groups,), 2, dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.arange(0, rows + 1, 2, dtype=torch.int32)
        metadata.q_cu_seq_lens_host_values = list(range(0, rows + 1, 2))
        metadata.expanded_decode_metadata = SimpleNamespace(
            enabled=True,
            block_table=torch.full((rows, 1), 11, dtype=torch.int32),
            kv_seq_lens=torch.ones(rows, dtype=torch.int32),
            kv_seq_lens_host=torch.ones(rows, dtype=torch.int32),
            kv_seq_lens_host_values=[1] * rows,
            paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
            paged_kv_indices=torch.full((rows,), 11, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(rows, dtype=torch.int32),
            paged_attention_tiling_data=None,
        )
        decoded = runner.prepare_execution(input_ids, metadata)
        assert decoded is not None
        graph_key = runner.warmup(input_ids, input_ids, metadata, decode_metadata=decoded)
        runner.execute(input_ids, input_ids, metadata, graph_key=graph_key, decode_metadata=decoded)
        entry = runner._graphs[graph_key]
        expected = list(range(rows + 1)) + [rows] * (entry.batch_size - rows)
        assert entry.static_metadata.q_cu_seq_lens.tolist() == expected
        assert entry.static_metadata.q_cu_seq_lens_host_values == expected
        assert entry.expanded_query_cu_seq_lens.tolist() == list(range(entry.batch_size + 1))
        assert entry.static_metadata.q_seq_lens_host.tolist() == [2] * 4

    runner._capture.assert_called_once()

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

"""Unit tests for xllm.python.model_executor.executor.

Tests the device-conditional backend dispatch, ModelExecutor construction
validation, and execution routing — using CPU mocks so no GPU/NPU required.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# conftest.py stands in for xllm.python, whose import would bind the active
# platform's kernel package and reach for operators from the C++ binary.
from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _resolve_graph_backend,
)
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_graph_buckets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[LayerCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._prepared = True

    def execute(self, q, k, v, layer) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


class _PagedStubAttentionBackend(StubAttentionBackend):
    @property
    def page_size(self) -> int:
        return 4


def _make_attention_layer(
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    scale=0.125,
    sliding_window=0,
    layer_id=0,
) -> Attention:
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        sliding_window=sliding_window,
        layer_id=layer_id,
    )


class _FakeModel(nn.Module):
    """Model with configurable number of uniform Attention layers."""

    def __init__(self, num_layers: int = 2, device: str = "cpu", **attn_kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)  # execution_model placeholder
        self.layers = nn.ModuleList([_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)])
        self._param = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, input_ids, positions):
        return input_ids


class _FakeModelHeterogeneous(nn.Module):
    """Model with non-uniform Attention layers (should fail validation)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attn1 = _make_attention_layer(num_heads=8, layer_id=0)
        self.attn2 = _make_attention_layer(num_heads=4, layer_id=1)
        self._param = nn.Parameter(torch.zeros(1))


class _FakeModelNoAttention(nn.Module):
    """Model without any Attention layers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self._param = nn.Parameter(torch.zeros(1))


# ---------------------------------------------------------------------------
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class TestNpuGraphBackendResolution:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_enable_graph_selects_aclgraph_on_npu(self, _mock_is_npu):
        config = {"enable_graph": True, "python_graph_backend": "off"}
        assert _resolve_graph_backend(config) == "aclgraph"


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_device_creates_npu_backend(self, _mock_is_npu):
        attn = _make_attention_layer()
        backend = _create_attention_backend(attn, torch.device("npu"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 2
        assert backend.init_kwargs["head_dim"] == 64

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=False,
    )
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_cuda",
        return_value=True,
    )
    def test_cuda_device_creates_flashinfer_backend(self, _mock_is_cuda, _mock_is_npu):
        attn = _make_attention_layer()
        module = types.ModuleType("xllm.python.attention.flashinfer")
        module.FlashInferBackend = StubAttentionBackend
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("cuda"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)


# ---------------------------------------------------------------------------
# Tests: ModelExecutor construction
# ---------------------------------------------------------------------------


class TestModelExecutorConstruction:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_valid_model_creates_executor(self, _mock_backend):
        model = _FakeModel(num_layers=3)
        config = {"python_graph_backend": "off"}
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)

        assert executor._num_attention_layers == 3
        assert executor.decode_graph_runner is None
        assert executor.inductor_runner is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_no_attention_layers_raises(self, _mock_backend):
        model = _FakeModelNoAttention()
        with pytest.raises(ValueError, match="does not contain an Attention layer"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_heterogeneous_attention_raises(self, _mock_backend):
        model = _FakeModelHeterogeneous()
        with pytest.raises(ValueError, match="identical attention configuration"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_graph_backend_off_variants(self, _mock_backend):
        for off_value in ("off", "", "none", "0"):
            model = _FakeModel(num_layers=1)
            executor = ModelExecutor(model, {"python_graph_backend": off_value}, max_seqs_per_batch=4)
            assert executor.decode_graph_runner is None
            assert executor.inductor_runner is None

    @patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_cuda_graph_is_supported(self, mock_create, mock_graph_runner):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 2,
                "dp_rank": 1,
                "max_position_embeddings": 128,
                "python_graph_backend": "cudagraphs",
            },
            max_seqs_per_batch=4,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            4,
            128,
            2,
            1,
        )

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_rejects_unsupported_graph_backend(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        with pytest.raises(NotImplementedError, match="supports cudagraphs and aclgraph only"):
            ModelExecutor(
                model,
                {
                    "dp_size": 2,
                    "max_position_embeddings": 128,
                    "python_graph_backend": "inductor",
                },
                max_seqs_per_batch=4,
            )


class TestDecodeCudaGraphDataParallelKeys:
    @staticmethod
    def _runner(dp_rank: int = 0) -> DecodeCudaGraphRunner:
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.max_batch = 16
        runner.dp_size = 2
        runner.dp_rank = dp_rank
        runner._graphs = {}
        return runner

    @staticmethod
    def _metadata(token_counts: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_token_counts=token_counts,
        )

    def test_graph_key_uses_global_max_data_parallel_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)

        first = runner._graph_key(input_ids, self._metadata([3, 1]))
        second = runner._graph_key(input_ids, self._metadata([3, 2]))

        assert first == (4, (4, 4))
        assert second == first

    def test_data_parallel_warmup_uses_local_batch_capacity(self):
        assert _decode_graph_buckets(16, 2) == [1, 2, 4, 8]
        assert _decode_graph_buckets(20, 2) == [1, 2, 4, 8, 16]

    def test_single_rank_graph_key_reuses_padded_bucket(self):
        runner = self._runner()
        runner.dp_size = 1
        runner.dp_rank = 0

        first = runner._graph_key(torch.zeros(3, dtype=torch.int32), self._metadata([3]))
        second = runner._graph_key(torch.zeros(4, dtype=torch.int32), self._metadata([4]))

        assert first == (4, (4,))
        assert second == first

    def test_graph_key_accepts_empty_data_parallel_rank(self):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([5, 0])) == (
            8,
            (8, 8),
        )

    def test_graph_key_rejects_unbalanced_unwarmed_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(9, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([9, 7])) is None

    def test_can_execute_requires_warmed_graph(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        graph_key = runner._graph_key(input_ids, metadata)

        assert not runner.can_execute(input_ids, metadata)
        runner._graphs[graph_key] = object()
        assert runner.can_execute(input_ids, metadata)

    @pytest.mark.parametrize("token_counts", ([3], [3, -1], [3, 2]))
    def test_graph_key_rejects_invalid_data_parallel_metadata(self, token_counts):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata(token_counts)) is None


# ---------------------------------------------------------------------------
# Tests: DecodeAclGraphRunner speculative metadata
# ---------------------------------------------------------------------------


class TestDecodeAclGraphSpeculativeMetadata:
    @staticmethod
    def _runner() -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=4,
            max_model_len=8,
        )

    @staticmethod
    def _metadata() -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(4, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
            q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_cu_seq_lens=torch.tensor([0, 4, 12], dtype=torch.int32),
            kv_seq_lens_host=torch.tensor([4, 8], dtype=torch.int32),
            kv_seq_lens_host_values=[4, 8],
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([4, 8], dtype=torch.int32),
            q_seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                enabled=True,
                kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                block_table=torch.tensor(
                    [[10, 11], [10, 11], [20, 21], [20, 21]],
                    dtype=torch.int32,
                ),
                paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
                paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                kv_seq_lens_host_values=[3, 4, 7, 8],
            ),
            is_prefill=False,
            is_chunked_prefill=True,
        )

    def test_expanded_metadata_selects_matching_paged_kv_rows(self) -> None:
        runner = self._runner()
        (
            block_table,
            kv_seq_lens,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = runner._decode_metadata(self._metadata())

        assert block_table.tolist() == [
            [10, 11],
            [10, 11],
            [20, 21],
            [20, 21],
        ]
        assert kv_seq_lens.tolist() == [3, 4, 7, 8]
        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_expanded_chunked_verify_can_use_decode_graph(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)

        assert runner.can_execute(input_ids, self._metadata())

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "block_table",
                torch.arange(8, dtype=torch.int32),
                "block_table must be two-dimensional",
            ),
            (
                "kv_seq_lens",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens must contain one value per sequence",
            ),
            (
                "kv_seq_lens_host",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens_host must contain one value per sequence",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4], dtype=torch.int32),
                "paged_kv_indptr must contain one offset per sequence",
            ),
            (
                "paged_kv_indices",
                torch.tensor([[10, 10], [20, 21]], dtype=torch.int32),
                "paged_kv_indices must be a non-empty flat page list",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 3], dtype=torch.int32),
                "paged_kv_last_page_len must contain one value per sequence",
            ),
        ],
    )
    def test_expanded_metadata_shape_mismatch_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "paged_kv_indptr",
                torch.tensor([1, 1, 2, 4, 6], dtype=torch.int32),
                "must start at zero",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 2, 1, 4, 6], dtype=torch.int32),
                "must be monotonic",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32),
                "terminal page offset must match page count",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 0, 4], dtype=torch.int32),
                "last-page lengths must be positive",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 5, 4], dtype=torch.int32),
                "must not exceed block size",
            ),
        ],
    )
    def test_expanded_paged_metadata_invariant_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    def test_expanded_page_count_exceeding_capacity_fails(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata.kv_seq_lens_host_values = [3, 4, 7, 9]

        with pytest.raises(RuntimeError, match="exceeds block-table capacity"):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("input_ids", "slot_mapping", "message"),
        [
            (
                torch.arange(3, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                "input_ids must contain one token per sequence",
            ),
            (
                torch.arange(4, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32),
                "slot_mapping must contain one slot per token",
            ),
        ],
    )
    def test_token_layout_mismatch_fails(
        self,
        input_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=message):
            self._runner()._validate_decode_token_layout(
                input_ids,
                torch.arange(4, dtype=torch.int32),
                slot_mapping,
                sequence_count=4,
            )


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestBindKvCaches:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_correct_count(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])
        assert len(backend._kv_caches) == 2

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_wrong_count_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="layer count does not match"):
            executor.bind_kv_caches([kv])

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_idempotent(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])
        executor.bind_kv_caches([kv])  # should not raise or re-bind


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


class TestExecuteRouting:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_bind_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        metadata = MagicMock(spec=AttentionMetadata)
        with pytest.raises(RuntimeError, match="KV caches are not bound"):
            executor.execute(torch.zeros(1), torch.zeros(1), metadata)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_routes_to_eager_runner(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        executor.eager_runner = MagicMock()
        grad_enabled = None

        def execute(*_args):
            nonlocal grad_enabled
            grad_enabled = torch.is_grad_enabled()
            return torch.ones(5)

        executor.eager_runner.execute.side_effect = execute

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.eager_runner.execute.assert_called_once()
        assert grad_enabled is False
        assert torch.equal(result, torch.ones(5))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_inductor_runner_takes_priority_over_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        executor.inductor_runner = MagicMock()
        executor.inductor_runner.execute.return_value = torch.ones(3)

        metadata = MagicMock(spec=AttentionMetadata)
        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.inductor_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(3))

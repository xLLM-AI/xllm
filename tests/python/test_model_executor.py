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

"""Unit tests for xllm.python.model_executor.executor.

Tests backend dispatch, ModelExecutor construction validation, and execution
routing with test-specific mocks after real runtime initialization.
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _resolve_graph_backend,
    _validate_npu_cp_model_config,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_graph_buckets,
)
from xllm.python.model_executor.runners.eager import EagerRunner  # noqa: E402

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


class _MlaStubAttentionBackend(StubAttentionBackend):
    @property
    def is_mla(self) -> bool:
        return True


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


class _FailingLayerSynchronizer:
    def record_event(self, layer_id: int) -> bool:
        return False


def test_attention_backend_defaults_to_non_mla() -> None:
    assert StubAttentionBackend().is_mla is False


def test_record_layer_event_propagates_record_failure() -> None:
    context = ForwardContext(
        attention_backend=StubAttentionBackend(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        layer_synchronizer=_FailingLayerSynchronizer(),
    )

    with (
        forward_context(context),
        pytest.raises(RuntimeError, match="failed to record layer completion event for layer 3"),
    ):
        record_layer_event(3)


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

    def test_glm_mtp_allows_aclgraph_for_cross_draft_topk_state(self) -> None:
        config = {
            "model_type": "glm_moe_dsa_mtp",
            "enable_graph": True,
            "python_graph_backend": "aclgraph",
        }

        assert _resolve_graph_backend(config) == "aclgraph"


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_creates_dsa_backend(self, _mock_is_npu):
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4",
            "compress_ratios": [1, 4, 128],
            "num_hidden_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("npu"), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [1, 4, 128]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_device_creates_npu_backend(self, _mock_is_npu):
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        backend = _create_attention_backend(
            attn,
            torch.device("npu"),
            torch.float16,
            {"enable_mla": False},
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 256
        assert backend.init_kwargs["is_mla"] is False

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_prefill_cp_uses_npu_backend_with_dcp_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 4, "enable_mla": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["is_mla"] is True
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_decode_cp1_uses_sfa_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 1, "enable_mla": True},
                max_num_reqs=3,
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 3

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

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_capacity_respects_decode_batch_limit(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=256,
            num_decoding_tokens=4,
            acl_graph_decode_batch_size_limit=16,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            64,
            128,
            1,
            0,
            16,
            4,
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
    def _metadata(
        token_counts: list[int] | tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
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

    def test_graph_key_accepts_dummy_execution_row_for_empty_data_parallel_rank(self):
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

        with pytest.raises(RuntimeError):
            runner._graph_key(input_ids, self._metadata(token_counts))


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

    def test_decode_metadata_rebuilds_token_row_paging_metadata(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata = None
        metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        metadata.block_table = torch.tensor(
            [[10, 11], [10, 11], [20, 21], [20, 21]],
            dtype=torch.int32,
        )
        metadata.kv_seq_lens = torch.tensor([3, 4, 7, 8], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [3, 4, 7, 8]
        metadata.paged_kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([10, 20, 21], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([4, 4], dtype=torch.int32)

        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._runner()._decode_metadata(metadata)

        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_expanded_chunked_verify_can_use_decode_graph(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)

        assert runner.can_execute(input_ids, self._metadata())

    def test_decode_batch_limit_uses_speculative_tokens_and_dp_global_max(
        self,
    ) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_decoding_tokens=4,
        )
        metadata = SimpleNamespace(
            dp_execution_token_counts=(32, 64),
        )

        assert runner._decode_batch_sizes(
            torch.zeros(32, dtype=torch.int32),
            metadata,
        ) == (8, 16)

    def test_mtp3_batch_eight_uses_32_row_graph_bucket(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_decoding_tokens=4,
        )
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=(),
            linear_state_indices=torch.arange(8, dtype=torch.int32),
        )

        with (
            patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
            patch(
                "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
                return_value=object(),
            ),
        ):
            # A KDA expanded batch requires a recurrent-state protocol.
            with patch.object(_PagedStubAttentionBackend, "supports_linear_spec_verify_graph", False):
                assert not runner.can_execute(torch.zeros(32, dtype=torch.int32), metadata)
            with patch.object(_PagedStubAttentionBackend, "supports_linear_spec_verify_graph", True):
                assert runner.can_execute(torch.zeros(32, dtype=torch.int32), metadata)

    def test_warmup_captures_with_scheduler_metadata_once(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = self._metadata()
        graph_key = runner._graph_key(
            padded_batch_size=4,
            is_expanded=True,
            input_embedding=None,
        )

        with patch.object(runner, "_prepare_graph_entry") as prepare:
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once_with(
                input_ids,
                positions,
                metadata,
                None,
                graph_key=graph_key,
            )

            runner._graphs[graph_key] = object()
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once()

        # Capture warmup advances KDA state; the first real replay must start
        # from the original recurrent state, not from a synthetic warmup step.
        conv = torch.arange(4, dtype=torch.float32)
        ssm = torch.arange(4, dtype=torch.float32) + 10
        runner.layer_caches = [SimpleNamespace(conv=conv, ssm=ssm)]
        entry = SimpleNamespace(static_metadata=SimpleNamespace(linear_state_indices=torch.tensor([1])))
        caller_stream = MagicMock()
        graph_stream = MagicMock()

        def advance_state(_entry, _stream):
            conv[1] += 10
            ssm[1] += 20

        with (
            patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: caller_stream), create=True),
            patch("xllm.python.model_executor.runners.acl_graph.AclGraphRunner._capture", side_effect=advance_state),
        ):
            runner._capture(entry, graph_stream)

        assert conv.tolist() == [0, 1, 2, 3]
        assert ssm.tolist() == [10, 11, 12, 13]
        caller_stream.wait_stream.assert_called_once_with(graph_stream)

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
                "input_ids must contain one token per metadata row",
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
                metadata_row_count=4,
            )

    def test_replay_returns_detached_static_output(self) -> None:
        runner = self._runner()
        batch_size = 3
        padded_batch_size = 4
        static_output = torch.arange(12).reshape(padded_batch_size, 3)
        graph = MagicMock()
        entry = SimpleNamespace(
            batch_size=padded_batch_size,
            graph=graph,
            static_output=static_output,
            static_metadata=SimpleNamespace(),
            graph_tasks=[],
            execution_state=SimpleNamespace(persistent_buffers={}),
        )
        graph_key = runner._graph_key(
            padded_batch_size,
            is_expanded=False,
            input_embedding=None,
        )
        runner._graphs[graph_key] = entry

        replay_stream = MagicMock()
        update_stream = MagicMock()
        replay_done_event = MagicMock()
        current_stream = MagicMock()
        runner._stream = replay_stream
        runner._update_stream = update_stream
        runner._replay_done_event = replay_done_event
        runner._update_done_event = MagicMock()
        fake_npu = SimpleNamespace(
            current_stream=MagicMock(return_value=current_stream),
            stream=MagicMock(return_value=nullcontext()),
        )
        metadata = SimpleNamespace(expanded_decode_metadata=None)

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(
                runner,
                "_prepare_graph_entry",
                return_value=entry,
            ),
        ):
            output = runner.execute(
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, dtype=torch.int32),
                metadata,
            )

        assert output.shape == (batch_size, 3)
        assert output.data_ptr() != static_output.data_ptr()
        output[0, 0] = -1
        assert static_output[0, 0].item() == 0
        replay_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(replay_stream)
        graph.replay.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestNormalizeLayerCaches:
    def test_legacy_five_slot_cache_keeps_generic_layout(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 6))

        cache = normalize_layer_caches([tensors])[0]

        assert cache.key is tensors[0]
        assert cache.value is tensors[1]
        assert cache.index is tensors[2]
        assert cache.conv is tensors[3]
        assert cache.ssm is tensors[4]
        assert cache.swa is None
        assert cache.compress_kv_state is None
        assert cache.compress_score_state is None
        assert cache.compress_index_kv_state is None
        assert cache.compress_index_score_state is None
        assert cache.indexer_scale is None

    def test_deepseek_v4_eleven_slot_cache_maps_all_slots(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 12))

        cache = normalize_layer_caches([tensors])[0]

        assert (
            cache.key,
            cache.value,
            cache.index,
            cache.conv,
            cache.ssm,
            cache.swa,
            cache.compress_kv_state,
            cache.compress_score_state,
            cache.compress_index_kv_state,
            cache.compress_index_score_state,
            cache.indexer_scale,
        ) == tensors

    def test_empty_deepseek_v4_slots_are_normalized_to_none(self):
        cache = normalize_layer_caches([(torch.ones(1), torch.ones(1), *(torch.empty(0),) * 9)])[0]

        assert cache.key is not None
        assert cache.value is not None
        assert cache.index is None
        assert cache.indexer_scale is None


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


@pytest.mark.parametrize(
    ("model_type", "is_draft", "speculative_tokens", "decoding_tokens"),
    [
        ("glm_moe_dsa", False, 1, 1),
        ("glm_moe_dsa", False, 0, 3),
        ("glm_moe_dsa_mtp", True, 0, 1),
    ],
)
@pytest.mark.parametrize("kv_split", [0, 1, 2])
def test_glm_cp_mtp_admission_requires_replicated_kv(
    model_type: str,
    is_draft: bool,
    speculative_tokens: int,
    decoding_tokens: int,
    kv_split: int,
) -> None:
    config = {
        "model_type": model_type,
        "is_draft_engine": is_draft,
        "num_speculative_tokens": speculative_tokens,
        "cp_size": 2,
        "kv_split_size": kv_split,
        # Sharded KV is valid for ordinary disaggregated prefill, but not MTP.
        "instance_role": "PREFILL",
        "enable_disagg_pd": True,
    }
    if kv_split == 1:
        _validate_npu_cp_model_config(config, decoding_tokens)
    else:
        with pytest.raises(NotImplementedError, match="MTP requires replicated KV caches"):
            _validate_npu_cp_model_config(config, decoding_tokens)


@pytest.mark.parametrize(("is_draft", "algorithm"), [(False, "mtp"), (True, "eagle3"), (True, "dflash")])
def test_glm_cp_draft_admission_requires_mtp_draft_engine(is_draft: bool, algorithm: str) -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "is_draft_engine": is_draft,
        "speculative_algorithm": algorithm,
        "cp_size": 2,
        "kv_split_size": 1,
    }
    with pytest.raises(NotImplementedError, match="does not support model_type"):
        _validate_npu_cp_model_config(config, 1)


def test_glm_non_mtp_cp_prefill_retains_sharded_kv_admission() -> None:
    _validate_npu_cp_model_config(
        {
            "model_type": "glm_moe_dsa",
            "cp_size": 2,
            "kv_split_size": 2,
            "instance_role": "PREFILL",
            "enable_disagg_pd": True,
        },
        1,
    )


def _mtp_graph_metadata(lengths: list[int], expanded: bool, device: torch.device) -> SimpleNamespace:
    page_ids = [2, 5, 7, 9]
    page_counts = [(length + 3) // 4 for length in lengths]
    indptr = [0]
    for count in page_counts:
        indptr.append(indptr[-1] + count)
    values = {
        "kv_seq_lens": lengths,
        "block_table": [page_ids for _ in lengths],
        "paged_kv_indptr": indptr,
        "paged_kv_indices": [page for count in page_counts for page in page_ids[:count]],
        "paged_kv_last_page_len": [(length - 1) % 4 + 1 for length in lengths],
    }
    tensors = {name: torch.tensor(value, dtype=torch.int32, device=device) for name, value in values.items()}
    metadata = SimpleNamespace(
        **tensors,
        slot_mapping=torch.tensor(
            [page_ids[(length - 1) // 4] * 4 + (length - 1) % 4 for length in lengths],
            dtype=torch.int32,
            device=device,
        ),
        q_cu_seq_lens=torch.arange(len(lengths) + 1, dtype=torch.int32, device=device),
        kv_cu_seq_lens=None,
        kv_seq_lens_host_values=lengths,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=expanded,
        is_spec_verify=expanded,
        dp_execution_token_counts=(),
    )
    if expanded:
        metadata.expanded_decode_metadata = SimpleNamespace(
            **tensors,
            enabled=True,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=lengths,
        )
        # Parent metadata still describes one request. The runner must consume
        # the expanded per-token lengths and page rows for verification.
        metadata.block_table = metadata.block_table[-1:]
        metadata.kv_seq_lens = metadata.kv_seq_lens[-1:]
        metadata.q_cu_seq_lens = torch.tensor([0, len(lengths)], dtype=torch.int32, device=device)
    return metadata


@torch.inference_mode()
def test_cp_mtp_decode_acl_replays_persistent_rows_and_detaches_outputs() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")

    class _GraphBackend(_MlaStubAttentionBackend):
        @property
        def page_size(self) -> int:
            return 4

    class _GraphModel(nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            positions: torch.Tensor,
            hidden: torch.Tensor | None = None,
            topk: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, None, torch.Tensor]:
            context = get_forward_context()
            assert context.cp_context is None
            metadata = context.metadata
            indices = positions.view(-1, 1, 1) + 1 if topk is None else topk + 1
            values = tokens + positions + metadata.slot_mapping + metadata.kv_seq_lens
            values = values + metadata.block_table[:, 0] * 10
            values = values + indices.sum(dim=(1, 2))
            if hidden is None:
                hidden = torch.zeros_like(tokens, dtype=torch.float32).view(-1, 1).expand(-1, 2)
            return hidden + values.to(hidden.dtype).view(-1, 1), None, indices

    device = torch.device("npu:0")
    runner = DecodeAclGraphRunner(_GraphModel(), _GraphBackend(), device, max_batch=4, max_model_len=16)
    bindings: dict[tuple, tuple[int, ...]] = {}
    retained: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    # Verify K+1 rows, repair two rows, then draft one row; repeat after a page
    # boundary and a new request reuses the same graph buckets. No per-replay
    # CPU synchronization is inserted between these invocations.
    for cycle in range(3):
        for phase, rows in enumerate((3, 2, 1)):
            lengths = [cycle * 4 + 3 + row for row in range(rows)]
            metadata = _mtp_graph_metadata(lengths, phase == 0, device)
            tokens_cpu = torch.arange(rows, dtype=torch.int32) + cycle * 10 + phase + 1
            positions_cpu = torch.tensor(lengths, dtype=torch.int32) - 1
            hidden_cpu = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2) + cycle * 100
            topk_cpu = torch.arange(rows * 2, dtype=torch.int32).view(rows, 1, 2) + cycle * 7
            tokens = tokens_cpu.to(device)
            positions = positions_cpu.to(device)
            hidden = None if phase == 0 else hidden_cpu.to(device)
            topk = topk_cpu.to(device) if phase == 2 else None
            assert runner.can_execute(tokens, metadata, hidden, topk)
            with forward_context(ForwardContext(None, device, None, [], cp_context=object())):
                output, auxiliary, indices = runner.execute(tokens, positions, metadata, hidden, topk)
            assert auxiliary is None
            key = runner._graph_key(runner._padded_batch_size(rows, metadata), phase == 0, hidden, topk)
            entry = runner._graphs[key]
            buffers = (
                entry.static_input_ids,
                entry.static_positions,
                entry.static_input_embedding,
                entry.static_mtp_topk_indices,
                entry.static_metadata.slot_mapping,
                entry.static_metadata.kv_seq_lens,
                entry.static_metadata.block_table,
            )
            addresses = tuple(tensor.data_ptr() for tensor in buffers if tensor is not None)
            assert bindings.setdefault(key, addresses) == addresses
            assert output.data_ptr() != entry.static_output[0].data_ptr()
            assert indices.data_ptr() != entry.static_output[2].data_ptr()
            expected_indices = positions_cpu.view(-1, 1, 1) + 1 if phase != 2 else topk_cpu + 1
            slots_cpu = torch.tensor(
                [[2, 5, 7, 9][(length - 1) // 4] * 4 + (length - 1) % 4 for length in lengths],
                dtype=torch.int32,
            )
            values = tokens_cpu + positions_cpu + slots_cpu + torch.tensor(lengths, dtype=torch.int32) + 20
            values = values + expected_indices.sum(dim=(1, 2))
            expected = (torch.zeros_like(hidden_cpu) if phase == 0 else hidden_cpu) + values.view(-1, 1)
            retained.append((output, indices, expected, expected_indices))

    assert len(runner._graphs) == 3
    for output, indices, expected, expected_indices in retained:
        torch.testing.assert_close(output.cpu(), expected)
        torch.testing.assert_close(indices.cpu(), expected_indices)


def _make_eager_runner(*, is_mla: bool = True) -> EagerRunner:
    runner = object.__new__(EagerRunner)
    backend_type = _MlaStubAttentionBackend if is_mla else StubAttentionBackend
    runner.attention_backend = backend_type()
    runner.cp_size = 4
    runner.cp_rank = 2
    runner.device = torch.device("cpu")
    runner.layer_caches = []
    runner.model = MagicMock(return_value=torch.ones(2))
    return runner


def test_eager_runner_preserves_qwen_pure_prefill_cp_context_contract() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=None,
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [3, 5],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_preserves_qwen_missing_length_fallback() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=None,
        kv_seq_lens_host=None,
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_builds_cp_context_for_chunked_prefill() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([11, 13], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [11, 13],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_rejects_mixed_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=True,
        is_spec_verify=False,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="mixed batches"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("prefill_flags", [(False, True), (True, False), (False, False)])
def test_eager_runner_keeps_spec_verify_rows_replicated(cp_size: int, prefill_flags: tuple[bool, bool]) -> None:
    runner = _make_eager_runner()
    runner.cp_size = cp_size
    runner.cp_rank = cp_size - 1
    input_ids = torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32)
    positions = torch.tensor([126, 127, 128, 7, 8, 9], dtype=torch.int32)
    embedding = torch.arange(12, dtype=torch.float32).view(6, 2)
    topk = torch.arange(12, dtype=torch.int32).view(6, 1, 2)
    metadata = SimpleNamespace(
        is_prefill=prefill_flags[0],
        is_chunked_prefill=prefill_flags[1],
        is_mixed=False,
        is_spec_verify=True,
        q_seq_lens_host=torch.tensor([3, 3], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([129, 10], dtype=torch.int32),
        slot_mapping=torch.tensor([254, 255, 384, 519, 520, 521], dtype=torch.int32),
    )

    def execute_model(
        tokens: torch.Tensor,
        token_positions: torch.Tensor,
        hidden: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_forward_context()
        assert context.cp_context is None
        assert context.metadata is metadata
        assert runner.attention_backend._prepared
        torch.testing.assert_close(tokens, input_ids)
        torch.testing.assert_close(token_positions, positions)
        torch.testing.assert_close(indices, topk)
        return hidden + (tokens + token_positions + context.metadata.slot_mapping).view(-1, 1), indices + 1

    runner.model.side_effect = execute_model
    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        side_effect=AssertionError("verification must not create a prefill CP plan"),
    ):
        hidden, indices = runner.execute(input_ids, positions, metadata, embedding, mtp_topk_indices=topk)

    torch.testing.assert_close(hidden, embedding + (input_ids + positions + metadata.slot_mapping).view(-1, 1))
    torch.testing.assert_close(indices, topk + 1)


@pytest.mark.parametrize(
    ("is_mla", "is_chunked_prefill", "is_mixed", "is_spec_verify"),
    [
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_eager_runner_preserves_non_cp_fallback(
    is_mla: bool,
    is_chunked_prefill: bool,
    is_mixed: bool,
    is_spec_verify: bool,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=is_spec_verify,
    )

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().cp_context is None
        return input_ids + positions

    runner.model.side_effect = execute_model
    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        output = runner.execute(torch.ones(1), torch.ones(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()
    torch.testing.assert_close(output, torch.full((1,), 2.0))


@pytest.mark.parametrize(
    ("q_seq_lens_host", "kv_seq_lens_host"),
    [
        (None, None),
        (None, torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([1], dtype=torch.int32), None),
    ],
)
def test_eager_runner_rejects_missing_cp_lengths(
    q_seq_lens_host: torch.Tensor | None,
    kv_seq_lens_host: torch.Tensor | None,
) -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=q_seq_lens_host,
        kv_seq_lens_host=kv_seq_lens_host,
    )

    with pytest.raises(RuntimeError, match="requires host query and KV"):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    assert not runner.attention_backend._prepared


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

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_mtp_topk_state_routes_to_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        executor.inductor_runner = MagicMock()
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(2)
        metadata = MagicMock(spec=AttentionMetadata)
        mtp_topk = torch.zeros((1, 1, 2), dtype=torch.int32)
        input_ids = torch.zeros(1)
        positions = torch.zeros(1)

        result = executor.execute(
            input_ids,
            positions,
            metadata,
            mtp_topk_indices=mtp_topk,
        )

        executor.inductor_runner.execute.assert_not_called()
        args = executor.eager_runner.execute.call_args.args
        assert args == (input_ids, positions, metadata, None, None, mtp_topk)
        assert torch.equal(result, torch.ones(2))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_acl_graph_warmup_uses_scheduler_inputs(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        input_ids = torch.zeros(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = MagicMock(spec=AttentionMetadata)
        graph_runner = MagicMock()
        graph_runner.can_execute.return_value = True
        graph_runner.execute.return_value = torch.ones(4)
        executor.decode_graph_runner = graph_runner

        result = executor.execute(input_ids, positions, metadata)

        graph_runner.warmup.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
        )
        graph_runner.execute.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
            graph_key=graph_runner.warmup.return_value,
        )
        assert torch.equal(result, torch.ones(4))


@patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend())
def test_context_parallel_rejects_data_parallel_combination(_mock_create):
    with pytest.raises(NotImplementedError, match="Python CP requires dp_size == 1"):
        ModelExecutor(
            _FakeModel(num_layers=1),
            {"cp_size": 2, "dp_size": 2, "python_graph_backend": "off"},
            max_seqs_per_batch=4,
        )


@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_mtp_topk_routes_to_acl_graph(mock_create):
    mock_create.return_value = StubAttentionBackend()
    executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
    executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
    graph_runner = create_autospec(DecodeAclGraphRunner, instance=True)
    graph_runner.can_execute.return_value = True
    executor.decode_graph_runner = graph_runner
    executor.eager_runner = MagicMock()
    input_ids = torch.arange(2, dtype=torch.int32)
    metadata = MagicMock(spec=AttentionMetadata)
    topk = torch.ones((2, 1, 8), dtype=torch.int32)

    result = executor.execute(input_ids, input_ids, metadata, mtp_topk_indices=topk)

    graph_runner.can_execute.assert_called_once_with(input_ids, metadata, None, mtp_topk_indices=topk)
    graph_runner.execute.assert_called_once_with(
        input_ids,
        input_ids,
        metadata,
        None,
        mtp_topk_indices=topk,
        graph_key=graph_runner.warmup.return_value,
    )
    assert result is graph_runner.execute.return_value
    executor.eager_runner.execute.assert_not_called()


class _PreparedStubAttentionBackend(StubAttentionBackend):
    @property
    def supports_prepared_metadata(self) -> bool:
        return True

    def prepare_metadata(self, metadata: AttentionMetadata, *, device_kv_lengths: bool = False) -> object:
        return SimpleNamespace(source=metadata.q_cu_seq_lens_host_values, device_kv_lengths=device_kv_lengths)


def test_executor_prepares_private_metadata_after_cache_binding() -> None:
    backend = _PreparedStubAttentionBackend()
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(_FakeModel(), {"model_type": "qwen3", "kv_split_size": 0}, max_seqs_per_batch=2)
    assert executor.supports_prepared_metadata
    metadata = SimpleNamespace(q_cu_seq_lens_host_values=[1, 2])
    with pytest.raises(RuntimeError, match="initialized"):
        executor.prepare_metadata(metadata)
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    executor.prepare_metadata(metadata)
    assert metadata.prepared_attention_state.source == [1, 2]
    assert not backend._prepared


@pytest.mark.parametrize(
    "unsupported",
    [{"cp_size": 2}, {"kv_split_size": 2}, {"model_type": "llama"}],
)
def test_executor_rejects_unsupported_prepared_metadata(unsupported: dict[str, object]) -> None:
    backend = _PreparedStubAttentionBackend()
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(_FakeModel(), {"model_type": "qwen3", **unsupported}, max_seqs_per_batch=2)
    assert not executor.supports_prepared_metadata
    metadata = SimpleNamespace(q_cu_seq_lens_host_values=[1, 2])
    with pytest.raises(RuntimeError, match="supported Qwen3 or GLM"):
        executor.prepare_metadata(metadata)
    assert not hasattr(metadata, "prepared_attention_state")
    assert not backend._prepared


@pytest.mark.parametrize("model_type", ["qwen3", "glm_moe_dsa"])
@pytest.mark.parametrize(
    "tp,rank,ep,moe_tp",
    [(1, 0, 1, 1), (2, 0, 1, 2), (2, 1, 1, 2), (2, 1, 2, 1), (16, 15, 16, 1), (16, 15, 1, 16)],
)
def test_executor_accepts_prepared_tensor_and_expert_parallel(
    model_type: str, tp: int, rank: int, ep: int, moe_tp: int
) -> None:
    backend = _PreparedStubAttentionBackend()
    config = {
        "model_type": model_type,
        "tp_size": tp,
        "tp_rank": rank,
        "ep_size": ep,
        "moe_tp_size": moe_tp,
        "world_size": tp,
        "kv_split_size": 0,
        "dp_size": 1,
        "cp_size": 1,
    }
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert executor.supports_prepared_metadata
    cache = torch.empty(2, 4, 1, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    metadata = SimpleNamespace(q_cu_seq_lens_host_values=[1, 2])
    executor.prepare_metadata(metadata)
    assert metadata.prepared_attention_state.source == [1, 2]


@pytest.mark.parametrize("split", ["kv_split_size", "layerwise_split_size"])
def test_executor_rejects_prepared_glm_split_topologies(split: str) -> None:
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_PreparedStubAttentionBackend()
    ):
        config = {"model_type": "glm_moe_dsa", "kv_split_size": 1, split: 2}
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert not executor.supports_prepared_metadata


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
@pytest.mark.parametrize("graph_backend", ["off", "aclgraph"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_executor_accepts_prepared_glm_prefill_cp(model_type: str, graph_backend: str, cp_size: int) -> None:
    config = {
        "model_type": model_type,
        "is_draft_engine": model_type == "glm_moe_dsa_mtp",
        "num_speculative_tokens": 2,
        "speculative_algorithm": "mtp",
        "kv_split_size": 1,
        "cp_size": cp_size,
        "cp_rank": cp_size - 1,
        "enable_task_pipeline": True,
        "python_graph_backend": graph_backend,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_PreparedStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2, num_decoding_tokens=3)

    assert executor.supports_prepared_metadata
    assert executor.eager_runner.cp_size == cp_size
    assert executor.eager_runner.cp_rank == cp_size - 1
    assert (executor.prepared_graph_runner is not None) == (graph_backend == "aclgraph")
    assert executor.decode_graph_runner is None


@pytest.mark.parametrize(
    "model_type,kv_split,group_size,is_npu",
    [
        ("qwen3", 1, 1, True),
        ("glm_moe_dsa", 1, 2, True),
        ("glm_moe_dsa", 2, 1, True),
        ("glm_moe_dsa", 1, 1, False),
    ],
)
def test_executor_rejects_unsupported_prepared_cp(
    model_type: str, kv_split: int, group_size: int, is_npu: bool
) -> None:
    group = MagicMock()
    group.size.return_value = group_size
    config = {
        "model_type": model_type,
        "cp_size": 2,
        "kv_split_size": kv_split,
        "enable_task_pipeline": True,
        "enable_disagg_pd": True,
        "instance_role": "PREFILL",
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=is_npu),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=group),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_PreparedStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)

    assert not executor.supports_prepared_metadata


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
@pytest.mark.parametrize("graph_backend", ["off", "aclgraph"])
@pytest.mark.parametrize("kv_split", [2, 4])
def test_executor_accepts_prepared_glm_dcp(model_type: str, graph_backend: str, kv_split: int) -> None:
    backend = _PreparedStubAttentionBackend()
    group = MagicMock()
    group.size.return_value = kv_split
    config = {
        "model_type": model_type,
        "kv_split_size": kv_split,
        "dp_size": 1,
        "cp_size": 1,
        "enable_task_pipeline": True,
        "python_graph_backend": graph_backend,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=group),
        patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert executor.supports_prepared_metadata
    assert (executor.prepared_graph_runner is not None) == (graph_backend == "aclgraph")
    assert executor.decode_graph_runner is None


@pytest.mark.parametrize(
    "model_type,kv_split,group_size,is_npu",
    [
        ("qwen3", 2, 2, True),
        ("DFlashDraftModel", 2, 2, True),
        ("DFlash2DraftModel", 2, 2, True),
        ("glm_moe_dsa", 2, None, True),
        ("glm_moe_dsa", 2, 1, True),
        ("glm_moe_dsa", 2, 4, True),
        ("glm_moe_dsa", 1, 2, True),
        ("glm_moe_dsa", 2, 2, False),
    ],
)
def test_executor_rejects_unsupported_prepared_dcp(
    model_type: str, kv_split: int, group_size: int | None, is_npu: bool
) -> None:
    group = None if group_size is None else MagicMock()
    if group is not None:
        group.size.return_value = group_size
    config = {"model_type": model_type, "kv_split_size": kv_split, "enable_task_pipeline": True}
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=is_npu),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=group),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_PreparedStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert not executor.supports_prepared_metadata


def test_executor_rejects_prepared_dcp_with_data_parallelism() -> None:
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        pytest.raises(NotImplementedError, match="Python DCP requires dp_size == 1"),
    ):
        ModelExecutor(
            _FakeModel(),
            {"model_type": "glm_moe_dsa", "kv_split_size": 2, "dp_size": 2, "enable_task_pipeline": True},
            max_seqs_per_batch=2,
        )


def test_prepared_executor_rejects_mtp_state_before_model_execution() -> None:
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_PreparedStubAttentionBackend()
    ):
        executor = ModelExecutor(_FakeModel(), {"model_type": "glm_moe_dsa"}, max_seqs_per_batch=2)
    cache = torch.empty(2, 4, 1, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    metadata = SimpleNamespace(
        q_cu_seq_lens_host_values=[1, 2],
        is_prefill=False,
        is_chunked_prefill=False,
    )
    executor.prepare_metadata(metadata)
    executor.eager_runner = MagicMock()
    executor.decode_graph_runner = MagicMock()
    with pytest.raises(ValueError, match="MTP top-k"):
        executor.execute(torch.zeros(2), torch.zeros(2), metadata, mtp_topk_indices=torch.zeros(2))
    executor.eager_runner.execute.assert_not_called()


@pytest.mark.parametrize("model_type", ["qwen3", "glm_moe_dsa"])
def test_executor_accepts_prepared_data_parallel(model_type: str) -> None:
    config = {"model_type": model_type, "dp_size": 2, "dp_rank": 1, "graph_backend": "off"}
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_PreparedStubAttentionBackend()
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert executor.supports_prepared_metadata


@pytest.mark.parametrize("dp_size,cp_size", [(1, 1), (2, 1), (1, 2), (1, 4)])
def test_prepared_mtp_preserves_hidden_and_topk(dp_size: int, cp_size: int) -> None:
    backend = _PreparedStubAttentionBackend()
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "is_draft_engine": True,
        "dp_size": dp_size,
        "cp_size": cp_size,
        "kv_split_size": 1,
        "enable_task_pipeline": True,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
        patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    assert executor.supports_prepared_metadata
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    metadata = SimpleNamespace(q_cu_seq_lens_host_values=[1, 2])
    executor.prepare_metadata(metadata)
    tokens = torch.tensor([13, 17], dtype=torch.int32)
    positions = torch.tensor([7, 19], dtype=torch.int32)
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
    topk = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    executor.eager_runner = MagicMock()
    result = executor.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk)
    executor.eager_runner.execute.assert_called_once_with(tokens, positions, metadata, hidden, None, topk)
    assert result is executor.eager_runner.execute.return_value
    with pytest.raises(RuntimeError, match="runner is not enabled"):
        executor.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk, enable_graph=True)


@pytest.mark.parametrize("model_type", ["DFlashDraftModel", "DFlash2DraftModel"])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_prepared_block_draft_uses_device_lengths(model_type: str, dp_size: int) -> None:
    backend = _PreparedStubAttentionBackend()
    config = {"model_type": model_type, "dp_size": dp_size, "enable_task_pipeline": True}
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=4)
    assert executor.supports_prepared_metadata
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    first = SimpleNamespace(q_cu_seq_lens_host_values=[4], dp_execution_token_counts=(4, 8))
    second = SimpleNamespace(q_cu_seq_lens_host_values=[4, 8], dp_execution_token_counts=(8, 4))
    executor.prepare_metadata(first)
    executor.prepare_metadata(second)
    assert first.prepared_attention_state.device_kv_lengths
    assert first.prepared_attention_state.source == [4]
    assert first.dp_execution_token_counts == (4, 8)
    assert second.prepared_attention_state.source == [4, 8]
    assert not backend._prepared


@pytest.mark.parametrize("dp_size,cp_size", [(1, 1), (2, 1), (1, 2), (1, 4)])
def test_prepared_mtp_graph_receives_bound_hidden_and_topk(dp_size: int, cp_size: int) -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "is_draft_engine": True,
        "dp_size": dp_size,
        "dp_rank": 0,
        "cp_size": cp_size,
        "kv_split_size": 1,
        "enable_task_pipeline": True,
        "python_graph_backend": "aclgraph",
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_PreparedStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=5, acl_graph_decode_batch_size_limit=4)
    assert executor.supports_prepared_metadata
    assert executor.prepared_graph_runner.max_batch == min((5 + dp_size - 1) // dp_size, 4) * 2
    assert executor.decode_graph_runner is None
    executor._kv_bound = True
    executor.prepared_graph_runner = MagicMock()
    tokens = torch.tensor([13, 17], dtype=torch.int32)
    positions = torch.tensor([7, 19], dtype=torch.int32)
    hidden = torch.zeros(2, 4)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False, prepared_attention_state=object())
    executor.warmup_prepared_graph(tokens, positions, metadata, hidden, topk)
    executor.prepared_graph_runner.warmup_prepared.assert_called_once_with(tokens, positions, metadata, hidden, topk)
    executor.execute(tokens, positions, metadata, hidden, mtp_topk_indices=topk, enable_graph=True)
    executor.prepared_graph_runner.execute.assert_called_once_with(tokens, positions, metadata, hidden, None, topk)


@pytest.mark.parametrize("dp_size,cp_size", [(1, 1), (2, 1), (1, 2), (1, 4)])
def test_speculative_target_graph_capacity_expands_validation_tokens(dp_size: int, cp_size: int) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "dp_size": dp_size,
        "dp_rank": 0,
        "cp_size": cp_size,
        "kv_split_size": 1,
        "enable_task_pipeline": True,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 128,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_PreparedStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(
            _FakeModel(), config, max_seqs_per_batch=5, num_decoding_tokens=4, acl_graph_decode_batch_size_limit=4
        )
    assert executor.prepared_graph_runner.max_batch == min((5 + dp_size - 1) // dp_size, 4) * 4
    assert executor.decode_graph_runner is None


@pytest.mark.parametrize("model_type", ["qwen3", "glm_moe_dsa"])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("pipeline", [False, True])
def test_executor_selects_acl_graph_input_owner(model_type: str, dp_size: int, pipeline: bool) -> None:
    config = {
        "model_type": model_type,
        "dp_size": dp_size,
        "dp_rank": dp_size - 1,
        "python_graph_backend": "aclgraph",
        "enable_task_pipeline": pipeline,
        "max_position_embeddings": 128,
    }
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_PreparedStubAttentionBackend()
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=4 * dp_size)
    assert executor.supports_prepared_metadata
    if pipeline:
        assert executor.decode_graph_runner is None
        assert executor.prepared_graph_runner.dp_size == dp_size
        assert executor.prepared_graph_runner.max_batch == 4
    else:
        assert executor.prepared_graph_runner is None
        assert executor.decode_graph_runner.dp_size == dp_size
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    runner = executor.prepared_graph_runner if pipeline else executor.decode_graph_runner
    assert len(runner.layer_caches) == 2

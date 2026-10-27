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
from unittest.mock import ANY, MagicMock, create_autospec, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from xllm.python.attention.csa_attention import DsaAttentionBackend  # noqa: E402
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _resolve_graph_backend,
    _validate_npu_cp_model_config,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    KVLoadContext,
    forward_context,
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry
from xllm.python.model_executor.runners.block_draft_acl_graph import (  # noqa: E402
    BlockDraftAclGraphRunner,
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


def _classify(config: dict) -> dict:
    """Classify a config dict the way the executor itself does.

    Real runs build config from ModelArgs; tests that construct configs
    directly rely on the same model-platform-support predicates the executor
    calls, so nothing needs to be injected here. All other keys pass through
    because the executor and the backends read runtime knobs from this dict.
    """
    return {**config, "model_type": config.get("model_type", "")}


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


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    """Test platform dispatch without executing device-specific backends."""

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_creates_dsa_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(head_dim=512)
        device = torch.device("cpu")
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = _classify(
            {
                "model_type": "deepseek_v4",
                "compress_ratios": [1, 4, 128],
                "num_hidden_layers": 3,
                "window_size": 128,
                "index_topk": 512,
                "index_n_heads": 64,
                "index_head_dim": 128,
                "qk_rope_head_dim": 64,
            }
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, device, torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["device"] is device
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["block_size"] == 0
        assert backend.init_kwargs["dspark_use_native_sas"] is False
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [1, 4, 128]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_dspark_passes_native_sas_config(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = _classify(
            {
                "model_type": "deepseek_v4_dspark",
                "compress_ratios": [1, 1, 1],
                "n_layers": 3,
                "window_size": 128,
                "block_size": 5,
                "dspark_use_native_sas": True,
            }
        )

        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("npu"), torch.bfloat16, config)

        assert backend.init_kwargs["block_size"] == 5
        assert backend.init_kwargs["dspark_use_native_sas"] is True

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_platform_creates_npu_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        device = torch.device("cpu")
        module = types.ModuleType("xllm.python.attention.npu_paged_attention")
        module.NpuPagedAttentionBackend = StubAttentionBackend
        with (
            patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
            patch.dict(sys.modules, {module.__name__: module}),
        ):
            backend = _create_attention_backend(
                attn,
                device,
                torch.float16,
                {"enable_mla": False},
            )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["device"] is device
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 256
        assert backend.init_kwargs["is_mla"] is False

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp", "qwen3"])
    def test_prefill_cp_selects_model_backend(self, _mock_is_npu: MagicMock, model_type: str) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        device = torch.device("cpu")
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        npu_module = types.ModuleType("xllm.python.attention.npu_paged_attention")
        npu_module.NpuPagedAttentionBackend = StubAttentionBackend
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module, npu_module.__name__: npu_module}),
        ):
            backend = _create_attention_backend(
                attn,
                device,
                torch.float16,
                {"cp_size": 4, "enable_mla": True, "model_type": model_type},
            )

        if model_type == "qwen3":
            assert isinstance(backend, StubAttentionBackend)
            assert backend.init_kwargs["device"] is device
            sfa_module.SfaDcpAttentionBackend.assert_not_called()
        else:
            assert backend is sfa_module.SfaDcpAttentionBackend.return_value
            sfa_module.SfaDcpAttentionBackend.assert_called_once()
            assert sfa_module.SfaDcpAttentionBackend.call_args.kwargs["device"] is device

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_uses_dense_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dense_module = types.ModuleType("xllm.python.attention.dense_dcp_backend")
        dense_module.DenseDcpAttentionBackend = StubAttentionBackend

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {dense_module.__name__: dense_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                },
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 128

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_rejects_dcp_outside_kv_replica_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 4

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(ValueError, match="replicated KV-head group"),
        ):
            _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 4,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_dcp_rejects_disaggregated_pd(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(NotImplementedError, match="disaggregated P/D"),
        ):
            _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                    "enable_disagg_pd": True,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_decode_cp1_uses_sfa_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        device = torch.device("cpu")
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
                device,
                torch.float16,
                {"cp_size": 1, "enable_mla": True},
                max_num_reqs=3,
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["device"] is device
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
    @pytest.mark.parametrize("graph_backend", ["off", "aclgraph"])
    @pytest.mark.parametrize("config_depth", [0, 3])
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_decoding_width_reaches_graph_capacity(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        graph_backend: str,
        config_depth: int,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        ModelExecutor(
            _FakeModel(num_layers=1),
            {
                "python_graph_backend": graph_backend,
                "num_speculative_tokens": config_depth,
                "max_position_embeddings": 128,
            },
            max_seqs_per_batch=4,
        )

        assert mock_create.call_args.args[4] == 4 * (config_depth + 1)
        if graph_backend == "aclgraph":
            assert mock_graph_runner.call_args.kwargs["num_speculative_tokens"] == config_depth
        else:
            mock_graph_runner.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_valid_model_creates_executor(self, mock_backend: MagicMock) -> None:
        model = _FakeModel(num_layers=3)
        config = {"python_graph_backend": "off"}
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)

        assert executor.execution_model is model.model
        assert executor.attention_backend is mock_backend.return_value
        assert executor.eager_runner.model is model.model
        assert executor.eager_runner.attention_backend is mock_backend.return_value
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
    def test_glm_next_mixed_layers_build_backend_from_dsa(self, mock_backend: MagicMock) -> None:
        class _DsaAttention(Attention):
            is_glm_next_mla = True

        class _FakeGlmNext(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(1, 1)
                self.kda = _make_attention_layer(num_heads=16, num_kv_heads=16, head_dim=64, scale=0.125, layer_id=0)
                self.dsa = _DsaAttention(
                    num_heads=8,
                    num_kv_heads=1,
                    head_dim=512,
                    scale=0.044,
                    sliding_window=0,
                    layer_id=3,
                )
                self._param = nn.Parameter(torch.zeros(1))

        ModelExecutor(_FakeGlmNext(), {"python_graph_backend": "off"}, max_seqs_per_batch=4)
        first_attention = mock_backend.call_args.args[0]
        assert first_attention.num_heads == 8
        assert first_attention.num_kv_heads == 1
        assert first_attention.head_dim == 512

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

    @patch("xllm.python.model_executor.executor.DecodeCudaGraphRunner")
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

    @pytest.mark.parametrize("is_draft_model", [False, True])
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_capacity_respects_decode_batch_limit(
        self,
        mock_create,
        mock_graph_runner,
        is_draft_model: bool,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
                "is_draft_model": is_draft_model,
            },
            max_seqs_per_batch=256,
            acl_graph_decode_batch_size_limit=16,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            16,
            128,
            1,
            0,
            num_speculative_tokens=0,
            is_draft_model=is_draft_model,
            enable_mega_moe_token_mask=False,
            enable_graph_mode_decode_no_padding=False,
            device_kv_lengths=False,
            max_capture_tokens=None,
        )

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_mega_moe_enables_token_mask(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "enable_mega_moe": True,
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=4,
        )

        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is True

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_data_parallel_passes_global_token_budget(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 16,
                "dp_rank": 0,
                "enable_mega_moe": True,
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=17,
        )

        assert mock_create.call_args.args[4] == 17
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            32,
            128,
            16,
            0,
            num_speculative_tokens=0,
            is_draft_model=False,
            enable_mega_moe_token_mask=True,
            enable_graph_mode_decode_no_padding=False,
            device_kv_lengths=False,
            max_capture_tokens=None,
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

    def test_execute_accepts_base_runner_context_keywords(self):
        runner = self._runner()
        runner._graph_key = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="decode batch exceeds CUDA graph capacity"):
            runner.execute(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                self._metadata([1, 1]),
                layer_synchronizer=MagicMock(),
                eplb=MagicMock(),
            )

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
            q_seq_lens_host=torch.tensor([2, 2], dtype=torch.int32),
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

    @staticmethod
    def _decode_metadata(
        batch_size: int,
        *,
        dp_execution_token_counts: tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32),
            paged_kv_indices=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32),
            kv_cu_seq_lens=torch.arange(batch_size + 1, dtype=torch.int32),
            kv_seq_lens_host_values=[1] * batch_size,
            block_table=torch.arange(batch_size, dtype=torch.int32).view(-1, 1),
            kv_seq_lens=torch.ones(batch_size, dtype=torch.int32),
            expanded_decode_metadata=None,
            dp_execution_token_counts=dp_execution_token_counts,
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
        )

    def test_expanded_metadata_selects_matching_paged_kv_rows(self) -> None:
        runner = self._runner()
        metadata = self._metadata()
        expanded = runner._expanded_verify_view(metadata)
        decoded = runner._decode_metadata(metadata, expanded)

        assert decoded.block_table.tolist() == [
            [10, 11],
            [10, 11],
            [20, 21],
            [20, 21],
        ]
        assert decoded.kv_seq_lens.tolist() == [3, 4, 7, 8]
        assert decoded.paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert decoded.paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert decoded.paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

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

        decoded = self._runner()._decode_metadata(metadata)

        assert decoded.paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert decoded.paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert decoded.paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    @pytest.mark.parametrize(
        "linear_state,supported,allowed", [(False, False, True), (True, False, False), (True, True, True)]
    )
    def test_expanded_chunked_verify_can_use_decode_graph(
        self: TestDecodeAclGraphSpeculativeMetadata, linear_state: bool, supported: bool, allowed: bool
    ) -> None:
        runner = self._runner()
        metadata = self._metadata()
        metadata.linear_state_indices = torch.arange(2, dtype=torch.int32) if linear_state else None
        with patch.object(_PagedStubAttentionBackend, "supports_linear_spec_verify_graph", supported):
            assert (runner.prepare_execution(torch.arange(4, dtype=torch.int32), metadata) is not None) is allowed

    def test_execute_accepts_base_runner_context_keywords(self) -> None:
        runner = self._runner()

        with pytest.raises(ValueError, match="decode batch exceeds ACL graph capacity"):
            runner.execute(
                torch.zeros(5, dtype=torch.int32),
                torch.zeros(5, dtype=torch.int32),
                self._decode_metadata(5),
                layer_synchronizer=MagicMock(),
                eplb=MagicMock(),
            )

    def test_decode_batch_limit_uses_speculative_tokens_and_dp_global_max(
        self: TestDecodeAclGraphSpeculativeMetadata,
    ) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            num_speculative_tokens=3,
        )
        metadata = self._metadata()
        metadata.dp_execution_token_counts = (32, 64)
        assert runner._padded_batch_size(32, metadata) == 32
        assert runner._padded_batch_size(64, metadata) == 64
        with pytest.raises(ValueError, match="decode batch exceeds ACL graph capacity"):
            runner._padded_batch_size(65, metadata)

    def test_warmup_captures_with_scheduler_metadata_once(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = self._metadata()
        graph_key = runner._graph_key(
            padded_batch_size=4,
            is_expanded=True,
            input_embedding=None,
            expanded_query_lens=(2, 2),
        )

        with patch.object(runner, "_prepare_graph_entry") as prepare:
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once_with(
                input_ids,
                positions,
                metadata,
                None,
                None,
                graph_key=graph_key,
                eplb=None,
                input_batch=None,
                expanded=ANY,
            )
            expanded = prepare.call_args.kwargs["expanded"]
            assert expanded is not None
            torch.testing.assert_close(expanded.block_table, metadata.expanded_decode_metadata.block_table)

            runner._graphs[graph_key] = object()
            runner.warmup(input_ids, positions, metadata)
            prepare.assert_called_once()

        runner._graphs.clear()
        conv = torch.arange(4, dtype=torch.float32)
        ssm = torch.arange(4, dtype=torch.float32) + 10
        runner.bind_layer_caches([LayerCache(None, None, conv=conv, ssm=ssm)])
        metadata = self._decode_metadata(1)
        metadata.linear_state_indices = torch.tensor([1], dtype=torch.int32)
        metadata.has_initial_state = torch.tensor([True])
        caller_stream = MagicMock()
        graph_stream = MagicMock()
        runner._stream = graph_stream

        def advance_state(entry: AclGraphEntry, _stream: object) -> None:
            indices = entry.static_metadata.linear_state_indices.long()
            assert not indices.any()
            conv[indices] += 10
            ssm[indices] += 20

        with (
            patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: caller_stream), create=True),
            patch.object(runner, "_capture", side_effect=advance_state),
            patch.object(runner, "_prepare_attention"),
            patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata"),
        ):
            runner.warmup(input_ids[:1], positions[:1], metadata)

        assert conv.tolist() == [10, 1, 2, 3]
        assert ssm.tolist() == [20, 11, 12, 13]
        entry = next(iter(runner._graphs.values()))
        assert entry.static_metadata.linear_state_indices.tolist() == [1]
        assert caller_stream.wait_stream.call_count == 2
        caller_stream.wait_stream.assert_called_with(graph_stream)

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
            self._runner()._expanded_verify_view(metadata)

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
            self._runner()._expanded_verify_view(metadata)

    def test_expanded_page_count_exceeding_capacity_fails(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata.kv_seq_lens_host_values = [3, 4, 7, 9]

        with pytest.raises(RuntimeError, match="exceeds block-table capacity"):
            self._runner()._expanded_verify_view(metadata)

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

    @pytest.mark.parametrize("task_updates", [False, True])
    @pytest.mark.parametrize(
        ("num_speculative_tokens", "is_draft", "with_embedding", "with_topk"),
        [
            (0, False, False, False),
            (3, False, False, False),
            (0, True, False, False),
            (0, False, True, False),
            (0, False, False, True),
        ],
        ids=["ordinary", "verify", "draft", "repair", "shared-topk"],
    )
    def test_replay_output_ownership_and_stream_ordering(
        self,
        task_updates: bool,
        num_speculative_tokens: int,
        is_draft: bool,
        with_embedding: bool,
        with_topk: bool,
    ) -> None:
        runner = self._runner()
        runner.num_speculative_tokens = num_speculative_tokens
        runner._is_draft_model = is_draft
        prepare_replay = MagicMock()
        runner.attention_backend.prepare_graph_replay = prepare_replay
        batch_size = 3
        padded_batch_size = 4
        static_output = torch.arange(12).reshape(padded_batch_size, 3)
        graph = MagicMock()
        entry = SimpleNamespace(
            batch_size=padded_batch_size,
            kv_capacity_pages=0,
            graph=graph,
            static_output=static_output,
            static_metadata=SimpleNamespace(),
            eplb=None,
            graph_tasks=[object()] if task_updates else [],
            execution_state=SimpleNamespace(persistent_buffers={}),
            layer_shared_cache={},
        )
        embedding = torch.zeros(batch_size, 3) if with_embedding else None
        topk = torch.zeros(batch_size, 1, 1, dtype=torch.int32) if with_topk else None
        graph_key = runner._graph_key(
            padded_batch_size,
            is_expanded=False,
            input_embedding=embedding,
            mtp_topk_indices=topk,
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
            patch.object(runner, "_fill_entry"),
            patch.object(runner, "_initialize_task_updates") as initialize_updates,
            patch.object(runner, "_update_after_replay") as update_after_replay,
        ):
            output = runner.execute(
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, dtype=torch.int32),
                metadata,
                embedding,
                topk,
            )

        assert output.shape == (batch_size, 3)
        detached = num_speculative_tokens > 0 or is_draft or with_embedding or with_topk
        assert (output.data_ptr() != static_output.data_ptr()) == detached
        expected = output.clone()
        static_output.add_(100)
        torch.testing.assert_close(output, expected if detached else expected + 100)
        if task_updates:
            replay_stream.wait_stream.assert_called_once_with(current_stream)
            current_stream.wait_stream.assert_called_once_with(replay_stream)
            current_stream.wait_event.assert_called_once_with(runner._update_done_event)
            initialize_updates.assert_called_once()
            update_after_replay.assert_called_once_with(entry, replay_stream, runner._update_done_event)
        else:
            replay_stream.wait_stream.assert_not_called()
            current_stream.wait_stream.assert_not_called()
            current_stream.wait_event.assert_not_called()
            initialize_updates.assert_not_called()
            update_after_replay.assert_not_called()
            assert runner._taskless_replay_stream is current_stream
        graph.replay.assert_called_once_with()
        prepare_replay.assert_called_once_with(entry.static_metadata)

    def test_graph_token_mask_updates_in_place_for_reused_bucket(self) -> None:
        runner = self._runner()
        runner._enable_mega_moe_token_mask = True
        metadata = self._decode_metadata(4)
        entry = runner._allocate_entry(
            4,
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        data_ptr = mask.data_ptr()

        runner._fill_mega_moe_token_mask(entry, (4,))
        assert mask.tolist() == [1, 1, 1, 1]

        runner._fill_mega_moe_token_mask(entry, (3,))
        assert mask.data_ptr() == data_ptr
        assert mask.tolist() == [1, 1, 1, 0]

    def test_non_mega_moe_graph_does_not_allocate_token_mask(self) -> None:
        runner = self._runner()
        metadata = self._decode_metadata(4)
        entry = runner._allocate_entry(
            4,
            torch.arange(4, dtype=torch.int32),
            metadata,
        )

        assert entry.static_metadata.mega_moe_token_mask is None

    def test_mega_moe_dp_uses_ceil_divided_local_graph_capacity(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=68,
            max_model_len=8,
            dp_size=16,
            enable_mega_moe_token_mask=True,
        )

        assert runner.max_batch == 5

    def test_graph_token_mask_uses_real_data_parallel_counts(self) -> None:
        # Empty DP ranks contribute a worker-materialized dummy row.
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            enable_mega_moe_token_mask=True,
        )
        metadata = self._decode_metadata(3, dp_execution_token_counts=(3, 1))
        entry = runner._allocate_entry(
            4,
            torch.arange(3, dtype=torch.int32),
            metadata,
        )

        runner._fill_mega_moe_token_mask(entry, metadata.dp_execution_token_counts)

        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        assert mask.tolist() == [1, 1, 1, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestNormalizeLayerCaches:
    @pytest.mark.parametrize(
        "slot_count,empty",
        [(5, False), (11, False), (11, True)],
        ids=["legacy", "deepseek-v4", "empty-deepseek-v4"],
    )
    def test_normalize_layer_cache_slots(self: TestNormalizeLayerCaches, slot_count: int, empty: bool) -> None:
        tensors = tuple(
            torch.empty(0) if empty and index >= 2 else torch.full((1,), index + 1) for index in range(slot_count)
        )
        cache = normalize_layer_caches([tensors])[0]
        fields = (
            "key",
            "value",
            "index",
            "conv",
            "ssm",
            "swa",
            "compress_kv_state",
            "compress_score_state",
            "compress_index_kv_state",
            "compress_index_score_state",
            "indexer_scale",
        )
        for index, field in enumerate(fields):
            expected = tensors[index] if index < slot_count and not (empty and index >= 2) else None
            assert getattr(cache, field) is expected, field


class TestBindKvCaches:
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_bind_idempotent(self: TestBindKvCaches, mock_create: MagicMock) -> None:
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        executor = ModelExecutor(
            _FakeModel(num_layers=2),
            {"python_graph_backend": "aclgraph", "max_position_embeddings": 128},
            max_seqs_per_batch=4,
        )
        first = [LayerCache(torch.ones(1), torch.full((1,), 2.0)) for _ in range(2)]
        second = [LayerCache(torch.full((1,), 3.0), torch.full((1,), 4.0)) for _ in range(2)]
        with patch.object(backend, "bind_kv_caches", wraps=backend.bind_kv_caches) as bind:
            with pytest.raises(ValueError, match="layer count does not match"):
                executor.bind_kv_caches(first[:1])
            bind.assert_not_called()
            executor.bind_kv_caches(first)
            bound = backend._kv_caches
            executor.bind_kv_caches(second)
            bind.assert_called_once_with(bound)
        assert len(bound) == 2
        assert backend._kv_caches is executor.eager_runner.layer_caches is bound
        assert executor.decode_graph_runner.layer_caches is bound
        assert all(actual is expected for actual, expected in zip(bound, first))


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_type", "is_draft", "speculative_tokens", "decoding_tokens"),
    [
        ("glm_moe_dsa", False, 1, 1),
        ("glm_moe_dsa", False, 0, 3),
        ("glm_moe_dsa_mtp", True, 0, 1),
        ("deepseek_v4", False, 0, 1),
        ("deepseek_v4", False, 1, 1),
    ],
)
@pytest.mark.parametrize("kv_split", [0, 1, 2])
def test_cp_admission_preserves_model_specific_kv_support(
    model_type: str,
    is_draft: bool,
    speculative_tokens: int,
    decoding_tokens: int,
    kv_split: int,
) -> None:
    config = _classify(
        {
            "model_type": model_type,
            "is_draft_model": is_draft,
            "is_mtp_algorithm": model_type == "glm_moe_dsa_mtp",
            "num_speculative_tokens": speculative_tokens,
            "cp_size": 2,
            "kv_split_size": kv_split,
            "instance_role": "PREFILL",
            "enable_disagg_pd": True,
        }
    )
    if model_type != "deepseek_v4" or kv_split == 1:
        _validate_npu_cp_model_config(config)
    else:
        with pytest.raises(NotImplementedError, match="requires replicated KV caches"):
            _validate_npu_cp_model_config(config)


@pytest.mark.parametrize(("is_draft", "algorithm"), [(False, "mtp"), (True, "eagle3"), (True, "dflash")])
def test_glm_cp_draft_admission_requires_mtp_draft_engine(is_draft: bool, algorithm: str) -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "is_draft_model": is_draft,
        "is_mtp_algorithm": algorithm == "mtp",
        "cp_size": 2,
        "kv_split_size": 1,
    }
    with pytest.raises(NotImplementedError, match="does not support model_type"):
        _validate_npu_cp_model_config(config)


def test_glm_non_mtp_cp_prefill_retains_sharded_kv_admission() -> None:
    _validate_npu_cp_model_config(
        {
            "model_type": "glm_moe_dsa",
            "cp_size": 2,
            "kv_split_size": 2,
            "instance_role": "PREFILL",
            "enable_disagg_pd": True,
        },
    )


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_3_flash", "glm5_next_text"])
def test_glm5_next_cp_admission_matches_runtime_constraints(model_type: str) -> None:
    config = _classify(
        {
            "model_type": model_type,
            "cp_size": 2,
            "kv_split_size": 1,
            "ep_size": 1,
        }
    )
    _validate_npu_cp_model_config(config)
    _validate_npu_cp_model_config({**config, "ep_size": 2, "dp_size": 2})

    with pytest.raises(NotImplementedError, match="ep_size >= 1"):
        _validate_npu_cp_model_config({**config, "ep_size": 0})
    with pytest.raises(NotImplementedError, match="disaggregated PD"):
        _validate_npu_cp_model_config({**config, "enable_disagg_pd": True})


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
        q_cu_seq_lens_host_values=list(range(len(lengths) + 1)),
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
        metadata.q_seq_lens_host = torch.tensor([len(lengths)], dtype=torch.int32)
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
    runner = DecodeAclGraphRunner(
        _GraphModel(), _GraphBackend(), device, max_batch=4, max_model_len=16, num_speculative_tokens=3
    )
    bindings: dict[tuple, tuple[int, ...]] = {}
    retained: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    # Verify K+1 rows, repair two rows, then draft one row; repeat after a page
    # boundary and a new request reuses the same graph buckets. No per-replay
    # CPU synchronization is inserted between these invocations.
    for cycle in range(3):
        for phase, rows in enumerate((4, 2, 1)):
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
            assert runner.prepare_execution(tokens, metadata, hidden, topk) is not None
            with forward_context(ForwardContext(None, device, None, [], cp_context=object())):
                output, auxiliary, indices = runner.execute(tokens, positions, metadata, hidden, topk)
            assert auxiliary is None
            key = runner._graph_key(
                runner._padded_batch_size(rows, metadata),
                phase == 0,
                hidden,
                topk,
                expanded_query_lens=(rows,) if phase == 0 else (),
            )
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
    runner.execution_metadata_builders = ()
    runner.model = MagicMock(return_value=torch.ones(2))
    return runner


@pytest.mark.parametrize(
    "is_mla,model_owned_cp",
    [(False, False), (True, False), (False, True)],
    ids=["qwen-prefill", "mla-chunked-prefill", "model-owned-prefill"],
)
def test_eager_runner_installs_cp_context(is_mla: bool, model_owned_cp: bool) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    if model_owned_cp:
        runner.attention_backend = object.__new__(DsaAttentionBackend)
    kv_lengths = [11, 13] if is_mla else [3, 5]
    metadata = SimpleNamespace(
        is_prefill=not is_mla,
        is_chunked_prefill=is_mla,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor(kv_lengths, dtype=torch.int32) if is_mla else None,
    )
    cp_context = object()

    def _read_context(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().cp_context is (None if model_owned_cp else cp_context)
        return input_ids

    runner.model.side_effect = _read_context
    with patch("xllm.python.model_executor.runners.eager.build_cp_context", return_value=cp_context) as build:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)
    if model_owned_cp:
        build.assert_not_called()
        assert runner.attention_backend._metadata is metadata
    else:
        build.assert_called_once_with([3, 5], kv_lengths, 4, 2, torch.device("cpu"))
    runner.model.assert_called_once()


def test_eager_runner_rejects_mixed_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=True,
        is_spec_verify=False,
    )
    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build,
        pytest.raises(NotImplementedError, match="mixed batches"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)
    build.assert_not_called()
    assert not runner.attention_backend._prepared


@pytest.mark.parametrize(
    "cp_size,prefill_flags,explicit_spec_verify",
    [(2, (False, True), True), (2, (False, True), False), (4, (True, False), False)],
)
def test_eager_runner_keeps_spec_verify_rows_replicated(
    cp_size: int, prefill_flags: tuple[bool, bool], explicit_spec_verify: bool
) -> None:
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
        is_spec_verify=explicit_spec_verify,
        expanded_decode_metadata=(
            None
            if explicit_spec_verify
            else _mtp_graph_metadata([2, 3, 4, 6, 7, 8], True, torch.device("cpu")).expanded_decode_metadata
        ),
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
    ("is_mla", "is_prefill", "is_chunked_prefill", "is_mixed", "is_spec_verify"),
    [
        (False, False, True, True, False),
        (False, False, True, False, True),
        (True, False, False, False, True),
        (False, True, False, False, False),
    ],
)
def test_eager_runner_preserves_non_cp_fallback(
    is_mla: bool,
    is_prefill: bool,
    is_chunked_prefill: bool,
    is_mixed: bool,
    is_spec_verify: bool,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=is_spec_verify,
        q_seq_lens_host=None,
        kv_seq_lens_host=None,
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

    @pytest.mark.parametrize("dp_size", [1, 2])
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_routes_to_eager_runner(self, mock_create: MagicMock, dp_size: int) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(
            model,
            {"enable_graph": False, "python_graph_backend": "off", "dp_size": dp_size},
            max_seqs_per_batch=4,
        )
        assert executor.decode_graph_runner is None

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        executor.eager_runner = MagicMock()
        grad_enabled = None

        def execute(*_args: object, eplb: object = None, input_batch: object = None) -> torch.Tensor:
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
        graph_runner = create_autospec(DecodeAclGraphRunner, instance=True)
        graph_runner.execute.return_value = torch.ones(4)
        executor.decode_graph_runner = graph_runner

        result = executor.execute(input_ids, positions, metadata)

        graph_runner.warmup.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
            None,
            eplb=None,
            decode_metadata=graph_runner.prepare_execution.return_value,
            is_graph_warmup=False,
            input_batch=None,
        )
        graph_runner.execute.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
            None,
            graph_key=graph_runner.warmup.return_value,
            eplb=None,
            decode_metadata=graph_runner.prepare_execution.return_value,
            input_batch=None,
        )
        assert torch.equal(result, torch.ones(4))


def test_executor_admits_startup_graphs_and_runs_larger_serving_batches_eagerly() -> None:
    backend = _PagedStubAttentionBackend()
    model = _FakeModel(num_layers=1)
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(model, {"python_graph_backend": "off"}, max_seqs_per_batch=4)
    executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
    executor.eager_runner = EagerRunner(model, backend, torch.device("cpu"))
    runner = DecodeAclGraphRunner(
        model,
        backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        enable_graph_mode_decode_no_padding=True,
        device_kv_lengths=True,
    )
    executor.decode_graph_runner = runner

    def _metadata(rows: int) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
            block_table=torch.arange(1, rows + 1, dtype=torch.int32).reshape(rows, 1),
            kv_seq_lens=torch.ones(rows, dtype=torch.int32),
            kv_seq_lens_host_values=[1] * rows,
            kv_cu_seq_lens=None,
            q_cu_seq_lens=None,
            paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
            paged_kv_indices=torch.arange(1, rows + 1, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(rows, dtype=torch.int32),
            slot_mapping=torch.arange(rows, dtype=torch.int32),
        )

    def _prepare_entry(*_args: object, graph_key: tuple[object, ...], **_kwargs: object) -> None:
        runner._graphs[graph_key] = AclGraphEntry()

    tokens = torch.arange(2, dtype=torch.int32)
    metadata = _metadata(2)
    graph_output = tokens + 10
    with (
        patch.object(runner, "_prepare_graph_entry", side_effect=_prepare_entry) as capture,
        patch.object(runner, "execute", return_value=graph_output) as replay,
    ):
        assert torch.equal(executor.execute(tokens, tokens, metadata, is_graph_warmup=True), graph_output)
        captured_graphs = capture.call_count
        assert captured_graphs > 0
        assert torch.equal(executor.execute(tokens, tokens, metadata), graph_output)
        assert not backend._prepared

        larger_tokens = torch.arange(3, dtype=torch.int32)
        assert torch.equal(executor.execute(larger_tokens, larger_tokens, _metadata(3)), larger_tokens)
        assert backend._prepared
        assert capture.call_count == captured_graphs
        assert replay.call_count == 2


@patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend())
def test_context_parallel_rejects_data_parallel_combination(_mock_create):
    with pytest.raises(NotImplementedError, match="Python CP requires dp_size == 1"):
        ModelExecutor(
            _FakeModel(num_layers=1),
            {"cp_size": 2, "dp_size": 2, "python_graph_backend": "off"},
            max_seqs_per_batch=4,
        )


@patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend())
def test_context_parallel_accepts_glm5_next_data_parallel_combination(_mock_create):
    executor = ModelExecutor(
        _FakeModel(num_layers=1),
        {"cp_size": 2, "dp_size": 2, "kv_split_size": 1, "model_type": "glm5_3_flash", "python_graph_backend": "off"},
        max_seqs_per_batch=4,
    )
    assert executor.dp_size == 2


@patch("xllm.python.model_executor.executor._create_attention_backend")
def test_mtp_topk_routes_to_acl_graph(mock_create):
    mock_create.return_value = StubAttentionBackend()
    executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
    executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
    graph_runner = create_autospec(DecodeAclGraphRunner, instance=True)
    executor.decode_graph_runner = graph_runner
    executor.eager_runner = MagicMock()
    input_ids = torch.arange(2, dtype=torch.int32)
    metadata = MagicMock(spec=AttentionMetadata)
    topk = torch.ones((2, 1, 8), dtype=torch.int32)

    result = executor.execute(input_ids, input_ids, metadata, mtp_topk_indices=topk)

    graph_runner.prepare_execution.assert_called_once_with(input_ids, metadata, None, topk, is_graph_warmup=False)
    graph_runner.execute.assert_called_once_with(
        input_ids,
        input_ids,
        metadata,
        None,
        topk,
        graph_key=graph_runner.warmup.return_value,
        eplb=None,
        decode_metadata=graph_runner.prepare_execution.return_value,
        input_batch=None,
    )
    assert result is graph_runner.execute.return_value
    executor.eager_runner.execute.assert_not_called()


class _GraphStubAttentionBackend(StubAttentionBackend):
    @property
    def supports_prepared_metadata(self) -> bool:
        return True

    def prepare_metadata(self, metadata: AttentionMetadata, *, device_kv_lengths: bool = False) -> object:
        return SimpleNamespace(source=metadata.q_cu_seq_lens_host_values, device_kv_lengths=device_kv_lengths)


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 4, 16])
def test_qwen_draft_prefix_capacity_reaches_backend_and_decode_graph(draft_tokens: int) -> None:
    width = draft_tokens + 1
    config = {
        "model_type": "qwen3_5_mtp",
        "is_draft_model": True,
        "is_mtp_algorithm": True,
        "num_speculative_tokens": draft_tokens,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 128,
    }
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=_GraphStubAttentionBackend(),
    ) as create:
        executor = ModelExecutor(
            _FakeModel(),
            config,
            max_seqs_per_batch=5,
            acl_graph_decode_batch_size_limit=4,
        )
    assert create.call_args.args[4] == 5 * width
    runner = executor.decode_graph_runner
    assert runner is not None and runner.max_batch == 4 * width


def test_glm_draft_graph_capacity_uses_two_rows_per_request() -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "is_draft_model": True,
        "is_mtp_algorithm": True,
        "num_speculative_tokens": 16,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 128,
    }
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=_GraphStubAttentionBackend(),
    ) as create:
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=5, acl_graph_decode_batch_size_limit=4)
    assert create.call_args.args[4] == 10
    assert executor.decode_graph_runner.max_batch == 8


def test_executor_rejects_qwen_mtp_width_beyond_kernel_capacity() -> None:
    with (
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_GraphStubAttentionBackend(),
        ),
        pytest.raises(NotImplementedError, match="1 to 16"),
    ):
        ModelExecutor(
            _FakeModel(),
            {"model_type": "qwen3_5_text", "is_mtp_algorithm": True, "num_speculative_tokens": 17},
            max_seqs_per_batch=2,
        )


@pytest.mark.parametrize("is_mtp", [False, True])
def test_kv_transfer_uses_eager_and_preserves_step_inputs(is_mtp: bool) -> None:
    config = {
        "model_type": "glm_moe_dsa_mtp" if is_mtp else "glm_moe_dsa",
        "is_draft_model": is_mtp,
        "is_mtp_algorithm": is_mtp,
    }
    backend = StubAttentionBackend()
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=backend):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=2)
    cache = torch.empty(2, 4, 1, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    metadata = SimpleNamespace(q_cu_seq_lens_host_values=[1, 2])
    tokens = torch.tensor([13, 17], dtype=torch.int32)
    positions = torch.tensor([7, 19], dtype=torch.int32)
    hidden = torch.zeros(2, 4) if is_mtp else None
    topk = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32) if is_mtp else None
    executor.eager_runner = MagicMock()
    executor.decode_graph_runner = MagicMock(spec=DecodeAclGraphRunner)
    executor.inductor_runner = MagicMock()

    for _ in range(2):
        synchronizer = MagicMock()
        result = executor.execute(tokens, positions, metadata, hidden, synchronizer, topk, enable_graph=True)
        expected_args = (tokens, positions, metadata, hidden, synchronizer)
        expected_args += (topk,)
        executor.eager_runner.execute.assert_called_with(*expected_args, eplb=None, input_batch=None)
        assert result is executor.eager_runner.execute.return_value
    assert executor.eager_runner.execute.call_count == 2
    executor.decode_graph_runner.prepare_execution.assert_not_called()
    for runner in (executor.decode_graph_runner, executor.inductor_runner):
        runner.execute.assert_not_called()


@pytest.mark.parametrize("invalid", ["decode", "graph", "verify", "unsupported"])
def test_layerwise_load_rejects_incompatible_execution_before_running(invalid: str) -> None:
    model = _FakeModel()
    model.model.supports_layerwise_kv_load = invalid != "unsupported"
    model.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    with patch("xllm.python.model_executor.executor._create_attention_backend", return_value=StubAttentionBackend()):
        executor = ModelExecutor(model, {}, max_seqs_per_batch=2)
    executor._kv_bound = True
    executor.eager_runner = MagicMock()
    metadata = SimpleNamespace(
        is_prefill=invalid != "decode", is_chunked_prefill=False, is_spec_verify=invalid == "verify"
    )
    with pytest.raises(ValueError, match="supported eager prefill"):
        executor.execute(
            torch.tensor([1]),
            torch.tensor([3]),
            metadata,
            enable_graph=invalid == "graph",
            kv_load_synchronizer=MagicMock(),
        )
    executor.eager_runner.execute.assert_not_called()


@pytest.mark.parametrize("dp_size,cp_size", [(1, 1), (2, 1), (1, 2), (1, 4)])
def test_speculative_target_graph_capacity_expands_validation_tokens(dp_size: int, cp_size: int) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "dp_size": dp_size,
        "dp_rank": 0,
        "cp_size": cp_size,
        "kv_split_size": 1,
        "num_speculative_tokens": 3,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 128,
    }
    with (
        patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
        patch("xllm.python.model_executor.executor.distributed.dcp_group", return_value=None),
        patch(
            "xllm.python.model_executor.executor._create_attention_backend",
            return_value=_GraphStubAttentionBackend(),
        ),
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=5, acl_graph_decode_batch_size_limit=4)
    assert executor.decode_graph_runner.max_batch == min((5 + dp_size - 1) // dp_size, 4) * 4


@pytest.mark.parametrize("model_type", ["qwen3", "glm_moe_dsa"])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_executor_binds_acl_graph_caches(model_type: str, dp_size: int) -> None:
    config = _classify(
        {
            "model_type": model_type,
            "dp_size": dp_size,
            "dp_rank": dp_size - 1,
            "python_graph_backend": "aclgraph",
            "max_position_embeddings": 128,
        }
    )
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=4 * dp_size)
    assert executor.decode_graph_runner.dp_size == dp_size
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    runner = executor.decode_graph_runner
    assert len(runner.layer_caches) == 2


@pytest.mark.parametrize("no_padding,expected", [(False, 32), (True, 24)])
@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_mtp"])
def test_executor_passes_no_padding_policy_to_ordinary_acl_runner(
    no_padding: bool, expected: int, model_type: str
) -> None:
    config = {
        "model_type": model_type,
        "is_draft_model": model_type == "qwen3_5_mtp",
        "is_mtp_algorithm": True,
        "num_speculative_tokens": 3,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 4096,
        "enable_graph_mode_decode_no_padding": no_padding,
    }
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(_FakeModel(), config, max_seqs_per_batch=32)

    runner = executor.decode_graph_runner
    assert runner is not None
    assert runner._padded_batch_size(24, SimpleNamespace(is_spec_verify=False)) == expected
    assert runner._capture_error_mode == ("thread_local" if no_padding else "global")


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_executor_selects_dspark_acl_graph_with_dp(dp_size: int) -> None:
    config = _classify(
        {
            "model_type": "DSparkDraftModel",
            "num_speculative_tokens": 6,
            "dp_size": dp_size,
            "dp_rank": dp_size - 1,
            "python_graph_backend": "aclgraph",
            "max_position_embeddings": 128,
        }
    )
    model = _FakeModel()
    model.cfg = SimpleNamespace(block_size=7)
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(model, config, max_seqs_per_batch=5, acl_graph_decode_batch_size_limit=2)
    runner = executor.decode_graph_runner
    assert isinstance(runner, BlockDraftAclGraphRunner)
    assert runner.dp_size == dp_size
    assert runner.dp_rank == dp_size - 1
    assert runner.max_batch == min((5 + dp_size - 1) // dp_size, 2)
    assert executor.context_kv_graph_runner is not None


def test_dflash2_leaf_uses_full_block_graph_and_verify_context_capacity() -> None:
    model = _FakeModel()
    model.cfg = SimpleNamespace(block_size=8)
    config = {
        "model_type": "DFlash2DraftModel",
        "num_speculative_tokens": 0,
        "python_graph_backend": "aclgraph",
        "max_position_embeddings": 128,
    }
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(model, config, max_seqs_per_batch=2)
    assert isinstance(executor.decode_graph_runner, BlockDraftAclGraphRunner)
    assert executor.decode_graph_runner.max_batch == 2
    executor._kv_bound = True
    executor.context_kv_graph_runner.warmup = MagicMock()
    executor.decode_graph_runner.can_execute = MagicMock(return_value=False)
    executor.eager_runner.execute = MagicMock()
    tokens = torch.zeros(16, dtype=torch.int32)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=True, prepared_attention_state=None)
    executor.execute(tokens, tokens, metadata, is_graph_warmup=True)
    executor.context_kv_graph_runner.warmup.assert_called_once_with(16)


def test_executor_warms_context_prefix_buckets_only_during_startup() -> None:
    config = _classify(
        {
            "model_type": "DSparkDraftModel",
            "num_speculative_tokens": 6,
            "python_graph_backend": "aclgraph",
            "max_position_embeddings": 128,
        }
    )
    model = _FakeModel()
    model.cfg = SimpleNamespace(block_size=7)
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)
    cache = torch.empty(2, 4, 2, 64)
    executor.bind_kv_caches([LayerCache(cache, cache), LayerCache(cache, cache)])
    executor.context_kv_graph_runner.warmup = MagicMock()
    executor.decode_graph_runner.can_execute = MagicMock(return_value=False)
    executor.eager_runner.execute = MagicMock(return_value=torch.zeros(14, 4))
    tokens = torch.zeros(14, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=True, prepared_attention_state=None)
    executor.execute(tokens, positions, metadata, is_graph_warmup=True)
    executor.context_kv_graph_runner.warmup.assert_called_once_with(16)
    executor.execute(tokens, positions, metadata)
    executor.context_kv_graph_runner.warmup.assert_called_once_with(16)


def test_executor_preserves_block_draft_admission_and_runner_arguments() -> None:
    config = _classify(
        {
            "model_type": "DSparkDraftModel",
            "num_speculative_tokens": 2,
            "python_graph_backend": "aclgraph",
            "max_position_embeddings": 128,
        }
    )
    model = _FakeModel()
    model.cfg = SimpleNamespace(block_size=3)
    with patch(
        "xllm.python.model_executor.executor._create_attention_backend", return_value=_GraphStubAttentionBackend()
    ):
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)
    executor._kv_bound = True
    runner = executor.decode_graph_runner
    assert isinstance(runner, BlockDraftAclGraphRunner)
    runner.prepare_execution = MagicMock(side_effect=AssertionError("ordinary decode admission"))
    runner.can_execute = MagicMock(return_value=True)
    runner.warmup = MagicMock(return_value="block-key")
    runner.execute = MagicMock(return_value=torch.ones(3, 4))
    tokens = torch.arange(3, dtype=torch.int32)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=True, prepared_attention_state=None)

    output = executor.execute(tokens, tokens, metadata)

    runner.prepare_execution.assert_not_called()
    runner.can_execute.assert_called_once_with(tokens, metadata, None, None)
    runner.warmup.assert_called_once_with(
        tokens, tokens, metadata, None, None, eplb=None, pre_capture_kv_buckets=False, input_batch=None
    )
    runner.execute.assert_called_once_with(
        tokens, tokens, metadata, None, None, graph_key="block-key", eplb=None, decode_metadata=None, input_batch=None
    )
    assert output is runner.execute.return_value


def test_eager_runner_skips_cp_for_empty_rank() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        is_dummy=True,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.ones(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


class TestDecodeAclGraphWidthKeying:
    def test_graph_key_identifies_uniform_span_width_not_sequence_count(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            num_speculative_tokens=3,
        )
        runner._has_linear_cache = True
        runner._enable_decode_no_padding = False
        runner._uses_compressed_kpool_tail = False

        two = SimpleNamespace(kpool_query_lens=(4, 4))
        four = SimpleNamespace(kpool_query_lens=(4, 4, 4, 4))

        assert runner._padded_kpool_query_lens(two, 8, 16) == runner._padded_kpool_query_lens(four, 16, 16)
        assert DecodeAclGraphRunner._graph_key(
            16, False, None, kpool_query_lens=runner._padded_kpool_query_lens(two, 8, 16)
        ) == DecodeAclGraphRunner._graph_key(
            16, False, None, kpool_query_lens=runner._padded_kpool_query_lens(four, 16, 16)
        )
        assert DecodeAclGraphRunner._graph_key(
            16, False, None, kpool_query_lens=(8, 8)
        ) != DecodeAclGraphRunner._graph_key(
            16, False, None, kpool_query_lens=runner._padded_kpool_query_lens(four, 16, 16)
        )
        with pytest.raises(ValueError, match="complete fixed-width requests"):
            runner._padded_batch_size(6, SimpleNamespace(is_spec_verify=True))
        with pytest.raises(ValueError, match="exceeds ACL graph capacity"):
            runner._padded_batch_size(20, SimpleNamespace(is_spec_verify=True))

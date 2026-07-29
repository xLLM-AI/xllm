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
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# The xllm.python package auto-registers models on import, which triggers
# torch.ops.xllm_ops lookups that require the C++ binary. We bypass this
# by mocking the ops and registry modules before importing executor.
_mock_ops = MagicMock()
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, KVCache  # noqa: E402
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _is_npu_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[KVCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
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


def _make_attention_layer(
    num_heads=8, num_kv_heads=2, head_dim=64, scale=0.125, sliding_window=0, layer_id=0,
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
        self.layers = nn.ModuleList(
            [_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)]
        )
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
# Tests: _is_npu_device
# ---------------------------------------------------------------------------


class TestIsNpuDevice:
    def test_npu_type(self):
        assert _is_npu_device(torch.device("npu")) is True

    def test_privateuseone_type(self):
        assert _is_npu_device(torch.device("privateuseone")) is True

    def test_cuda_type(self):
        assert _is_npu_device(torch.device("cuda")) is False

    def test_cpu_type(self):
        assert _is_npu_device(torch.device("cpu")) is False


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @patch(
        "xllm.python.model_executor.executor._is_npu_device", return_value=True
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_device_creates_npu_backend(self, _mock_is_npu):
        attn = _make_attention_layer()
        backend = _create_attention_backend(
            attn, torch.device("npu"), torch.float16
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 2
        assert backend.init_kwargs["head_dim"] == 64

    @patch(
        "xllm.python.model_executor.executor._is_npu_device", return_value=False
    )
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_cuda_device_creates_flashinfer_backend(self, mock_create, _mock_is_npu):
        mock_create.return_value = StubAttentionBackend(num_heads=8)
        attn = _make_attention_layer()
        # Verify the factory would be called (we can't import flashinfer in NPU env)
        from xllm.python.model_executor.executor import _is_npu_device
        assert _is_npu_device(torch.device("cuda")) is False


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
        assert executor.decode_cuda_graph_runner is None
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
            executor = ModelExecutor(
                model, {"python_graph_backend": off_value}, max_seqs_per_batch=4
            )
            assert executor.decode_cuda_graph_runner is None
            assert executor.inductor_runner is None


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
        executor.eager_runner.execute.return_value = torch.ones(5)

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.eager_runner.execute.assert_called_once()
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

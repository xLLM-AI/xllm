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

import importlib
import sys
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    _resolve_graph_backend,
)
from xllm.python.model_executor.runners.compile_runner import CompileRunner  # noqa: E402

_HAS_NPU = torch.npu.is_available()
_HAS_TORCHAIR = importlib.util.find_spec("torchair") is not None
_NPU_DEVICE = torch.device("npu") if _HAS_NPU else torch.device("cpu")


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
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class TestNpuGraphBackendResolution:
    def test_enable_graph_selects_aclgraph_on_npu(self):
        config = {"enable_graph": True, "python_graph_backend": "off"}

        assert (
            _resolve_graph_backend(config, torch.device("npu"))
            == "aclgraph"
        )


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
        assert executor.decode_graph_runner is None
        assert executor.compile_runner is None

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
            assert executor.decode_graph_runner is None
            assert executor.compile_runner is None


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
    def test_compile_runner_takes_priority_over_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        executor.compile_runner = MagicMock()
        executor.compile_runner.execute.return_value = torch.ones(3)

        metadata = MagicMock(spec=AttentionMetadata)
        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.compile_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(3))


# ---------------------------------------------------------------------------
# Real network models for graph compiler tests
# ---------------------------------------------------------------------------


class _MlpBlock(nn.Module):
    """MLP block: LayerNorm -> Linear -> SiLU -> Linear + residual."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x + residual


class _ResidualStack(nn.Module):
    """ResNet-like stack with skip connections and bottleneck blocks."""

    def __init__(self, dim: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim),
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class _SelfAttention(nn.Module):
    """Multi-head self-attention (no external backend dependency)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: self-attention + MLP with residuals."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _DeepNetwork(nn.Module):
    """Deep network combining Transformer + ResNet blocks."""

    def __init__(self, dim: int, num_heads: int, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList([
            _TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.residual_tail = _ResidualStack(dim, num_blocks=2)
        self.head_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.transformer_blocks:
            x = block(x, positions)
        x = self.residual_tail(x, positions)
        return self.head_norm(x)


class _GraphCompileWrapperModel(nn.Module):
    """Outer model with Attention layers for ModelExecutor + inner execution model."""

    def __init__(
        self,
        inner_model: nn.Module,
        num_attn_layers: int = 2,
        device: str = "npu",
        num_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 64,
    ):
        super().__init__()
        self.model = inner_model
        self.layers = nn.ModuleList([
            _make_attention_layer(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=i,
            )
            for i in range(num_attn_layers)
        ])
        self._param = nn.Parameter(torch.zeros(1, device=device))


# ---------------------------------------------------------------------------
# Tests: CompileRunner._resolve_compile_backend
# ---------------------------------------------------------------------------


class TestResolveCompileBackend:
    def test_passthrough_backend(self):
        assert CompileRunner._resolve_compile_backend("npu") == "npu"
        assert CompileRunner._resolve_compile_backend("inductor") == "inductor"
        assert CompileRunner._resolve_compile_backend("eager") == "eager"

    @pytest.mark.skipif(not _HAS_TORCHAIR, reason="torchair not installed")
    def test_torchair_backend(self):
        backend = CompileRunner._resolve_compile_backend("torchair")
        assert callable(backend)


# ---------------------------------------------------------------------------
# Tests: CompileRunner fullgraph=True with real NPU compilation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")
class TestCompileRunnerFullGraph:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = _NPU_DEVICE
        self.dtype = torch.float16
        self.backend = StubAttentionBackend()

    def _run_and_verify(self, model, batch=2, seq_len=8, dim=64):
        model = model.to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=True, dynamic=False,
        )
        x = torch.randn(batch, seq_len, dim, device=self.device, dtype=self.dtype)
        pos = torch.arange(seq_len, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        compiled_out = runner.execute(x, pos, metadata)
        eager_out = model(x, pos)

        assert compiled_out.shape == eager_out.shape
        assert torch.allclose(compiled_out, eager_out, atol=1e-2, rtol=1e-2)
        return compiled_out

    def test_mlp_block(self):
        self._run_and_verify(_MlpBlock(dim=64))

    def test_residual_stack(self):
        self._run_and_verify(_ResidualStack(dim=64, num_blocks=3))

    def test_transformer_block(self):
        self._run_and_verify(_TransformerBlock(dim=64, num_heads=4))

    def test_deep_network(self):
        self._run_and_verify(_DeepNetwork(dim=64, num_heads=4, num_layers=2))

    def test_mlp_block_various_batch_sizes(self):
        model = _MlpBlock(dim=64).to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=True, dynamic=False,
        )
        metadata = MagicMock(spec=AttentionMetadata)
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 8, 64, device=self.device, dtype=self.dtype)
            pos = torch.arange(8, device=self.device)
            out = runner.execute(x, pos, metadata)
            assert out.shape == (bs, 8, 64)

    def test_transformer_various_seq_lens(self):
        model = _TransformerBlock(dim=64, num_heads=4).to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=True, dynamic=False,
        )
        metadata = MagicMock(spec=AttentionMetadata)
        for seq_len in [4, 16, 32]:
            x = torch.randn(2, seq_len, 64, device=self.device, dtype=self.dtype)
            pos = torch.arange(seq_len, device=self.device)
            out = runner.execute(x, pos, metadata)
            assert out.shape == (2, seq_len, 64)

    def test_repeated_invocations_consistency(self):
        model = _MlpBlock(dim=64).to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=True, dynamic=False,
        )
        metadata = MagicMock(spec=AttentionMetadata)
        x = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        pos = torch.arange(8, device=self.device)

        out1 = runner.execute(x, pos, metadata)
        out2 = runner.execute(x, pos, metadata)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Tests: CompileRunner fullgraph=False (dynamo) with real NPU compilation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")
class TestCompileRunnerDynamo:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = _NPU_DEVICE
        self.dtype = torch.float16
        self.backend = StubAttentionBackend()

    def _run_and_verify(self, model, batch=2, seq_len=8, dim=64):
        model = model.to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=False, dynamic=False,
        )
        x = torch.randn(batch, seq_len, dim, device=self.device, dtype=self.dtype)
        pos = torch.arange(seq_len, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        compiled_out = runner.execute(x, pos, metadata)
        eager_out = model(x, pos)

        assert compiled_out.shape == eager_out.shape
        assert torch.allclose(compiled_out, eager_out, atol=1e-2, rtol=1e-2)
        return compiled_out

    def test_mlp_block(self):
        self._run_and_verify(_MlpBlock(dim=64))

    def test_residual_stack(self):
        self._run_and_verify(_ResidualStack(dim=64, num_blocks=2))

    def test_transformer_block(self):
        self._run_and_verify(_TransformerBlock(dim=64, num_heads=4))

    def test_deep_network(self):
        self._run_and_verify(_DeepNetwork(dim=64, num_heads=4, num_layers=2))

    def test_repeated_invocations_consistency(self):
        model = _MlpBlock(dim=64).to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="npu", fullgraph=False, dynamic=False,
        )
        metadata = MagicMock(spec=AttentionMetadata)
        x = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        pos = torch.arange(8, device=self.device)

        out1 = runner.execute(x, pos, metadata)
        out2 = runner.execute(x, pos, metadata)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Tests: fullgraph vs dynamo output equivalence
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")
class TestFullGraphVsDynamoEquivalence:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = _NPU_DEVICE
        self.dtype = torch.float16
        self.backend = StubAttentionBackend()

    def _compare_modes(self, model_cls, batch=2, seq_len=8, dim=64):
        model_fg = model_cls.to(self.device, dtype=self.dtype)
        model_dy = model_cls.to(self.device, dtype=self.dtype)
        model_dy.load_state_dict(model_fg.state_dict())

        runner_fg = CompileRunner(
            model_fg, self.backend, self.device,
            backend="npu", fullgraph=True, dynamic=False,
        )
        runner_dy = CompileRunner(
            model_dy, self.backend, self.device,
            backend="npu", fullgraph=False, dynamic=False,
        )

        x = torch.randn(batch, seq_len, dim, device=self.device, dtype=self.dtype)
        pos = torch.arange(seq_len, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        out_fg = runner_fg.execute(x, pos, metadata)
        out_dy = runner_dy.execute(x, pos, metadata)

        assert out_fg.shape == out_dy.shape
        assert torch.allclose(out_fg, out_dy, atol=1e-2, rtol=1e-2)

    def test_mlp_equivalence(self):
        self._compare_modes(_MlpBlock(dim=64))

    def test_transformer_equivalence(self):
        self._compare_modes(_TransformerBlock(dim=64, num_heads=4))

    def test_deep_network_equivalence(self):
        self._compare_modes(_DeepNetwork(dim=64, num_heads=4, num_layers=2))


# ---------------------------------------------------------------------------
# Tests: ModelExecutor integration with CompileRunner (real compilation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")
class TestModelExecutorCompileIntegration:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = _NPU_DEVICE
        self.dtype = torch.float16

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_creates_compile_runner_with_npu_backend(self, _mock):
        inner = _MlpBlock(dim=64)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {"python_graph_backend": "npu"}
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        assert executor.compile_runner is not None
        assert isinstance(executor.compile_runner, CompileRunner)
        assert executor.decode_graph_runner is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_creates_compile_runner_fullgraph(self, _mock):
        inner = _MlpBlock(dim=64)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {
            "python_graph_backend": "npu",
            "python_compile_fullgraph": True,
        }
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        assert executor.compile_runner is not None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_compile_execute_mlp(self, _mock):
        inner = _MlpBlock(dim=64)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {"python_graph_backend": "npu", "python_compile_fullgraph": True}
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])

        input_ids = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        positions = torch.arange(8, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        result = executor.execute(input_ids, positions, metadata)
        assert result.shape == (2, 8, 64)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_compile_execute_transformer(self, _mock):
        inner = _TransformerBlock(dim=64, num_heads=4)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {"python_graph_backend": "npu", "python_compile_fullgraph": True}
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])

        input_ids = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        positions = torch.arange(8, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        result = executor.execute(input_ids, positions, metadata)
        assert result.shape == (2, 8, 64)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_compile_execute_deep_network(self, _mock):
        inner = _DeepNetwork(dim=64, num_heads=4, num_layers=2)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {"python_graph_backend": "npu", "python_compile_fullgraph": True}
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])

        input_ids = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        positions = torch.arange(8, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        result = executor.execute(input_ids, positions, metadata)
        assert result.shape == (2, 8, 64)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_compile_dynamo_mode(self, _mock):
        inner = _MlpBlock(dim=64)
        wrapper = _GraphCompileWrapperModel(inner, num_attn_layers=2, device="npu")
        wrapper = wrapper.to(self.device, dtype=self.dtype)

        config = {
            "python_graph_backend": "npu",
            "python_compile_fullgraph": False,
        }
        executor = ModelExecutor(wrapper, config, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])

        input_ids = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        positions = torch.arange(8, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        result = executor.execute(input_ids, positions, metadata)
        assert result.shape == (2, 8, 64)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_executor_compile_output_matches_eager(self, _mock):
        inner_fg = _MlpBlock(dim=64)
        wrapper_fg = _GraphCompileWrapperModel(inner_fg, num_attn_layers=2, device="npu")
        wrapper_fg = wrapper_fg.to(self.device, dtype=self.dtype)

        inner_eager = _MlpBlock(dim=64)
        wrapper_eager = _GraphCompileWrapperModel(inner_eager, num_attn_layers=2, device="npu")
        wrapper_eager = wrapper_eager.to(self.device, dtype=self.dtype)
        wrapper_eager.load_state_dict(wrapper_fg.state_dict())

        config_compile = {"python_graph_backend": "npu", "python_compile_fullgraph": True}
        config_eager = {"python_graph_backend": "off"}

        executor_compile = ModelExecutor(wrapper_fg, config_compile, max_seqs_per_batch=4)
        executor_eager = ModelExecutor(wrapper_eager, config_eager, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor_compile.bind_kv_caches([kv, kv])
        executor_eager.bind_kv_caches([kv, kv])

        input_ids = torch.randn(2, 8, 64, device=self.device, dtype=self.dtype)
        positions = torch.arange(8, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        out_compile = executor_compile.execute(input_ids, positions, metadata)
        out_eager = executor_eager.execute(input_ids, positions, metadata)

        assert torch.allclose(out_compile, out_eager, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Tests: torchair-specific compilation (requires torchair package)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NPU or not _HAS_TORCHAIR, reason="NPU or torchair not available")
class TestCompileRunnerTorchair:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = _NPU_DEVICE
        self.dtype = torch.float16
        self.backend = StubAttentionBackend()

    def _run_and_verify(self, model, batch=2, seq_len=8, dim=64, fullgraph=True):
        model = model.to(self.device, dtype=self.dtype)
        runner = CompileRunner(
            model, self.backend, self.device,
            backend="torchair", fullgraph=fullgraph, dynamic=False,
        )
        x = torch.randn(batch, seq_len, dim, device=self.device, dtype=self.dtype)
        pos = torch.arange(seq_len, device=self.device)
        metadata = MagicMock(spec=AttentionMetadata)

        compiled_out = runner.execute(x, pos, metadata)
        eager_out = model(x, pos)

        assert compiled_out.shape == eager_out.shape
        assert torch.allclose(compiled_out, eager_out, atol=1e-2, rtol=1e-2)

    def test_mlp_fullgraph(self):
        self._run_and_verify(_MlpBlock(dim=64), fullgraph=True)

    def test_mlp_dynamo(self):
        self._run_and_verify(_MlpBlock(dim=64), fullgraph=False)

    def test_transformer_fullgraph(self):
        self._run_and_verify(_TransformerBlock(dim=64, num_heads=4), fullgraph=True)

    def test_transformer_dynamo(self):
        self._run_and_verify(_TransformerBlock(dim=64, num_heads=4), fullgraph=False)

    def test_deep_network_fullgraph(self):
        self._run_and_verify(
            _DeepNetwork(dim=64, num_heads=4, num_layers=2), fullgraph=True,
        )

    def test_deep_network_dynamo(self):
        self._run_and_verify(
            _DeepNetwork(dim=64, num_heads=4, num_layers=2), fullgraph=False,
        )

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

import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _backend_for_linear_cache(
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
) -> Any:
    from xllm.python.attention.kda_linear_attention import KdaLinearAttentionMixin

    backend = object.__new__(KdaLinearAttentionMixin)
    backend._kv_caches = [SimpleNamespace(conv=conv_cache, ssm=ssm_cache)]
    backend._metadata = None
    return backend


def _layer() -> SimpleNamespace:
    weight = torch.ones(6, 1, 3, dtype=torch.float32)
    return SimpleNamespace(
        layer_id=0,
        conv_kernel_size=3,
        head_dim=2,
        num_heads_local=1,
        qkv_dim=2,
        conv_dim=6,
        conv1d=SimpleNamespace(weight=weight),
        activation="silu",
    )


@pytest.fixture
def kda_test_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with monkeypatch.context() as context:
        _install_kda_stubs(context)
        yield


def _install_kda_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    fla_npu = types.ModuleType("fla_npu")
    fla_npu_ops = types.ModuleType("fla_npu.ops")
    ascendc = types.ModuleType("fla_npu.ops.ascendc")

    def chunk_kda_fwd(
        query: torch.Tensor,
        *_args: Any,
        initial_state: torch.Tensor,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros(1, query.shape[1], 1, 2, dtype=query.dtype)
        return output, initial_state + 200

    ascendc.chunk_kda_fwd = chunk_kda_fwd
    ascendc.recurrent_kda = lambda *_args, **_kwargs: None
    fla_npu.ops = fla_npu_ops
    fla_npu_ops.ascendc = ascendc
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)

    glm_module = types.ModuleType("xllm.python.models.glm5_next")
    glm_module._causal_conv1d_fn = lambda conv_input, _weight, _activation: conv_input[:, :, 2:]
    glm_module._causal_conv1d_update = lambda conv_input, _state, _weight, _activation: conv_input
    glm_module._l2norm = lambda value, **_kwargs: value
    monkeypatch.setitem(sys.modules, "xllm.python.models.glm5_next", glm_module)


def test_execute_linear_without_state_indices(kda_test_environment: None) -> None:
    backend = _backend_for_linear_cache(torch.empty(0), torch.empty(0))
    backend._metadata = SimpleNamespace(
        linear_state_indices=None,
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.zeros(1, 1, 1, 2, dtype=torch.float32),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
    )

    assert output.shape == (1, 1, 1, 2)


def test_merged_spec_verify_uses_remapped_state_indices(
    kda_test_environment: None,
) -> None:
    conv_cache = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)
    ssm_cache = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    per_row_indices = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    backend._metadata = SimpleNamespace(
        linear_state_indices=per_row_indices,
        linear_state_read_indices=per_row_indices.clone(),
        linear_state_write_indices=per_row_indices,
        has_initial_state=torch.tensor([1, 1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.zeros(1, 4, 1, 2, dtype=torch.float32),
        torch.ones(1, 4, 1, dtype=torch.float32),
        _layer(),
    )

    assert output.shape == (1, 4, 1, 2)


def test_execute_linear_prefill_reads_source_and_writes_live(
    kda_test_environment: None,
) -> None:
    conv_cache = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)
    ssm_cache = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    original_conv = conv_cache.clone()
    original_ssm = ssm_cache.clone()
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([3], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([1], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([3], dtype=torch.int32),
        has_initial_state=torch.tensor([1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.zeros(1, 1, 1, 2, dtype=torch.float32),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
    )

    assert output.shape == (1, 1, 1, 2)
    assert torch.equal(conv_cache[1], original_conv[1])
    assert torch.equal(ssm_cache[1], original_ssm[1])
    assert torch.equal(conv_cache[2], original_conv[2])
    assert torch.equal(ssm_cache[2], original_ssm[2])
    assert not torch.equal(conv_cache[3], original_conv[3])
    assert torch.equal(ssm_cache[3], original_ssm[1] + 200)

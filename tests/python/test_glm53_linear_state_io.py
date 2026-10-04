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


def _forget_gate(lower_bound: float | None = -5.0) -> SimpleNamespace:
    """Stub Glm5NextForgetGate: raw_projection is identity on the handed raw,
    gate_from_raw applies the safe-gate the backend fuses in-kernel on the
    plain path (only the MTP / non-fused paths materialize it here)."""
    num_heads, head_dim = 1, 2

    def gate_from_raw(raw: torch.Tensor) -> torch.Tensor:
        g = raw.float() + torch.zeros(num_heads * head_dim, dtype=torch.float32).view(1, 1, num_heads, head_dim)
        decay = torch.exp(torch.zeros(num_heads, dtype=torch.float32).view(1, 1, num_heads, 1))
        if lower_bound is not None:
            return lower_bound * torch.sigmoid(decay * g)
        softplus = torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))
        return -decay * softplus

    return SimpleNamespace(
        A_log=torch.zeros(num_heads, dtype=torch.float32),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
        safe_gate_lower_bound=lower_bound,
        gate_from_raw=gate_from_raw,
    )


def _layer(lower_bound: float | None = -5.0) -> SimpleNamespace:
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
        forget_gate=_forget_gate(lower_bound),
    )


@pytest.fixture
def kda_test_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict]]:
    with monkeypatch.context() as context:
        kernel_calls = _install_kda_stubs(context)
        yield kernel_calls


def _install_kda_stubs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    fla_npu = types.ModuleType("fla_npu")
    fla_npu_ops = types.ModuleType("fla_npu.ops")
    ascendc = types.ModuleType("fla_npu.ops.ascendc")

    kernel_calls: list[dict] = []

    def chunk_kda_fwd(
        query: torch.Tensor,
        *_args: Any,
        initial_state: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # _args[2] is the gate tensor the backend hands the kernel (raw on the
        # fused plain path, materialized gate otherwise); record the fuse flags.
        gate_arg = _args[2] if len(_args) > 2 else None
        kernel_calls.append({"op": "chunk", "gate": gate_arg, **kwargs})
        output = torch.zeros(1, query.shape[1], 1, 2, dtype=query.dtype)
        return output, initial_state + 200

    def recurrent_kda(*args: Any, **kwargs: Any) -> None:
        gate_arg = args[3] if len(args) > 3 else None
        kernel_calls.append({"op": "recurrent", "gate": gate_arg, **kwargs})
        return None

    ascendc.chunk_kda_fwd = chunk_kda_fwd
    ascendc.recurrent_kda = recurrent_kda
    fla_npu.ops = fla_npu_ops
    fla_npu_ops.ascendc = ascendc
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)

    glm_module = types.ModuleType("xllm.python.models.glm5_next")
    glm_module._causal_conv1d_fn = lambda conv_input, _weight, _activation: conv_input[:, :, 2:]
    glm_module._causal_conv1d_update = lambda conv_input, _state, _weight, _activation: conv_input
    import xllm.python as python_package

    monkeypatch.setitem(python_package.__dict__, "kernels", SimpleNamespace(l2_norm=lambda value: value))
    monkeypatch.setitem(sys.modules, "xllm.python.models.glm5_next", glm_module)

    return kernel_calls


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
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
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
        torch.ones(1, 4, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
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
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 1, 1, 2)
    assert torch.equal(conv_cache[1], original_conv[1])
    assert torch.equal(ssm_cache[1], original_ssm[1])
    assert torch.equal(conv_cache[2], original_conv[2])
    assert torch.equal(ssm_cache[2], original_ssm[2])
    assert not torch.equal(conv_cache[3], original_conv[3])
    assert torch.equal(ssm_cache[3], original_ssm[1] + 200)


def _plain_prefill_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([0], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def test_plain_prefill_fuses_gate_in_kernel(kda_test_environment: list[dict]) -> None:
    # Plain (non-MTP) prefill: the kernel must compute the safe-gate itself
    # (use_gate_in_kernel=True + safe_gate + lower_bound) and receive the raw
    # projection verbatim — never a python-materialized gate (the double
    # safe-gate bug). The gate is not materialized on this path at all.
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.float32)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(lower_bound=-5.0),
        raw_gate_proj=raw,
    )

    chunk_calls = [c for c in kda_test_environment if c["op"] == "chunk"]
    assert len(chunk_calls) == 1
    call = chunk_calls[0]
    assert call["use_gate_in_kernel"] is True
    assert call["safe_gate"] is True
    assert call["lower_bound"] == -5.0
    # The kernel gets the raw projection (0.5), not lb*sigmoid(...) of it.
    assert torch.allclose(call["gate"].reshape(-1), raw.reshape(-1))


def test_lower_bound_out_of_range_falls_back_to_python_gate(
    kda_test_environment: list[dict],
) -> None:
    # AscendC safe_gate requires lower_bound in [-5, 0); an out-of-range config
    # must fall back to the materialized python gate (use_gate_in_kernel=False)
    # rather than passing an illegal lower_bound to the kernel.
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.float32)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(lower_bound=-6.0),
        raw_gate_proj=raw,
    )

    chunk_calls = [c for c in kda_test_environment if c["op"] == "chunk"]
    assert len(chunk_calls) == 1
    call = chunk_calls[0]
    assert call["use_gate_in_kernel"] is False
    assert "lower_bound" not in call
    # The kernel gets the materialized gate lb*sigmoid(decay*raw), not the raw.
    expected = _forget_gate(-6.0).gate_from_raw(raw)
    assert torch.allclose(call["gate"].reshape(-1), expected.reshape(-1))


def test_chunked_spec_verify_does_not_fuse_gate(
    kda_test_environment: list[dict],
) -> None:
    # MTP chunked spec-verify (per-row duplicated slots) must NOT fuse: its
    # correctness relies on the materialized gate (a gate==0 row is a state
    # no-op), so feeding raw + safe_gate would double-apply the safe-gate.
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

    backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 1, dtype=torch.float32),
        _layer(lower_bound=-5.0),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
    )

    chunk_calls = [c for c in kda_test_environment if c["op"] == "chunk"]
    assert chunk_calls, "spec-verify prefill should still call chunk_kda_fwd"
    assert all(c["use_gate_in_kernel"] is False for c in chunk_calls)

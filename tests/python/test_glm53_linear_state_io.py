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
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    backend = object.__new__(NpuPagedAttentionBackend)
    backend._kv_caches = [SimpleNamespace(conv=conv_cache, ssm=ssm_cache)]
    backend._metadata = None
    backend._kda_checkpoint_stride = ssm_cache.shape[0] // conv_cache.shape[0] if conv_cache.shape[0] else None
    backend._kda_query_start_locs = {}
    backend._kda_checkpoint_offsets = {}
    return backend


def _forget_gate(lower_bound: float | None = -5.0) -> SimpleNamespace:
    """Stub KdaForgetGate: raw_projection is identity on the handed raw,
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
        head_dim=head_dim,
        num_heads=num_heads,
        qkv_dim=num_heads * head_dim,
        A_log=torch.zeros(num_heads, dtype=torch.float32),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
        safe_gate_lower_bound=lower_bound,
        gate_from_raw=gate_from_raw,
    )


def _kda_inputs(lower_bound: float | None = -5.0) -> tuple[int, torch.Tensor, SimpleNamespace, str]:
    conv = torch.ones(3, 6, dtype=torch.bfloat16)
    return 0, conv, _forget_gate(lower_bound), "silu"


@pytest.fixture
def kda_test_environment(monkeypatch: pytest.MonkeyPatch, causal_conv1d_reference: list[dict]) -> Iterator[list[dict]]:
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

    def recurrent_kda(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        gate_arg = args[3] if len(args) > 3 else None
        kernel_calls.append({"op": "recurrent", "gate": gate_arg, "beta": args[4], **kwargs})
        return torch.zeros_like(args[0]), kwargs["initial_state"] + 100

    ascendc.chunk_kda_fwd = chunk_kda_fwd
    ascendc.recurrent_kda = recurrent_kda
    fla_npu.ops = fla_npu_ops
    fla_npu_ops.ascendc = ascendc
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)

    import xllm.python as python_package

    monkeypatch.setitem(
        python_package.__dict__,
        "kernels",
        SimpleNamespace(l2_norm=lambda value: value, causal_conv1d_update_v2=lambda value, *args, **kwargs: value),
    )

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
        *_kda_inputs(),
        raw_gate_proj=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 1, 1, 2)


def test_merged_spec_verify_uses_remapped_state_indices(
    kda_test_environment: list[dict],
) -> None:
    conv_cache = torch.arange(4 * 3 * 6, dtype=torch.bfloat16).reshape(4, 3, 6)
    ssm_cache = torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    per_row_indices = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        linear_state_indices=per_row_indices,
        linear_state_read_indices=per_row_indices.clone(),
        linear_state_write_indices=per_row_indices,
        has_initial_state=torch.tensor([1, 1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=None,
        block_table=None,
    )
    backend._metadata = metadata

    output = backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 1, dtype=torch.float32),
        *_kda_inputs(),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 4, 1, 2)
    assert len(kda_test_environment) == 1
    verify_call = kda_test_environment[0]
    assert verify_call["op"] == "recurrent"
    assert verify_call["inplace_final_state"] is True
    assert verify_call["ssm_state_indices"].tolist() == [
        [4, 5, 6, 7],
        [12, 13, 14, 15],
    ]
    assert verify_call["cu_seqlens"].tolist() == [0, 2, 4]
    assert verify_call["num_accepted_tokens"].tolist() == [1, 1]


def test_execute_linear_prefill_reads_source_and_writes_live(
    kda_test_environment: None,
) -> None:
    conv_cache = torch.arange(4 * 2 * 6, dtype=torch.bfloat16).reshape(4, 2, 6)
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
        *_kda_inputs(),
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
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        *_kda_inputs(lower_bound=-5.0),
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
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        *_kda_inputs(lower_bound=-6.0),
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
    conv_cache = torch.arange(4 * 3 * 6, dtype=torch.bfloat16).reshape(4, 3, 6)
    ssm_cache = torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    per_row_indices = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        linear_state_indices=per_row_indices,
        linear_state_read_indices=per_row_indices.clone(),
        linear_state_write_indices=per_row_indices,
        has_initial_state=torch.tensor([1, 1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=None,
        block_table=None,
    )
    backend._metadata = metadata

    backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 1, dtype=torch.float32),
        *_kda_inputs(-5.0),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
    )

    assert len(kda_test_environment) == 1
    verify_call = kda_test_environment[0]
    assert verify_call["op"] == "recurrent"
    assert verify_call["use_gate_in_kernel"] is False
    assert verify_call["use_beta_sigmoid_in_kernel"] is False
    assert torch.equal(verify_call["gate"], torch.full((4, 1, 2), -2.5))
    assert torch.equal(verify_call["beta"], torch.ones(4, 1))


@pytest.mark.parametrize("in_graph", [False, True])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("chunked_verify", [False, True])
def test_spec_verify_uses_combined_state_without_environment_flags(
    kda_test_environment: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    in_graph: bool,
    width: int,
    chunked_verify: bool,
) -> None:
    from xllm.python.attention import kda_linear_attention

    monkeypatch.setattr(kda_linear_attention, "in_acl_graph", lambda: in_graph)
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6), torch.zeros(4, 1, 2, 2))
    slots = torch.tensor([1, 3], dtype=torch.int32)
    total_rows = 2 * width
    backend._metadata = SimpleNamespace(
        linear_state_indices=slots.repeat_interleave(width) if in_graph else slots,
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=chunked_verify,
        is_spec_verify=chunked_verify,
        q_cu_seq_lens=torch.arange(
            0, total_rows + 1, width if chunked_verify and not in_graph else 1, dtype=torch.int32
        ),
        q_seq_lens=torch.full((2,), width, dtype=torch.int32),
        kv_seq_lens=torch.arange(total_rows, dtype=torch.int32) + 10,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=SimpleNamespace() if in_graph else None,
    )
    expected_output = torch.zeros(1, total_rows, 1, 2)
    verify_calls = []

    def spec_verify(*args: Any) -> torch.Tensor:
        verify_calls.append(args)
        return expected_output

    monkeypatch.setattr(backend, "_spec_verify", spec_verify)
    output = backend.execute_linear(
        torch.ones(1, 6, total_rows, dtype=torch.bfloat16),
        torch.ones(1, total_rows, 1),
        *_kda_inputs(),
        raw_gate_proj=torch.zeros(1, total_rows, 1, 2),
    )

    assert output is expected_output
    assert len(verify_calls) == 1
    assert torch.equal(verify_calls[0][5], slots.to(torch.int64))
    assert not kda_test_environment


def test_spec_verify_rejects_nonuniform_widths(kda_test_environment: list[dict]) -> None:
    conv_cache = torch.ones(4, 2, 6)
    ssm_cache = torch.ones(4, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 1, 3, 3, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        is_prefill=False,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.arange(6, dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    with pytest.raises(RuntimeError, match="uniform rows per sequence"):
        backend.execute_linear(
            torch.ones(1, 6, 5, dtype=torch.bfloat16),
            torch.ones(1, 5, 1),
            *_kda_inputs(),
            raw_gate_proj=torch.zeros(1, 5, 1, 2),
        )

    assert not kda_test_environment
    assert torch.equal(conv_cache, torch.ones_like(conv_cache))
    assert torch.equal(ssm_cache, torch.ones_like(ssm_cache))


def test_plain_decode_keeps_fused_gate_without_verify_state(kda_test_environment: list[dict]) -> None:
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6, dtype=torch.bfloat16), torch.zeros(4, 1, 2, 2))
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.arange(3, dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 2, dtype=torch.bfloat16),
        torch.ones(1, 2, 1),
        *_kda_inputs(),
        raw_gate_proj=torch.zeros(1, 2, 1, 2),
    )

    assert output.shape == (1, 2, 1, 2)
    assert len(kda_test_environment) == 1
    assert kda_test_environment[0]["op"] == "recurrent"
    assert kda_test_environment[0]["use_gate_in_kernel"] is True
    assert kda_test_environment[0]["cu_seqlens"].tolist() == [0, 1, 2]


def test_varlen_prefill_uses_one_native_convolution(
    kda_test_environment: list[dict], causal_conv1d_reference: list[dict]
) -> None:
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6, dtype=torch.bfloat16), torch.zeros(4, 1, 2, 2))
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 6, dtype=torch.bfloat16),
        torch.ones(1, 6, 1),
        *_kda_inputs(),
        raw_gate_proj=torch.zeros(1, 6, 1, 2),
    )

    assert output.shape == (1, 6, 1, 2)
    assert len(causal_conv1d_reference) == 1
    assert causal_conv1d_reference[0]["query_start_loc"] == [0, 1, 3, 6]
    assert causal_conv1d_reference[0]["run_mode"] == 0
    assert [call["op"] for call in kda_test_environment] == ["chunk"] * 3
    assert [call["cu_seqlens"].tolist() for call in kda_test_environment] == [[0, 1], [0, 2], [0, 3]]


def _verify_metadata(slots: torch.Tensor, width: int) -> SimpleNamespace:
    per_row_slots = slots.repeat_interleave(width)
    return SimpleNamespace(
        linear_state_indices=per_row_slots,
        is_prefill=False,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.arange(per_row_slots.numel() + 1, dtype=torch.int32),
    )


@pytest.mark.parametrize("widths", [(1, 2, 4), (2, 4, 4), (4, 4, 4), (1, 4, 4)])
def test_consecutive_verify_widths_preserve_accepted_state(
    kda_test_environment: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    widths: tuple[int, ...],
) -> None:
    from fla_npu.ops import ascendc

    conv_cache = torch.zeros(4, 5, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    slots = torch.tensor([3, 1])
    base_indices = slots * 4
    ssm_cache[base_indices] = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
    expected_ssm = ssm_cache[base_indices].clone()
    previous_checkpoints = None
    previous_width = 1

    def _make_recurrent_kda(
        expected_state: torch.Tensor,
        current_width: int,
        expected_accepted: torch.Tensor,
    ) -> Any:
        def recurrent_kda(
            query: torch.Tensor,
            *_args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state_indices = kwargs["ssm_state_indices"].long()
            kernel_accepted = kwargs["num_accepted_tokens"].long()
            assert torch.equal(kernel_accepted.cpu(), expected_accepted.long())
            assert bool(((kernel_accepted >= 1) & (kernel_accepted <= current_width)).all())
            selected_indices = state_indices[
                torch.arange(2),
                kernel_accepted - 1,
            ]
            state_pool = kwargs["initial_state"]
            initial_state = state_pool[selected_indices].clone()
            torch.testing.assert_close(initial_state, expected_state, rtol=0, atol=0)
            for row in range(current_width):
                state_pool.index_copy_(
                    0,
                    state_indices[:, row],
                    initial_state + row + 1,
                )
            return torch.zeros_like(query), state_pool

        return recurrent_kda

    for width in widths:
        accepted = torch.ones(2, dtype=torch.int32)
        if previous_checkpoints is not None:
            accepted = torch.tensor([previous_width, 1], dtype=torch.int32)
            expected_ssm = previous_checkpoints[
                torch.arange(2),
                accepted.to(torch.int64) - 1,
            ]

        monkeypatch.setattr(
            ascendc,
            "recurrent_kda",
            _make_recurrent_kda(expected_ssm, width, accepted),
        )
        metadata = _verify_metadata(slots, width)
        metadata.num_accepted_tokens = accepted
        output = backend._spec_verify(
            torch.ones(1, 6, 2 * width, dtype=torch.bfloat16),
            torch.zeros(1, 2 * width, 1, 2),
            torch.ones(1, 2 * width, 1),
            _kda_inputs()[1],
            "silu",
            slots,
            metadata,
            conv_cache,
            ssm_cache,
            ascendc.recurrent_kda,
        )

        assert output.shape == (1, 2 * width, 1, 2)
        checkpoint_indices = base_indices[:, None] + torch.arange(width)
        previous_checkpoints = ssm_cache[checkpoint_indices].clone()
        expected_checkpoints = expected_ssm[:, None] + torch.arange(1, width + 1).view(
            1,
            width,
            1,
            1,
            1,
        )
        torch.testing.assert_close(previous_checkpoints, expected_checkpoints)
        previous_width = width


def test_verify_rejects_width_above_capacity_before_mutation(kda_test_environment: list[dict]) -> None:
    conv_cache = torch.ones(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.ones(4, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _verify_metadata(torch.tensor([1, 3]), 5)

    with pytest.raises(RuntimeError, match="conv checkpoint capacity"):
        backend.execute_linear(
            torch.ones(1, 6, 10, dtype=torch.bfloat16),
            torch.ones(1, 10, 1),
            *_kda_inputs(),
            raw_gate_proj=torch.zeros(1, 10, 1, 2),
        )

    assert not kda_test_environment
    assert torch.equal(conv_cache, torch.ones_like(conv_cache))
    assert torch.equal(ssm_cache, torch.ones_like(ssm_cache))


def test_spec_conv_treats_reserved_padding_slot_as_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.kernels_npu.causal_conv1d import causal_conv1d_update_v2

    captured_pad_slot_ids = []

    def native_conv(*args: Any) -> torch.Tensor:
        captured_pad_slot_ids.append(args[8])
        return args[0]

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_update_v2", native_conv, raising=False)
    causal_conv1d_update_v2(
        torch.ones(2, 6),
        torch.ones(6, 3),
        torch.zeros(2, 5, 6),
        torch.tensor([1, 0]),
        torch.tensor([0, 1, 2]),
        1,
        torch.ones(2, dtype=torch.int32),
    )

    assert captured_pad_slot_ids == [-1]


@pytest.mark.parametrize(("query_lengths", "slot_repeats"), [((2, 2), 2), ((1, 1, 1, 1), 1)])
def test_checkpoint_indices_follow_forward_lifetime(
    kda_test_environment: list[dict], query_lengths: tuple[int, ...], slot_repeats: int
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    backend = NpuPagedAttentionBackend(
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        scale=1.0,
        sliding_window=-1,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16),
                value=torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16),
            ),
            LayerCache(
                key=None, value=None, conv=torch.zeros(4, 5, 6, dtype=torch.bfloat16), ssm=torch.zeros(16, 1, 2, 2)
            ),
            LayerCache(
                key=None, value=None, conv=torch.zeros(4, 5, 6, dtype=torch.bfloat16), ssm=torch.zeros(16, 1, 2, 2)
            ),
        ]
    )
    metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 3], dtype=torch.int32).repeat_interleave(slot_repeats),
        num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32),
        q_seq_lens=torch.tensor(query_lengths),
        q_cu_seq_lens=torch.cat((torch.zeros(1, dtype=torch.int32), torch.tensor(query_lengths).cumsum(0))),
        kv_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=True,
        expanded_decode_metadata=None,
    )
    backend._metadata = metadata
    capture_cache: dict[object, object] = {}
    for cache in (capture_cache.copy(), capture_cache):
        context = ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches, layer_shared_cache=cache)
        with forward_context(context):
            for layer_id in (1, 2):
                backend.execute_linear(
                    torch.ones(1, 6, 4, dtype=torch.bfloat16),
                    torch.ones(1, 4, 1),
                    layer_id,
                    *_kda_inputs()[1:],
                    raw_gate_proj=torch.zeros(1, 4, 1, 2),
                )
        # Indices are derived per call from the per-step slot ids (no
        # cross-layer tensor reuse), so compare values.
    captured_indices = kda_test_environment[-1]["ssm_state_indices"]
    assert captured_indices.tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]

    metadata.linear_state_indices.copy_(torch.tensor([2, 0]).repeat_interleave(slot_repeats))
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        backend.execute_linear(
            torch.ones(1, 6, 4, dtype=torch.bfloat16),
            torch.ones(1, 4, 1),
            1,
            *_kda_inputs()[1:],
            raw_gate_proj=torch.zeros(1, 4, 1, 2),
        )
    next_indices = kda_test_environment[-1]["ssm_state_indices"]
    assert next_indices.tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]

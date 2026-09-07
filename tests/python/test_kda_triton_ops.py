# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Numerical and signature checks for the local SGLang-compatible KDA Triton implementations."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA Triton tests require CUDA", allow_module_level=True)

_ROOT = Path(__file__).parents[2] / "xllm" / "python"
_package = types.ModuleType("xllm.python.kernels_cuda")
_package.__path__ = [str(_ROOT / "kernels_cuda")]
sys.modules["xllm.python.kernels_cuda"] = _package

from xllm.python.kernels_cuda.triton.fla.kda import (  # noqa: E402
    RCP_LN2,
    chunk_kda,
    kda_gate_chunk_cumsum,
)
from xllm.python.kernels_cuda.triton.kda import (  # noqa: E402
    fused_recurrent_kda_packed_decode,
)


def _gate_reference(raw, a_log, dt_bias, chunk_size=64, lower_bound=None):
    x = raw.float() + dt_bias.float().view(1, 1, raw.shape[2], raw.shape[3])
    if lower_bound is None:
        gate = -torch.exp(a_log.float()).view(1, 1, raw.shape[2], 1) * torch.nn.functional.softplus(x)
    else:
        gate = lower_bound * torch.sigmoid(torch.exp(a_log.float()).view(1, 1, raw.shape[2], 1) * x)
    out = torch.empty_like(gate)
    for b in range(raw.shape[0]):
        for start in range(0, raw.shape[1], chunk_size):
            out[b, start : start + chunk_size] = gate[b, start : start + chunk_size].cumsum(0)
    return out


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("varlen", [False, True])
def test_kda_gate_chunk_cumsum_matches_fp32_reference(lower_bound, varlen):
    torch.manual_seed(11)
    lengths = [17, 64, 65, 81] if varlen else [81, 81]
    total = sum(lengths)
    raw = torch.randn(1, total, 2, 7, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(2, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(2 * 7, device="cuda", dtype=torch.float32) * 0.1
    cu = (
        torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        if varlen
        else None
    )
    actual = kda_gate_chunk_cumsum(raw, a_log, 64, dt_bias=dt_bias, cu_seqlens=cu, lower_bound=lower_bound)
    x = raw.float() + dt_bias.view(1, 1, 2, 7)
    gate = (
        lower_bound * torch.sigmoid(torch.exp(a_log).view(1, 1, 2, 1) * x)
        if lower_bound is not None
        else -torch.exp(a_log).view(1, 1, 2, 1) * torch.nn.functional.softplus(x)
    )
    expected = torch.empty_like(gate)
    if cu is None:
        for start in range(0, total, 64):
            expected[:, start : min(start + 64, total)] = gate[:, start : min(start + 64, total)].cumsum(1)
    else:
        for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
            for c in range(start, end, 64):
                expected[:, c : min(c + 64, end)] = gate[:, c : min(c + 64, end)].cumsum(1)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=2e-3)


def test_kda_gate_chunk_cumsum_converts_natural_log_to_log2():
    torch.manual_seed(13)
    raw = torch.randn(1, 65, 2, 7, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(2, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(2 * 7, device="cuda", dtype=torch.float32) * 0.1
    actual = kda_gate_chunk_cumsum(
        raw,
        a_log,
        64,
        scale=RCP_LN2,
        dt_bias=dt_bias,
    )
    expected = _gate_reference(raw, a_log, dt_bias) * RCP_LN2
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=2e-3)


def _decode_reference(mixed, a, b, a_log, dt_bias, state, indices, scale, lower_bound=None, norm=False):
    B = mixed.shape[0]
    HV, V, K = state.shape[-3:]
    H = (mixed.shape[1] - HV * V) // (2 * K)
    q = mixed[:, : H * K].view(B, H, K).float()
    k = mixed[:, H * K : 2 * H * K].view(B, H, K).float()
    v = mixed[:, 2 * H * K :].view(B, HV, V).float()
    if norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(HV // H, 1)
    k = k.repeat_interleave(HV // H, 1)
    out = torch.zeros(B, 1, HV, V, device=mixed.device, dtype=mixed.dtype)
    for i in range(B):
        idx = int(indices[i])
        if idx < 0:
            continue
        h = state[idx].float()
        x = a[i].float().view(HV, K) + dt_bias.float().view(HV, K)
        gate = (
            lower_bound * torch.sigmoid(torch.exp(a_log.float())[:, None] * x)
            if lower_bound is not None
            else -torch.exp(a_log.float())[:, None] * torch.nn.functional.softplus(x)
        )
        h = h * gate.exp()[:, None, :]
        delta = (v[i] - torch.einsum("hvk,hk->hv", h, k[i])) * torch.sigmoid(b[i].float())[:, None]
        h = h + delta[:, :, None] * k[i][:, None, :]
        state[idx].copy_(h)
        out[i, 0] = torch.einsum("hvk,hk->hv", h, q[i] * scale).to(mixed.dtype)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("norm", [False, True])
def test_fused_recurrent_kda_packed_decode_matches_fp32_reference(dtype, lower_bound, norm):
    torch.manual_seed(19)
    B, H, HV, K, V = 4, 2, 4, 7, 9
    mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=dtype).contiguous()
    a = torch.randn(B, HV * K, device="cuda", dtype=dtype).contiguous()
    b = torch.randn(B, HV, device="cuda", dtype=dtype).contiguous()
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32)
    indices = torch.tensor([0, 1, -1, 2], device="cuda", dtype=torch.int32)
    state = torch.randn(3, HV, V, K, device="cuda", dtype=torch.float32)
    expected_state = state.clone()
    expected = _decode_reference(mixed, a, b, a_log, dt_bias, expected_state, indices, K**-0.5, lower_bound, norm=norm)
    actual = torch.empty(B, 1, HV, V, device="cuda", dtype=dtype)
    result, returned_state = fused_recurrent_kda_packed_decode(
        mixed, a, b, a_log, dt_bias, K**-0.5, state, actual, indices, norm, lower_bound
    )
    assert result.data_ptr() == actual.data_ptr()
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(returned_state, expected_state, rtol=2e-2, atol=2e-2)


def test_fused_recurrent_kda_decode_rejects_wrong_output_contract():
    with pytest.raises(ValueError, match="shape"):
        fused_recurrent_kda_packed_decode(
            torch.empty(1, 12, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 8, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 2, device="cuda", dtype=torch.bfloat16),
            torch.empty(2, device="cuda"),
            torch.empty(8, device="cuda"),
            0.125,
            torch.empty(2, 2, 2, 4, device="cuda"),
            torch.empty(1, 1, 1, 2, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, device="cuda", dtype=torch.int32),
        )


def _chunk_reference(
    q,
    k,
    v,
    raw_g,
    beta,
    a_log,
    dt_bias,
    state,
    lengths,
    lower_bound,
    beta_is_raw=False,
    normalize_qk=False,
):
    q = q.float()
    k = k.float()
    if normalize_qk:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v = v.float()
    if a_log is None:
        gate = raw_g.float()
    else:
        x = raw_g.float() + dt_bias.float().view(1, 1, v.shape[2], raw_g.shape[3])
        gate = (
            lower_bound * torch.sigmoid(torch.exp(a_log).view(1, 1, v.shape[2], 1) * x)
            if lower_bound is not None
            else -torch.exp(a_log).view(1, 1, v.shape[2], 1) * torch.nn.functional.softplus(x)
        )
    beta = beta.float().sigmoid() if beta_is_raw else beta.float()
    out = torch.empty_like(v)
    offset = 0
    for seq, length in enumerate(lengths):
        h = state[seq].float()
        for t in range(offset, offset + length):
            h = h * gate[0, t].exp()[:, None, :]
            delta = (v[0, t] - torch.einsum("hvk,hk->hv", h, k[0, t])) * beta[0, t].float()[:, None]
            h = h + delta[:, :, None] * k[0, t][:, None, :]
            out[0, t] = torch.einsum("hvk,hk->hv", h, q[0, t]) * q.shape[-1] ** -0.5
        state[seq].copy_(h)
        offset += length
    return out


@pytest.mark.parametrize(
    ("mode", "lengths"),
    [
        ("fixed", [17, 17]),
        ("fixed", [64, 64]),
        ("fixed", [65, 65]),
        ("fixed", [81, 81]),
        ("varlen", [17, 64]),
    ],
)
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_chunk_kda_matches_fp32_reference_fixed_and_varlen(mode, lengths, lower_bound):
    torch.manual_seed(23)
    H = HV = 2
    K, V = 7, 9
    is_varlen = mode == "varlen"
    batch = 1 if is_varlen else len(lengths)
    tokens = sum(lengths) if is_varlen else lengths[0]
    q = torch.randn(batch, tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, HV, V, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(batch, tokens, HV, K, device="cuda", dtype=torch.bfloat16) * 0.2
    beta = torch.sigmoid(torch.randn(batch, tokens, HV, device="cuda", dtype=torch.bfloat16))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state = torch.randn(len(lengths), HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    if is_varlen:
        expected = _chunk_reference(q, k, v, raw_g, beta, a_log, dt_bias, expected_state, lengths, lower_bound)
    else:
        expected_rows = []
        for sequence in range(batch):
            row_state = expected_state[sequence : sequence + 1]
            expected_rows.append(
                _chunk_reference(
                    q[sequence : sequence + 1],
                    k[sequence : sequence + 1],
                    v[sequence : sequence + 1],
                    raw_g[sequence : sequence + 1],
                    beta[sequence : sequence + 1],
                    a_log,
                    dt_bias,
                    row_state,
                    [tokens],
                    lower_bound,
                )
            )
        expected = torch.cat(expected_rows, dim=0)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        cu_seqlens=cu if is_varlen else None,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(state, expected_state, rtol=6e-2, atol=6e-2)


def test_chunk_kda_matches_fp32_reference_k128_v128():
    torch.manual_seed(31)
    H, HV, K, V, T = 1, 1, 128, 128, 17
    q = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.03
    raw_g = torch.randn(1, T, HV, K, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(torch.randn(1, T, HV, device="cuda", dtype=torch.bfloat16))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state = torch.randn(1, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    expected_state = state.clone()
    expected = _chunk_reference(q, k, v, raw_g, beta, a_log, dt_bias, expected_state, [T], -5.0)
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(state, expected_state, rtol=7e-2, atol=7e-2)


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_chunk_kda_matches_fp32_reference_gqa(varlen, lower_bound):
    torch.manual_seed(33)
    H, HV, K, V = 1, 2, 7, 9
    lengths = [65, 65] if not varlen else [17, 81]
    batch = 1 if varlen else len(lengths)
    tokens = sum(lengths) if varlen else lengths[0]
    q = torch.randn(batch, tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    # SGLang expects the chunk caller to expand Q/K for GQA.
    q = q.repeat_interleave(HV // H, dim=-2)
    k = k.repeat_interleave(HV // H, dim=-2)
    v = torch.randn(batch, tokens, HV, V, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(batch, tokens, HV, K, device="cuda", dtype=torch.bfloat16) * 0.2
    beta = torch.sigmoid(torch.randn(batch, tokens, HV, device="cuda", dtype=torch.bfloat16))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state = torch.randn(len(lengths), HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    if varlen:
        expected = _chunk_reference(q, k, v, raw_g, beta, a_log, dt_bias, expected_state, lengths, lower_bound)
    else:
        expected_rows = []
        for sequence in range(batch):
            expected_rows.append(
                _chunk_reference(
                    q[sequence : sequence + 1],
                    k[sequence : sequence + 1],
                    v[sequence : sequence + 1],
                    raw_g[sequence : sequence + 1],
                    beta[sequence : sequence + 1],
                    a_log,
                    dt_bias,
                    expected_state[sequence : sequence + 1],
                    [tokens],
                    lower_bound,
                )
            )
        expected = torch.cat(expected_rows, dim=0)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        cu_seqlens=cu if varlen else None,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(state, expected_state, rtol=7e-2, atol=7e-2)


def _chunk_reference_with_intermediate_states(q, k, v, raw_g, beta, a_log, dt_bias, state, lengths, lower_bound):
    q = q.float()
    k = k.float()
    v = v.float()
    x = raw_g.float() + dt_bias.float().view(1, 1, v.shape[2], raw_g.shape[3])
    gate = (
        lower_bound * torch.sigmoid(torch.exp(a_log).view(1, 1, v.shape[2], 1) * x)
        if lower_bound is not None
        else -torch.exp(a_log).view(1, 1, v.shape[2], 1) * torch.nn.functional.softplus(x)
    )
    out = torch.empty_like(v)
    snapshots = []
    offset = 0
    for seq, length in enumerate(lengths):
        h = state[seq].float()
        for chunk_start in range(0, length, 64):
            snapshots.append(h.clone())
            for t in range(offset + chunk_start, offset + min(chunk_start + 64, length)):
                h = h * gate[0, t].exp()[:, None, :]
                delta = (v[0, t] - torch.einsum("hvk,hk->hv", h, k[0, t])) * beta[0, t].float()[:, None]
                h = h + delta[:, :, None] * k[0, t][:, None, :]
                out[0, t] = torch.einsum("hvk,hk->hv", h, q[0, t]) * q.shape[-1] ** -0.5
        state[seq].copy_(h)
        offset += length
    return out, torch.stack(snapshots).unsqueeze(0).to(state.dtype)


def test_chunk_kda_returns_sglang_intermediate_chunk_states():
    torch.manual_seed(37)
    H, HV, K, V = 1, 1, 7, 9
    lengths = [17, 64]
    total = sum(lengths)
    q = torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, HV, V, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(1, total, HV, K, device="cuda", dtype=torch.bfloat16) * 0.2
    beta = torch.sigmoid(torch.randn(1, total, HV, device="cuda", dtype=torch.bfloat16))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state = torch.randn(2, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    expected, expected_h = _chunk_reference_with_intermediate_states(
        q, k, v, raw_g, beta, a_log, dt_bias, expected_state, lengths, None
    )
    cu = torch.tensor([0, 17, 81], device="cuda", dtype=torch.int32)
    actual, actual_h = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(2, device="cuda", dtype=torch.int32),
        cu_seqlens=cu,
        A_log=a_log,
        dt_bias=dt_bias,
        output_intermediate_states=True,
    )
    assert actual_h.shape == (1, 2, HV, V, K)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(actual_h.float(), expected_h.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(state, expected_state, rtol=6e-2, atol=6e-2)


def test_fused_recurrent_kda_packed_decode_mutates_same_state_slot_sequentially():
    torch.manual_seed(41)
    B, H, HV, K, V = 1, 1, 2, 7, 9
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.1
    state = torch.randn(1, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    for _ in range(2):
        mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(B, HV * K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
        expected = _decode_reference(
            mixed,
            a,
            b,
            a_log,
            dt_bias,
            expected_state,
            torch.zeros(B, device="cuda", dtype=torch.int32),
            K**-0.5,
            norm=True,
        )
        actual = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
        result, returned_state = fused_recurrent_kda_packed_decode(
            mixed,
            a,
            b,
            a_log,
            dt_bias,
            K**-0.5,
            state,
            actual,
            torch.zeros(B, device="cuda", dtype=torch.int32),
            True,
        )
        torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(returned_state, expected_state, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_fused_recurrent_kda_packed_decode_is_cuda_graph_capturable(lower_bound):
    torch.manual_seed(43)
    B, H, HV, K, V = 2, 1, 2, 7, 9
    dtype = torch.bfloat16
    mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=dtype).contiguous()
    a = torch.randn(B, HV * K, device="cuda", dtype=dtype).contiguous()
    b = torch.randn(B, HV, device="cuda", dtype=dtype).contiguous()
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32)
    indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    state = torch.randn(3, HV, V, K, device="cuda", dtype=torch.float32)
    out = torch.empty(B, 1, HV, V, device="cuda", dtype=dtype)

    # Compile the Triton kernel before capture.  All index inspection is done
    # by the caller before graph capture; replay only launches device kernels.
    fused_recurrent_kda_packed_decode(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        K**-0.5,
        state,
        out,
        indices,
        True,
        lower_bound,
    )
    torch.cuda.synchronize()

    initial_state = state.clone()
    expected_state = initial_state.clone()
    expected = _decode_reference(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        expected_state,
        indices,
        K**-0.5,
        lower_bound,
        norm=True,
    )
    state.copy_(initial_state)
    out.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_state = fused_recurrent_kda_packed_decode(
            mixed,
            a,
            b,
            a_log,
            dt_bias,
            K**-0.5,
            state,
            out,
            indices,
            True,
            lower_bound,
        )

    graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == out.data_ptr()
    assert graph_state.data_ptr() == state.data_ptr()
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("varlen", [False, True])
def test_chunk_kda_accepts_preactivated_gates_without_a_log(varlen):
    torch.manual_seed(47)
    H = HV = 1
    K, V = 7, 9
    lengths = [17, 65] if varlen else [65, 65]
    batch = 1 if varlen else len(lengths)
    tokens = sum(lengths) if varlen else lengths[0]
    q = torch.randn(batch, tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, HV, V, device="cuda", dtype=torch.bfloat16) * 0.1
    activated_gate = -torch.nn.functional.softplus(
        torch.randn(batch, tokens, HV, K, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(torch.randn(batch, tokens, HV, device="cuda", dtype=torch.bfloat16))
    state = torch.randn(len(lengths), HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()

    if varlen:
        expected = _chunk_reference(
            q,
            k,
            v,
            activated_gate,
            beta,
            None,
            None,
            expected_state,
            lengths,
            None,
        )
    else:
        expected_rows = []
        for sequence in range(batch):
            expected_rows.append(
                _chunk_reference(
                    q[sequence : sequence + 1],
                    k[sequence : sequence + 1],
                    v[sequence : sequence + 1],
                    activated_gate[sequence : sequence + 1],
                    beta[sequence : sequence + 1],
                    None,
                    None,
                    expected_state[sequence : sequence + 1],
                    [tokens],
                    None,
                )
            )
        expected = torch.cat(expected_rows, dim=0)

    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=activated_gate,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        cu_seqlens=cu if varlen else None,
        A_log=None,
        dt_bias=None,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(state, expected_state, rtol=6e-2, atol=6e-2)


@pytest.mark.parametrize("varlen", [False, True])
def test_chunk_kda_beta_is_raw_matches_activated_beta(varlen):
    torch.manual_seed(53)
    H = HV = 1
    K, V = 7, 9
    lengths = [17, 65] if varlen else [65, 65]
    batch = 1 if varlen else len(lengths)
    tokens = sum(lengths) if varlen else lengths[0]
    q = torch.randn(batch, tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, HV, V, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_gate = torch.randn(batch, tokens, HV, K, device="cuda", dtype=torch.bfloat16) * 0.2
    raw_beta = torch.randn(batch, tokens, HV, device="cuda", dtype=torch.bfloat16)
    activated_beta = raw_beta.float().sigmoid()
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state_raw = torch.randn(len(lengths), HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    state_activated = state_raw.clone()
    indices = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    raw_result = chunk_kda(
        q=q,
        k=k,
        v=v.clone(),
        g=raw_gate,
        beta=raw_beta,
        initial_state=state_raw,
        initial_state_indices=indices,
        cu_seqlens=cu if varlen else None,
        A_log=a_log,
        dt_bias=dt_bias,
        beta_is_raw=True,
    )
    activated_result = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=activated_beta,
        initial_state=state_activated,
        initial_state_indices=indices,
        cu_seqlens=cu if varlen else None,
        A_log=a_log,
        dt_bias=dt_bias,
        beta_is_raw=False,
    )
    torch.testing.assert_close(raw_result.float(), activated_result.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize_qk", [False, True])
def test_chunk_kda_matches_fp32_reference_for_dtype_and_qk_norm(dtype, normalize_qk):
    torch.manual_seed(59)
    H = HV = 1
    K, V, T = 7, 9, 65
    q = torch.randn(1, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, T, HV, V, device="cuda", dtype=dtype) * 0.1
    raw_gate = torch.randn(1, T, HV, K, device="cuda", dtype=dtype) * 0.2
    beta = torch.sigmoid(torch.randn(1, T, HV, device="cuda", dtype=dtype))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    state = torch.randn(1, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    expected = _chunk_reference(
        q,
        k,
        v,
        raw_gate,
        beta,
        a_log,
        dt_bias,
        expected_state,
        [T],
        None,
        normalize_qk=normalize_qk,
    )
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        A_log=a_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=normalize_qk,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(state, expected_state, rtol=6e-2, atol=6e-2)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("tokens", [4096, 4161])
def test_chunk_kda_exercises_small_grid_threshold(lower_bound, tokens):
    torch.manual_seed(61)
    batch, H = 4, 1
    HV, K, V = 1, 17, 17
    q = torch.randn(batch, tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, HV, V, device="cuda", dtype=torch.bfloat16) * 0.03
    raw_gate = torch.randn(batch, tokens, HV, K, device="cuda", dtype=torch.bfloat16) * 0.05
    beta = torch.sigmoid(torch.randn(batch, tokens, HV, device="cuda", dtype=torch.bfloat16))
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.03
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.03
    state = torch.randn(batch, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    expected_state = state.clone()
    expected_rows = []
    for sequence in range(batch):
        expected_rows.append(
            _chunk_reference(
                q[sequence : sequence + 1],
                k[sequence : sequence + 1],
                v[sequence : sequence + 1],
                raw_gate[sequence : sequence + 1],
                beta[sequence : sequence + 1],
                a_log,
                dt_bias,
                expected_state[sequence : sequence + 1],
                [tokens],
                lower_bound,
            )
        )
    expected = torch.cat(expected_rows, dim=0)
    actual = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-1, atol=5e-1)
    torch.testing.assert_close(state, expected_state, rtol=1e-1, atol=5e-1)


@pytest.mark.parametrize("batch", [32, 128])
def test_fused_recurrent_kda_packed_decode_larger_batches(batch):
    torch.manual_seed(67)
    H, HV, K, V = 1, 2, 7, 9
    dtype = torch.bfloat16
    mixed = torch.randn(batch, 2 * H * K + HV * V, device="cuda", dtype=dtype)
    a = torch.randn(batch, HV * K, device="cuda", dtype=dtype)
    b = torch.randn(batch, HV, device="cuda", dtype=dtype)
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    state = torch.randn(batch, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    expected_state = state.clone()
    expected = _decode_reference(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        expected_state,
        indices,
        K**-0.5,
        norm=True,
    )
    actual = torch.empty(batch, 1, HV, V, device="cuda", dtype=dtype)
    result, returned_state = fused_recurrent_kda_packed_decode(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        K**-0.5,
        state,
        actual,
        indices,
        True,
    )
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(returned_state, expected_state, rtol=2e-2, atol=2e-2)


def test_fused_recurrent_kda_packed_decode_supports_state_pool_stride():
    torch.manual_seed(71)
    B, H, HV, K, V = 2, 1, 2, 7, 9
    dtype = torch.float16
    mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=dtype)
    a = torch.randn(B, HV * K, device="cuda", dtype=dtype)
    b = torch.randn(B, HV, device="cuda", dtype=dtype)
    a_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HV * K, device="cuda", dtype=torch.float32) * 0.05
    indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    state_storage = torch.randn(5, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    state = state_storage[::2]
    expected_state = state.clone()
    expected = _decode_reference(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        expected_state,
        indices,
        K**-0.5,
        norm=True,
    )
    actual = torch.empty(B, 1, HV, V, device="cuda", dtype=dtype)
    result, returned_state = fused_recurrent_kda_packed_decode(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        K**-0.5,
        state,
        actual,
        indices,
        True,
    )
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(returned_state, expected_state, rtol=2e-2, atol=2e-2)


def test_kda_public_signatures_match_sglang():
    decode = inspect.signature(fused_recurrent_kda_packed_decode).parameters
    assert tuple(decode) == (
        "mixed_qkv",
        "a",
        "b",
        "A_log",
        "dt_bias",
        "scale",
        "initial_state",
        "out",
        "ssm_state_indices",
        "use_qk_l2norm_in_kernel",
        "lower_bound",
    )
    assert decode["use_qk_l2norm_in_kernel"].default is False
    assert decode["lower_bound"].default is None

    cumsum = inspect.signature(kda_gate_chunk_cumsum).parameters
    assert tuple(cumsum) == (
        "g",
        "A_log",
        "chunk_size",
        "scale",
        "dt_bias",
        "cu_seqlens",
        "output_dtype",
        "chunk_indices",
        "lower_bound",
    )
    assert cumsum["scale"].default is None
    assert cumsum["dt_bias"].default is None
    assert cumsum["cu_seqlens"].default is None
    assert cumsum["output_dtype"].default == torch.float
    assert cumsum["chunk_indices"].default is None
    assert cumsum["lower_bound"].default is None

    prefill = inspect.signature(chunk_kda).parameters
    assert tuple(prefill) == (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "initial_state_indices",
        "use_qk_l2norm_in_kernel",
        "cu_seqlens",
        "A_log",
        "dt_bias",
        "lower_bound",
        "output_intermediate_states",
        "beta_is_raw",
        "kwargs",
    )
    assert prefill["scale"].default is None
    assert prefill["initial_state"].default is None
    assert prefill["initial_state_indices"].default is None
    assert prefill["use_qk_l2norm_in_kernel"].default is False
    assert prefill["cu_seqlens"].default is None
    assert prefill["A_log"].default is None
    assert prefill["dt_bias"].default is None
    assert prefill["lower_bound"].default is None
    assert prefill["output_intermediate_states"].default is False
    assert prefill["beta_is_raw"].default is False
    assert prefill["kwargs"].kind is inspect.Parameter.VAR_KEYWORD

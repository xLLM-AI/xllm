# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Real GDN outputs, full state pools, checkpoints and changed-input replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import triton.runtime.jit
from xllm_kernel.ops import (
    fused_qkvzba_split_reshape_cat,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)


@pytest.fixture(autouse=True)
def _require_startup_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_compile(*args: Any, **kwargs: Any) -> None:
        pytest.fail("GDN launch compiled a specialization after startup preparation")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "_do_compile", _unexpected_compile)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    normalize: bool,
    scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Independent CPU sequence recurrence using batched matrix products."""
    q, k, v, g, beta = (tensor.cpu().float() for tensor in (q, k, v, g, beta))
    repeat = v.shape[1] // q.shape[1]
    q = q.repeat_interleave(repeat, dim=1)
    k = k.repeat_interleave(repeat, dim=1)
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = initial.clone()
    outputs, checkpoints = [], []
    for token in range(q.shape[0]):
        state = state * g[token].exp()[:, None, None]
        prediction = torch.bmm(k[token, :, None, :], state).squeeze(1)
        residual = (v[token] - prediction) * beta[token, :, None]
        state = state + torch.bmm(k[token, :, :, None], residual[:, None, :])
        outputs.append(torch.bmm((q[token] * scale)[:, None, :], state).squeeze(1))
        checkpoints.append(state.clone())
    return torch.stack(outputs).bfloat16(), checkpoints


def _inputs(tokens: int, heads: int, value_heads: int) -> tuple[torch.Tensor, ...]:
    return (
        (torch.randn(1, tokens, heads, 128) * 0.05).bfloat16().npu(),
        (torch.randn(1, tokens, heads, 128) * 0.05).bfloat16().npu(),
        (torch.randn(1, tokens, value_heads, 128) * 0.2).bfloat16().npu(),
        (-torch.rand(1, tokens, value_heads) * 0.1).npu(),
        torch.rand(1, tokens, value_heads).bfloat16().npu(),
    )


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode", ["decode", "final_state", "verify"])
@pytest.mark.parametrize("graph", [False, True])
def test_recurrence_state_and_accepted_checkpoints(ratio: int, normalize: bool, mode: str, graph: bool) -> None:
    torch.manual_seed(42)
    tokens = 2 if mode == "decode" else 5
    q, k, v, g, beta = _inputs(tokens, 1, ratio)
    baseline = torch.randn(12, ratio, 128, 128) * 0.02
    state = baseline.npu()
    lengths = torch.tensor([0, 1 if mode == "decode" else 2, tokens], dtype=torch.int64).npu()
    # A noncontiguous checkpoint view exercises both sequence and token strides.
    index_storage = torch.full((2, 6), -99, dtype=torch.int32).npu()
    indices = index_storage[:, ::2] if mode == "verify" else torch.tensor([4, 9], dtype=torch.int32).npu()
    accepted = torch.tensor([2, 3], dtype=torch.int32).npu() if mode == "verify" else None
    if mode == "verify":
        indices.copy_(torch.tensor([[4, 1, 8], [6, 3, 9]], dtype=torch.int32))
    scale = 0.2

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            state,
            lengths,
            indices,
            num_accepted_tokens=accepted,
            inplace_final_state=mode != "final_state",
            scale=scale,
            use_qk_l2norm_in_kernel=normalize,
        )

    stream = torch.npu.Stream()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        if graph:
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                output, final_state = _run()
        for round_index in range(3):
            q.copy_(torch.randn_like(q) * 0.05)
            state.copy_(baseline)
            if accepted is not None:
                accepted.copy_(torch.tensor([round_index + 1, 3 - round_index], dtype=torch.int32))
                if round_index == 1:
                    indices.copy_(torch.tensor([[8, 4, 1], [9, 6, 3]], dtype=torch.int32))
                lengths.copy_(torch.tensor([0, 2 + round_index % 2, tokens], dtype=torch.int64))
            input_before = tuple(tensor.cpu() for tensor in (q, k, v, g, beta, lengths, indices))
            if graph:
                captured.replay()
            else:
                output, final_state = _run()
            stream.synchronize()
            expected_pool = baseline.clone()
            expected_final, expected_output = [], []
            boundaries = lengths.cpu().tolist()
            index_cpu = indices.cpu()
            for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                slot = (
                    int(index_cpu[seq, int(accepted[seq].cpu()) - 1]) if accepted is not None else int(index_cpu[seq])
                )
                result, checkpoints = _reference(
                    q[0, start:end],
                    k[0, start:end],
                    v[0, start:end],
                    g[0, start:end],
                    beta[0, start:end],
                    baseline[slot],
                    normalize,
                    scale,
                )
                expected_output.append(result)
                expected_final.append(checkpoints[-1])
                if mode == "verify":
                    for token, checkpoint in enumerate(checkpoints):
                        expected_pool[int(index_cpu[seq, token])] = checkpoint
                elif mode == "decode":
                    expected_pool[slot] = checkpoints[-1]
            torch.testing.assert_close(output.cpu()[0], torch.cat(expected_output), rtol=0.016, atol=2e-4)
            torch.testing.assert_close(state.cpu(), expected_pool, rtol=5e-4, atol=2e-5)
            assert output.is_contiguous() and output.data_ptr() != state.data_ptr()
            if mode == "final_state":
                torch.testing.assert_close(final_state.cpu(), torch.stack(expected_final), rtol=5e-4, atol=2e-5)
                assert final_state.data_ptr() != state.data_ptr()
            else:
                assert final_state.data_ptr() == state.data_ptr()
            for tensor, before in zip((q, k, v, g, beta, lengths, indices), input_before):
                torch.testing.assert_close(tensor.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_split_layout_alias_and_replay(ratio: int, graph: bool) -> None:
    torch.manual_seed(43)
    tokens, heads, dim = 5, 2, 128
    value_heads = heads * ratio
    packed = torch.randn(tokens, heads * (2 * dim + 2 * ratio * dim), dtype=torch.bfloat16).npu()
    ba = torch.randn(tokens, value_heads * 2, dtype=torch.bfloat16).npu()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = fused_qkvzba_split_reshape_cat(packed, ba, heads, value_heads)
    for _ in range(3):
        packed.copy_(torch.randn_like(packed))
        ba.copy_(torch.randn_like(ba))
        before = packed.cpu(), ba.cpu()
        if graph:
            captured.replay()
        else:
            output = fused_qkvzba_split_reshape_cat(packed, ba, heads, value_heads)
        torch.npu.synchronize()
        # Reshape/split independently describes the per-head source layout.
        q, k, v, z = before[0].view(tokens, heads, -1).split([dim, dim, ratio * dim, ratio * dim], dim=-1)
        b, a = before[1].view(tokens, heads, 2 * ratio).chunk(2, dim=-1)
        expected = (
            torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1),
            z.reshape(tokens, value_heads, dim),
            b.reshape(tokens, value_heads),
            a.reshape(tokens, value_heads),
        )
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            assert actual.is_contiguous()
        assert len({tensor.data_ptr() for tensor in (*output, packed, ba)}) == 6
        torch.testing.assert_close(packed.cpu(), before[0], rtol=0, atol=0)
        torch.testing.assert_close(ba.cpu(), before[1], rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_fused_gating_varlen_negative_slot_and_full_state(normalize: bool, graph: bool) -> None:
    torch.manual_seed(44)
    q, k, v, _, _ = _inputs(5, 2, 6)
    a = torch.randn(1, 5, 6, dtype=torch.bfloat16).npu()
    b = torch.randn_like(a)
    a_log = torch.randn(6).npu()
    dt_bias = torch.randn(6).npu()
    baseline = torch.randn(8, 6, 128, 128) * 0.02
    state = baseline.npu()
    indices = torch.tensor([5, -1, 2], dtype=torch.int32).npu()
    lengths = torch.tensor([0, 2, 5, 5], dtype=torch.int32).npu()

    def _run() -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            a_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            state,
            indices,
            lengths,
            0.125,
            normalize,
            0.5,
            8.0,
        )

    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = _run()
    for round_index in range(3):
        state.copy_(baseline)
        a.copy_(torch.randn_like(a) * 5)
        a[0, 0, 0] = 40
        a[0, 1, 1] = -40
        indices.copy_(torch.tensor([5 - round_index, -1, 2], dtype=torch.int32))
        lengths.copy_(torch.tensor([0, round_index + 1, 5, 5], dtype=torch.int32))
        if graph:
            captured.replay()
        else:
            output = _run()
        torch.npu.synchronize()
        gate = -a_log.cpu().exp() * torch.nn.functional.softplus(
            a.cpu().float() + dt_bias.cpu(), beta=0.5, threshold=8.0
        )
        beta = b.cpu().float().sigmoid()
        boundaries = lengths.cpu().tolist()
        expected_pool, expected_output = baseline.clone(), []
        for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start == end:
                continue
            slot = int(indices[seq].cpu())
            initial = baseline[slot] if slot >= 0 else torch.zeros_like(baseline[0])
            result, checkpoints = _reference(
                q[0, start:end],
                k[0, start:end],
                v[0, start:end],
                gate[0, start:end],
                beta[0, start:end],
                initial,
                normalize,
                0.125,
            )
            expected_output.append(result)
            if slot >= 0:
                expected_pool[slot] = checkpoints[-1]
        torch.testing.assert_close(output.cpu()[0], torch.cat(expected_output), rtol=0.016, atol=2e-4)
        torch.testing.assert_close(state.cpu(), expected_pool, rtol=5e-4, atol=2e-5)


@pytest.mark.parametrize("invalid", ["q_dtype", "state_layout", "verify_copy", "g_shape"])
def test_rejected_recurrent_contract_leaves_state_unchanged(invalid: str) -> None:
    q, k, v, g, beta = _inputs(2, 1, 2)
    state = torch.randn(3, 2, 128, 128).npu()
    before = state.cpu()
    lengths = torch.tensor([0, 1, 2], dtype=torch.int64).npu()
    indices = torch.tensor([0, 2], dtype=torch.int32).npu()
    accepted = None
    if invalid == "q_dtype":
        q = q.half()
    elif invalid == "state_layout":
        state = state.transpose(-1, -2)
    elif invalid == "g_shape":
        g = g[:, :, :1].contiguous()
    else:
        accepted = torch.ones(2, dtype=torch.int32, device="npu")
        indices = indices[:, None]
    with pytest.raises(ValueError):
        fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            state,
            lengths,
            indices,
            num_accepted_tokens=accepted,
            inplace_final_state=invalid != "verify_copy",
        )
    actual = state.transpose(-1, -2) if invalid == "state_layout" else state
    torch.testing.assert_close(actual.cpu(), before, rtol=0, atol=0)


def test_unsupported_projection_ratio_is_rejected() -> None:
    packed = torch.randn(2, 1536, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(2, 10, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError, match="GQA ratio"):
        fused_qkvzba_split_reshape_cat(packed, ba, 1, 5)


def test_standalone_package_without_host_imports(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
from typing import Any

class BlockHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split('.')[0] == 'xllm':
            raise AssertionError(f'unexpected host dependency: {fullname}')

sys.meta_path.insert(0, BlockHost())
import torch
import torch_npu
from xllm_kernel import initialize
from xllm_kernel.ops import fused_qkvzba_split_reshape_cat, fused_recurrent_gated_delta_rule

operations = ('fused_recurrent_gated_delta_rule', 'fused_sigmoid_gating_delta_rule_update', 'fused_qkvzba_split_reshape_cat')
plans = initialize(device='npu', operations=operations)
assert all(plan.spec.binding == 'python' for plan in plans.values())
packed = torch.zeros(2, 768, dtype=torch.bfloat16, device='npu')
ba = torch.zeros(2, 4, dtype=torch.bfloat16, device='npu')
output = fused_qkvzba_split_reshape_cat(packed, ba, 1, 2)
assert all(tensor.count_nonzero().item() == 0 for tensor in output)
q = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16, device='npu')
v = torch.zeros(1, 2, 2, 128, dtype=torch.bfloat16, device='npu')
g = torch.full((1, 2, 2), -0.1, dtype=torch.float32, device='npu')
beta = torch.ones(1, 2, 2, dtype=torch.bfloat16, device='npu')
state = torch.ones(3, 2, 128, 128, dtype=torch.float32, device='npu')
cu = torch.arange(3, dtype=torch.int64, device='npu')
indices = torch.tensor([2, 0], dtype=torch.int32, device='npu')
output, final = fused_recurrent_gated_delta_rule(q, q, v, g, beta, state, cu, indices)
assert final.data_ptr() == state.data_ptr()
assert output.count_nonzero().item() == 0
expected = torch.ones_like(state)
expected[0] *= torch.exp(torch.tensor(-0.1))
expected[2] *= torch.exp(torch.tensor(-0.1))
torch.testing.assert_close(state, expected, rtol=5e-4, atol=2e-5)
assert not any(name == 'xllm' or name.startswith('xllm.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def test_unaligned_contiguous_inputs_reuse_startup_specializations() -> None:
    torch.manual_seed(46)
    q, k, v, g, beta = _inputs(2, 1, 2)

    def _offset(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.new_empty(tensor.numel() + 1)
        value = storage[1:].view_as(tensor)
        value.copy_(tensor)
        assert value.is_contiguous() and value.storage_offset() == 1
        return value

    state = torch.randn(3, 2, 128, 128).npu() * 0.02
    aligned_state = state.clone()
    lengths = torch.arange(3, dtype=torch.int64, device="npu")
    indices = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    expected, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, aligned_state, lengths, indices)
    actual, _ = fused_recurrent_gated_delta_rule(
        _offset(q), _offset(k), _offset(v), _offset(g), _offset(beta), state, lengths, indices
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, aligned_state, rtol=0, atol=0)
    packed = torch.randn(2, 768, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(2, 4, dtype=torch.bfloat16, device="npu")
    expected_split = fused_qkvzba_split_reshape_cat(packed, ba, 1, 2)
    actual_split = fused_qkvzba_split_reshape_cat(_offset(packed), _offset(ba), 1, 2)
    for actual_part, expected_part in zip(actual_split, expected_split):
        torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)

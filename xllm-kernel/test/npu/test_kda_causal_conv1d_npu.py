# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""KDA V2 outputs, full backing state and changed-metadata graph replay."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
import triton.runtime.jit
from xllm_kernel.ops.attention.kda import causal_conv1d_update_v2


@pytest.fixture(autouse=True)
def _require_startup_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_compile(*args: Any, **kwargs: Any) -> None:
        pytest.fail("KDA convolution compiled after startup preparation")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "_do_compile", _unexpected_compile)


def _reference(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    lengths: list[int],
    slots: list[int],
    accepted: list[int] | None,
    bias: torch.Tensor | None,
    activation: bool,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent grouped CPU convolution plus explicit checkpoint windows."""
    output = torch.zeros_like(tokens)
    valid = torch.zeros(tokens.shape[0], dtype=torch.bool)
    expected_state = state.clone()
    start = 0
    width = weight.shape[1]
    for seq, (length, slot) in enumerate(zip(lengths, slots)):
        end = start + length
        if length and slot != pad_slot_id:
            offset = 0 if accepted is None else accepted[seq] - 1
            history = state[slot, offset : offset + width - 1]
            window = torch.cat((history, tokens[start:end])).half().float()
            result = (
                F.conv1d(
                    window.t().unsqueeze(0),
                    weight.float().unsqueeze(1),
                    None if bias is None else bias.float(),
                    groups=tokens.shape[1],
                )
                .squeeze(0)
                .t()
                .bfloat16()
            )
            output[start:end] = F.silu(result) if activation else result
            valid[start:end] = True
            rows = width - 1 + (length - 1 if accepted is not None else 0)
            if rows:
                if length >= rows:
                    updated = tokens[end - rows : end]
                else:
                    shift = length if accepted is None else 1
                    prefix = state[slot, offset + shift : offset + shift + rows - length]
                    updated = torch.cat((prefix, tokens[start:end]))
                expected_state[slot, :rows] = updated
        start = end
    return output, expected_state, valid


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("mode", ["dense_decode", "dense_verify", "varlen", "packed_verify"])
@pytest.mark.parametrize("state_view", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_output_full_state_and_replay(width: int, mode: str, state_view: bool, graph: bool) -> None:
    torch.manual_seed(51)
    dim = 257 if width % 2 else 17
    packed = mode in ("varlen", "packed_verify")
    verify = mode in ("dense_verify", "packed_verify")
    max_len = 1 if mode == "dense_decode" else 3
    lengths = [2, 0, 3, 1] if packed else [max_len] * 4
    tokens = sum(lengths)
    capacity = width - 1 + (max_len - 1 if verify else 0) + 3
    baseline = (torch.randn(7, capacity, dim) * 0.2).bfloat16()
    # Transposing a channel-first allocation checks copy-back through a view.
    backing = baseline.transpose(1, 2).contiguous().npu() if state_view else baseline.npu()
    state = backing.transpose(1, 2) if state_view else backing
    value_storage = torch.randn(tokens, dim + 1, dtype=torch.bfloat16).npu()
    value = value_storage[:, 1:] if packed else torch.randn(4, dim, max_len, dtype=torch.bfloat16).npu()
    if mode == "dense_decode":
        value = value.squeeze(-1)
    weight = (torch.randn(width, dim) * 0.2).bfloat16().npu().t()
    bias = torch.randn(dim, dtype=torch.bfloat16).npu() if width % 2 else None
    activation = width % 3 != 0
    # A nonempty padded sequence tests masking independently of empty sequences.
    pad_slot_id = -1 if width % 2 else 6
    slot_values = [0, 4, 2, pad_slot_id]
    slot_storage = torch.full((4, 4), 5, dtype=torch.int64).npu()
    indices = slot_storage[:, ::2] if width % 2 else slot_storage[:, 0]
    if indices.ndim == 2:
        indices[:, 0].copy_(torch.tensor(slot_values))
    else:
        indices.copy_(torch.tensor(slot_values))
    qsl = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64).npu() if packed else None
    accepted = torch.ones(4, dtype=torch.int64).npu() if verify else None
    state_pointer = state.data_ptr()

    def _run() -> torch.Tensor:
        return causal_conv1d_update_v2(
            value, weight, state, indices, qsl, max_len, accepted, bias, activation, pad_slot_id
        )

    stream = torch.npu.Stream()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        if graph:
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                output = _run()
        for round_index in range(3):
            value.copy_(torch.randn_like(value) * 0.2)
            state.copy_(baseline)
            slot_values[:3] = [0, 4, 2] if round_index % 2 == 0 else [4, 2, 0]
            if indices.ndim == 2:
                indices[:, 0].copy_(torch.tensor(slot_values))
            else:
                indices.copy_(torch.tensor(slot_values))
            if packed:
                lengths = [2, 0, 3, 1] if round_index % 2 == 0 else [3, 1, 0, 2]
                qsl.copy_(torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]))
            counts = [1 + round_index, 3 - round_index, 2, 1] if verify else None
            if accepted is not None:
                accepted.copy_(torch.tensor(counts))
            before = tuple(t.cpu() for t in (value, weight, indices, slot_storage))
            if graph:
                captured.replay()
            else:
                output = _run()
            stream.synchronize()
            dense = value.unsqueeze(-1) if value.ndim == 2 and not packed else value
            reference_tokens = value.cpu() if packed else dense.transpose(1, 2).reshape(tokens, dim).cpu()
            expected, pool, valid = _reference(
                reference_tokens,
                weight.cpu(),
                baseline,
                lengths,
                slot_values,
                counts,
                None if bias is None else bias.cpu(),
                activation,
                pad_slot_id,
            )
            torch.testing.assert_close(output.cpu()[valid], expected[valid], rtol=0.016, atol=0.008)
            torch.testing.assert_close(state.cpu(), pool, rtol=0, atol=0)
            torch.testing.assert_close(backing.cpu(), pool.transpose(1, 2) if state_view else pool, rtol=0, atol=0)
            assert state.data_ptr() == state_pointer
            assert output.shape == (tokens, dim) and output.is_contiguous()
            assert output.data_ptr() not in {value.data_ptr(), weight.data_ptr(), state.data_ptr()}
            for tensor, original in zip((value, weight, indices, slot_storage), before):
                torch.testing.assert_close(tensor.cpu(), original, rtol=0, atol=0)


def test_default_slots_use_reserved_slot_zero() -> None:
    value = torch.ones(2, 17, dtype=torch.bfloat16).npu()
    weight = torch.ones(17, 2, dtype=torch.bfloat16).npu()
    state = torch.zeros(3, 1, 17, dtype=torch.bfloat16).npu()
    result = causal_conv1d_update_v2(value, weight, state, activation=False)
    torch.testing.assert_close(result.cpu(), torch.ones(2, 17, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(state.cpu()[:2], torch.ones(2, 1, 17, dtype=torch.bfloat16), rtol=0, atol=0)
    assert torch.count_nonzero(state.cpu()[2]) == 0


@pytest.mark.parametrize("invalid", ["weight", "capacity", "indices", "accepted", "lengths", "bias", "dtype"])
def test_invalid_contract_fails_before_state_mutation(invalid: str) -> None:
    value = torch.ones(2, 17, dtype=torch.bfloat16).npu()
    weight = torch.ones(17, 3, dtype=torch.bfloat16).npu()
    state = torch.randn(3, 5, 17, dtype=torch.bfloat16).npu()
    indices = torch.tensor([0, 1], dtype=torch.int32).npu()
    qsl = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
    accepted = torch.ones(2, dtype=torch.int32).npu()
    bias = None
    if invalid == "weight":
        weight = torch.ones(17, 7, dtype=torch.bfloat16).npu()
    elif invalid == "capacity":
        state = state[:, :1]
    elif invalid == "indices":
        indices = indices[:, None]
    elif invalid == "accepted":
        accepted = accepted[:1]
    elif invalid == "lengths":
        qsl = qsl[None]
    elif invalid == "bias":
        bias = torch.ones(18, dtype=torch.bfloat16).npu()
    elif invalid == "dtype":
        value = value.float()
    before = state.cpu()
    with pytest.raises(ValueError):
        causal_conv1d_update_v2(value, weight, state, indices, qsl, 1, accepted, bias)
    torch.testing.assert_close(state.cpu(), before, rtol=0, atol=0)

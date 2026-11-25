# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise indexed GDN convolution and layout-preserving gating."""

from importlib.util import find_spec

import pytest
import torch
from xllm_kernel.ops.attention.gdn import causal_conv1d_decode, causal_conv1d_prefill, fused_gdn_gating


@torch.inference_mode()
def test_prefill_reads_and_updates_the_selected_history_slots() -> None:
    dtype = torch.bfloat16
    values = torch.rand((5, 128), dtype=dtype)
    weight = torch.rand((4, 128), dtype=dtype)
    state = torch.rand((4, 3, 128), dtype=dtype)
    device_state = state.npu()
    output = causal_conv1d_prefill(values.npu(), weight.npu(), device_state, [0, 3, 5], [3, 1], [1, 0])
    expected_state = state.clone()
    output_parts = []
    for start, end, slot, read_initial in ((0, 3, 3, True), (3, 5, 1, False)):
        history = state[slot] if read_initial else torch.zeros_like(state[slot])
        combined = torch.cat((history, values[start:end]))
        expected_state[slot] = combined[-3:]
        windows = combined.unfold(0, 4, 1).transpose(-1, -2)
        output_parts.append(torch.nn.functional.silu((windows.float() * weight.float()).sum(dim=-2)))
    torch.testing.assert_close(output.cpu(), torch.cat(output_parts).to(dtype), atol=0.015, rtol=0.01)
    torch.testing.assert_close(device_state.cpu(), expected_state, atol=0, rtol=0)


@pytest.mark.skipif(find_spec("tilelang") is None, reason="requires TileLang")
@pytest.mark.parametrize("channels", [128, 2304])
@torch.inference_mode()
def test_decode_updates_original_history_on_graph_replay(channels: int) -> None:
    dtype = torch.bfloat16
    value = torch.rand((3, channels), dtype=dtype, device="npu")
    weight = torch.rand((4, channels), dtype=dtype, device="npu")
    bias = torch.rand(channels, dtype=dtype, device="npu")
    state = torch.rand((4, 3, channels), dtype=dtype, device="npu")
    slots = torch.tensor([3, 1, 0], dtype=torch.int64, device="npu")
    state_pointer = state.data_ptr()

    def forward() -> torch.Tensor:
        return causal_conv1d_decode(value, weight, state, slots, bias=bias)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = forward()

    for indices in ((1, 3, 0), (2, 0, 3), (3, 1, 2), (0, 2, 1)):
        before = state.cpu()
        slots.copy_(torch.tensor(indices))
        value.copy_(torch.rand_like(value))
        graph.replay()
        expected_state = before.clone()
        expected = []
        for row, slot in enumerate(indices):
            combined = torch.cat((before[slot], value[row : row + 1].cpu()))
            expected_state[slot] = combined[-3:]
            expected.append(
                torch.nn.functional.silu((combined.float() * weight.cpu().float()).sum(0) + bias.cpu().float())
            )
        torch.testing.assert_close(output.cpu(), torch.stack(expected).to(dtype), atol=0.02, rtol=0.015)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
        assert state.data_ptr() == state_pointer


@pytest.mark.skipif(find_spec("tilelang") is None, reason="requires TileLang")
@torch.inference_mode()
def test_gdn_gating_refreshes_outputs_on_replay() -> None:
    a_log = torch.randn(32, device="npu") * 0.1
    dt_bias = torch.randn_like(a_log) * 0.1
    a = torch.randn((3, 32), dtype=torch.bfloat16, device="npu")
    b = torch.randn_like(a)

    fused_gdn_gating(a_log, a, b, dt_bias)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        g, beta = fused_gdn_gating(a_log, a, b, dt_bias)
    for _ in range(2):
        a.copy_(torch.randn_like(a))
        b.copy_(torch.randn_like(b))
        graph.replay()
        expected_g = -a_log.cpu().exp() * torch.nn.functional.softplus(a.cpu().float() + dt_bias.cpu())
        expected_beta = b.cpu().float().sigmoid().to(b.dtype)
        torch.testing.assert_close(g.cpu()[0], expected_g, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(beta.cpu()[0], expected_beta, atol=0.004, rtol=0)

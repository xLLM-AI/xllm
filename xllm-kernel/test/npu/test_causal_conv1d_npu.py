# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Validate channel-last convolution output, state and graph replay."""

import pytest
import torch
from torch.nn import functional
from xllm_kernel.ops.attention.kda import causal_conv1d


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_width", [2, 4])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("activation_mode", [0, 1])
@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_causal_conv1d_output_state_and_graph_replay(
    dtype: torch.dtype,
    kernel_width: int,
    rows: int,
    activation_mode: int,
    run_mode: int,
    packed: bool,
) -> None:
    device = torch.device("npu", torch.npu.current_device())
    generator = torch.Generator().manual_seed(42)
    sequence_lengths = (rows, rows + 1) if packed else (rows, rows)
    input_shape = (sum(sequence_lengths), 16) if packed else (2, rows, 16)
    query_start_loc = [0, rows, sum(sequence_lengths)] if packed else []
    inputs = torch.randn(input_shape, generator=generator).to(device=device, dtype=dtype)
    weight = torch.randn(kernel_width, 16, generator=generator).to(device=device, dtype=dtype)
    state = torch.randn(2, kernel_width - 1, 16, generator=generator).to(device=device, dtype=dtype)
    for _ in range(3):
        causal_conv1d(inputs, weight, state, query_start_loc, activation_mode, run_mode)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = causal_conv1d(inputs, weight, state, query_start_loc, activation_mode, run_mode)
    for _ in range(3):
        input_cpu = torch.randn(input_shape, generator=generator).to(dtype)
        state_cpu = torch.randn(2, kernel_width - 1, 16, generator=generator).to(dtype)
        expected_outputs = []
        expected_states = []
        for initial_state, sequence_input in zip(state_cpu, input_cpu.reshape(-1, 16).split(sequence_lengths)):
            window = torch.cat((initial_state, sequence_input), dim=0)
            convolved = functional.conv1d(
                window.float().t().unsqueeze(0), weight.cpu().float().t().unsqueeze(1), groups=16
            )
            expected_outputs.append(convolved.squeeze(0).t())
            expected_states.append(window[-(kernel_width - 1) :])
        expected = torch.cat(expected_outputs).reshape(input_shape)
        if activation_mode == 1:
            expected = functional.silu(expected)
        inputs.copy_(input_cpu)
        state.copy_(state_cpu)
        graph.replay()
        torch.npu.synchronize()
        eager_state = state_cpu.to(device)
        eager = causal_conv1d(inputs, weight, eager_state, query_start_loc, activation_mode, run_mode)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
        torch.testing.assert_close(state.cpu(), torch.stack(expected_states), rtol=0, atol=0)
        torch.testing.assert_close(captured.cpu(), expected.to(dtype), rtol=2 * torch.finfo(dtype).eps, atol=1e-3)

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check packed QKV and chunk GDN prefill against independent references."""

import torch
from xllm_kernel.ops.attention.gdn import causal_conv1d_qkv_prefill, chunk_gated_delta_rule


def test_causal_conv1d_qkv_prefill_output_state_and_graph_replay() -> None:
    generator = torch.Generator().manual_seed(2797)
    lengths = (3, 5)
    channels = 512
    x = torch.zeros((8, channels), dtype=torch.bfloat16, device="npu")
    weight = (torch.randn((4, channels), generator=generator) * 0.2).bfloat16()
    weight_npu = weight.npu()
    state = torch.zeros((3, 3, channels), dtype=x.dtype, device=x.device)

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return causal_conv1d_qkv_prefill(
            x,
            weight_npu,
            state,
            query_start_loc=[0, 3, 8],
            state_indices=[2, 0],
            has_initial_state=[1, 0],
            num_qk_heads=1,
            num_v_heads=2,
            head_k_dim=128,
            head_v_dim=128,
        )

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward()
    for _ in range(3):
        tokens = (torch.randn(x.shape, generator=generator) * 0.3).bfloat16()
        initial = (torch.randn(state.shape, generator=generator) * 0.2).bfloat16()
        x.copy_(tokens)
        state.copy_(initial)
        graph.replay()

        expected_state = initial.clone()
        outputs = []
        for slot, use_initial, token_rows in zip((2, 0), (True, False), tokens.split(lengths), strict=True):
            history = initial[slot] if use_initial else torch.zeros_like(initial[slot])
            window = torch.cat((history, token_rows))
            projected = torch.nn.functional.conv1d(
                window.float().t().unsqueeze(0), weight.float().t().unsqueeze(1), groups=channels
            )
            outputs.append(torch.nn.functional.silu(projected.squeeze(0).t()).bfloat16())
            expected_state[slot] = window[-3:]
        q, k, v = torch.cat(outputs).split((128, 128, 256), dim=-1)
        expected_qk = tuple(
            (head.float() * torch.rsqrt(head.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
            for head in (q, k)
        )
        expected = (*expected_qk, v)
        for output, reference in zip(captured, expected, strict=True):
            torch.testing.assert_close(output.cpu().reshape(8, -1), reference, atol=2e-3, rtol=0.02)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)


def _chunk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
    lengths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = (
        (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6))
        .to(x.dtype)
        .half()
        .float()
        .repeat_interleave(v.shape[1] // q.shape[1], dim=1)
        for x in (q, k)
    )
    v, beta, gate = v.half().float(), beta.half().float(), gate.bfloat16().float()
    all_outputs, states = [], []
    start = 0
    for index, length in enumerate(lengths):
        state = initial[index].half().float()
        for i in range(start, start + length):
            state = state * gate[i].exp()[:, None, None]
            prediction = (k[i, :, :, None] * state).sum(dim=1)
            error = (v[i] - prediction) * beta[i, :, None]
            state = state + k[i, :, :, None] * error[:, None, :]
            all_outputs.append((q[i, :, :, None] * state).sum(dim=1) * q.shape[-1] ** -0.5)
        states.append(state)
        start += length
    return torch.stack(all_outputs).bfloat16(), torch.stack(states)


def test_chunk_gdn_packed_output_final_state_and_graph_replay() -> None:
    generator = torch.Generator().manual_seed(2798)
    lengths = (7, 8)
    tokens = sum(lengths)
    q = (torch.randn((tokens, 1, 128), generator=generator) * 0.2).bfloat16()
    k = (torch.randn(q.shape, generator=generator) * 0.2).bfloat16()
    v = (torch.randn((tokens, 2, 128), generator=generator) * 0.15).bfloat16()
    gate = (-torch.rand((tokens, 2), generator=generator) * 0.1).float()
    beta = torch.rand((tokens, 2), generator=generator).bfloat16()
    initial = (torch.randn((2, 2, 128, 128), generator=generator) * 0.015).bfloat16()
    q_npu, k_npu, v_npu, gate_npu, beta_npu, initial_npu = (tensor.npu() for tensor in (q, k, v, gate, beta, initial))
    offsets = torch.tensor([0, lengths[0], tokens], dtype=torch.int32, device="npu")

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        return chunk_gated_delta_rule(
            q_npu, k_npu, v_npu, gate_npu, beta_npu, initial_npu, offsets, sequence_lengths=list(lengths)
        )

    eager = forward()
    automatic_lengths = chunk_gated_delta_rule(q_npu, k_npu, v_npu, gate_npu, beta_npu, initial_npu, offsets)
    torch.testing.assert_close(eager, automatic_lengths, atol=0, rtol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward()

    for _ in range(3):
        v = (torch.randn(v.shape, generator=generator) * 0.15).bfloat16()
        initial = (torch.randn(initial.shape, generator=generator) * 0.015).bfloat16()
        v_npu.copy_(v)
        initial_npu.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, forward(), atol=0, rtol=0)
        expected_out, expected_state = _chunk_reference(q, k, v, gate, beta, initial, lengths)
        torch.testing.assert_close(captured[0].cpu(), expected_out, atol=5e-4, rtol=0.02)
        torch.testing.assert_close(captured[1].cpu(), expected_state, atol=5e-4, rtol=0.02)
        torch.testing.assert_close(initial_npu.cpu(), initial, atol=0, rtol=0)

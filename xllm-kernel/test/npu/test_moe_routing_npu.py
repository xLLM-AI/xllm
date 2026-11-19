# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the packaged MoE routing and quantized expert providers."""

from collections.abc import Callable

import pytest
import torch
import torch_npu
from xllm_kernel.ops import (
    dequant_swiglu_quant,
    moe_gating_top_k_hash,
    moe_grouped_matmul_swiglu_quant,
    moe_init_routing_v3,
)


def _check_graph_replay(forward: Callable[[], tuple[torch.Tensor, ...]], update: Callable[[], None]) -> None:
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward()
    for _ in range(3):
        update()
        eager = forward()
        graph.replay()
        torch.npu.synchronize()
        for actual, reference in zip(captured, eager, strict=True):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("lookup", [False, True])
def test_moe_gating_matches_sqrt_softplus_reference(lookup: bool) -> None:
    generator = torch.Generator().manual_seed(2796)
    logits = torch.randn((9, 128), generator=generator)
    scores = torch.nn.functional.softplus(logits).sqrt()
    bias = torch.linspace(-0.2, 0.2, 128) if not lookup else None
    token_ids = torch.arange(9, dtype=torch.int64).flip(0) if lookup else None
    table = torch.stack([torch.randperm(128, generator=generator)[:4] for _ in range(9)]).int() if lookup else None
    indices = table[token_ids].long() if lookup else (scores + bias).topk(4).indices
    selected = scores.gather(-1, indices)
    expected = selected / selected.sum(-1, keepdim=True) * 1.6

    logits_npu = logits.npu()
    bias_npu = bias.npu() if bias is not None else None
    token_ids_npu = token_ids.npu() if token_ids is not None else None
    table_npu = table.npu() if table is not None else None

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return moe_gating_top_k_hash(
            logits_npu,
            k=4,
            bias=bias_npu,
            input_ids=token_ids_npu,
            tid2eid=table_npu,
            k_group=1,
            group_count=1,
            routed_scaling_factor=1.6,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=True,
        )

    actual, actual_ids, norm = forward()
    torch.testing.assert_close(actual_ids.cpu(), indices.int())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(norm.cpu(), scores, atol=2e-5, rtol=2e-5)

    def update() -> None:
        logits_npu.copy_(torch.randn(logits.shape, generator=generator))

    _check_graph_replay(forward, update)


@pytest.mark.parametrize("swiglu_mode", [0, 1])
def test_dequant_swiglu_quant_matches_reference(swiglu_mode: int) -> None:
    generator = torch.Generator().manual_seed(2797)
    gates = torch.randint(-1600, 1601, (9, 512), generator=generator, dtype=torch.int32)
    channel = torch.rand(512, generator=generator).mul_(0.08)
    token = torch.rand(9, generator=generator).mul_(0.2)
    gates_npu, channel_npu, token_npu = gates.npu(), channel.npu(), token.npu()
    alpha, bias = (1.25, 0.5) if swiglu_mode else (1.0, 0.0)

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        return dequant_swiglu_quant(
            gates_npu,
            channel_npu,
            token_npu,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=alpha,
            glu_bias=bias,
        )

    actual, actual_scale = forward()
    gate, up = (gates.float() * channel * token.unsqueeze(-1)).chunk(2, dim=-1)
    if swiglu_mode:
        gate = gate.clamp(max=7.0)
        up = up.clamp(-7.0, 7.0) + bias
    expected = gate * torch.sigmoid(gate * alpha) * up
    scale = expected.abs().amax(dim=-1) / 127
    expected_int8 = (expected / scale.unsqueeze(-1)).round().clamp(-127, 127)
    torch.testing.assert_close(actual.cpu().float(), expected_int8, atol=1, rtol=0)
    torch.testing.assert_close(actual_scale.cpu(), scale, atol=2e-6, rtol=0.003)

    def update() -> None:
        gates_npu.copy_(torch.randint(-1600, 1601, gates.shape, generator=generator, dtype=torch.int32))

    _check_graph_replay(forward, update)


@pytest.mark.usefixtures("require_fused_moe_gmm1")
def test_routing_v3_grouped_swiglu_and_graph_replay() -> None:
    generator = torch.Generator().manual_seed(2798)
    num_experts = 4
    width = 256
    hidden = torch.randn((32, width), generator=generator).bfloat16()
    experts = torch.arange(32, dtype=torch.int32).remainder(num_experts).reshape(-1, 1)
    weights = torch.randint(-4, 5, (num_experts, width, 2 * width), generator=generator, dtype=torch.int8)
    weight_scale = torch.rand((num_experts, 2 * width), generator=generator).mul_(0.04).add_(0.01)
    torch.npu.config.allow_internal_format = True
    nz_weight = torch_npu.npu_format_cast(weights.npu(), torch_npu.Format.FRACTAL_NZ)
    hidden_npu, experts_npu, weight_scale_npu = hidden.npu(), experts.npu(), weight_scale.npu()

    def forward() -> tuple[torch.Tensor, ...]:
        expanded, row_idx, groups, scale = moe_init_routing_v3(
            hidden_npu, experts_npu, hidden_npu.shape[0], num_experts, [0, num_experts]
        )
        output, out_scale = moe_grouped_matmul_swiglu_quant(expanded, nz_weight, weight_scale_npu, scale, groups)
        return expanded, row_idx, groups, scale, output, out_scale

    expanded, row_idx, groups, scale, output, out_scale = forward()
    source_to_expanded = row_idx.cpu().long()
    assert sorted(source_to_expanded.tolist()) == list(range(32))
    rows = source_to_expanded.argsort()
    torch.testing.assert_close(groups.cpu(), torch.tensor([[0, 8], [1, 8], [2, 8], [3, 8]]))
    torch.testing.assert_close(experts[rows, 0], torch.arange(4, dtype=torch.int32).repeat_interleave(8))
    expanded_cpu = expanded.cpu().float() * scale.cpu().unsqueeze(-1)
    torch.testing.assert_close(expanded_cpu, hidden[rows].float(), atol=0.018, rtol=0)

    projections = torch.bmm(expanded.cpu().float().unsqueeze(1), weights[experts[rows, 0]].float()).squeeze(1)
    projections *= scale.cpu().unsqueeze(-1) * weight_scale[experts[rows, 0]]
    gate, up = projections.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    expected_scale = expected.abs().amax(dim=-1) / 127
    torch.testing.assert_close(out_scale.cpu(), expected_scale, atol=2e-5, rtol=0.003)
    expected_int8 = (expected / expected_scale.unsqueeze(-1)).round().clamp(-127, 127)
    torch.testing.assert_close(output.cpu().float(), expected_int8, atol=1, rtol=0)

    def update() -> None:
        hidden_npu.copy_(torch.randn(hidden.shape, generator=generator))
        experts_npu.copy_(torch.randperm(32, generator=generator).remainder(num_experts).reshape(-1, 1))

    _check_graph_replay(forward, update)

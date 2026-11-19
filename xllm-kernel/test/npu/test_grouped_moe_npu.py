# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Validate fused grouped MoE GMM1 independently of the router."""

import pytest
import torch
import torch_npu
from xllm_kernel.ops import moe_grouped_matmul_swiglu_quant


def _assert_grouped_moe_output(
    actual: tuple[torch.Tensor, torch.Tensor],
    tokens: torch.Tensor,
    weights: torch.Tensor,
    channel_scale: torch.Tensor,
    token_scale: torch.Tensor,
    groups: torch.Tensor,
) -> None:
    experts = torch.repeat_interleave(groups[:, 0], groups[:, 1])
    projection = torch.bmm(tokens.float().unsqueeze(1), weights[experts].float()).squeeze(1)
    gate, up = (projection * channel_scale[experts].float() * token_scale[:, None]).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    expected_scale = activated.abs().amax(dim=-1) / 127
    expected_codes = (activated / expected_scale[:, None]).round().clamp(-127, 127)

    codes, scale = actual
    assert codes.dtype == torch.int8
    torch.testing.assert_close(scale.cpu(), expected_scale, atol=2e-5, rtol=0.003)
    torch.testing.assert_close(codes.cpu().float(), expected_codes, atol=1, rtol=0)


@pytest.mark.usefixtures("require_fused_moe_gmm1")
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_grouped_moe_honors_expert_counts_and_quantization_scales(scale_dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(2794)
    tokens = torch.randint(-64, 65, (32, 256), generator=generator, dtype=torch.int8)
    weights = torch.randint(-5, 6, (4, 256, 512), generator=generator, dtype=torch.int8)
    channel_scale = torch.rand((4, 512), generator=generator).mul_(0.04).add_(0.01).to(scale_dtype)
    token_scale = torch.linspace(0.01, 0.04, tokens.shape[0])
    groups = torch.tensor([[0, 5], [1, 7], [2, 9], [3, 11]], dtype=torch.int64)

    torch.npu.config.allow_internal_format = True
    nz_weights = torch_npu.npu_format_cast(weights.npu(), torch_npu.Format.FRACTAL_NZ)
    tokens_npu, channel_scale_npu, token_scale_npu, groups_npu = (
        value.npu() for value in (tokens, channel_scale, token_scale, groups)
    )

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_grouped_matmul_swiglu_quant(tokens_npu, nz_weights, channel_scale_npu, token_scale_npu, groups_npu)

    torch.npu.synchronize()
    _assert_grouped_moe_output(forward(), tokens, weights, channel_scale, token_scale, groups)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward()

    for counts in ((3, 7, 12, 10), (9, 5, 4, 14), (2, 12, 8, 10)):
        groups[:, 1] = torch.tensor(counts)
        token_scale = token_scale.roll(1)
        groups_npu.copy_(groups)
        token_scale_npu.copy_(token_scale)
        graph.replay()
        torch.npu.synchronize()
        _assert_grouped_moe_output(captured, tokens, weights, channel_scale, token_scale, groups)

    torch.testing.assert_close(groups_npu.cpu(), groups)
    torch.testing.assert_close(token_scale_npu.cpu(), token_scale, atol=0, rtol=0)

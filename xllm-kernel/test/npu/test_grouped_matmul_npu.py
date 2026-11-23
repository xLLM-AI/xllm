# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check grouped matmul's caller-owned output and graph replay."""

import pytest
import torch
from xllm_kernel.ops.gemm import grouped_matmul, grouped_matmul_out, has_grouped_matmul_out


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_grouped_matmul_out_replays_live_expert_counts(group_list_type: int) -> None:
    if not has_grouped_matmul_out():
        pytest.skip("CANN grouped matmul out is unavailable")
    generator = torch.Generator().manual_seed(741)
    x = torch.randint(-5, 6, (16, 256), dtype=torch.int8, generator=generator).npu()
    weight = torch.randint(-5, 6, (2, 256, 128), dtype=torch.int8, generator=generator).npu()
    scale = torch.full((2, 128), 0.03125, dtype=torch.bfloat16, device="npu")
    per_token_scale = torch.full((16,), 0.125, device="npu")
    groups = torch.tensor([8, 16] if group_list_type == 0 else [8, 8], dtype=torch.int64, device="npu")
    out = torch.empty((16, 128), dtype=torch.bfloat16, device="npu")

    def run() -> torch.Tensor:
        return grouped_matmul_out(
            x,
            weight,
            scale,
            per_token_scale,
            groups,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output=out,
        )

    def reference() -> torch.Tensor:
        return grouped_matmul(
            x=[x],
            weight=[weight],
            scale=[scale],
            per_token_scale=[per_token_scale],
            group_list=groups,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=out.dtype,
        )[0]

    assert run() is out
    torch.testing.assert_close(out.cpu(), reference().cpu(), rtol=0, atol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    assert captured is out

    for split in (3, 13):
        groups.copy_(torch.tensor([split, 16] if group_list_type == 0 else [split, 16 - split]))
        previous = out.clone()
        graph.replay()
        torch.testing.assert_close(out.cpu(), reference().cpu(), rtol=0, atol=0)
        assert not torch.equal(previous, out)

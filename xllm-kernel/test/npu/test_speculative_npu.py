# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

import pytest
import torch
from xllm_kernel.ops.speculative import greedy_prefix_verify


@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("target_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("bonus_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [0, 7, 65])
@torch.inference_mode()
def test_greedy_replay_resets_suffix_and_supports_strided_inputs(
    draft_dtype: torch.dtype, target_dtype: torch.dtype, bonus_dtype: torch.dtype, width: int
) -> None:
    draft = torch.arange(3 * width * 2, dtype=draft_dtype, device="npu").reshape(3, width * 2)[:, ::2]
    target = draft.to(target_dtype).clone()
    bonus = torch.full((1, 1), 999, dtype=bonus_dtype, device="npu").expand(3, 1)
    greedy_prefix_verify(draft, target, bonus)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = greedy_prefix_verify(draft, target, bonus)
    for rejected in (width, 0, width // 2):
        target.copy_(draft)
        if rejected < width:
            target[:, rejected].add_(1)
        graph.replay()
        expected = torch.full((3, width + 1), -1, dtype=torch.int32)
        expected[:, :rejected] = target[:, :rejected].cpu().to(torch.int32)
        expected[:, rejected] = target[:, rejected].cpu() if rejected < width else bonus[:, 0].cpu()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

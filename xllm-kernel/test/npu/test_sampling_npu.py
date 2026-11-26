# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise in-place sampling with thresholds refreshed between graph replays."""

import pytest
import torch
from xllm_kernel.ops.sampling import top_k_top_p


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_top_k_top_p_preserves_storage_and_reads_live_thresholds(dtype: torch.dtype) -> None:
    # Keep every unfiltered probability representable, including FP16.
    source = (torch.arange(128, dtype=dtype) / 16).repeat(3, 1)
    logits = source.npu()
    top_k = torch.tensor([1, 4, 128], dtype=torch.int32, device="npu")
    top_p = torch.ones(3, dtype=torch.float32, device="npu")
    pointer = logits.data_ptr()
    top_k_top_p(logits, top_k, top_p)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        top_k_top_p(logits, top_k, top_p)
    for ks, ps in (([2, 7, 128], [1.0, 1.0, 1.0]), ([128, 128, 128], [0.5, 0.8, 1.0])):
        logits.copy_(source)
        top_k.copy_(torch.tensor(ks))
        top_p.copy_(torch.tensor(ps))
        graph.replay()
        expected = source.clone()
        for row, (k, p) in enumerate(zip(ks, ps)):
            expected[row, : 128 - k] = -torch.inf
            if p < 1:
                # The cumulative mass crosses 0.5 at 12 tokens and 0.8 at 26,
                # away from the dtype-rounded cutoffs.
                keep = 12 if p == 0.5 else 26
                expected[row, : 128 - keep] = -torch.inf
        torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
        assert logits.data_ptr() == pointer
        assert top_k.cpu().tolist() == ks
        torch.testing.assert_close(top_p.cpu(), torch.tensor(ps), rtol=0, atol=0)

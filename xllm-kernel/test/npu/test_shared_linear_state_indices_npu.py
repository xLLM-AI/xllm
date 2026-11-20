# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Test graph-owned Conv/SSM role metadata with strided state tables."""

import pytest
import torch
import triton
from xllm_kernel.ops.attention.gdn.mega import shared_linear_state_indices_out


@pytest.mark.parametrize("groups,batch", [(1, 1), (3, 7), (23, 301)])
def test_shared_linear_state_indices_replays_into_same_buffer(
    groups: int, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_compile(*args: object, **kwargs: object) -> None:
        pytest.fail("shared GDN state gather compiled after startup")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "_do_compile", unexpected_compile)
    capacity, stride = 19, 27
    table = torch.arange(2 * groups * stride, dtype=torch.int32).reshape(2 * groups, stride)
    rows = table.npu()[:, :capacity]
    ids = torch.ones(batch, dtype=torch.int32, device=rows.device)
    out = torch.empty((groups, batch, 2), dtype=ids.dtype, device=ids.device)
    pointer = out.data_ptr()

    assert shared_linear_state_indices_out(ids, rows, out) is None
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        shared_linear_state_indices_out(ids, rows, out)

    for step in range(3):
        logical = ((torch.arange(batch, dtype=torch.int32) + step * 3) % (capacity + 2) - 1).contiguous()
        valid = (logical > 0) & (logical < capacity)
        physical = table[:, :capacity].index_select(1, torch.where(valid, logical, 0).long())
        expected = physical.view(groups, 2, batch).transpose(1, 2)
        expected = torch.where(valid[None, :, None], expected, -1)
        ids.copy_(logical)
        graph.replay()
        assert out.data_ptr() == pointer
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

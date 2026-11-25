# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check static decode metadata addresses and padding after replay."""

import torch
from xllm_kernel.ops.kv_cache import update_decode_graph_metadata


@torch.inference_mode()
def test_decode_metadata_replay_refreshes_inputs_and_padding() -> None:
    device = torch.device("npu")
    tokens = torch.tensor([5, 7], dtype=torch.int32, device=device)
    positions = torch.tensor([8, 13], dtype=torch.int32, device=device)
    slots = torch.tensor([17, 29], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([0, 1, 4], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    indices = torch.tensor([2, 3, 4, 5], dtype=torch.int32, device=device)
    last_page = torch.tensor([128, 12], dtype=torch.int32, device=device)
    outputs = tuple(torch.full((n,), -99, dtype=torch.int32, device=device) for n in (4, 4, 4, 5, 4, 5, 8, 4))

    def run() -> torch.Tensor:
        return update_decode_graph_metadata(tokens, positions, slots, kv_lens, indptr, indices, last_page, *outputs, 4)

    assert run() is outputs[0]
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()

    assert result is outputs[0]
    for step in (1, 2):
        tokens.fill_(step)
        positions.fill_(step + 1)
        slots.fill_(step + 2)
        kv_lens.copy_(torch.tensor([0, step, step + 7]))
        indptr.copy_(torch.tensor([0, step, 4]))
        indices.fill_(step + 3)
        last_page.fill_(step + 4)
        graph.replay()
        expected = (
            [step, step, 0, 0],
            [step + 1, step + 1, 0, 0],
            [step + 2, step + 2, -1, -1],
            [0, step, step + 7, step + 7, step + 7],
            [step, 7, 0, 0],
            [0, step, 4, 4, 4],
            [step + 3] * 4 + [-99] * 4,
            [step + 4, step + 4, 1, 1],
        )
        assert tuple(tensor.cpu().tolist() for tensor in outputs) == expected

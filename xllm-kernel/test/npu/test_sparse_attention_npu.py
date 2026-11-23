# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Numerical and replay checks for the packaged DSA and SFA operators."""

import pytest
import torch
from xllm_kernel.ops.attention.dsa import sparse_flash_attention_lse, sparse_flash_attention_lse_out
from xllm_kernel.ops.attention.dsv4 import compressor, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_compressor_pools_kv_and_writes_tail_state(graph_mode: bool) -> None:
    generator = torch.Generator().manual_seed(781)
    dtype = torch.float16
    ratio, head_dim, hidden_dim, rope_dim = 128, 512, 1024, 64
    x = (torch.randn(1, ratio + 2, hidden_dim, generator=generator) * 0.1).to(dtype)
    x_npu = x.npu()
    wkv = (torch.randn(head_dim, hidden_dim, generator=generator) * 0.05).to(dtype)
    wgate = torch.zeros_like(wkv)
    kv_state = torch.zeros(3, ratio, head_dim, device="npu")
    score_state = torch.zeros_like(kv_state)
    table = torch.tensor([[1, 2]], dtype=torch.int32).npu()
    ape = torch.arange(ratio, dtype=torch.float32)[:, None].expand(-1, head_dim).contiguous() / ratio
    cos = torch.zeros(1, 2, rope_dim, dtype=dtype)
    sin = torch.ones_like(cos)
    args = (
        x_npu,
        wkv.npu(),
        wgate.npu(),
        kv_state,
        score_state,
        ape.npu(),
        torch.ones(head_dim, dtype=dtype).npu(),
        sin.npu(),
        cos.npu(),
        table,
        table,
        None,
        None,
        None,
        rope_dim,
        ratio,
        1,
        1e-6,
        1,
        False,
    )
    result = compressor(*args)
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = compressor(*args)

    for scale in (1, -1):
        live = x * scale
        x_npu.copy_(live)
        if graph_mode:
            graph.replay()
        else:
            result = compressor(*args)
        projection = live.float() @ wkv.float().T
        pooled = (projection[0, :ratio] * ape.softmax(dim=0)).sum(dim=0)
        norm = pooled * (pooled.square().mean() + 1e-6).rsqrt()
        rope_tail = norm[-rope_dim:]
        half = rope_dim // 2
        expected = torch.cat((norm[:-rope_dim], -rope_tail[half:], rope_tail[:half]))
        torch.testing.assert_close(result[0].cpu()[0, 0].float(), expected, atol=0.025, rtol=0.025)
        assert all(buffer.numel() == 0 for buffer in result[1:])
        torch.testing.assert_close(kv_state.cpu()[2, :2], projection[0, ratio:], atol=0.002, rtol=0.02)
        torch.testing.assert_close(score_state.cpu()[2, :2], ape[:2], atol=1e-5, rtol=0)
        assert torch.count_nonzero(kv_state.cpu()[0]).item() == 0


@pytest.mark.parametrize("ratio", [1, 128], ids=["swa", "compressed"])
@torch.inference_mode()
def test_shared_kv_metadata_replays_with_live_lengths_and_pages(ratio: int) -> None:
    dtype = torch.float16
    heads, head_dim, page_size = 64, 512, 128
    q = torch.zeros(1, 1, heads, head_dim, dtype=dtype).npu()
    ori = torch.arange(3, dtype=dtype)[:, None, None, None].expand(3, page_size, 1, head_dim).contiguous()
    cmp = torch.full((1, page_size, 1, head_dim), 8, dtype=dtype)
    has_cmp = ratio > 1
    ori_npu = ori.npu()
    cmp_npu = cmp.npu() if has_cmp else None
    ori_table = torch.tensor([[2]], dtype=torch.int32).npu()
    cmp_table = torch.zeros_like(ori_table) if has_cmp else None
    seq = torch.tensor([page_size], dtype=torch.int32).npu()
    sinks = torch.zeros(heads, dtype=torch.float32).npu()
    scale = head_dim**-0.5

    def attend() -> torch.Tensor:
        metadata = sparse_attn_sharedkv_metadata(
            heads,
            1,
            head_dim,
            None,
            None,
            None,
            None,
            seq,
            1,
            1,
            page_size,
            0,
            0,
            ratio,
            4,
            3,
            page_size - 1,
            0,
            "BSND",
            "PA_ND",
            True,
            has_cmp,
        )
        assert metadata.shape == (1024,)
        assert metadata.dtype == torch.int32
        output, _ = sparse_attn_sharedkv(
            q,
            ori_npu,
            cmp_npu,
            None,
            None,
            ori_table,
            cmp_table,
            None,
            None,
            None,
            None,
            seq,
            sinks,
            metadata,
            scale,
            ratio,
            4,
            3,
            page_size - 1,
            0,
            "BSND",
            "PA_ND",
            False,
        )
        return output

    eager = attend()
    expected = (2 * page_size + (8 if has_cmp else 0)) / (page_size + (2 if has_cmp else 1))
    torch.testing.assert_close(eager.cpu(), torch.full(q.shape, expected, dtype=dtype), atol=0.003, rtol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = attend()

    for page, length in ((1, page_size), (2, 64)):
        ori_table.fill_(page)
        seq.fill_(length)
        graph.replay()
        compressed_tokens = length // ratio if has_cmp else 0
        expected = (page * length + 8 * compressed_tokens) / (length + compressed_tokens + 1)
        torch.testing.assert_close(output.cpu(), torch.full(q.shape, expected, dtype=dtype), atol=0.003, rtol=0)


@torch.inference_mode()
def test_sparse_attention_lse_matches_reference_and_replays_to_output_buffer() -> None:
    generator = torch.Generator().manual_seed(2460)
    dtype = torch.bfloat16
    page_size, heads, head_dim, rope_dim = 128, 8, 512, 64
    q = (torch.randn(1, heads, head_dim, generator=generator) * 0.2).to(dtype)
    k = (torch.randn(1, page_size, 1, head_dim, generator=generator) * 0.2).to(dtype)
    q_rope = (torch.randn(1, heads, rope_dim, generator=generator) * 0.4).to(dtype)
    k_rope = (torch.randn(1, page_size, 1, rope_dim, generator=generator) * 0.4).to(dtype)
    k_npu = k.npu()
    indices = torch.zeros((1, 1, 16), dtype=torch.int32).npu()
    table = torch.zeros((1, 1), dtype=torch.int32).npu()
    q_seq = torch.tensor([1], dtype=torch.int32).npu()
    kv_seq = torch.tensor([page_size], dtype=torch.int32).npu()
    scale = head_dim**-0.5
    args = (
        q.npu(),
        k_npu,
        k_npu,
        indices,
        table,
        q_seq,
        kv_seq,
        q_rope.npu(),
        k_rope.npu(),
        scale,
        1,
        "TND",
        "PA_BSND",
        3,
    )

    def reference(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = k[0, selected, 0].float()
        keys_rope = k_rope[0, selected, 0].float()
        score = (q[0].float() @ keys.T + q_rope[0].float() @ keys_rope.T) * scale
        maximum = score.max(dim=-1).values
        total = (score - maximum[:, None]).exp().sum(dim=-1)
        output = (score.softmax(dim=-1) @ keys).unsqueeze(0).to(dtype)
        return output, maximum.view(1, 1, heads), total.view(1, 1, heads)

    selected = torch.arange(0, 16, dtype=torch.int32)
    indices.copy_(selected.view_as(indices))
    actual = sparse_flash_attention_lse(*args, return_softmax_lse=True)
    for result, expected in zip(actual, reference(selected.long()), strict=True):
        torch.testing.assert_close(result.cpu(), expected, atol=0.008, rtol=0.008)

    output = torch.empty_like(args[0])
    assert sparse_flash_attention_lse_out(*args, output) is output
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert sparse_flash_attention_lse_out(*args, output) is output
    for start in (8, 32, 64):
        selected = torch.arange(start, start + 16, dtype=torch.int32)
        indices.copy_(selected.view_as(indices))
        graph.replay()
        torch.testing.assert_close(output.cpu(), reference(selected.long())[0], atol=0.008, rtol=0.008)

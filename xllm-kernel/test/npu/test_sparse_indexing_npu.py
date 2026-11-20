# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the packaged AICPU metadata, quantized indexer and cache scatter."""

import pytest
import torch
from xllm_kernel.ops import quant_lightning_indexer, quant_lightning_indexer_metadata, scatter_nd_update


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_quantized_indexer_resolves_paged_keys(graph_mode: bool) -> None:
    device = torch.device("npu", torch.npu.current_device())
    is_ascend950 = torch.npu.get_device_name(device).startswith("Ascend950")
    quant_dtype = torch.float8_e4m3fn if is_ascend950 else torch.int8
    scale_dtype = torch.float32 if is_ascend950 else torch.float16
    page_size, pages, num_heads, head_dim, top_k = 128, 3, 64, 128, 8
    key_count = page_size + 2
    query = torch.ones((1, num_heads, head_dim), dtype=torch.int8).to(quant_dtype).to(device)
    key = torch.ones((pages, page_size, 1, head_dim), dtype=torch.int8).to(quant_dtype).to(device)
    query_scale = torch.ones((1, num_heads), dtype=scale_dtype, device=device)
    weights = torch.zeros_like(query_scale)
    weights[:, 0] = 1
    scales = (torch.arange(pages * page_size, dtype=torch.float32).view(pages, page_size, 1) + 1) / page_size
    key_scale = scales.to(device=device, dtype=scale_dtype)
    query_lens = torch.tensor([1], dtype=torch.int32, device=device)
    key_lens = torch.tensor([key_count], dtype=torch.int32, device=device)
    table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)

    def select() -> torch.Tensor:
        metadata = quant_lightning_indexer_metadata(
            num_heads_q=num_heads,
            num_heads_k=1,
            head_dim=head_dim,
            actual_seq_lengths_query=query_lens,
            actual_seq_lengths_key=key_lens,
            max_seqlen_q=1,
            max_seqlen_k=key_count,
            sparse_count=top_k,
            cmp_ratio=1,
        )
        assert metadata.dtype == torch.int32
        assert metadata.shape == (1024,)
        return quant_lightning_indexer(
            query, key, weights, query_scale, key_scale, metadata, query_lens, key_lens, table, top_k
        )

    output = select()
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = select()

    for pages_in_order, live_keys in (([2, 0], key_count), ([0, 2], key_count), ([1, 2], page_size)):
        key_lens.fill_(live_keys)
        table.copy_(torch.tensor([pages_in_order], dtype=torch.int32))
        if graph_mode:
            graph.replay()
        else:
            output = select()
        expected = scales[pages_in_order].flatten()[:live_keys].topk(top_k).indices.to(torch.int32)
        assert output.dtype == torch.int32
        assert output.shape == (1, 1, top_k)
        torch.testing.assert_close(output.cpu()[0, 0].sort().values, expected.sort().values, atol=0, rtol=0)


@pytest.mark.parametrize("dtype,index_dtype", [(torch.int8, torch.int32), (torch.bfloat16, torch.int64)])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_cache_scatter_updates_only_selected_rows(
    dtype: torch.dtype, index_dtype: torch.dtype, graph_mode: bool
) -> None:
    width = 128
    backing = torch.full((8, width + 32), -7, dtype=dtype, device="npu")
    cache = backing[:, :width]
    address = cache.data_ptr()
    indices = torch.empty((3, 1), dtype=index_dtype, device="npu")
    updates = torch.empty((3, width), dtype=dtype, device="npu")
    indices.fill_(-1)
    updates.zero_()
    assert scatter_nd_update(cache, indices, updates) is None
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            scatter_nd_update(cache, indices, updates)

    expected = torch.full(backing.shape, -7, dtype=dtype)
    for step, rows in enumerate(([5, -1, 1], [2, 6, -1], [-1, 4, 0])):
        values = torch.tensor([step + 1, step + 11, step + 21], dtype=dtype)
        indices.copy_(torch.tensor(rows, dtype=index_dtype).unsqueeze(-1))
        updates.copy_(values.unsqueeze(-1).expand(-1, width))
        if graph_mode:
            graph.replay()
        else:
            scatter_nd_update(cache, indices, updates)
        for row, value in zip(rows, values, strict=True):
            if row >= 0:
                expected[row, :width] = value
        assert cache.data_ptr() == address
        torch.testing.assert_close(backing.cpu(), expected, atol=0, rtol=0)

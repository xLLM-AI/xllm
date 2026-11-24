# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Paged KV writes through the startup-selected NPU provider."""

import pytest
import torch
from xllm_kernel.ops.kv_cache import reshape_paged_cache
from xllm_kernel.platform import current_platform


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_paged_cache_writes_selected_slots(dtype: torch.dtype, graph_mode: bool) -> None:
    is_a5 = current_platform.get_ascend_soc_generation() == "a5"
    heads, head_dim = (3, 17) if is_a5 else (2, 128)
    blocks, page_size = 4, 128
    capacity = blocks * page_size
    generator = torch.Generator().manual_seed(2911)
    key = torch.randn((5, heads, head_dim), dtype=dtype, generator=generator)
    value = torch.randn((5, heads, head_dim), dtype=dtype, generator=generator)
    key_npu, value_npu = key.npu(), value.npu()
    expected_key = torch.full((blocks, page_size, heads, head_dim), -2, dtype=dtype)
    expected_value = torch.full_like(expected_key, 3)
    key_cache, value_cache = expected_key.npu(), expected_value.npu()
    slots = torch.full((key.size(0),), -1, dtype=torch.int32).npu()
    key_pointer, value_pointer = key_cache.data_ptr(), value_cache.data_ptr()

    def write() -> torch.Tensor:
        return reshape_paged_cache(slots, key_npu, value_npu, key_cache, value_cache)

    assert write() is key_cache
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            assert write() is key_cache

    for step, mapping in enumerate(((page_size - 1, -1, page_size, 0, capacity - 1), (2, 257, -1, 311, 3))):
        host_slots = torch.tensor(mapping, dtype=torch.int32)
        slots.copy_(host_slots)
        key_rows = key + step
        value_rows = value - step
        key_npu.copy_(key_rows)
        value_npu.copy_(value_rows)
        if graph_mode:
            graph.replay()
        else:
            assert write() is key_cache

        valid = host_slots >= 0
        offsets = host_slots[valid].long()
        expected_key.flatten(0, 1).index_copy_(0, offsets, key_rows[valid])
        expected_value.flatten(0, 1).index_copy_(0, offsets, value_rows[valid])
        assert key_cache.data_ptr() == key_pointer
        assert value_cache.data_ptr() == value_pointer
        torch.testing.assert_close(key_cache.cpu(), expected_key, atol=0, rtol=0)
        torch.testing.assert_close(value_cache.cpu(), expected_value, atol=0, rtol=0)


@torch.inference_mode()
def test_a5_cache_ignores_out_of_range_slots() -> None:
    if current_platform.get_ascend_soc_generation() != "a5":
        pytest.skip("requires the Ascend950 cache provider")
    shape = (2, 128, 1, 64)
    cache = torch.full(shape, 7, dtype=torch.bfloat16).npu()
    rows = torch.zeros((3, *shape[-2:]), dtype=cache.dtype).npu()
    slots = torch.tensor([shape[0] * shape[1], -1, 0], dtype=torch.int32).npu()

    assert reshape_paged_cache(slots, rows, rows, cache, cache) is cache
    expected = torch.full(shape, 7, dtype=cache.dtype)
    expected[0, 0].zero_()
    torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)

# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real paged-attention live-prefix and first-body equivalence comparisons.

Run only after reserving an idle device through the container test launcher.
Independent CPU attention references preserve the full valid KV prefix and
reject stale cache state, an incorrect selected row, and an unmasked tail.
"""

from __future__ import annotations

import os
from itertools import accumulate
from math import prod
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="requires the real NPU backend")

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)

_OUTPUT_RTOL = 0.02
_OUTPUT_ATOL = 1e-3
_INDEX_GUARD_SIZE = 37
_INDEX_GUARD_VALUE = -123456789


def _guarded_shared_indices(shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full((prod(shape) + 2 * _INDEX_GUARD_SIZE,), _INDEX_GUARD_VALUE, dtype=torch.int32, device=device)
    return storage, storage.narrow(0, _INDEX_GUARD_SIZE, prod(shape)).view(shape)


def _assert_guarded_shared_indices(storage: torch.Tensor, expected: torch.Tensor) -> None:
    actual = storage.cpu()
    guard = [_INDEX_GUARD_VALUE] * _INDEX_GUARD_SIZE
    assert actual[:_INDEX_GUARD_SIZE].tolist() == guard
    assert actual[-_INDEX_GUARD_SIZE:].tolist() == guard
    torch.testing.assert_close(
        actual[_INDEX_GUARD_SIZE:-_INDEX_GUARD_SIZE].view(expected.shape), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(("groups", "batch", "row_gap"), [(1, 1, 0), (3, 259, 7)], ids=["c1", "strided-tail"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_shared_linear_index_fusion_refreshes_rows_and_ids(groups: int, batch: int, row_gap: int, graph: bool) -> None:
    device = torch.device("npu:0")
    capacity = 19
    row_storage = torch.full((2 * groups + 2, capacity + row_gap), -71, dtype=torch.int32, device=device)
    rows = row_storage[1:-1, :capacity]
    ids = torch.empty(batch, dtype=torch.int32, device=device)
    storage, output = _guarded_shared_indices((groups, batch, 2), device)
    addresses = ids.data_ptr(), rows.data_ptr(), output.data_ptr()
    captured = None
    for phase in range(3):
        # Python integer values distinguish adjacent rows above float32's exact
        # range. The row gaps and output guards also exercise partial stores.
        row_values = [
            [0, *((1 << 24) + 1009 * row + 3 * logical + 2003 * phase for logical in range(1, capacity))]
            for row in range(2 * groups)
        ]
        choices = (1, capacity - 1, 0, -1, 2, -(1 << 31))
        logical_ids = (
            [1 + phase % 2] if batch == 1 else [choices[(index + phase) % len(choices)] for index in range(batch)]
        )
        rows.copy_(torch.tensor(row_values, dtype=torch.int32, device=device))
        ids.copy_(torch.tensor(logical_ids, dtype=torch.int32, device=device))
        if captured is None:
            torch.ops.xllm_ops.shared_linear_state_indices_out(ids, rows, output)
            if graph:
                torch.npu.synchronize()
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    torch.ops.xllm_ops.shared_linear_state_indices_out(ids, rows, output)
        else:
            captured.replay()
        torch.npu.synchronize()
        expected = torch.tensor(
            [
                [
                    [row_values[2 * group + role][logical] if logical > 0 else -1 for role in range(2)]
                    for logical in logical_ids
                ]
                for group in range(groups)
            ],
            dtype=torch.int32,
        )
        _assert_guarded_shared_indices(storage, expected)
        assert addresses == (ids.data_ptr(), rows.data_ptr(), output.data_ptr())


@pytest.mark.parametrize(
    ("layers", "batch", "columns", "tokens", "row_gap", "page_size"),
    [(1, 1, 1, 1, 0, 128), (3, 2, 131, 259, 7, 128), (2, 1, 7, 11, 5, 1)],
    ids=["c1", "strided-tail", "large-physical-pages"],
)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_shared_full_index_fusion_refreshes_tables_and_slots(
    layers: int, batch: int, columns: int, tokens: int, row_gap: int, page_size: int, graph: bool
) -> None:
    device = torch.device("npu:0")
    capacity, padding_row = 19, 10
    row_storage = torch.full((layers + 2, capacity + row_gap), -71, dtype=torch.int32, device=device)
    rows = row_storage[1:-1, :capacity]
    table = torch.empty((batch, columns), dtype=torch.int32, device=device)
    slots = torch.empty(tokens, dtype=torch.int32, device=device)
    table_storage, output_tables = _guarded_shared_indices((layers, batch, columns), device)
    slot_storage, output_slots = _guarded_shared_indices((layers, tokens), device)
    addresses = tuple(tensor.data_ptr() for tensor in (rows, table, slots, output_tables, output_slots))
    captured = None

    def _run() -> None:
        torch.ops.xllm_ops.shared_full_indices_out(
            table, slots, rows, page_size, padding_row, output_tables, output_slots
        )

    for phase in range(3):
        base = (1 << 24) + 1 if page_size == 1 else (1 << 17) + 1
        row_values = [
            [0, *(base + 1009 * layer + 3 * logical + 2003 * phase for logical in range(1, capacity))]
            for layer in range(layers)
        ]
        choices = (1, 0, -1, capacity - 1, 2, -(1 << 31))
        table_values = (
            [1 + phase % 2]
            if batch * columns == 1
            else [choices[(index + phase) % len(choices)] for index in range(batch * columns)]
        )
        slot_choices = (
            page_size + min(1, page_size - 1),
            0,
            -1,
            3 * page_size - 1,
            (capacity - 1) * page_size + min(7, page_size - 1),
            -(1 << 31),
        )
        slot_values = (
            [(1 + phase % 2) * page_size + min(1, page_size - 1)]
            if tokens == 1
            else [slot_choices[(index + phase) % len(slot_choices)] for index in range(tokens)]
        )
        rows.copy_(torch.tensor(row_values, dtype=torch.int32, device=device))
        table.copy_(torch.tensor(table_values, dtype=torch.int32, device=device).view(batch, columns))
        slots.copy_(torch.tensor(slot_values, dtype=torch.int32, device=device))
        if captured is None:
            _run()
            if graph:
                torch.npu.synchronize()
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    _run()
        else:
            captured.replay()
        torch.npu.synchronize()
        expected_tables = torch.tensor(
            [
                [
                    row_values[layer][logical] if logical > 0 else padding_row if logical == 0 else logical
                    for logical in table_values
                ]
                for layer in range(layers)
            ],
            dtype=torch.int32,
        ).view(layers, batch, columns)
        expected_slots = torch.tensor(
            [
                [
                    slot if slot < 0 else row_values[layer][slot // page_size] * page_size + slot % page_size
                    for slot in slot_values
                ]
                for layer in range(layers)
            ],
            dtype=torch.int32,
        )
        _assert_guarded_shared_indices(table_storage, expected_tables)
        _assert_guarded_shared_indices(slot_storage, expected_slots)
        assert addresses == tuple(tensor.data_ptr() for tensor in (rows, table, slots, output_tables, output_slots))


@torch.inference_mode()
def test_shared_linear_index_fusion_preserves_empty_output_guards() -> None:
    device = torch.device("npu:0")
    rows = torch.zeros((4, 8), dtype=torch.int32, device=device)
    ids = torch.empty(0, dtype=torch.int32, device=device)
    storage, output = _guarded_shared_indices((2, 0, 2), device)
    torch.ops.xllm_ops.shared_linear_state_indices_out(ids, rows, output)
    torch.npu.synchronize()
    _assert_guarded_shared_indices(storage, torch.empty((2, 0, 2), dtype=torch.int32))


@pytest.mark.parametrize(("batch", "tokens"), [(0, 0), (0, 3), (2, 0)])
@torch.inference_mode()
def test_shared_full_index_fusion_handles_independent_empty_outputs(batch: int, tokens: int) -> None:
    device = torch.device("npu:0")
    rows = torch.tensor([[0, 6], [0, 14]], dtype=torch.int32, device=device)
    table = torch.tensor([[0, 1, -1]] * batch, dtype=torch.int32, device=device).view(batch, 3)
    slots = torch.tensor([0, 129, -1][:tokens], dtype=torch.int32, device=device)
    table_storage, output_tables = _guarded_shared_indices((2, batch, 3), device)
    slot_storage, output_slots = _guarded_shared_indices((2, tokens), device)
    torch.ops.xllm_ops.shared_full_indices_out(table, slots, rows, 128, 10, output_tables, output_slots)
    torch.npu.synchronize()
    expected_tables = torch.tensor([[[10, physical, -1]] * batch for physical in (6, 14)], dtype=torch.int32).view(
        2, batch, 3
    )
    expected_slots = torch.tensor([[0, physical * 128 + 1, -1][:tokens] for physical in (6, 14)], dtype=torch.int32)
    _assert_guarded_shared_indices(table_storage, expected_tables)
    _assert_guarded_shared_indices(slot_storage, expected_slots)


@pytest.mark.parametrize(
    ("batch", "columns", "fused"),
    [(5, 256, True), (8, 255, True), (8, 256, False), (32, 256, False)],
    ids=["c1-captured-table", "fusion-limit", "above-fusion-limit", "large-parallel-map"],
)
@torch.inference_mode()
def test_slab_full_mapping_size_dispatch_preserves_graph_updates(
    batch: int, columns: int, fused: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.attention.backend import LayerCache, SharedKVCacheLayout
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    device = torch.device("npu:0")
    layers, capacity = 16, 32
    backend = NpuPagedAttentionBackend(1, 1, 2, 1.0, 0, False, device, torch.float16)
    backend._shared_kv_layout = SharedKVCacheLayout(128, 8, 4, 8, 2, capacity, 262144, num_slabs=4)
    rows = torch.zeros((layers, capacity), dtype=torch.int32, device=device)
    cache = torch.empty((1, 128, 1, 2), device=device)
    backend._kv_caches = [LayerCache(cache, cache, full_block_rows=row) for row in rows.unbind(0)]
    table = torch.empty((batch, columns), dtype=torch.int32, device=device)
    slots = torch.empty(batch, dtype=torch.int32, device=device)
    backend._block_table_i32 = table
    metadata = SimpleNamespace(slot_mapping=slots, has_kv_shard=False)
    backend._metadata = metadata
    context = ForwardContext(backend, device, metadata, backend._kv_caches, execution_state=AclGraphExecutionState({}))
    native = torch.ops.xllm_ops.shared_full_indices_out
    fused_calls: list[int] = []

    def _fused(
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        row_maps: torch.Tensor,
        page_size: int,
        padding_row: int,
        output_tables: torch.Tensor,
        output_slots: torch.Tensor,
    ) -> None:
        fused_calls.append(output_tables.numel() + output_slots.numel())
        native(block_table, slot_mapping, row_maps, page_size, padding_row, output_tables, output_slots)

    monkeypatch.setattr(torch.ops.xllm_ops, "shared_full_indices_out", _fused)
    captured = None
    pointers = None
    with forward_context(context):
        for phase in range(3):
            logical = 1 + phase % 2
            physical = [16 + 8 * layer + 32 * phase for layer in range(layers)]
            rows[:, logical].copy_(torch.tensor(physical, dtype=torch.int32, device=device))
            table.fill_(logical)
            table[:, -2] = 0
            table[:, -1] = -1
            slots.fill_(logical * 128 + phase)
            slots[0] = 0
            slots[-1] = -1
            if captured is None:
                backend._prepare_shared_full_tables()
                torch.npu.synchronize()
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    backend._prepare_shared_full_tables()
            else:
                captured.replay()
            torch.npu.synchronize()
            outputs = (*backend._shared_full_block_tables.values(), *backend._shared_full_slot_mappings.values())
            current_pointers = tuple(output.data_ptr() for output in outputs)
            if pointers is None:
                pointers = current_pointers
            assert current_pointers == pointers
            padding = backend._shared_kv_layout.full_read_padding_row
            for layer, (output_table, output_slots) in enumerate(
                zip(backend._shared_full_block_tables.values(), backend._shared_full_slot_mappings.values())
            ):
                assert output_table.cpu().tolist() == [[physical[layer]] * (columns - 2) + [padding, -1]] * batch
                assert output_slots.cpu().tolist() == [0] + [physical[layer] * 128 + phase] * (batch - 2) + [-1]
    assert fused_calls == ([16 * (batch * columns + batch)] * 2 if fused else [])


@torch.inference_mode()
def test_prepared_slab_replay_reads_current_maps_into_each_graph_entry() -> None:
    from xllm.python.attention.backend import LayerCache, SharedKVCacheLayout
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    device = torch.device("npu:0")
    backend = NpuPagedAttentionBackend(1, 1, 2, 1.0, 0, False, device, torch.float16)
    rows = torch.zeros(2, 32, dtype=torch.int32, device=device)
    rows[:, 1:3].copy_(torch.tensor([[16, 18], [48, 50]], dtype=torch.int32, device=device))
    caches = [
        LayerCache(
            torch.empty(1, 128, 1, 2, device=device),
            torch.empty(1, 128, 1, 2, device=device),
            full_block_rows=row,
        )
        for row in rows.unbind(0)
    ]
    backend.bind_kv_caches(caches)
    layout = SharedKVCacheLayout(128, 8, 4, 8, 2, 32, 262144, num_slabs=4)
    backend._shared_kv_layout = layout
    contexts = []
    for _ in range(2):
        metadata = SimpleNamespace(
            block_table=torch.tensor([[1, 0, -1], [2, 0, -1]], dtype=torch.int32, device=device),
            slot_mapping=torch.tensor([129, -1], dtype=torch.int32, device=device),
            q_seq_lens=torch.ones(2, dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=[0, 1, 2],
            kv_seq_lens=torch.tensor([6, 4], dtype=torch.int32, device=device),
            kv_seq_lens_host_values=[6, 4],
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        contexts.append(ForwardContext(backend, device, metadata, caches, execution_state=AclGraphExecutionState({})))

    captures, outputs, pointers = [], [], []
    for context in contexts:
        with forward_context(context):
            backend.prepare(context.metadata, graph_mode=True)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                backend.prepare(context.metadata, graph_mode=True)
            captures.append(captured)
            tensors = (*backend._shared_full_block_tables.values(), *backend._shared_full_slot_mappings.values())
            outputs.append(tensors)
            pointers.append(tuple(tensor.data_ptr() for tensor in tensors))
    torch.npu.synchronize()
    assert all(first != second for first, second in zip(pointers[0], pointers[1], strict=True))

    # Logical IDs and the native row tables retain their addresses while a
    # released lease acquires new physical rows. Replay must read both anew.
    rows[:, 2].copy_(torch.tensor([72, 80], dtype=torch.int32, device=device))
    contexts[0].metadata.block_table[0, 0] = 2
    contexts[0].metadata.slot_mapping[0] = 258
    with forward_context(contexts[0]):
        backend.prepare_graph_replay(contexts[0].metadata)
        assert pointers[0] == tuple(
            tensor.data_ptr()
            for tensor in (*backend._shared_full_block_tables.values(), *backend._shared_full_slot_mappings.values())
        )
    captures[0].replay()
    torch.npu.synchronize()
    padding = layout.full_read_padding_row
    assert outputs[0][0].cpu().tolist() == [[72, padding, -1], [72, padding, -1]]
    assert outputs[0][1].cpu().tolist() == [[80, padding, -1], [80, padding, -1]]
    assert outputs[0][2].cpu().tolist() == [9218, -1]
    assert outputs[0][3].cpu().tolist() == [10242, -1]
    assert outputs[1][0].cpu().tolist() == [[16, padding, -1], [18, padding, -1]]
    assert outputs[1][2].cpu().tolist() == [2049, -1]

    with forward_context(contexts[1]):
        backend.prepare_graph_replay(contexts[1].metadata)
    captures[1].replay()
    torch.npu.synchronize()
    assert outputs[1][0].cpu().tolist() == [[16, padding, -1], [72, padding, -1]]
    assert outputs[1][1].cpu().tolist() == [[48, padding, -1], [80, padding, -1]]
    assert outputs[1][2].cpu().tolist() == [2049, -1]
    assert outputs[0][2].cpu().tolist() == [9218, -1]


@torch.inference_mode()
def test_ordinary_slab_replay_refreshes_host_tables_per_graph_entry() -> None:
    from xllm.python.attention.backend import LayerCache, SharedKVCacheLayout
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    device = torch.device("npu:0")
    backend = NpuPagedAttentionBackend(2, 1, 64, 0.125, 0, False, device, torch.bfloat16)
    rows = torch.zeros(32, dtype=torch.int32, device=device)
    rows[1:3].copy_(torch.tensor([16, 18], dtype=torch.int32, device=device))
    cache = torch.empty(128, 128, 1, 64, dtype=torch.bfloat16, device=device)
    backend.bind_kv_caches([LayerCache(cache, cache, full_block_rows=rows)])
    layout = SharedKVCacheLayout(128, 8, 4, 8, 2, 32, 262144, num_slabs=4)
    backend._shared_kv_layout = layout
    contexts, captures, outputs = [], [], []
    for _ in range(2):
        metadata = SimpleNamespace(
            block_table=torch.tensor([[1, 0, -1], [2, 0, -1]], dtype=torch.int32, device=device),
            slot_mapping=torch.tensor([129, -1], dtype=torch.int32, device=device),
            q_seq_lens=torch.ones(2, dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=[0, 1, 2],
            kv_seq_lens=torch.tensor([6, 4], dtype=torch.int32, device=device),
            kv_seq_lens_host_values=[6, 4],
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
        )
        context = ForwardContext(
            backend, device, metadata, backend._kv_caches, execution_state=AclGraphExecutionState({})
        )
        with forward_context(context):
            backend.prepare(metadata, graph_mode=True)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                output = torch.cat(
                    (backend._shared_full_block_tables[0].flatten(), backend._shared_full_slot_mappings[0])
                )
            captures.append(captured)
            outputs.append(output)
        contexts.append(context)
    torch.npu.synchronize()
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    padding = layout.full_read_padding_row

    for step, index in enumerate((0, 1, 0)):
        context = contexts[index]
        physical = 72 + 2 * step
        rows[2] = physical
        context.metadata.block_table[0, 0] = 2
        context.metadata.slot_mapping[0] = 258
        with forward_context(context):
            backend.prepare_graph_replay(context.metadata)
        captures[index].replay()
        torch.npu.synchronize()
        assert outputs[index].cpu().tolist() == [physical, padding, -1, physical, padding, -1, physical * 128 + 2, -1]
    assert outputs[1].cpu().tolist() == [74, padding, -1, 74, padding, -1, 9474, -1]


@pytest.mark.parametrize("layer_id", [0, 3], ids=["first-full-role", "last-full-role"])
@pytest.mark.parametrize("width", [1, 3], ids=["decode", "expanded-verify"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_shared_full_attention_matches_dense_after_chunked_prefill(layer_id: int, width: int, graph: bool) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    device = torch.device("npu:0")
    heads, kv_heads, dimension, page_size, page_count = 4, 2, 64, 128, 8
    full_row_bytes, conv_row_bytes, ssm_row_bytes, page_bytes = 32768, 5376, 65536, 393216
    arena = torch.zeros(page_count * page_bytes, device=device, dtype=torch.uint8)

    def arena_view(offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        byte_count = prod(shape) * torch.empty((), dtype=dtype).element_size()
        return arena.narrow(0, offset, byte_count).view(dtype).view(shape)

    full_rows = (page_count - 1) * page_bytes // full_row_bytes + 1
    shared_caches = [
        LayerCache(
            key=arena_view(2 * role * full_row_bytes, (full_rows, page_size, kv_heads, dimension), torch.bfloat16),
            value=arena_view(
                (2 * role + 1) * full_row_bytes, (full_rows, page_size, kv_heads, dimension), torch.bfloat16
            ),
        )
        for role in range(4)
    ]
    shared_caches.append(
        LayerCache(
            key=None,
            value=None,
            conv=arena_view(
                0,
                (((page_count - 1) * page_bytes + conv_row_bytes - 1) // conv_row_bytes + 1, 7, 384),
                torch.bfloat16,
            ),
            ssm=arena_view(
                ssm_row_bytes, ((page_count - 1) * page_bytes // ssm_row_bytes + 5, 1, 128, 128), torch.float32
            ),
            linear_state_slots=torch.arange(page_count, device=device, dtype=torch.int32),
        )
    )
    dense_caches = [
        LayerCache(
            key=torch.zeros(page_count, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16),
            value=torch.zeros(page_count, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16),
        )
        for _ in range(4)
    ]
    cache_sets = [dense_caches, shared_caches]
    backends = [
        NpuPagedAttentionBackend(heads, kv_heads, dimension, dimension**-0.5, 0, False, device, torch.bfloat16)
        for _ in cache_sets
    ]
    for backend, caches in zip(backends, cache_sets):
        backend.bind_kv_caches(caches)
    layout = backends[1].shared_kv_layout
    assert backends[0].shared_kv_layout is None
    assert layout is not None and layout.slot_bytes == page_bytes and layout.conv_row_stride is None
    attention = Attention(heads, kv_heads, dimension, dimension**-0.5, 0, layer_id)
    generator = torch.Generator().manual_seed(37031 + layer_id * 7 + width)
    table = [[5, 2], [1, 6]]
    logical_pages = torch.tensor([1, 2, 5, 6], device=device, dtype=torch.int64)
    physical_pages = layout.full_block_table(logical_pages)
    protected = shared_caches[(layer_id + 1) % 4]
    protected_keys = protected.key.index_select(0, physical_pages).clone()
    protected_values = protected.value.index_select(0, physical_pages).clone()
    # The allocator can assign this same arena page to Linear states. Full
    # cache writes must not reach it or the neighboring Full role payload.
    linear_page = arena.narrow(0, 7 * page_bytes, page_bytes)
    linear_page.fill_(0x5A)
    protected_linear = linear_page.clone()
    histories: list[tuple[torch.Tensor, torch.Tensor]] = [
        (
            torch.empty(0, kv_heads, dimension, dtype=torch.bfloat16),
            torch.empty(0, kv_heads, dimension, dtype=torch.bfloat16),
        )
        for _ in table
    ]

    def logical_slots(blocks: list[list[int]], starts: list[int], counts: list[int]) -> list[int]:
        # Scheduler slots use logical block ids and retain the token offset.
        return [
            row[position // page_size] * page_size + position % page_size
            for row, start, count in zip(blocks, starts, counts)
            for position in range(start, start + count)
        ]

    def metadata_for(
        backend: NpuPagedAttentionBackend,
        blocks: list[list[int]],
        counts: list[int],
        lengths: list[int],
        slots: list[int],
        *,
        prefill: bool = False,
        chunked: bool = False,
        verify: bool = False,
        device_lengths: bool = False,
    ) -> SimpleNamespace:
        ends = list(accumulate(counts, initial=0))
        metadata = SimpleNamespace(
            slot_mapping=torch.tensor(slots, device=device, dtype=torch.int32),
            block_table=torch.tensor(blocks, device=device, dtype=torch.int32),
            q_seq_lens=torch.tensor(counts, device=device, dtype=torch.int32),
            q_cu_seq_lens=torch.tensor(ends, device=device, dtype=torch.int32),
            q_cu_seq_lens_host_values=ends,
            kv_seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            kv_seq_lens_host_values=[2 * page_size] * len(lengths) if device_lengths else lengths,
            is_prefill=prefill,
            is_chunked_prefill=chunked,
            is_spec_verify=verify,
            is_mixed=False,
            has_kv_shard=False,
            expanded_decode_metadata=None,
            linear_state_indices=torch.tensor([3, 4], device=device, dtype=torch.int32) if verify else None,
            num_accepted_tokens=torch.ones(2, device=device, dtype=torch.int32) if verify else None,
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=device_lengths)
        return metadata

    def reference(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, start: int) -> torch.Tensor:
        repeated_keys = keys.float().repeat_interleave(heads // kv_heads, dim=1)
        repeated_values = values.float().repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("thd,khd->htk", query.float(), repeated_keys) * dimension**-0.5
        future = torch.arange(keys.shape[0])[None, :] > start + torch.arange(query.shape[0])[:, None]
        scores.masked_fill_(future[None], float("-inf"))
        return torch.einsum("htk,khd->thd", scores.softmax(-1), repeated_values).reshape(query.shape[0], -1)

    def check_cache_pages() -> None:
        for field in ("key", "value"):
            actual = getattr(shared_caches[layer_id], field).index_select(0, physical_pages)
            expected = getattr(dense_caches[layer_id], field).index_select(0, logical_pages)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(protected.key.index_select(0, physical_pages), protected_keys, rtol=0, atol=0)
        torch.testing.assert_close(protected.value.index_select(0, physical_pages), protected_values, rtol=0, atol=0)
        torch.testing.assert_close(linear_page, protected_linear, rtol=0, atol=0)

    # The second chunk crosses 127 -> 128 and writes nonzero in-page offsets
    # on nonadjacent physical blocks before the later cached-prefix reads.
    for phase, counts in enumerate(([127, 130], [3, 2])):
        starts = [keys.shape[0] for keys, _ in histories]
        total = sum(counts)
        query_host = (torch.randn(total, heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        key_host = (torch.randn(total, kv_heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        value_host = (torch.randn(total, kv_heads, dimension, generator=generator) * 0.1).to(torch.bfloat16)
        expected, offset = [], 0
        for row, count in enumerate(counts):
            value_host[offset : offset + count].add_(
                row * 1.5 + torch.arange(starts[row], starts[row] + count).view(-1, 1, 1) / page_size
            )
            keys, values = histories[row]
            keys = torch.cat((keys, key_host[offset : offset + count]))
            values = torch.cat((values, value_host[offset : offset + count]))
            histories[row] = keys, values
            expected.append(reference(query_host[offset : offset + count], keys, values, starts[row]))
            offset += count
        query, keys, values = query_host.to(device), key_host.to(device), value_host.to(device)
        outputs = []
        for backend, caches in zip(backends, cache_sets):
            metadata = metadata_for(
                backend,
                table,
                counts,
                [start + count for start, count in zip(starts, counts)],
                logical_slots(table, starts, counts),
                prefill=phase == 0,
                chunked=phase == 1,
            )
            with forward_context(ForwardContext(backend, device, metadata, caches)):
                backend.prepare(metadata)
                outputs.append(attention(query, keys, values))
        torch.testing.assert_close(outputs[1], outputs[0], rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(outputs[1].float().cpu(), torch.cat(expected), rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        check_cache_pages()

    rows = 2 * width
    query = torch.zeros(rows, heads, dimension, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(rows, kv_heads, dimension, device=device, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    read_slots = logical_slots(table[::-1], [132, 130], [width, width])
    read_lengths = [start + column + 1 for start in (132, 130) for column in range(width)]
    read_metadata = [
        metadata_for(
            backend,
            [row for row in table[::-1] for _ in range(width)],
            [1] * rows,
            read_lengths,
            read_slots,
            verify=width > 1,
            device_lengths=True,
        )
        for backend in backends
    ]
    states = [AclGraphExecutionState({}) for _ in backends]

    def read_forward(index: int) -> torch.Tensor:
        metadata = read_metadata[index]
        context = ForwardContext(
            backends[index], device, metadata, cache_sets[index], execution_state=states[index] if graph else None
        )
        with forward_context(context):
            backends[index].prepare(metadata)
            return attention(query, keys, values)

    captures, captured_outputs = [], []
    if graph:
        for index in range(2):
            read_forward(index)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                captured_outputs.append(read_forward(index))
            captures.append(captured)
        torch.npu.synchronize()

    for iteration, order in enumerate(([1, 0], [0, 1], [1, 0])):
        current_table = [table[row] for row in order]
        starts = [histories[row][0].shape[0] - iteration for row in order]
        query_host = (torch.randn(rows, heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        key_host = (torch.randn(rows, kv_heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        value_host = (torch.randn(rows, kv_heads, dimension, generator=generator) * 0.1 + iteration + 3).to(
            torch.bfloat16
        )
        query.copy_(query_host.to(device))
        keys.copy_(key_host.to(device))
        values.copy_(value_host.to(device))
        slots = torch.tensor(logical_slots(current_table, starts, [width, width]), device=device, dtype=torch.int32)
        lengths = torch.tensor(
            [start + column + 1 for start in starts for column in range(width)], device=device, dtype=torch.int32
        )
        expanded_table = torch.tensor(
            [row for row in current_table for _ in range(width)], device=device, dtype=torch.int32
        )
        for metadata in read_metadata:
            metadata.slot_mapping.copy_(slots)
            metadata.block_table.copy_(expanded_table)
            metadata.kv_seq_lens.copy_(lengths)
        outputs = []
        for index in range(2):
            if graph:
                captures[index].replay()
                outputs.append(captured_outputs[index])
            else:
                outputs.append(read_forward(index))
        expected = []
        for row, source in enumerate(order):
            begin, end = row * width, (row + 1) * width
            prefix_keys, prefix_values = histories[source]
            expected.append(
                reference(
                    query_host[begin:end],
                    torch.cat((prefix_keys[: starts[row]], key_host[begin:end])),
                    torch.cat((prefix_values[: starts[row]], value_host[begin:end])),
                    starts[row],
                )
            )
        torch.testing.assert_close(outputs[1], outputs[0], rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(outputs[1].float().cpu(), torch.cat(expected), rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        check_cache_pages()
    torch.npu.synchronize()


@pytest.mark.parametrize("layer_id", [0, 3], ids=["first-full-role", "last-full-role"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_slab_full_device_lengths_ignore_poisoned_writable_scratch(layer_id: int, graph: bool) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    device = torch.device("npu:0")
    heads, kv_heads, dimension, page_size, width, num_slots = 4, 2, 64, 128, 5, 8
    full_row_bytes, conv_row_bytes, ssm_row_bytes, stride = 32768, 4096, 65536, 3
    slab_bytes, num_slabs = stride * ssm_row_bytes, 8
    arena_bytes = num_slabs * slab_bytes
    arena = torch.zeros(arena_bytes, device=device, dtype=torch.uint8)

    def _arena_view(offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        byte_count = prod(shape) * torch.empty((), dtype=dtype).element_size()
        return arena.narrow(0, offset, byte_count).view(dtype).view(shape)

    shared_caches = []
    row_maps: dict[int, torch.Tensor] = {}
    for index in range(4):
        if index in (0, 3):
            rows = torch.zeros(num_slots, dtype=torch.int32, device=device)
            rows[1:4].copy_(torch.tensor([12, 24, 36], dtype=torch.int32, device=device) + (2 if index else 0))
            row_maps[index] = rows
            shared_caches.append(
                LayerCache(
                    key=_arena_view(0, (arena_bytes // full_row_bytes, page_size, kv_heads, dimension), torch.bfloat16),
                    value=_arena_view(
                        full_row_bytes,
                        (arena_bytes // full_row_bytes - 1, page_size, kv_heads, dimension),
                        torch.bfloat16,
                    ),
                    full_block_rows=rows,
                    shared_checkpoint_stride=stride,
                )
            )
        else:
            shared_caches.append(
                LayerCache(
                    key=None,
                    value=None,
                    conv=_arena_view(0, (arena_bytes // conv_row_bytes, 4, 512), torch.bfloat16),
                    ssm=_arena_view(0, (arena_bytes // ssm_row_bytes, 1, 128, 128), torch.float32),
                    linear_state_slots=torch.zeros(2, num_slots, dtype=torch.int32, device=device),
                    shared_checkpoint_stride=stride,
                )
            )
    dense_caches = [
        LayerCache(
            key=torch.zeros(num_slots, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16),
            value=torch.zeros(num_slots, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16),
        )
        for _ in shared_caches
    ]
    cache_sets = [dense_caches, shared_caches]
    backends = [
        NpuPagedAttentionBackend(heads, kv_heads, dimension, dimension**-0.5, 0, False, device, torch.bfloat16)
        for _ in cache_sets
    ]
    for backend, caches in zip(backends, cache_sets):
        backend.bind_kv_caches(caches)
    layout = backends[1].shared_kv_layout
    assert layout is not None and layout.num_slabs == num_slabs
    assert layout.full_read_padding_row == 6
    attention = Attention(heads, kv_heads, dimension, dimension**-0.5, 0, layer_id)
    generator = torch.Generator().manual_seed(70301 + layer_id)
    prefix_tokens = 2 * page_size
    prefix_keys = (torch.randn(prefix_tokens, kv_heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
    prefix_values = (
        torch.randn(prefix_tokens, kv_heads, dimension, generator=generator) * 0.1
        + torch.arange(prefix_tokens).view(-1, 1, 1) / page_size
    ).to(torch.bfloat16)
    logical_prefix = torch.tensor([2, 1], dtype=torch.int64, device=device)
    physical_prefix = layout.full_block_table(logical_prefix, row_maps[layer_id]).to(torch.int64)
    for index, caches in enumerate(cache_sets):
        selected_rows = logical_prefix if index == 0 else physical_prefix
        caches[layer_id].key.index_copy_(
            0, selected_rows, prefix_keys.view(2, page_size, kv_heads, dimension).to(device)
        )
        caches[layer_id].value.index_copy_(
            0, selected_rows, prefix_values.view(2, page_size, kv_heads, dimension).to(device)
        )
    # A new Full lease initializes its whole K/V pair once; its unwritten
    # 123-token tail stays finite. Native allocator tests cover lease reuse.
    new_row = int(row_maps[layer_id][3].item())
    shared_full = shared_caches[layer_id]
    protected_full = shared_caches[3 if layer_id == 0 else 0]
    protected_rows = row_maps[3 if layer_id == 0 else 0][1:4].to(torch.int64)
    protected_keys = protected_full.key.index_select(0, protected_rows).clone()
    protected_values = protected_full.value.index_select(0, protected_rows).clone()
    linear_slab = arena.narrow(0, 7 * slab_bytes, slab_bytes)
    linear_slab.fill_(0x3F)
    protected_linear = linear_slab.clone()
    query = torch.zeros(width, heads, dimension, dtype=torch.bfloat16, device=device)
    keys = torch.zeros(width, kv_heads, dimension, dtype=torch.bfloat16, device=device)
    values = torch.zeros_like(keys)
    live_lengths = list(range(prefix_tokens + 1, prefix_tokens + width + 1))
    metadata_sets = []
    for backend in backends:
        metadata = SimpleNamespace(
            slot_mapping=torch.arange(3 * page_size, 3 * page_size + width, dtype=torch.int32, device=device),
            block_table=torch.tensor([[2, 1, 3, 0, 0, 0, 0, 0]] * width, dtype=torch.int32, device=device),
            q_seq_lens=torch.ones(width, dtype=torch.int32, device=device),
            q_cu_seq_lens=torch.arange(width + 1, dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=list(range(width + 1)),
            kv_seq_lens=torch.tensor(live_lengths, dtype=torch.int32, device=device),
            kv_seq_lens_host_values=[num_slots * page_size] * width,
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=True,
            is_mixed=False,
            has_kv_shard=False,
            expanded_decode_metadata=None,
            linear_state_indices=torch.ones(1, dtype=torch.int32, device=device),
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        metadata_sets.append(metadata)
    states = [AclGraphExecutionState({}) for _ in backends]

    def _forward(index: int) -> torch.Tensor:
        context = ForwardContext(
            backends[index],
            device,
            metadata_sets[index],
            cache_sets[index],
            execution_state=states[index] if graph else None,
        )
        with forward_context(context):
            backends[index].prepare(metadata_sets[index])
            return attention(query, keys, values)

    captures, captured_outputs = [], []
    if graph:
        for index in range(2):
            _forward(index)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                captured_outputs.append(_forward(index))
            captures.append(captured)
        torch.npu.synchronize()

    # Writable scratch can retain NaNs from Linear state or graph warmup.
    # The fixed-capacity FIA table must read the separate finite padding pair.
    shared_full.key[0].fill_(float("nan"))
    shared_full.value[0].fill_(float("nan"))
    shared_caches[1].conv[0].fill_(float("nan"))
    shared_caches[1].ssm[:stride].fill_(float("nan"))
    assert torch.isnan(shared_full.value[0]).any().item()
    padding = layout.full_read_padding_row
    torch.testing.assert_close(shared_full.key[padding], torch.zeros_like(shared_full.key[padding]), rtol=0, atol=0)
    torch.testing.assert_close(shared_full.value[padding], torch.zeros_like(shared_full.value[padding]), rtol=0, atol=0)

    for iteration in range(2):
        query_host = (torch.randn(width, heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        key_host = (torch.randn(width, kv_heads, dimension, generator=generator) * 0.2).to(torch.bfloat16)
        value_host = (torch.randn(width, kv_heads, dimension, generator=generator) * 0.1 + iteration + 2).to(
            torch.bfloat16
        )
        query.copy_(query_host.to(device))
        keys.copy_(key_host.to(device))
        values.copy_(value_host.to(device))
        outputs = []
        for index in range(2):
            if graph:
                captures[index].replay()
                outputs.append(captured_outputs[index])
            else:
                outputs.append(_forward(index))
        all_keys = torch.cat((prefix_keys, key_host)).float().repeat_interleave(heads // kv_heads, dim=1)
        all_values = torch.cat((prefix_values, value_host)).float().repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("thd,khd->htk", query_host.float(), all_keys) * dimension**-0.5
        invalid = torch.arange(prefix_tokens + width)[None] >= torch.tensor(live_lengths)[:, None]
        scores.masked_fill_(invalid[None], float("-inf"))
        expected = torch.einsum("htk,khd->thd", scores.softmax(-1), all_values).reshape(width, -1)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(outputs[1].float().cpu(), expected, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(
            shared_full.key.index_select(0, physical_prefix).cpu(),
            prefix_keys.view(2, page_size, kv_heads, dimension),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            shared_full.value.index_select(0, physical_prefix).cpu(),
            prefix_values.view(2, page_size, kv_heads, dimension),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(shared_full.key[new_row, :width], keys, rtol=0, atol=0)
        torch.testing.assert_close(shared_full.value[new_row, :width], values, rtol=0, atol=0)
        torch.testing.assert_close(
            shared_full.key[new_row, width:], torch.zeros_like(shared_full.key[new_row, width:]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            shared_full.value[new_row, width:], torch.zeros_like(shared_full.value[new_row, width:]), rtol=0, atol=0
        )
        torch.testing.assert_close(protected_full.key.index_select(0, protected_rows), protected_keys, rtol=0, atol=0)
        torch.testing.assert_close(
            protected_full.value.index_select(0, protected_rows), protected_values, rtol=0, atol=0
        )
        torch.testing.assert_close(linear_slab, protected_linear, rtol=0, atol=0)
        torch.testing.assert_close(
            shared_full.value[padding], torch.zeros_like(shared_full.value[padding]), rtol=0, atol=0
        )
    torch.npu.synchronize()


@pytest.mark.parametrize("rows", [1, 5], ids=["draft-step", "draft-prefix"])
@pytest.mark.parametrize("shared_draft", [False, True], ids=["dense", "shared-companion"])
@torch.inference_mode()
def test_ordinary_draft_graph_reads_growing_kv_prefix(
    rows: int, shared_draft: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.attention.paged_kv_metadata import build_row_aligned_paged_kv_metadata
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    device = torch.device("npu:0")
    heads, kv_heads, dimension, page_size, capacity = 4, 2, 64, 128, 4096
    backend = NpuPagedAttentionBackend(heads, kv_heads, dimension, dimension**-0.5, 0, False, device, torch.bfloat16)
    values_host = torch.arange(33 * page_size, dtype=torch.float32) / 2048
    values_host = values_host.view(33, page_size, 1, 1).expand(33, page_size, kv_heads, dimension)
    if shared_draft:
        full_row_bytes = page_size * kv_heads * dimension * 2
        slice_bytes, bank_bytes = 6 * full_row_bytes, 96 * full_row_bytes
        arena = torch.zeros(3 * bank_bytes, device=device, dtype=torch.uint8)
        # The first two banks belong to the target. Poisoning them detects
        # a draft view that accidentally retains the target bank's base.
        arena[: 2 * bank_bytes].view(torch.bfloat16).fill_(float("nan"))
        view_bytes = bank_bytes - full_row_bytes
        key = arena.narrow(0, 2 * bank_bytes, view_bytes).view(torch.bfloat16)
        value = arena.narrow(0, 2 * bank_bytes + full_row_bytes, view_bytes).view(torch.bfloat16)
        key = key.view(95, page_size, kv_heads, dimension)
        value = value.view_as(key)
        physical_pages = 12 + 2 * torch.arange(33, dtype=torch.int64, device=device)
        # K and V views overlap. Write only the allocated V lanes so the
        # paired K lanes and the immutable read-padding pair stay zero.
        value.index_copy_(0, physical_pages, values_host.contiguous().to(device=device, dtype=key.dtype))
        caches = [
            LayerCache(
                key=key,
                value=value,
                full_block_rows=torch.zeros(64, dtype=torch.int32, device=device),
                shared_checkpoint_stride=3,
                shared_bank_bytes=bank_bytes,
                shared_slice_bytes=slice_bytes,
                shared_conv_scratch_row=0,
                shared_ssm_scratch_row=0,
                shared_full_read_padding_page=6,
            )
        ]
    else:
        key = torch.zeros(33, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16)
        value = values_host.contiguous().to(device=device, dtype=key.dtype)
        caches = [LayerCache(key=key, value=value)]
    backend.bind_kv_caches(caches)
    assert (backend.shared_kv_layout is not None) == shared_draft
    attention = Attention(heads, kv_heads, dimension, dimension**-0.5, 0, 0)
    query = torch.zeros(rows, heads * dimension, device=device, dtype=key.dtype)
    projected = torch.zeros(rows, kv_heads * dimension, device=device, dtype=key.dtype)

    class _DraftAttention(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return attention(query, projected, projected)

    runner = DecodeAclGraphRunner(
        _DraftAttention(),
        backend,
        device,
        rows,
        capacity,
        is_draft_model=True,
        enable_graph_mode_decode_no_padding=True,
        device_kv_lengths=True,
    )
    runner.bind_layer_caches(caches)
    tokens = torch.zeros(rows, dtype=torch.int32, device=device)
    table = torch.arange(1, 33, dtype=torch.int32, device=device).expand(rows, -1).contiguous()

    def _metadata_for(length: int, *, dummy: bool = False) -> SimpleNamespace:
        lengths = list(range(length, length + rows))
        indptr, indices, last_page = build_row_aligned_paged_kv_metadata(table, lengths, page_size)
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
            expanded_decode_metadata=None,
            block_table=table,
            kv_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
            kv_seq_lens_host_values=lengths,
            kv_cu_seq_lens=None,
            q_seq_lens=torch.ones(rows, dtype=torch.int32, device=device),
            q_cu_seq_lens=torch.arange(rows + 1, dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=list(range(rows + 1)),
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last_page,
            slot_mapping=torch.full((rows,), -1, dtype=torch.int32, device=device),
            linear_state_indices=None,
        )
        if shared_draft:
            metadata.is_dummy = dummy
            if not dummy:
                metadata.shared_full_block_table = (12 + 2 * table).contiguous()
                metadata.shared_full_slot_mapping = metadata.slot_mapping
                metadata.shared_full_read_padding_page = 6
        return metadata

    runner.warmup(tokens, tokens, _metadata_for(8, dummy=shared_draft), is_graph_warmup=True)
    # With zero queries and keys, attention must average the entire live
    # prefix. This distinguishes a frozen short capture bound from the mask.
    for length in (129, 1875, 7, 1987):
        metadata = _metadata_for(length)
        decoded = runner.prepare_execution(tokens, metadata)
        assert decoded is not None
        graph_key = runner.warmup(tokens, tokens, metadata, decode_metadata=decoded)
        actual = runner.execute(tokens, tokens, metadata, graph_key=graph_key, decode_metadata=decoded)
        expected = (
            torch.tensor(
                [(page_size + (live_length - 1) / 2) / 2048 for live_length in metadata.kv_seq_lens_host_values]
            )
            .view(rows, 1)
            .expand(rows, heads * dimension)
        )
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("policy", ["dspark", "dspark_causal", "dflash2"])
@pytest.mark.parametrize("kv_heads,dimension", [(4, 64), (1, 128)])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("qk_scale", [0.1, 1.0])
@torch.inference_mode()
def test_windowed_draft_attention_matches_global_reference_across_long_contexts(
    policy: str, kv_heads: int, dimension: int, graph: bool, qk_scale: float
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.layers.attention import Attention

    device = torch.device("npu:0")
    batch, width, heads, page_size, source_pages = 2, 8, 4, 128, 514
    generator = torch.Generator().manual_seed(71029)
    shape = (batch * source_pages, page_size, kv_heads, dimension)
    key_host = (torch.randn(shape, generator=generator) * qk_scale).to(torch.bfloat16)
    value_host = torch.empty(shape, dtype=torch.bfloat16)
    query_host = (torch.randn((batch * width, heads, dimension), generator=generator) * qk_scale).to(torch.bfloat16)
    table_host = torch.randperm(batch * source_pages, generator=generator).to(torch.int32).view(batch, source_pages)
    global_positions = torch.arange(source_pages * page_size).view(source_pages, page_size, 1, 1)
    head_bias = torch.arange(kv_heads).view(1, 1, -1, 1) * 0.2
    channel_bias = torch.arange(dimension).remainder(7).view(1, 1, 1, -1) * 0.03
    for row in range(batch):
        # A slowly varying global-position signal distinguishes a true sliding
        # window from an accidentally unmasked full prefix at BF16 tolerance.
        logical_values = torch.sin((global_positions + row * 47) / 512.0) + head_bias + channel_bias
        value_host.index_copy_(0, table_host[row].to(torch.int64), logical_values.to(torch.bfloat16))
    key, value, query = key_host.to(device), value_host.to(device), query_host.to(device)
    backend = NpuPagedAttentionBackend(heads, kv_heads, dimension, dimension**-0.5, 2048, False, device, torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=key, value=value)])
    layer = Attention(
        heads,
        kv_heads,
        dimension,
        dimension**-0.5,
        2048,
        0,
        causal=policy == "dspark_causal",
        fia_sparse_mode=4 if policy == "dflash2" else 0,
        fia_pre_tokens=2047,
        fia_next_tokens=width - 1,
        fia_use_attention_mask=True,
    )
    slots = []

    def forward(metadata: SimpleNamespace) -> torch.Tensor:
        backend.prepare(metadata)
        return backend._prefill(query, query, query, key, value, metadata, batch * width, layer)

    def original_forward(metadata: SimpleNamespace) -> torch.Tensor:
        # Match the pre-windowing BNSD path on the same physical KV cache.
        # The independent CPU oracle below checks both implementations.
        backend.prepare(metadata)
        mask = backend._block_attention_mask(metadata.prepared_attention_state, layer)
        q = query.view(batch, width, heads, dimension).transpose(1, 2).contiguous()
        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            key.view(key.shape[0], page_size, -1),
            value.view(value.shape[0], page_size, -1),
            atten_mask=mask,
            block_table=metadata.block_table,
            actual_seq_lengths=[width] * batch,
            actual_seq_lengths_kv=[source_pages * page_size] * batch,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            scale=dimension**-0.5,
            input_layout="BNSD",
            block_size=page_size,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            softmax_lse_flag=False,
        )
        return output.transpose(1, 2).reshape(batch * width, heads * dimension)

    pool = torch.npu.graph_pool_handle() if graph else None
    for _ in range(2):
        metadata = SimpleNamespace(
            slot_mapping=torch.full((batch * width,), -1, dtype=torch.int32, device=device),
            q_seq_lens=torch.full((batch,), width, dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=[width, 2 * width],
            block_table=table_host.to(device),
            kv_seq_lens=torch.full((batch,), width, dtype=torch.int32, device=device),
            kv_seq_lens_host_values=[source_pages * page_size] * batch,
            is_prefill=False,
            is_chunked_prefill=True,
            is_spec_verify=False,
            has_kv_shard=False,
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        captured = None
        output = None
        original_graph = None
        original_output = None
        if graph:
            forward(metadata)
            original_forward(metadata)
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured, pool=pool):
                output = forward(metadata)
            original_graph = torch.npu.NPUGraph()
            with torch.npu.graph(original_graph, pool=pool):
                original_output = original_forward(metadata)
            torch.npu.synchronize()
        slots.append((metadata, captured, output, original_graph, original_output))

    for length in (8, 127, 129, 4095, 4097, 16383, 16385, 32767, 32769, 65535, 65537, 137):
        for slot, (metadata, captured, output, original_graph, original_output) in enumerate(slots):
            lengths = [length, max(width, length - 127)]
            # The second Slot has a different logical-to-physical mapping.
            # Replacing table contents must not require another capture.
            table = table_host if slot == 0 else table_host.flip(1)
            metadata.block_table.copy_(table.to(device))
            metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            if captured is None:
                actual = forward(metadata)
            else:
                captured.replay()
                actual = output
            actual_host = actual.float().cpu().view(batch, width, heads, dimension)
            if original_graph is None:
                original = original_forward(metadata)
            else:
                original_graph.replay()
                original = original_output
            original_host = original.float().cpu().view(batch, width, heads, dimension)
            torch.testing.assert_close(actual_host, original_host, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
            for row, kv_length in enumerate(lengths):
                anchor = kv_length - width
                for column in range(width):
                    if policy == "dflash2":
                        lower = max(0, anchor + column - 2047)
                        upper = min(kv_length, anchor + column + width)
                    else:
                        lower = max(0, anchor - 2048)
                        upper = anchor + column + 1 if policy == "dspark_causal" else kv_length
                    positions = torch.arange(lower, upper, dtype=torch.int64)
                    physical_pages = table[row, positions // page_size].to(torch.int64)
                    offsets = positions % page_size
                    keys = key_host[physical_pages, offsets].float().repeat_interleave(heads // kv_heads, dim=1)
                    values = value_host[physical_pages, offsets].float().repeat_interleave(heads // kv_heads, dim=1)
                    q = query_host[row * width + column].float()
                    scores = torch.einsum("hd,thd->ht", q, keys) * dimension**-0.5
                    expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
                    torch.testing.assert_close(actual_host[row, column], expected, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
                    torch.testing.assert_close(
                        original_host[row, column], expected, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL
                    )
    torch.npu.synchronize()


@pytest.mark.parametrize("graph", [False, True])
def test_live_device_attention_excludes_rejected_kv_tail(graph: bool) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(41023)
    key_host = torch.randn((2, 128, 1, 64), generator=generator) * 0.1
    value_host = torch.randn((2, 128, 1, 64), generator=generator) * 0.1
    # Values beyond either true prefix make a missing tail mask conspicuous.
    value_host[:, 8:].fill_(100.0)
    query_host = torch.randn((2, 2, 64), generator=generator) * 0.1
    key = key_host.to(device=device, dtype=torch.bfloat16)
    value = value_host.to(device=device, dtype=torch.bfloat16)
    query = query_host.to(device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([3, 5], device=device, dtype=torch.int32)
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=device,
        dtype=torch.bfloat16,
    )
    backend.bind_kv_caches([LayerCache(key=key, value=value)])
    metadata = SimpleNamespace(
        slot_mapping=torch.zeros(2, device=device, dtype=torch.int32),
        q_seq_lens=torch.ones(2, device=device, dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, 1, 2],
        block_table=torch.tensor([[0], [1]], device=device, dtype=torch.int32),
        kv_seq_lens=lengths,
        kv_seq_lens_host_values=[100, 100],
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        has_kv_shard=False,
    )
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    backend.prepare(metadata)

    def decode() -> torch.Tensor:
        return backend._decode(query, key, value, metadata, 2)

    captured = None
    graph_output = None
    if graph:
        decode()
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            graph_output = decode()
        torch.npu.synchronize()
    for prefix_lengths in ([3, 5], [1, 2], [7, 4]):
        lengths.copy_(torch.tensor(prefix_lengths, device=device, dtype=torch.int32))
        if captured is not None:
            captured.replay()
            actual = graph_output
        else:
            actual = decode()
        expected = []
        for row, prefix_length in enumerate(prefix_lengths):
            q = query[row].float().cpu()
            k = key[row, :prefix_length, 0].float().cpu()
            v = value[row, :prefix_length, 0].float().cpu()
            expected.append(torch.softmax(q @ k.t() * 0.125, dim=-1) @ v)
        torch.testing.assert_close(
            actual.float().cpu().view(2, 2, 64), torch.stack(expected), rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL
        )
    torch.npu.synchronize()


@pytest.mark.parametrize("draft_width", [2, 3, 4, 16])
@pytest.mark.parametrize("graph", [False, True])
def test_refreshed_attention_hidden_matches_reexecuted_first_body(draft_width: int, graph: bool) -> None:
    """A real cache write/FIA round agrees with the redundant two-row body.

    This isolates the attention identity used by native prefix-output reuse.
    The C++ pipeline tests cover request identity, owned output publication and
    admission-time logits; this test does not replace a full-model comparison.
    """
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(86031 + draft_width)
    batch, prefix, heads, dimension = 2, 5, 2, 64
    width = draft_width + 1
    rows = batch * width
    hidden_size = heads * dimension

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * 0.1

    verifier_hidden = random((batch, width, hidden_size))
    token_embedding = random((batch, width, hidden_size))
    fused_inputs = verifier_hidden * 0.5 + token_embedding * 0.25
    query_host = (fused_inputs @ random((hidden_size, hidden_size))).to(torch.bfloat16).view(rows, heads, dimension)
    key_rows_host = (fused_inputs @ random((hidden_size, dimension))).to(torch.bfloat16).view(rows, 1, dimension)
    value_rows_host = fused_inputs @ random((hidden_size, dimension))
    value_rows_host += torch.arange(1, width + 1).view(1, width, 1) * 0.25
    value_rows_host = value_rows_host.to(torch.bfloat16).view(rows, 1, dimension)
    key_host = random((batch, 128, 1, dimension)).to(torch.bfloat16)
    value_host = random((batch, 128, 1, dimension)).to(torch.bfloat16)
    value_host[:, prefix:].fill_(100.0)
    key, value = key_host.to(device), value_host.to(device)
    query, key_rows, value_rows = query_host.to(device), key_rows_host.to(device), value_rows_host.to(device)
    head = random((hidden_size, 19)).to(device=device, dtype=torch.bfloat16)
    accepted = torch.ones(batch, device=device, dtype=torch.int32)
    selected = torch.empty(batch, device=device, dtype=torch.int64)
    owned = torch.empty(batch, hidden_size, device=device, dtype=torch.bfloat16)
    row_starts = torch.arange(batch, device=device, dtype=torch.int64) * width
    slots = (
        torch.arange(prefix, prefix + width, device=device, dtype=torch.int32).repeat(batch)
        + torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(width) * 128
    )

    def backend_for(count: int, table: list[int], lengths: torch.Tensor) -> NpuPagedAttentionBackend:
        backend = NpuPagedAttentionBackend(
            num_heads=heads,
            num_kv_heads=1,
            head_dim=dimension,
            scale=0.125,
            sliding_window=0,
            is_mla=False,
            device=device,
            dtype=torch.bfloat16,
        )
        backend.bind_kv_caches([LayerCache(key=key, value=value)])
        metadata = SimpleNamespace(
            slot_mapping=torch.zeros(count, device=device, dtype=torch.int32),
            q_seq_lens=torch.ones(count, device=device, dtype=torch.int32),
            q_cu_seq_lens_host_values=list(range(count + 1)),
            block_table=torch.tensor(table, device=device, dtype=torch.int32).view(count, 1),
            kv_seq_lens=lengths,
            kv_seq_lens_host_values=[100] * count,
            is_prefill=False,
            is_chunked_prefill=False,
            is_spec_verify=False,
            has_kv_shard=False,
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        backend.prepare(metadata)
        return backend

    refresh_lengths = torch.arange(prefix + 1, prefix + width + 1, device=device, dtype=torch.int32).repeat(batch)
    refresh_backend = backend_for(rows, [0] * width + [1] * width, refresh_lengths)
    first_lengths = torch.ones(2 * batch, device=device, dtype=torch.int32)
    first_backend = backend_for(2 * batch, [0, 0, 1, 1], first_lengths)

    def refresh() -> torch.Tensor:
        torch.ops.xllm_ops.reshape_paged_cache(slots, key_rows, value_rows, key, value)
        output = refresh_backend._decode(query, key, value, refresh_backend._metadata, rows)
        selected.copy_(row_starts + accepted.to(torch.int64) - 1)
        owned.copy_(output.view(rows, hidden_size).index_select(0, selected))
        return output

    captured = None
    graph_output = None
    if graph:
        refresh()
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            graph_output = refresh()
        torch.npu.synchronize()

    for length in range(1, width + 1):
        accepted_host = [length, width + 1 - length]
        accepted.copy_(torch.tensor(accepted_host, device=device, dtype=torch.int32))
        if captured is not None:
            captured.replay()
            output = graph_output
        else:
            output = refresh()
        # Destroy the borrowed full-width output before it could be reused as
        # the first proposal. The selected per-request rows must be independent.
        output.fill_(-11.0)
        cached_hidden = owned.clone()
        cached_logits = cached_hidden @ head

        first_indices, first_slots, first_lengths_host, expected = [], [], [], []
        for sequence, committed in enumerate(accepted_host):
            last = sequence * width + committed - 1
            previous = sequence * width + max(0, committed - 2)
            first_indices.extend((previous, last))
            # A non-full acceptance redirects the unused repair write beyond
            # the live prefix, exactly as the native two-row first body does.
            repair = prefix + committed - 2 if committed == width else prefix + committed
            first_slots.extend((sequence * 128 + repair, sequence * 128 + prefix + committed - 1))
            first_lengths_host.extend((prefix + committed - 1, prefix + committed))
            q = query_host[last].float()
            k = torch.cat((key_host[sequence, :prefix, 0], key_rows_host[sequence * width : last + 1, 0])).float()
            v = torch.cat((value_host[sequence, :prefix, 0], value_rows_host[sequence * width : last + 1, 0])).float()
            expected.append(torch.softmax(q @ k.t() * 0.125, dim=-1) @ v)
            torch.testing.assert_close(
                key[sequence, : prefix + committed, 0].cpu(), k.to(torch.bfloat16), rtol=0, atol=0
            )
            torch.testing.assert_close(
                value[sequence, : prefix + committed, 0].cpu(), v.to(torch.bfloat16), rtol=0, atol=0
            )
            old_k = key_host[sequence, : prefix + committed, 0].float()
            old_v = value_host[sequence, : prefix + committed, 0].float()
            wrong_without_refresh = torch.softmax(q @ old_k.t() * 0.125, dim=-1) @ old_v
            assert not torch.allclose(wrong_without_refresh, expected[-1], rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
            if committed > 1:
                previous_q = query_host[last - 1].float()
                wrong_previous_row = torch.softmax(previous_q @ k[:-1].t() * 0.125, dim=-1) @ v[:-1]
                assert not torch.allclose(wrong_previous_row, expected[-1], rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
            # The reference first body must really perform its current-row KV
            # write. A read-only repeated query would now see this poison.
            key[sequence, prefix + committed - 1].zero_()
            value[sequence, prefix + committed - 1].fill_(99.0)

        gather = torch.tensor(first_indices, device=device, dtype=torch.int64)
        first_lengths.copy_(torch.tensor(first_lengths_host, device=device, dtype=torch.int32))
        torch.ops.xllm_ops.reshape_paged_cache(
            torch.tensor(first_slots, device=device, dtype=torch.int32),
            key_rows.index_select(0, gather),
            value_rows.index_select(0, gather),
            key,
            value,
        )
        repeated = (
            first_backend._decode(query.index_select(0, gather), key, value, first_backend._metadata, 2 * batch)
            .view(batch, 2, hidden_size)[:, 1]
            .contiguous()
        )
        expected_hidden = torch.stack(expected).view(batch, hidden_size)
        torch.testing.assert_close(cached_hidden.float().cpu(), expected_hidden, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(repeated, cached_hidden, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        torch.testing.assert_close(repeated @ head, cached_logits, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        for sequence, committed in enumerate(accepted_host):
            begin, end = sequence * width, sequence * width + committed
            expected_keys = torch.cat((key_host[sequence, :prefix, 0], key_rows_host[begin:end, 0]))
            expected_values = torch.cat((value_host[sequence, :prefix, 0], value_rows_host[begin:end, 0]))
            torch.testing.assert_close(key[sequence, : prefix + committed, 0].cpu(), expected_keys, rtol=0, atol=0)
            torch.testing.assert_close(value[sequence, : prefix + committed, 0].cpu(), expected_values, rtol=0, atol=0)
    torch.npu.synchronize()

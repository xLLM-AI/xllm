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

"""Real NPU FIA graph tiers against an independent full-context CPU oracle."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="requires the real NPU backend")

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)


@pytest.fixture(scope="module")
def _graph_owners() -> list[object]:
    # The allocator retires a shared pool when its last graph is destroyed.
    # Serving retains its runners together; retain these independent probes
    # likewise rather than reusing a retired process-global pool handle.
    return []


@pytest.mark.parametrize("prepared", [False, True], ids=["ordinary", "prepared"])
@pytest.mark.parametrize("rows", [2, 8], ids=["two-rows", "eight-expanded-rows"])
@torch.inference_mode()
def test_fia_capacity_graphs_preserve_full_context_across_tiers(
    prepared: bool, rows: int, _graph_owners: list[object]
) -> None:
    torch.npu.set_device(0)
    task_stream = torch.npu.Stream(device=torch.device("npu:0"))
    task_stream.wait_stream(torch.npu.current_stream())
    # Native Slot warmup captures on its non-default task stream. Prepared
    # graphs deliberately borrow that caller stream rather than owning one.
    with torch.npu.stream(task_stream):
        _assert_fia_capacity_graphs(prepared, rows, _graph_owners)
    torch.npu.current_stream().wait_stream(task_stream)
    torch.npu.synchronize()


def _assert_fia_capacity_graphs(prepared: bool, rows: int, graph_owners: list[object]) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import get_forward_context
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
    from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner

    device = torch.device("npu:0")
    page_size, pages, heads, dimension = 128, 65, 2, 64
    context = pages * page_size
    generator = torch.Generator().manual_seed(5124097 + rows)
    # Every logical sequence has a disjoint, permuted physical page layout.
    # A stale or truncated compact table therefore changes the numerical oracle.
    table_host = torch.randperm(rows * pages, generator=generator).view(rows, pages).to(torch.int32)
    key = torch.zeros(rows * pages, page_size, 1, dimension, device=device, dtype=torch.bfloat16)
    values_host = torch.empty(rows * pages, page_size, 1, dimension, dtype=torch.bfloat16)
    for row in range(rows):
        logical = torch.full((context,), 64.0 + row * 2)
        logical[:512] = 0.25 + row * 2
        logical[512:4096] = 16.0 + row * 2
        # A one-token crossing must be distinguishable despite BF16 rounding.
        logical[511] = 64.0 + row * 2
        logical[512] = 512.0 + row * 2
        logical[4096] = 2048.0 + row * 2
        for page in range(pages):
            physical = int(table_host[row, page])
            values_host[physical, :, 0] = logical[page * page_size : (page + 1) * page_size, None]
    value = values_host.to(device)
    query = torch.zeros(rows, heads, dimension, device=device, dtype=torch.bfloat16)
    backend = NpuPagedAttentionBackend(heads, 1, dimension, dimension**-0.5, 0, False, device, torch.bfloat16)
    cache = LayerCache(key=key, value=value)
    backend.bind_kv_caches([cache])

    class AttentionProbe(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            metadata = get_forward_context().metadata
            return backend._decode(query, key, value, metadata, tokens.numel())

    tokens = torch.zeros(rows, device=device, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    table = table_host.to(device)
    lengths = torch.full((rows,), 511, device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(rows, device=device, dtype=torch.int32),
        block_table=table,
        q_seq_lens=torch.ones(rows, device=device, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(1, rows + 1, device=device, dtype=torch.int32),
        q_cu_seq_lens_host_values=list(range(1, rows + 1)),
        kv_seq_lens=lengths,
        kv_seq_lens_host_values=[512] * rows,
        kv_cu_seq_lens=None,
        paged_kv_indptr=torch.arange(0, (rows + 1) * pages, pages, device=device, dtype=torch.int32),
        paged_kv_indices=table.reshape(-1).clone(),
        paged_kv_last_page_len=torch.full((rows,), 127, device=device, dtype=torch.int32),
        linear_state_indices=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        has_kv_shard=False,
        expanded_decode_metadata=None,
    )
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    runner = (
        PreparedAclGraphRunner(AttentionProbe(), backend, device, rows, device_kv_lengths=True)
        if prepared
        else DecodeAclGraphRunner(AttentionProbe(), backend, device, rows, 8192, device_kv_lengths=True)
    )
    runner.bind_layer_caches([cache])
    graph_owners.append(runner)
    if prepared:
        runner.warmup_prepared(tokens, positions, metadata)
        entries = runner._prepared_graphs
    else:
        runner.warmup(tokens, positions, metadata, is_graph_warmup=True)
        entries = runner._graphs
    torch.npu.synchronize()
    by_capacity = {entry.kv_capacity_pages: entry for entry in entries.values()}
    assert set(by_capacity) == {4, 32, 65}
    assert len(entries) == 3
    addresses = {capacity: entry.static_metadata.block_table.data_ptr() for capacity, entry in by_capacity.items()}
    assert len(set(addresses.values())) == 3
    for capacity, entry in by_capacity.items():
        assert entry.execution_state.paged_attention == {}, "BNSD graphs must not retain unused TND workspace"
        assert entry.static_metadata.block_table.shape == (rows, capacity)
        assert entry.static_metadata.block_table.is_contiguous()
        assert entry.static_metadata.block_table.data_ptr() != table.data_ptr()

    for host_bound, live, capacity, rotate_table in (
        (512, 511, 4, False),
        (512, 512, 4, True),
        (1024, 513, 32, False),
        (1024, 519, 32, True),
        (8192, 4097, 65, False),
        (8192, 4103, 65, True),
        (512, 7, 4, False),
    ):
        current_table = table_host.roll(1, dims=0) if rotate_table else table_host
        table.copy_(current_table)
        metadata.paged_kv_indices.copy_(current_table.to(device).reshape(-1))
        # Conservative Host bounds select a tier; exact live lengths remain
        # device tensors consumed by the captured mask without CPU readback.
        metadata.kv_seq_lens_host_values[:] = [host_bound] * rows
        lengths.fill_(live)
        metadata.paged_kv_last_page_len.fill_((live - 1) % page_size + 1)
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        output = runner.execute(tokens, positions, metadata)
        torch.npu.synchronize()
        assert output is not None
        actual = output.float().cpu().view(rows, heads, dimension)
        expected, truncated, unmasked = [], [], []
        for row in range(rows):
            physical = current_table[row].to(torch.long)
            sequence = values_host[physical, :, 0].reshape(context, dimension).float()
            # Zero Q and K imply uniform softmax over the entire valid prefix.
            # This reference neither calls FIA nor reuses its tail-mask logic.
            expected.append(sequence[:live].mean(dim=0).expand(heads, -1))
            truncated.append(sequence[: min(live, 512)].mean(dim=0).expand(heads, -1))
            unmasked.append(sequence[: capacity * page_size].mean(dim=0).expand(heads, -1))
        expected_tensor = torch.stack(expected)
        torch.testing.assert_close(actual, expected_tensor, rtol=0.01, atol=0.01)
        if live > 512:
            assert not torch.allclose(expected_tensor, torch.stack(truncated), rtol=0.01, atol=0.01)
        if live != capacity * page_size:
            assert not torch.allclose(expected_tensor, torch.stack(unmasked), rtol=0.01, atol=0.01)
        compact = by_capacity[capacity].static_metadata.block_table
        torch.testing.assert_close(compact.cpu(), current_table[:, :capacity])
        assert compact.data_ptr() == addresses[capacity]
        assert len(entries) == 3, "crossing or returning to a tier must not capture at runtime"
    torch.npu.synchronize()

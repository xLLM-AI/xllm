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

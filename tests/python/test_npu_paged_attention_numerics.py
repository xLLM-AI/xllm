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


@pytest.mark.parametrize("rows", [1, 3, 5], ids=["draft-step", "padded-prefix", "draft-prefix"])
@pytest.mark.parametrize("commit_in_graph", [False, True], ids=["prepared-lengths", "late-device-commit"])
@pytest.mark.parametrize("layers", [1, 16])
@pytest.mark.parametrize("use_v2", [False, True], ids=["fia-v1", "fia-v2"])
@pytest.mark.parametrize("native_updates", [False, True], ids=["python-updates", "native-updates"])
@torch.inference_mode()
def test_ordinary_draft_graph_reads_growing_kv_prefix(
    rows: int,
    commit_in_graph: bool,
    layers: int,
    use_v2: bool,
    native_updates: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.attention.paged_kv_metadata import build_row_aligned_paged_kv_metadata
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.forward_context import AclGraphCaptureContext, AclGraphNativeTaskGroup, AclGraphTask
    from xllm.python.model_executor.runners.acl_graph import AclGraphEntry
    from xllm.python.model_executor.runners.base import ModelExecutionOutput
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

    monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph._GRAPH_POOL", None)
    if not native_updates:

        def legacy_context(stream: object, tasks: list, *, batch_fia_task_updates: bool) -> AclGraphCaptureContext:
            return AclGraphCaptureContext(stream, tasks)

        monkeypatch.setattr("xllm.python.model_executor.runners.acl_graph.AclGraphCaptureContext", legacy_context)
    device = torch.device("npu:0")
    # Model-level batching uses Qwen3.8-27B's actual TP4 head geometry.
    heads, kv_heads, dimension = (4, 2, 64) if layers == 1 else (6, 1, 256)
    page_size, capacity = 128, 4096
    backend = NpuPagedAttentionBackend(heads, kv_heads, dimension, dimension**-0.5, 0, False, device, torch.bfloat16)
    if use_v2 and not backend._use_fia_v2:
        pytest.skip("FIA v2 is unavailable")
    backend._use_fia_v2 = use_v2
    key = torch.zeros(33, page_size, kv_heads, dimension, device=device, dtype=torch.bfloat16)
    values_host = torch.arange(33 * page_size, dtype=torch.float32) / 2048
    values_host = values_host.view(33, page_size, 1, 1).expand(33, page_size, kv_heads, dimension)
    value = values_host.contiguous().to(device=device, dtype=key.dtype)
    caches = [LayerCache(key=key, value=value + index / 16) for index in range(layers)]
    backend.bind_kv_caches(caches)
    attentions = [Attention(heads, kv_heads, dimension, dimension**-0.5, 0, index) for index in range(layers)]
    token_capacity = 8 if rows == 3 else rows
    committed_lengths = torch.zeros(token_capacity, dtype=torch.int32, device=device)
    committed_lengths[:rows].fill_(1)

    class _DraftAttention(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            count = input_ids.numel()
            query = torch.zeros(count, heads * dimension, device=device, dtype=key.dtype)
            projected = torch.zeros(count, kv_heads * dimension, device=device, dtype=key.dtype)
            return torch.cat([attention(query, projected, projected).clone() for attention in attentions], dim=-1)

    class _CommittedDraftRunner(DecodeAclGraphRunner):
        def _forward_static(self, entry: AclGraphEntry) -> ModelExecutionOutput:
            if commit_in_graph:
                count = entry.static_input_ids.numel()
                entry.static_metadata.kv_seq_lens.copy_(committed_lengths[:count])
            return super()._forward_static(entry)

    runner = _CommittedDraftRunner(
        _DraftAttention(),
        backend,
        device,
        token_capacity,
        capacity,
        is_spec_draft=True,
        enable_graph_mode_decode_no_padding=rows != 3,
        device_kv_lengths=True,
    )
    runner.bind_layer_caches(caches)
    tokens = torch.zeros(rows, dtype=torch.int32, device=device)
    table = torch.arange(1, 33, dtype=torch.int32, device=device).expand(rows, -1).contiguous()

    def _metadata_for(length: int) -> SimpleNamespace:
        lengths = list(range(length, length + rows))
        indptr, indices, last_page = build_row_aligned_paged_kv_metadata(table, lengths, page_size)
        return SimpleNamespace(
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

    runner.warmup(tokens, tokens, _metadata_for(8), is_graph_warmup=True)
    # With zero queries and keys, attention must average the entire live
    # prefix. This distinguishes a frozen short capture bound from the mask.
    for length, bound in ((129, 256), (1875, 2048), (7, 256), (1987, 4096 - rows + 1)):
        live_lengths = list(range(length, length + rows))
        committed_lengths[:rows].copy_(torch.tensor(live_lengths, dtype=torch.int32, device=device))
        metadata = _metadata_for(bound if commit_in_graph else length)
        decoded = runner.prepare_execution(tokens, metadata)
        assert decoded is not None
        graph_key = runner.warmup(tokens, tokens, metadata, decode_metadata=decoded)
        actual = runner.execute(tokens, tokens, metadata, graph_key=graph_key, decode_metadata=decoded)
        entry = runner._graphs[graph_key]
        if native_updates:
            assert len(entry.graph_tasks) == 1
            group = entry.graph_tasks[0]
            assert isinstance(group, AclGraphNativeTaskGroup)
            assert group.end - group.begin == group.batch.size() == layers
        else:
            assert len(entry.graph_tasks) == layers
            assert all(isinstance(task, AclGraphTask) for task in entry.graph_tasks)
        expected = (
            torch.tensor([(page_size + (live_length - 1) / 2) / 2048 for live_length in live_lengths])
            .view(rows, 1)
            .expand(rows, heads * dimension)
        )
        expected = torch.cat([expected + index / 16 for index in range(layers)], dim=-1)
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

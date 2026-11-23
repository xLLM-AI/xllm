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

"""Real RoPE MLA/SFA q8-causal versus packed-q1 target attention.

This exercises the production attention entry and prepared device metadata,
not a complete GLM checkpoint: projection quantization, indexer selection, and
multi-layer hidden-state propagation are outside this fixture. The short
prefix is fully selected by fixed sparse indices, including future candidates,
so correctness must come from the real SFA causal mask and live KV lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="real target SFA tests require torch_npu")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)

_HEADS = 8
_LATENT = 512
_ROPE = 64
_PAGE = 128
_WIDTH = 8
_TOPK = 2048
_SCALE = (_LATENT + _ROPE) ** -0.5
_PLANS = ((1, (1,)), (1, (4,)), (1, (8,)), (3, (1, 1, 1)), (3, (1, 4, 3)), (3, (8, 8, 8)))


@dataclass
class _Case:
    query: torch.Tensor
    query_rope: torch.Tensor
    latent: torch.Tensor
    rope: torch.Tensor
    prefixes: tuple[int, ...]
    widths: tuple[int, ...]


@dataclass
class _Slot:
    backend: NpuPagedAttentionBackend
    caches: list[LayerCache]
    context: ForwardContext
    metadata: SimpleNamespace
    query: torch.Tensor
    query_rope: torch.Tensor
    topk: torch.Tensor
    projection: torch.Tensor
    graph: object | None = None
    output: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    decisions: torch.Tensor | None = None


def _make_case(widths: tuple[int, ...], phase: int) -> _Case:
    generator = torch.Generator().manual_seed(731 + phase)
    batch = len(widths)
    prefixes = tuple(value + phase % 3 for value in (126, 129, 253)[:batch])
    query = torch.randn(batch * _WIDTH, _HEADS, _LATENT, generator=generator) * 0.1
    query_rope = torch.randn(batch * _WIDTH, _HEADS, _ROPE, generator=generator) * 0.1
    latent = torch.randn(batch, 3 * _PAGE, 1, _LATENT, generator=generator) * 0.1
    rope = torch.randn(batch, 3 * _PAGE, 1, _ROPE, generator=generator) * 0.1
    # Each request's three-channel signature yields a genuine linear head with
    # a large top-1 margin; the other 509 channels remain nontrivial attention.
    latent[..., :3] = -1
    for owner in range(batch):
        latent[owner, :, :, owner] = 2 + owner * 0.25
    return _Case(query.bfloat16(), query_rope.bfloat16(), latent.bfloat16(), rope.bfloat16(), prefixes, widths)


def _block_table(batch: int) -> torch.Tensor:
    return torch.tensor([[1 + owner * 3 + column for column in (0, 2, 1)] for owner in range(batch)], dtype=torch.int32)


def _projection() -> torch.Tensor:
    weight = torch.zeros(_HEADS * _LATENT, 3)
    for head in range(_HEADS):
        for channel in range(3):
            weight[head * _LATENT + channel, channel] = 1 / _HEADS
    return weight


def _make_slot(batch: int, rows: int, device: torch.device, *, packed: bool) -> _Slot:
    query_ends = list(range(1, rows + 1)) if packed else list(range(_WIDTH, rows + 1, _WIDTH))
    sequences = len(query_ends)
    key = torch.zeros(1 + batch * 3, _PAGE, 1, _LATENT, dtype=torch.bfloat16, device=device)
    value = torch.zeros(1 + batch * 3, _PAGE, 1, _ROPE, dtype=torch.bfloat16, device=device)
    caches = [
        LayerCache(
            key=key,
            value=value,
            index=torch.zeros(1 + batch * 3, _PAGE, 1, 128, dtype=key.dtype, device=device),
        )
    ]
    backend = NpuPagedAttentionBackend(_HEADS, 1, _LATENT, _SCALE, 0, True, device, key.dtype)
    backend.bind_kv_caches(caches)
    metadata = SimpleNamespace(
        q_seq_lens=torch.full((sequences,), 1 if packed else _WIDTH, dtype=torch.int32, device=device),
        q_cu_seq_lens=torch.tensor(query_ends, dtype=torch.int32, device=device),
        q_cu_seq_lens_host_values=query_ends,
        kv_seq_lens=torch.full((sequences,), 3 * _PAGE, dtype=torch.int32, device=device),
        kv_seq_lens_host_values=[3 * _PAGE] * sequences,
        block_table=torch.zeros(sequences, 3, dtype=torch.int32, device=device),
        slot_mapping=torch.full((rows,), -1, dtype=torch.int32, device=device),
        is_prefill=False,
        is_chunked_prefill=False,
        has_kv_shard=False,
        expanded_decode_metadata=None,
    )
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    context = ForwardContext(backend, device, metadata, caches, execution_state=AclGraphExecutionState({}))
    return _Slot(
        backend,
        caches,
        context,
        metadata,
        torch.zeros(rows, _HEADS, _LATENT, dtype=key.dtype, device=device),
        torch.zeros(rows, _HEADS, _ROPE, dtype=key.dtype, device=device),
        torch.full((rows, 1, _TOPK), -1, dtype=torch.int32, device=device),
        _projection().to(device),
    )


def _fill_slot(slot: _Slot, case: _Case, *, packed: bool) -> list[int]:
    batch = len(case.widths)
    table = _block_table(batch)
    key = torch.zeros_like(slot.caches[0].key, device="cpu")
    rope = torch.zeros_like(slot.caches[0].value, device="cpu")
    for owner in range(batch):
        for logical_page, physical_page in enumerate(table[owner].tolist()):
            key[physical_page] = case.latent[owner, logical_page * _PAGE : (logical_page + 1) * _PAGE]
            rope[physical_page] = case.rope[owner, logical_page * _PAGE : (logical_page + 1) * _PAGE]
    slot.caches[0].key.copy_(key)
    slot.caches[0].value.copy_(rope)
    rows = slot.query.shape[0]
    query = torch.zeros_like(slot.query, device="cpu")
    query_rope = torch.zeros_like(slot.query_rope, device="cpu")
    topk = torch.full((rows, 1, _TOPK), -1, dtype=torch.int32)
    tables = torch.zeros_like(slot.metadata.block_table, device="cpu")
    lengths = torch.ones_like(slot.metadata.kv_seq_lens, device="cpu")
    source_rows = [owner * _WIDTH + column for owner, width in enumerate(case.widths) for column in range(width)]
    mapping = source_rows if packed else list(range(batch * _WIDTH))
    for row, source in enumerate(mapping):
        owner, column = divmod(source, _WIDTH)
        query[row], query_rope[row] = case.query[source], case.query_rope[source]
        # Include all eight candidate indices. SFA must suppress future keys.
        length = case.prefixes[owner] + _WIDTH
        topk[row, 0, :length] = torch.arange(length, dtype=torch.int32)
        if packed:
            tables[row] = table[owner]
            lengths[row] = case.prefixes[owner] + column + 1
    if not packed:
        tables.copy_(table)
        lengths.copy_(torch.tensor([prefix + _WIDTH for prefix in case.prefixes], dtype=torch.int32))
    # Reserved block 0 has an initialized key, one visible KV row and no cache
    # publication. Do not use zero-length padding to bypass the SDK contract.
    topk[len(mapping) :, 0, 0] = 0
    slot.query.copy_(query)
    slot.query_rope.copy_(query_rope)
    slot.topk.copy_(topk)
    slot.metadata.block_table.copy_(tables)
    slot.metadata.kv_seq_lens.copy_(lengths)
    return source_rows


def _attend(slot: _Slot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with forward_context(slot.context):
        slot.backend.prepare(slot.metadata, graph_mode=True)
        output = slot.backend.execute_mla(
            slot.query,
            slot.query_rope,
            None,
            None,
            SimpleNamespace(layer_id=0, qk_rope_head_dim=_ROPE),
            topk=slot.topk,
            cache_is_preprocessed=True,
        )
        logits = output.float().flatten(1).matmul(slot.projection)
        return output, logits, logits.argmax(dim=-1)


def _cpu_causal_attention(case: _Case) -> torch.Tensor:
    output = torch.empty_like(case.query)
    for owner, prefix in enumerate(case.prefixes):
        for column in range(_WIDTH):
            row = owner * _WIDTH + column
            length = prefix + column + 1
            latent = case.latent[owner, :length, 0].float()
            rope = case.rope[owner, :length, 0].float()
            scores = (case.query[row].float().matmul(latent.T) + case.query_rope[row].float().matmul(rope.T)) * _SCALE
            output[row] = scores.softmax(dim=-1).matmul(latent).bfloat16()
    return output


def _addresses(slot: _Slot) -> tuple[int, ...]:
    return tuple(
        tensor.data_ptr()
        for tensor in (
            slot.query,
            slot.query_rope,
            slot.topk,
            slot.metadata.q_cu_seq_lens,
            slot.metadata.kv_seq_lens,
            slot.metadata.block_table,
            slot.caches[0].key,
            slot.caches[0].value,
            slot.output,
        )
    )


def _replay(slot: _Slot, stream: object) -> None:
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        slot.graph.replay()
    torch.npu.synchronize()


def _check(slot: _Slot, reference: _Slot, case: _Case, mapping: list[int]) -> torch.Tensor:
    expected = _cpu_causal_attention(case)
    reference_output, reference_logits, reference_decisions = _attend(reference)
    actual = slot.output.cpu()
    torch.testing.assert_close(reference_output.cpu(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[: len(mapping)], expected[mapping], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[: len(mapping)], reference_output.cpu()[mapping], rtol=2e-2, atol=2e-2)
    # SDK padding can contain negligible roundoff; unused rows never commit.
    padding = actual[len(mapping) :]
    assert torch.isfinite(padding).all()
    torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=1e-6)
    logits = reference_logits.cpu()[mapping]
    margins = logits.topk(2, dim=-1).values.diff(dim=-1).neg()
    assert torch.all(margins > 2), "deterministic projection must not hide near-tie top-1 differences"
    torch.testing.assert_close(slot.decisions.cpu()[: len(mapping)], reference_decisions.cpu()[mapping], rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    return actual


@pytest.mark.parametrize("batch,widths", _PLANS)
@torch.inference_mode()
def test_sfa_target_q8_matches_captured_packed_q1(batch: int, widths: tuple[int, ...]) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    device = torch.device("npu", torch.npu.current_device())
    rows = max(4, 1 << (sum(widths) - 1).bit_length())
    reference = _make_slot(batch, batch * _WIDTH, device, packed=False)
    slots = [_make_slot(batch, rows, device, packed=True) for _ in range(2)]
    stream = torch.npu.Stream(device=device)
    for slot in slots:
        _fill_slot(slot, _make_case(widths, 0), packed=True)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                _attend(slot)
        torch.npu.synchronize()
        slot.graph = torch.npu.NPUGraph()
        with torch.npu.graph(slot.graph, stream=stream):
            slot.output, slot.logits, slot.decisions = _attend(slot)
    torch.npu.synchronize()
    addresses = [_addresses(slot) for slot in slots]
    assert set(addresses[0]).isdisjoint(addresses[1])
    previous: list[torch.Tensor | None] = [None, None]
    for phase in range(3):
        live_widths = tuple(reversed(widths)) if phase % 2 else widths
        for index, slot in enumerate(slots):
            case = _make_case(live_widths, phase + index)
            mapping = _fill_slot(slot, case, packed=True)
            _fill_slot(reference, case, packed=False)
            _replay(slot, stream)
            previous[index] = _check(slot, reference, case, mapping)
            assert _addresses(slot) == addresses[index]
            if previous[1 - index] is not None:
                torch.testing.assert_close(slots[1 - index].output.cpu(), previous[1 - index], rtol=0, atol=0)

    # Change the final candidate's K/V and query, not the earlier queries.
    # It is present in every top-k list; only causal visibility can isolate it.
    slot = slots[0]
    case = _make_case(widths, 4)
    mapping = _fill_slot(slot, case, packed=True)
    _fill_slot(reference, case, packed=False)
    _replay(slot, stream)
    before = _check(slot, reference, case, mapping)
    reference_before = _attend(reference)[0].cpu().clone()
    case.latent[0, case.prefixes[0] + 7, 0, 4] += 32
    case.rope[0, case.prefixes[0] + 7] += 4
    case.query[7] += 1
    _fill_slot(slot, case, packed=True)
    _fill_slot(reference, case, packed=False)
    _replay(slot, stream)
    after = _check(slot, reference, case, mapping)
    early = [row for row, source in enumerate(mapping) if source < 7]
    torch.testing.assert_close(after[early], before[early], rtol=0, atol=0)
    reference_after = _attend(reference)[0].cpu()
    torch.testing.assert_close(reference_after[:7], reference_before[:7], rtol=0, atol=0)
    assert not torch.allclose(reference_after[7], reference_before[7], rtol=2e-2, atol=2e-2)

    if batch > 1:
        # Change another request's entire prefix, ensuring both layouts keep
        # per-request cache ownership rather than attending across packed rows.
        case.latent[-1, :, 0, 5] += 8
        _fill_slot(slot, case, packed=True)
        _fill_slot(reference, case, packed=False)
        _replay(slot, stream)
        isolated = _check(slot, reference, case, mapping)
        others = [row for row, source in enumerate(mapping) if source // _WIDTH != batch - 1]
        affected = [row for row, source in enumerate(mapping) if source // _WIDTH == batch - 1]
        torch.testing.assert_close(isolated[others], after[others], rtol=0, atol=0)
        assert not torch.allclose(isolated[affected], after[affected], rtol=2e-2, atol=2e-2)
    assert _addresses(slot) == addresses[0]

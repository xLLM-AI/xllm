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

"""CPU graph-recipe tests and real-NPU whole-round capture/replay checks."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.npu_paged_attention import (
    NpuPagedAttentionBackend,
    _PreparedMlaAttention,
    _PreparedPagedAttention,
)
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.dspark_ragged import (
    DSparkVerifyBuffers,
    allocate_verify_prefixes,
    build_verify_offsets,
    pack_verify_inputs,
)
from xllm.python.model_executor.dspark_ragged_commit import greedy_verify_commit_out
from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner
from xllm.python.model_executor.runners.unified_block_acl_graph import (
    AdaptiveUnifiedBlockGraphRegistry,
    _AdaptiveUnifiedBlockAclGraph,
    _verify_buckets,
)


@dataclass
class _Case:
    target_binding: tuple
    draft_binding: tuple
    tokens: torch.Tensor
    lengths: torch.Tensor
    inputs: DSparkAdaptiveGraphInputs
    context_slots: torch.Tensor

    def arguments(self) -> tuple:
        return self.target_binding, self.draft_binding, self.tokens, self.lengths, self.inputs, self.context_slots


def _binding(rows: int, device: str, *, draft: bool, draft_width: int) -> tuple:
    width = draft_width if draft else 1
    table = torch.ones((rows, 64), dtype=torch.int32, device=device)
    empty = torch.empty(0, dtype=torch.int32, device=device)
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=torch.ones(rows * width, dtype=torch.int32, device=device),
        paged_kv_indptr=empty,
        paged_kv_indices=empty,
        paged_kv_last_page_len=empty,
        block_table=table,
        q_seq_lens=torch.full((rows,), width, dtype=torch.int32, device=device),
        q_cu_seq_lens=torch.arange(1, rows + 1, dtype=torch.int32, device=device) * width,
        q_cu_seq_lens_host_values=[width * (index + 1) for index in range(rows)],
        kv_seq_lens=torch.full((rows,), 512 if draft else 513, dtype=torch.int32, device=device),
        kv_seq_lens_host_values=[512 if draft else 513] * rows,
        is_chunked_prefill=draft,
        prepared_attention_state=_PreparedPagedAttention(table, [], [], [], kv_capacity=512 if draft else 513),
    )
    return (
        torch.ones(rows * width, dtype=torch.int32, device=device),
        torch.zeros(rows * width, dtype=torch.int32, device=device),
        metadata,
    )


def _case(batch: int, graph_rows: int, device: str = "cpu", *, capacity: int = 16, draft_width: int = 7) -> _Case:
    physical_batch = next(rows for rows in (1, 2, 4, 8, 16) if rows >= batch)
    inputs = DSparkAdaptiveGraphInputs(
        arena=torch.zeros((12 * draft_width + 4) * capacity, dtype=torch.uint8, device=device),
        control=torch.tensor([2**32 + batch], dtype=torch.int64, device=device),
        anchors=torch.ones(capacity, dtype=torch.int64, device=device),
        base_positions=torch.zeros(capacity, dtype=torch.int32, device=device),
        base_kv_lengths=torch.ones(capacity, dtype=torch.int32, device=device),
        base_block_table=torch.ones((capacity, 64), dtype=torch.int32, device=device),
        offsets=torch.empty(capacity + 1, dtype=torch.int32, device=device),
        owners=torch.empty(graph_rows, dtype=torch.int32, device=device),
        columns=torch.empty(graph_rows, dtype=torch.int32, device=device),
        block_size=16,
        draft_width=draft_width,
    )
    return _Case(
        _binding(graph_rows, device, draft=False, draft_width=draft_width),
        _binding(physical_batch, device, draft=True, draft_width=draft_width),
        torch.empty((batch, draft_width + 1), dtype=torch.int64, device=device),
        torch.empty(batch, dtype=torch.int32, device=device),
        inputs,
        torch.empty(graph_rows, dtype=torch.int32, device=device),
    )


def _patch(case: _Case, shift: int, *, suppress: bool) -> None:
    batch = case.tokens.shape[0]
    draft_width = case.inputs.draft_width
    device = case.tokens.device
    anchors = torch.arange(batch, device=device, dtype=torch.int64) * 10 + shift + 2
    starts = torch.arange(batch, device=device, dtype=torch.int32) * 16 + shift
    case.inputs.control.fill_(batch + (2**32 if suppress else 0))
    case.inputs.anchors[:batch].copy_(anchors)
    case.inputs.base_positions[:batch].copy_(starts)
    case.inputs.base_kv_lengths[:batch].copy_(starts + 1)
    ids, positions, metadata = case.draft_binding
    ids.fill_(1)
    ids[: batch * draft_width].view(batch, draft_width)[:, 0].copy_(anchors)
    positions.zero_()
    positions[: batch * draft_width].view(batch, draft_width).copy_(
        starts[:, None] + torch.arange(draft_width, device=device)
    )
    # Like native patch_mtp_decode, restore these before every serving replay.
    metadata.slot_mapping.copy_(positions + 16)
    case.tokens.fill_(-77)
    case.lengths.fill_(-77)


def _variant(case: _Case, batch: int, graph_rows: int) -> _Case:
    """Create the final native-style views, retaining one Slot's storage."""
    bindings = []
    for binding, rows, width in (
        (case.target_binding, graph_rows, 1),
        (case.draft_binding, next(rows for rows in (1, 2, 4, 8, 16) if rows >= batch), case.inputs.draft_width),
    ):
        ids, positions, metadata = binding
        table = metadata.block_table[:rows]
        metadata = replace(
            metadata,
            slot_mapping=metadata.slot_mapping[: rows * width],
            block_table=table,
            q_seq_lens=metadata.q_seq_lens[:rows],
            q_cu_seq_lens=metadata.q_cu_seq_lens[:rows],
            q_cu_seq_lens_host_values=[width * (index + 1) for index in range(rows)],
            kv_seq_lens=metadata.kv_seq_lens[:rows],
            kv_seq_lens_host_values=metadata.kv_seq_lens_host_values[:rows],
            prepared_attention_state=replace(metadata.prepared_attention_state, block_table=table),
        )
        bindings.append((ids[: rows * width], positions[: rows * width], metadata))
    return _Case(
        *bindings,
        case.tokens[:batch],
        case.lengths[:batch],
        replace(case.inputs, owners=case.inputs.owners[:graph_rows], columns=case.inputs.columns[:graph_rows]),
        case.context_slots[:graph_rows],
    )


class _ToyBackend:
    """Exercise the production metadata factory, without attention kernels."""

    def __init__(self, *, mla: bool, block_draft: bool) -> None:
        self.page_size = 16
        self.logical_page_size = 16
        self.is_mla = mla
        self._is_mla = mla
        self._mla_block_draft = block_draft

    def prepare_metadata(
        self, metadata: StaticGraphAttentionMetadata, *, device_kv_lengths: bool = False
    ) -> _PreparedPagedAttention | _PreparedMlaAttention:
        return NpuPagedAttentionBackend.prepare_metadata(self, metadata, device_kv_lengths=device_kv_lengths)


class _ToyExecutor(torch.nn.Module):
    """Use real tensor heads/commit recipe without pretending to run GLM."""

    def __init__(self, device: str, calls: list[str], *, target: bool) -> None:
        super().__init__()
        resolved_device = torch.device(device)
        if resolved_device.type == "npu" and resolved_device.index is None:
            resolved_device = torch.device("npu", torch.npu.current_device())
        self.target = target
        self.calls = calls
        self.record_calls = resolved_device.type == "cpu"
        self.attention_backend = _ToyBackend(mla=target, block_draft=not target)
        self.eager_runner = SimpleNamespace(device=resolved_device, layer_caches=[])
        self.prepared_graph_runner = PreparedAclGraphRunner(
            self.model, self.attention_backend, resolved_device, 128, block_draft=not target
        )
        self.draft_cache = torch.zeros(512, device=resolved_device)
        self.context_cache = torch.zeros(512, device=resolved_device)

    @property
    def model(self) -> torch.nn.Module:
        return self

    def _record(self, name: str) -> None:
        if self.record_calls:
            self.calls.append(name)

    def create_mtp_role_backends(self, count: int) -> tuple:
        return (self.attention_backend,) * count

    def execute_mtp_role(
        self, ids: torch.Tensor, positions: torch.Tensor, metadata: object, **kwargs: object
    ) -> object:
        self._record("target" if self.target else "draft")
        if self.target:
            return ids.to(torch.int64) + 1, torch.stack((positions.float(), ids.float()), dim=1)
        slots = metadata.slot_mapping
        self.draft_cache.index_add_(0, slots.clamp_min(0).to(torch.int64), (slots >= 0).float())
        return positions.float().view(-1, 1)

    def compute_logits(self, hidden: torch.Tensor, selected: object) -> torch.Tensor:
        self._record("draft_logits")
        return torch.zeros((hidden.shape[0], 256), device=hidden.device)

    def dspark_markov_bias(self, previous: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros((previous.numel(), 256), device=previous.device)
        return bias.scatter_(1, (previous + 1).view(-1, 1), 20.0)

    def dspark_confidence_probs(self, hidden: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self._record("confidence")
        return torch.sigmoid(hidden[:, :, 0] * 0.001)

    def compute_greedy_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        self._record("target_head")
        return hidden

    def write_context_kv(
        self, hidden: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, caches: object, params: object
    ) -> None:
        self._record("context")
        self.context_cache.index_add_(0, slots.clamp_min(0).to(torch.int64), (slots >= 0).float())


@pytest.fixture
def cpu_recipe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Only CPU recipe tests replace kernels/capture; NPU tests do not."""
    calls: list[str] = []

    def allocate(arena: torch.Tensor, control: torch.Tensor, graph_rows: int, draft_width: int) -> None:
        calls.append("allocate")
        batch = int(control.item()) & (2**32 - 1)
        confidence = (
            arena[8 * draft_width * batch : 12 * draft_width * batch].view(torch.float32).view(batch, draft_width)
        )
        widths = arena[12 * draft_width * batch : (12 * draft_width + 4) * batch].view(torch.int32)
        allocate_verify_prefixes(
            confidence, min(draft_width * batch, graph_rows - batch), widths, torch.empty(batch + 1, dtype=torch.int32)
        )

    def pack(
        anchors: torch.Tensor,
        arena: torch.Tensor,
        base_positions: torch.Tensor,
        base_kv_lengths: torch.Tensor,
        base_block_table: torch.Tensor,
        control: torch.Tensor,
        offsets: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        slots: torch.Tensor,
        table: torch.Tensor,
        query_lengths: torch.Tensor,
        owners: torch.Tensor,
        columns: torch.Tensor,
        block_size: int,
        draft_width: int,
    ) -> None:
        calls.append("pack")
        batch = int(control.item()) & (2**32 - 1)
        widths = arena[12 * draft_width * batch : (12 * draft_width + 4) * batch].view(torch.int32)
        build_verify_offsets(widths, offsets[: batch + 1])
        pack_verify_inputs(
            anchors=anchors[:batch],
            proposals=arena[: 8 * draft_width * batch].view(torch.int64).view(batch, draft_width),
            base_positions=base_positions[:batch],
            base_kv_lengths=base_kv_lengths[:batch],
            base_block_table=base_block_table[:batch],
            widths=widths,
            offsets=offsets[: batch + 1],
            logical_rows=min((draft_width + 1) * batch, tokens.numel()),
            block_size=block_size,
            outputs=DSparkVerifyBuffers(tokens, positions, kv_seq_lens, slots, table, query_lengths, owners, columns),
        )
        if int(control.item()) >> 32:
            slots.fill_(-1)

    def commit(*args: torch.Tensor) -> None:
        calls.append("commit")
        greedy_verify_commit_out(*args)

    def capture(runner: _AdaptiveUnifiedBlockAclGraph) -> None:
        runner.hidden = runner._run()

    def replay(runner: _AdaptiveUnifiedBlockAclGraph) -> torch.Tensor:
        runner.hidden.copy_(runner._run())
        return runner.hidden

    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_allocate_verify_prefixes_graph", allocate)
    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_pack_verify_inputs_graph", pack)
    monkeypatch.setattr(torch.ops.xllm_ops, "dspark_greedy_verify_commit", commit)
    monkeypatch.setattr(_AdaptiveUnifiedBlockAclGraph, "capture", capture)
    monkeypatch.setattr(_AdaptiveUnifiedBlockAclGraph, "execute", replay)
    monkeypatch.setattr(_AdaptiveUnifiedBlockAclGraph, "close", lambda _: None)
    monkeypatch.setattr(torch.npu, "Stream", lambda **_: object())
    monkeypatch.setattr(torch.npu, "current_stream", lambda *_: SimpleNamespace(wait_stream=lambda _: None))
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: (0, 0))
    return calls


def _registry(device: str, calls: list[str]) -> AdaptiveUnifiedBlockGraphRegistry:
    target = _ToyExecutor(device, calls, target=True)
    draft = _ToyExecutor(device, calls, target=False)
    return AdaptiveUnifiedBlockGraphRegistry(
        target,
        draft,
        lambda: None,
        lambda: None,
        lambda payload: calls.append(f"consensus:{payload.numel()}"),
        lambda runner: runner.capture(),
    )


def _check_outputs(case: _Case, shift: int) -> None:
    batch = case.tokens.shape[0]
    draft_width = case.inputs.draft_width
    widths = case.inputs.arena[12 * draft_width * batch : (12 * draft_width + 4) * batch].view(torch.int32)
    assert int(widths.sum().item()) == min((draft_width + 1) * batch, case.target_binding[0].numel())
    torch.testing.assert_close(case.lengths, widths)
    columns = torch.arange(draft_width + 1, device=case.tokens.device)[None, :]
    anchors = torch.arange(batch, device=case.tokens.device)[:, None] * 10 + shift + 2
    expected = (anchors + columns + 1).masked_fill(columns >= widths[:, None], -1)
    torch.testing.assert_close(case.tokens, expected)


@pytest.mark.parametrize("batch,graph_rows", [(1, 1), (1, 8), (2, 4), (3, 8), (3, 32), (16, 128)])
@pytest.mark.parametrize("draft_width", [5, 7])
def test_cpu_recipe_runs_confidence_pruning_before_target_and_commits_only_live_rows(
    batch: int, graph_rows: int, draft_width: int, cpu_recipe: list[str]
) -> None:
    registry = _registry("cpu", cpu_recipe)
    if draft_width == 5 and batch == 16:
        graph_rows = 96
    case = _case(batch, graph_rows, draft_width=draft_width)
    _patch(case, 0, suppress=True)
    registry.warmup(*case.arguments())
    assert not torch.any(registry.draft.draft_cache)
    assert not torch.any(registry.draft.context_cache)
    assert torch.all(case.target_binding[2].slot_mapping == -1)
    assert torch.all(case.context_slots == -1)
    cpu_recipe.clear()
    _patch(case, 1, suppress=False)
    result = registry.execute(*case.arguments())
    assert result.shape == (graph_rows, 2)
    assert cpu_recipe == [
        "draft",
        "draft_logits",
        "confidence",
        f"consensus:{12 * draft_width * batch}",
        "allocate",
        "pack",
        "target",
        "target_head",
        "commit",
        "context",
    ]
    _check_outputs(case, 1)
    assert torch.any(registry.draft.draft_cache)
    assert torch.any(registry.draft.context_cache)
    address = result.data_ptr()
    _patch(case, 2, suppress=False)
    assert registry.execute(*case.arguments()).data_ptr() == address
    _check_outputs(case, 2)
    assert registry.cache_info()["size"] == 1


def test_registry_requires_startup_binding_and_grows_compact_kv_buckets(cpu_recipe: list[str]) -> None:
    registry = _registry("cpu", cpu_recipe)
    case = _case(3, 8)
    with pytest.raises(ValueError, match="not captured during startup"):
        registry.execute(*case.arguments())
    _patch(case, 0, suppress=True)
    registry.warmup(*case.arguments())
    runner = next(iter(registry.variants.values()))
    assert runner.draft_kv_bound == 512
    assert runner.target_kv_bound == 1024
    for _, _, metadata in (case.target_binding, case.draft_binding):
        metadata.kv_seq_lens_host_values[:] = [128] * metadata.block_table.shape[0]
    _patch(case, 3, suppress=False)
    registry.warmup(*case.arguments())
    registry.execute(*case.arguments())
    assert registry.cache_info()["size"] == 1
    assert runner.target_metadata.block_table.data_ptr() == case.target_binding[2].block_table.data_ptr()
    assert runner.draft_metadata.block_table.shape[1] == 32
    case.target_binding[2].kv_seq_lens_host_values[:] = [514] * 8
    _patch(case, 4, suppress=False)
    registry.execute(*case.arguments())
    assert registry.cache_info()["size"] == 1
    case.target_binding[2].kv_seq_lens_host_values[:] = [128] * 8
    case.draft_binding[2].kv_seq_lens_host_values[:] = [513] * 4
    original_slots = case.draft_binding[2].slot_mapping.clone()
    original_control = case.inputs.control.clone()
    registry.warmup(*case.arguments())
    assert registry.cache_info()["size"] == 2
    torch.testing.assert_close(case.draft_binding[2].slot_mapping, original_slots)
    torch.testing.assert_close(case.inputs.control, original_control)
    _patch(case, 5, suppress=False)
    registry.execute(*case.arguments())
    _check_outputs(case, 5)
    case.draft_binding[2].kv_seq_lens_host_values[:] = [128] * 4
    _patch(case, 6, suppress=False)
    registry.execute(*case.arguments())
    _check_outputs(case, 6)
    assert registry.cache_info()["size"] == 2
    case.target_binding[2].kv_seq_lens_host_values[:] = [1025] * 8
    with pytest.raises(ValueError, match="block-table capacity"):
        registry.execute(*case.arguments())


def test_runtime_unified_kv_capture_never_writes_live_caches(cpu_recipe: list[str]) -> None:
    registry = _registry("cpu", cpu_recipe)
    case = _case(3, 8)
    _patch(case, 0, suppress=True)
    registry.warmup(*case.arguments())
    _patch(case, 1, suppress=False)
    registry.execute(*case.arguments())
    previous = registry.draft.context_cache.clone()
    previous_draft = registry.draft.draft_cache.clone()
    previous_control = case.inputs.control.clone()
    previous_slots = case.draft_binding[2].slot_mapping.clone()
    case.draft_binding[2].kv_seq_lens_host_values[:] = [513] * 4
    registry.warmup(*case.arguments())
    torch.testing.assert_close(registry.draft.context_cache, previous)
    torch.testing.assert_close(registry.draft.draft_cache, previous_draft)
    torch.testing.assert_close(case.inputs.control, previous_control)
    torch.testing.assert_close(case.draft_binding[2].slot_mapping, previous_slots)
    registry.execute(*case.arguments())
    _check_outputs(case, 1)


def test_mla_draft_uses_backend_metadata_with_live_device_kv_lengths(cpu_recipe: list[str]) -> None:
    registry = _registry("cpu", cpu_recipe)
    backend = _ToyBackend(mla=True, block_draft=True)
    registry.draft.attention_backend = backend
    registry.draft.prepared_graph_runner.attention_backend = backend
    case = _case(3, 8)
    source = case.draft_binding[2]
    source.prepared_attention_state = backend.prepare_metadata(source, device_kv_lengths=True)
    assert isinstance(source.prepared_attention_state, _PreparedMlaAttention)
    _patch(case, 0, suppress=True)
    registry.warmup(*case.arguments())
    runner = next(iter(registry.variants.values()))
    prepared = runner.draft_metadata.prepared_attention_state
    assert isinstance(prepared, _PreparedMlaAttention)
    assert not hasattr(prepared, "kv_capacity")
    assert prepared.actual_seq_kv.data_ptr() == source.kv_seq_lens.data_ptr()
    assert prepared.actual_seq_q.data_ptr() == source.q_cu_seq_lens.data_ptr()
    assert prepared.block_table.data_ptr() == runner.draft_metadata.block_table.data_ptr()
    assert prepared.kv_lengths == [512] * 4
    assert torch.all(case.target_binding[2].slot_mapping == -1)
    assert torch.all(case.context_slots == -1)
    _patch(case, 1, suppress=False)
    registry.execute(*case.arguments())
    _check_outputs(case, 1)


def test_registry_retains_two_slots_without_eviction_and_rejects_new_storage(cpu_recipe: list[str]) -> None:
    registry = _registry("cpu", cpu_recipe)
    cases = [_case(3, 8) for _ in range(3)]
    for case in cases[:2]:
        _patch(case, 0, suppress=True)
        registry.warmup(*case.arguments())
    saved = None
    for index in (0, 1, 0):
        _patch(cases[index], index + 1, suppress=False)
        registry.execute(*cases[index].arguments())
        if index == 0:
            saved = cases[0].tokens.clone()
        else:
            torch.testing.assert_close(cases[0].tokens, saved)
    assert registry.cache_info()["size"] == 2
    assert registry.cache_info()["capacity"] == 128 * ((64).bit_length() + 1) ** 2
    with pytest.raises(ValueError, match="Slot capacity"):
        registry.warmup(*cases[2].arguments())
    changed = _case(3, 8)
    with pytest.raises(ValueError, match="not captured during startup"):
        registry.execute(*changed.arguments())


def test_same_slot_keeps_actual_batch_and_graph_rows_in_startup_key(cpu_recipe: list[str]) -> None:
    registry = _registry("cpu", cpu_recipe)
    storage = _case(3, 32)
    variants = [_variant(storage, batch, graph_rows) for batch, graph_rows in ((1, 8), (3, 8), (3, 32))]
    for case in variants:
        _patch(case, 0, suppress=True)
        registry.warmup(*case.arguments())
    assert registry.cache_info()["slots"] == 1
    for shift, case in enumerate((*variants, variants[0]), 1):
        _patch(case, shift, suppress=False)
        registry.execute(*case.arguments())
        _check_outputs(case, shift)
    assert registry.cache_info()["size"] == 3
    with pytest.raises(ValueError, match="not captured during startup"):
        registry.execute(*_variant(storage, 2, 8).arguments())


@pytest.mark.parametrize(
    "batch,expected", [(1, (1, 2, 4, 8)), (3, (4, 8, 16, 32)), (16, (16, 32, 64, 128)), (128, (128, 256, 512, 1024))]
)
def test_unified_verify_policy_has_four_finite_tiers(batch: int, expected: tuple[int, ...]) -> None:
    assert _verify_buckets(batch, 1024, 7) == expected


@pytest.mark.parametrize("batch,expected", [(8, (8, 16, 32, 48)), (16, (16, 32, 64, 96)), (128, (128, 256, 512, 768))])
def test_unified_draft5_policy_stops_at_its_full_width(batch: int, expected: tuple[int, ...]) -> None:
    assert _verify_buckets(batch, 1024, 5) == expected


@pytest.mark.parametrize("batch,graph_rows", [(1, 8), (2, 4), (3, 8), (3, 32)])
@pytest.mark.parametrize("draft_width", [5, 7])
def test_real_npu_whole_round_capture_replays_changed_inputs_and_suppression(
    batch: int, graph_rows: int, draft_width: int
) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    registry = _registry("npu", [])
    case = _case(batch, graph_rows, "npu", draft_width=draft_width)
    _patch(case, 0, suppress=True)
    try:
        registry.warmup(*case.arguments())
        registry.execute(*case.arguments())
        torch.npu.synchronize()
        assert not torch.any(registry.draft.draft_cache).item()
        assert not torch.any(registry.draft.context_cache).item()
        assert torch.all(case.target_binding[2].slot_mapping == -1).item()
        assert torch.all(case.context_slots == -1).item()
        address = None
        for shift in (1, 2):
            _patch(case, shift, suppress=False)
            result = registry.execute(*case.arguments())
            torch.npu.synchronize()
            if address is not None:
                assert result.data_ptr() == address
            address = result.data_ptr()
            _check_outputs(case, shift)
        assert torch.any(registry.draft.draft_cache).item()
        assert torch.any(registry.draft.context_cache).item()
        assert registry.cache_info()["size"] == 1
    finally:
        registry.close()

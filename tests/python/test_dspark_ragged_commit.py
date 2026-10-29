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

"""Ragged greedy acceptance, per-request bonus and packed context contracts."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import pytest
import torch

from xllm.python.model_executor import dspark_ragged_commit
from xllm.python.model_executor.dspark_ragged import (
    DSparkVerifyBuffers,
    allocate_verify_prefixes,
    create_verify_buffers,
    pack_verify_inputs,
)
from xllm.python.model_executor.dspark_ragged_commit import greedy_verify_commit_out


def _inputs(
    width_values: list[int], rejected_positions: list[int], capacity: int, draft_width: int = 7
) -> tuple[torch.Tensor, ...]:
    batch_size = len(width_values)
    proposals = torch.full((batch_size, draft_width), -999, dtype=torch.int64)
    target_ids = []
    offsets = [0]
    owners = []
    columns = []
    for owner, (width, first_reject) in enumerate(zip(width_values, rejected_positions)):
        for column in range(width - 1):
            token = (1 << 33) + owner * 100 + column
            proposals[owner, column] = token
            target_ids.append((1 << 35) + owner if column == first_reject else token)
        target_ids.append((1 << 34) + owner)
        offsets.append(len(target_ids))
        owners.extend([owner] * width)
        columns.extend(range(width))
    padding = capacity - len(target_ids)
    target_ids.extend([-888] * padding)
    owners.extend([-1] * padding)
    columns.extend([-1] * padding)
    return (
        proposals,
        torch.tensor(target_ids, dtype=torch.int64),
        torch.tensor(width_values, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(owners, dtype=torch.int32),
        torch.tensor(columns, dtype=torch.int32),
        torch.arange(capacity, dtype=torch.int32) + 40,
        torch.full((batch_size, draft_width + 1), 777, dtype=torch.int64),
        torch.full((batch_size,), -77, dtype=torch.int32),
        torch.full((capacity,), 123, dtype=torch.int32),
    )


def _oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proposals, target_ids, widths, offsets, owners, columns, source_slots = inputs[:7]
    tokens = torch.full((proposals.shape[0], proposals.shape[1] + 1), -1, dtype=torch.int64)
    lengths = torch.empty(proposals.shape[0], dtype=torch.int32)
    context_slots = torch.full(source_slots.shape, -1, dtype=torch.int32)
    for request, width in enumerate(widths.tolist()):
        begin = int(offsets[request])
        cut = width - 1
        for step in range(width - 1):
            if proposals[request, step] != target_ids[begin + step]:
                cut = step
                break
        lengths[request] = cut + 1
        tokens[request, : cut + 1] = target_ids[begin : begin + cut + 1]
    for row, owner in enumerate(owners.tolist()):
        if owner >= 0 and int(columns[row]) < int(lengths[owner]):
            context_slots[row] = source_slots[row]
    return tokens, lengths, context_slots


@pytest.mark.parametrize(
    "width,first_reject", [(width, first_reject) for width in range(1, 9) for first_reject in range(width)]
)
def test_every_pruned_width_and_first_rejection_matches_oracle(width: int, first_reject: int) -> None:
    inputs = _inputs([width], [first_reject], 8)
    snapshots = tuple(tensor.clone() for tensor in inputs[:7])
    expected = _oracle(inputs)
    addresses = tuple(tensor.data_ptr() for tensor in inputs[7:])
    greedy_verify_commit_out(*inputs)
    for actual, reference in zip(inputs[7:], expected):
        torch.testing.assert_close(actual, reference)
    for actual, original in zip(inputs[:7], snapshots):
        torch.testing.assert_close(actual, original)
    assert addresses == tuple(tensor.data_ptr() for tensor in inputs[7:])
    assert int(inputs[8][0]) == min(first_reject + 1, width)


def test_mixed_requests_use_their_own_bonus_and_context_prefix() -> None:
    inputs = _inputs([1, 3, 8], [0, 2, 3], 16)
    expected = _oracle(inputs)
    greedy_verify_commit_out(*inputs)
    for actual, reference in zip(inputs[7:], expected):
        torch.testing.assert_close(actual, reference)
    assert inputs[8].tolist() == [1, 3, 4]
    assert int(inputs[7][0, 0]) == (1 << 34)
    assert int(inputs[7][1, 2]) == (1 << 34) + 1
    assert int(inputs[7][2, 3]) == (1 << 35) + 2
    assert inputs[9].tolist() == [40, 41, 42, 43, 44, 45, 46, 47] + [-1] * 8
    # Committed length includes one target correction/bonus, while the
    # proposed-draft output receives width-1 draft tokens.
    proposed = inputs[2] - 1
    accepted = inputs[8] - 1
    assert proposed.tolist() == [0, 2, 7]
    assert accepted.tolist() == [0, 2, 3]


@pytest.mark.parametrize("capacity", [0, 4, 32])
def test_idle_shard_clears_context_padding_without_loading_request_zero(capacity: int) -> None:
    inputs = _inputs([], [], capacity)
    greedy_verify_commit_out(*inputs)
    assert inputs[7].shape == (0, 8)
    assert inputs[8].numel() == 0
    assert inputs[9].tolist() == [-1] * capacity


def test_two_slots_reuse_outputs_without_stale_suffix_or_shared_context() -> None:
    slots = [_inputs([8, 2], [7, 0], 16), _inputs([1, 4, 3], [0, 3, 0], 16)]
    addresses = [tuple(tensor.data_ptr() for tensor in inputs[7:]) for inputs in slots]
    for _ in range(3):
        for inputs in slots:
            expected = _oracle(inputs)
            greedy_verify_commit_out(*inputs)
            for actual, reference in zip(inputs[7:], expected):
                torch.testing.assert_close(actual, reference)
        slots[0][0][0, 0] += 1
    assert addresses == [tuple(tensor.data_ptr() for tensor in inputs[7:]) for inputs in slots]
    assert slots[0][7].data_ptr() != slots[1][7].data_ptr()
    assert slots[0][9].data_ptr() != slots[1][9].data_ptr()


def test_changing_prefix_widths_reuses_the_same_physical_bucket_and_addresses() -> None:
    inputs = _inputs([1, 3, 8], [0, 2, 3], 16)
    addresses = tuple(tensor.data_ptr() for tensor in inputs)
    for widths, rejections in (([8, 4, 1], [7, 1, 0]), ([1, 1, 1], [0, 0, 0]), ([5, 5, 5], [2, 4, 0])):
        fresh = _inputs(widths, rejections, 16)
        for destination, source in zip(inputs[:7], fresh[:7]):
            destination.copy_(source)
        expected = _oracle(inputs)
        greedy_verify_commit_out(*inputs)
        for actual, reference in zip(inputs[7:], expected):
            torch.testing.assert_close(actual, reference)
        assert addresses == tuple(tensor.data_ptr() for tensor in inputs)


@pytest.mark.parametrize("field", [0, 2, 5, 7, 8, 9])
def test_invalid_tensor_metadata_is_rejected_before_output_writes(field: int) -> None:
    inputs = list(_inputs([3, 5], [1, 4], 16))
    inputs[field] = inputs[field].to(torch.float32)
    snapshots = tuple(tensor.clone() for tensor in inputs[7:])
    with pytest.raises(ValueError, match="shape|dtype"):
        greedy_verify_commit_out(*inputs)
    for actual, original in zip(inputs[7:], snapshots):
        torch.testing.assert_close(actual, original)


def test_overlapping_source_and_context_output_are_rejected_before_writes() -> None:
    inputs = list(_inputs([3, 5], [1, 4], 16))
    inputs[9] = inputs[6]
    snapshots = tuple(tensor.clone() for tensor in inputs)
    with pytest.raises(ValueError, match="overlap"):
        greedy_verify_commit_out(*inputs)
    for actual, original in zip(inputs, snapshots):
        torch.testing.assert_close(actual, original)


def test_insufficient_anchor_capacity_is_rejected_before_output_writes() -> None:
    inputs = list(_inputs([3, 5], [1, 4], 16))
    inputs[1] = inputs[1][:1]
    snapshots = tuple(tensor.clone() for tensor in inputs[7:])
    with pytest.raises(ValueError, match="at least one anchor"):
        greedy_verify_commit_out(*inputs)
    for actual, original in zip(inputs[7:], snapshots):
        torch.testing.assert_close(actual, original)


def test_reference_never_reads_device_values_to_host() -> None:
    tree = ast.parse(inspect.getsource(dspark_ragged_commit))
    forbidden = {"item", "tolist", "cpu", "numpy", "nonzero", "masked_select"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        assert not (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden)
        assert not (isinstance(node.func, ast.Name) and node.func.id in {"int", "bool", "float"})


def _assert_npu_outputs(inputs: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    # Readbacks are test assertions only. The operator never transfers an
    # acceptance/width to Host to select a launch or allocate an output.
    for actual, reference in zip(inputs[7:], expected):
        torch.testing.assert_close(actual.cpu(), reference)


@pytest.mark.parametrize("draft_width", [5, 7])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@torch.inference_mode()
def test_reference_commit_supports_both_widths_at_every_rejection(draft_width: int, batch_size: int) -> None:
    for first_reject in range(draft_width + 1):
        widths = [draft_width + 1] + [1 + index % (draft_width + 1) for index in range(1, batch_size)]
        inputs = _inputs(
            widths, [min(first_reject, width - 1) for width in widths], batch_size * (draft_width + 1), draft_width
        )
        expected = _oracle(inputs)
        greedy_verify_commit_out(*inputs)
        _assert_npu_outputs(inputs, expected)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("first_reject", range(8))
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_npu_greedy_commit_matches_reference_at_every_rejection(
    batch_size: int, first_reject: int, draft_width: int
) -> None:
    widths = [draft_width + 1] + [1 + (index - 1) % (draft_width + 1) for index in range(1, batch_size)]
    cpu_inputs = _inputs(
        widths, [min(first_reject, width - 1) for width in widths], batch_size * (draft_width + 1), draft_width
    )
    expected = _oracle(cpu_inputs)
    inputs = tuple(tensor.to("npu:0") for tensor in cpu_inputs)
    addresses = tuple(tensor.data_ptr() for tensor in inputs)
    torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
    torch.npu.synchronize()
    _assert_npu_outputs(inputs, expected)
    assert addresses == tuple(tensor.data_ptr() for tensor in inputs)
    for actual, original in zip(inputs[:7], cpu_inputs[:7]):
        torch.testing.assert_close(actual.cpu(), original)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch_size,capacity_scale", [(128, 128), (1, 128), (3, 8)])
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_npu_maximum_and_padded_capacity_dispatch_matches_reference(
    batch_size: int, capacity_scale: int, draft_width: int
) -> None:
    capacity = capacity_scale * (draft_width + 1)
    widths = [1 + index % (draft_width + 1) for index in range(batch_size)]
    rejections = [min(index % 5, width - 1) for index, width in enumerate(widths)]
    cpu_inputs = _inputs(widths, rejections, capacity, draft_width)
    expected = _oracle(cpu_inputs)
    inputs = tuple(tensor.to("npu:0") for tensor in cpu_inputs)
    addresses = tuple(tensor.data_ptr() for tensor in inputs)
    # Physical G can be larger than 8*logical B, including DP-idle padding.
    # The dispatch bucket must cover G as well as logical request storage.
    torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
    torch.npu.synchronize()
    _assert_npu_outputs(inputs, expected)
    assert addresses == tuple(tensor.data_ptr() for tensor in inputs)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 16])
@pytest.mark.parametrize("budget_kind", ["zero", "partial", "full"])
@pytest.mark.parametrize("confidence_value", [0.0, 1.0])
@torch.inference_mode()
def test_npu_equal_confidence_budget_boundaries_commit_correctly(
    batch_size: int, budget_kind: str, confidence_value: float
) -> None:
    budget = {"zero": 0, "partial": min(batch_size * 7, batch_size + 2), "full": batch_size * 7}[budget_kind]
    # Stable position/request ties allocate complete layers, then one token to
    # each leading request. The expected mapping is known before Device work.
    widths = [1 + budget // batch_size + (index < budget % batch_size) for index in range(batch_size)]
    cpu_inputs = _inputs(widths, [width - 1 for width in widths], batch_size * 8)
    expected = _oracle(cpu_inputs)
    inputs = tuple(tensor.to("npu:0") for tensor in cpu_inputs)
    confidence = torch.full((batch_size, 7), confidence_value, dtype=torch.float32, device="npu:0")
    torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, inputs[2], budget)
    torch.testing.assert_close(inputs[2].cpu(), cpu_inputs[2])
    torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
    torch.npu.synchronize()
    _assert_npu_outputs(inputs, expected)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("capacity", [0, 8, 128])
@torch.inference_mode()
def test_npu_idle_shard_writes_only_safe_context_padding(capacity: int) -> None:
    cpu_inputs = _inputs([], [], capacity)
    inputs = tuple(tensor.to("npu:0") for tensor in cpu_inputs)
    torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
    torch.npu.synchronize()
    _assert_npu_outputs(inputs, _oracle(cpu_inputs))


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 16])
@torch.inference_mode()
def test_npu_two_slot_graphs_replay_changing_prefixes_without_reallocation(batch_size: int) -> None:
    capacity = batch_size * 8
    stream = torch.npu.Stream(device="npu:0")
    slots = []
    graphs = []
    for _ in range(2):
        cpu_inputs = _inputs([8] * batch_size, [7] * batch_size, capacity)
        inputs = tuple(tensor.to("npu:0") for tensor in cpu_inputs)
        slots.append(inputs)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            torch.ops.xllm_ops.dspark_greedy_verify_commit(*inputs)
        graphs.append(graph)
    addresses = [tuple(tensor.data_ptr() for tensor in inputs) for inputs in slots]
    for iteration in range(3):
        for slot_index, (inputs, graph) in enumerate(zip(slots, graphs)):
            widths = [1 + (index + iteration + slot_index) % 8 for index in range(batch_size)]
            rejections = [min(iteration + index % 3, width - 1) for index, width in enumerate(widths)]
            fresh = _inputs(widths, rejections, capacity)
            expected = _oracle(fresh)
            with torch.npu.stream(stream):
                for destination, source in zip(inputs[:7], fresh[:7]):
                    destination.copy_(source)
                graph.replay()
            stream.synchronize()
            _assert_npu_outputs(inputs, expected)
            assert tuple(tensor.data_ptr() for tensor in inputs) == addresses[slot_index]
    assert slots[0][7].data_ptr() != slots[1][7].data_ptr()
    assert slots[0][9].data_ptr() != slots[1][9].data_ptr()


def _chain_live_inputs(iteration: int, slot_index: int) -> dict[str, torch.Tensor]:
    anchors = torch.arange(3, dtype=torch.int64) * 100 + 1000 * slot_index + 10 * iteration + 100
    proposals = anchors[:, None] + torch.arange(1, 8, dtype=torch.int64)
    confidence = torch.full((3, 7), 0.05, dtype=torch.float32)
    owner = (iteration + slot_index) % 3
    if iteration % 2 == 0:
        confidence[owner].fill_(0.99)
    else:
        confidence.fill_(1.0)
    if iteration == 1:
        proposals[owner, 0] += 9
    elif iteration == 2:
        proposals[owner, 2] += 7
    elif iteration == 3:
        proposals[:, 1] += 7
    positions = torch.tensor([15, 31, 47], dtype=torch.int32) + iteration * 3
    return {
        "confidence": confidence,
        "anchors": anchors,
        "proposals": proposals,
        "base_positions": positions,
        "base_kv_lengths": positions + 1,
        "base_block_table": torch.arange(24, dtype=torch.int32).view(3, 8) + 1 + slot_index * 64,
    }


def _run_captured_chain(
    live: dict[str, torch.Tensor],
    widths: torch.Tensor,
    offsets: torch.Tensor,
    packed: DSparkVerifyBuffers,
    target_ids: torch.Tensor,
    results: tuple[torch.Tensor, ...],
    budget: int,
    block_size: int,
) -> None:
    # The complete Device producer/consumer chain is captured with no
    # intermediate Host read, fence or CPU-derived width/offset.
    torch.ops.xllm_ops.dspark_allocate_verify_prefixes(live["confidence"], widths, budget)
    torch.ops.xllm_ops.dspark_pack_verify_inputs(
        live["anchors"],
        live["proposals"],
        live["base_positions"],
        live["base_kv_lengths"],
        live["base_block_table"],
        widths,
        offsets,
        packed.tokens,
        packed.positions,
        packed.kv_seq_lens,
        packed.slot_mapping,
        packed.block_table,
        packed.q_seq_lens,
        packed.owners,
        packed.columns,
        block_size,
    )
    # Preallocated int64 destination performs the cast, then emulates
    # a deterministic target predicting input-token+1 at every row.
    target_ids.copy_(packed.tokens)
    target_ids.add_(1)
    torch.ops.xllm_ops.dspark_greedy_verify_commit(
        live["proposals"],
        target_ids,
        widths,
        offsets,
        packed.owners,
        packed.columns,
        packed.slot_mapping,
        *results,
    )


_CHAIN_PLANS = ((0, 4), (1, 4), (5, 8), (13, 16), (21, 32), (5, 16))


def _chain_reference(
    live: dict[str, torch.Tensor], budget: int, capacity: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, DSparkVerifyBuffers, torch.Tensor, tuple[torch.Tensor, ...]]:
    batch_size = live["anchors"].shape[0]
    widths = torch.empty(batch_size, dtype=torch.int32)
    offsets = torch.empty(batch_size + 1, dtype=torch.int32)
    allocate_verify_prefixes(live["confidence"], budget, widths, offsets)
    packed = create_verify_buffers(capacity, 8, "cpu")
    pack_verify_inputs(
        **{name: tensor for name, tensor in live.items() if name != "confidence"},
        widths=widths,
        offsets=offsets,
        logical_rows=batch_size + budget,
        block_size=block_size,
        outputs=packed,
    )
    target_ids = packed.tokens.to(torch.int64) + 1
    expected = _oracle(
        (live["proposals"], target_ids, widths, offsets, packed.owners, packed.columns, packed.slot_mapping)
    )
    return widths, offsets, packed, target_ids, expected


@pytest.mark.parametrize("budget,capacity", _CHAIN_PLANS)
def test_cpu_chain_oracle_covers_typical_and_boundary_graph_plans(budget: int, capacity: int) -> None:
    for iteration in range(4):
        live = _chain_live_inputs(iteration, 0)
        widths, offsets, packed, target_ids, expected = _chain_reference(live, budget, capacity, 16)
        results = (
            torch.full((3, 8), 777, dtype=torch.int64),
            torch.full((3,), -77, dtype=torch.int32),
            torch.full((capacity,), 123, dtype=torch.int32),
        )
        greedy_verify_commit_out(
            live["proposals"], target_ids, widths, offsets, packed.owners, packed.columns, packed.slot_mapping, *results
        )
        for actual, reference in zip(results, expected):
            torch.testing.assert_close(actual, reference)
        assert int(widths.sum()) == 3 + budget
        assert int(offsets[-1]) == 3 + budget
        assert torch.all((results[1] >= 1) & (results[1] <= widths))
        assert torch.all(packed.owners[3 + budget :] == -1)
        assert torch.all(results[2][3 + budget :] == -1)
        if budget == 0:
            torch.testing.assert_close(results[0][:, 0], live["anchors"] + 1)
            assert torch.all(results[1] == 1)


@dataclass
class _ChainSlot:
    live: dict[str, torch.Tensor]
    widths: torch.Tensor
    offsets: torch.Tensor
    packed: DSparkVerifyBuffers
    target_ids: torch.Tensor
    results: tuple[torch.Tensor, ...]
    graphs: dict[tuple[int, int], torch.npu.NPUGraph]


def _chain_views(slot: _ChainSlot, capacity: int) -> tuple[DSparkVerifyBuffers, torch.Tensor, tuple[torch.Tensor, ...]]:
    packed = DSparkVerifyBuffers(**{name: tensor[:capacity] for name, tensor in vars(slot.packed).items()})
    results = (slot.results[0], slot.results[1], slot.results[2][:capacity])
    return packed, slot.target_ids[:capacity], results


def _chain_buffers(slot: _ChainSlot) -> tuple[torch.Tensor, ...]:
    return (*slot.live.values(), slot.widths, slot.offsets, *vars(slot.packed).values(), slot.target_ids, *slot.results)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("budget,capacity", _CHAIN_PLANS)
@torch.inference_mode()
def test_npu_captured_allocation_pack_target_commit_chain_preserves_slot_isolation(
    budget: int, capacity: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_size, block_size = 3, 16
    # K is a captured scalar, while G fixes the packed physical shape. Each
    # (K,G) therefore gets its own prebuilt graph, even when G is identical.
    plans = list(dict.fromkeys(((budget, capacity), (5, 16))))
    max_capacity = max(rows for _, rows in plans)
    stream = torch.npu.Stream(device="npu:0")
    slots = []
    for slot_index in range(2):
        live = {name: tensor.to("npu:0") for name, tensor in _chain_live_inputs(0, slot_index).items()}
        slot = _ChainSlot(
            live=live,
            widths=torch.empty(batch_size, dtype=torch.int32, device="npu:0"),
            offsets=torch.empty(batch_size + 1, dtype=torch.int32, device="npu:0"),
            packed=create_verify_buffers(max_capacity, 8, "npu:0"),
            target_ids=torch.empty(max_capacity, dtype=torch.int64, device="npu:0"),
            results=(
                torch.empty((batch_size, 8), dtype=torch.int64, device="npu:0"),
                torch.empty(batch_size, dtype=torch.int32, device="npu:0"),
                torch.empty(max_capacity, dtype=torch.int32, device="npu:0"),
            ),
            graphs={},
        )
        for plan_budget, plan_capacity in plans:
            packed, target_ids, results = _chain_views(slot, plan_capacity)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(2):
                    _run_captured_chain(
                        slot.live, slot.widths, slot.offsets, packed, target_ids, results, plan_budget, block_size
                    )
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                _run_captured_chain(
                    slot.live, slot.widths, slot.offsets, packed, target_ids, results, plan_budget, block_size
                )
            slot.graphs[(plan_budget, plan_capacity)] = graph
        slots.append(slot)
    addresses = [tuple(tensor.data_ptr() for tensor in _chain_buffers(slot)) for slot in slots]
    assert set(addresses[0]).isdisjoint(addresses[1])
    assert len({id(graph) for slot in slots for graph in slot.graphs.values()}) == len(slots) * len(plans)

    def _reject_capture(*args: object, **kwargs: object) -> None:
        raise AssertionError("serving replay must select a prebuilt K/G graph without recapture")

    monkeypatch.setattr(torch.npu, "NPUGraph", _reject_capture)
    monkeypatch.setattr(torch.npu, "graph", _reject_capture)
    accepted_prefixes = set()
    for iteration in range(4):
        for plan_budget, plan_capacity in plans if iteration % 2 == 0 else reversed(plans):
            for slot_index in (0, 1) if iteration % 2 == 0 else (1, 0):
                slot = slots[slot_index]
                graph = slot.graphs[(plan_budget, plan_capacity)]
                packed, target_ids, results = _chain_views(slot, plan_capacity)
                cpu_live = _chain_live_inputs(iteration, slot_index)
                expected_widths, expected_offsets, expected_packed, expected_target_ids, expected = _chain_reference(
                    cpu_live, plan_budget, plan_capacity, block_size
                )
                accepted_prefixes.add(tuple(expected[1].tolist()))
                # Readbacks are outside the captured chain and additionally
                # detect cross-Slot writes or writes beyond the selected G.
                other_buffers = _chain_buffers(slots[1 - slot_index])
                other_values = tuple(tensor.cpu() for tensor in other_buffers)
                capacity_buffers = (*vars(slot.packed).values(), slot.target_ids, slot.results[2])
                tails = tuple(tensor[plan_capacity:].cpu() for tensor in capacity_buffers)
                with torch.npu.stream(stream):
                    for name, source in cpu_live.items():
                        slot.live[name].copy_(source)
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(slot.widths.cpu(), expected_widths)
                torch.testing.assert_close(slot.offsets.cpu(), expected_offsets)
                torch.testing.assert_close(target_ids.cpu(), expected_target_ids)
                for name, reference in vars(expected_packed).items():
                    torch.testing.assert_close(getattr(packed, name).cpu(), reference)
                _assert_npu_outputs(
                    (
                        slot.live["proposals"],
                        target_ids,
                        slot.widths,
                        slot.offsets,
                        packed.owners,
                        packed.columns,
                        packed.slot_mapping,
                        *results,
                    ),
                    expected,
                )
                for tensor, reference in zip(other_buffers, other_values):
                    torch.testing.assert_close(tensor.cpu(), reference)
                for tensor, reference in zip(capacity_buffers, tails):
                    torch.testing.assert_close(tensor[plan_capacity:].cpu(), reference)
                for index, current in enumerate(slots):
                    assert tuple(tensor.data_ptr() for tensor in _chain_buffers(current)) == addresses[index]
    assert len(accepted_prefixes) > 1

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

"""Reference contracts for fixed-capacity multi-request DSpark verification."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from xllm.python.model_executor import dspark_ragged
from xllm.python.model_executor.dspark_ragged import (
    allocate_verify_prefixes,
    create_verify_buffers,
    pack_verify_inputs,
)


def _allocation(confidence: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = confidence.shape[0]
    widths = torch.empty(batch_size, dtype=torch.int32, device=confidence.device)
    offsets = torch.empty(batch_size + 1, dtype=torch.int32, device=confidence.device)
    allocate_verify_prefixes(confidence, budget, widths, offsets)
    return widths, offsets


def _vanilla_widths(confidence: torch.Tensor, budget: int) -> list[int]:
    probabilities = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    survival = torch.cumprod(probabilities, dim=1).tolist()
    ranked = sorted(
        (float(score), column, owner) for owner, scores in enumerate(survival) for column, score in enumerate(scores)
    )
    ranked.sort(key=lambda value: (-value[0], value[1], value[2]))
    widths = [1] * confidence.shape[0]
    for _, _, owner in ranked[:budget]:
        widths[owner] += 1
    return widths


@pytest.mark.parametrize(
    "batch_size,graph_rows", [(0, 8), (1, 1), (1, 8), (1, 16), (2, 4), (2, 8), (3, 4), (3, 8), (3, 16)]
)
def test_device_control_budget_matches_fixed_graph_capacity(batch_size: int, graph_rows: int) -> None:
    confidence = torch.full((batch_size, 7), 1.0)
    budget = min(7 * batch_size, graph_rows - batch_size)
    widths, offsets = _allocation(confidence, budget)
    assert widths.tolist() == _vanilla_widths(confidence, budget)
    assert offsets[-1].item() == min(8 * batch_size, graph_rows)
    assert offsets.tolist() == [0] + torch.cumsum(widths, dim=0).tolist()


@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 1.0])
def test_prefix_allocation_is_exact_and_matches_vanilla(batch_size: int, fraction: float) -> None:
    generator = torch.Generator().manual_seed(17 + batch_size)
    confidence = torch.rand(batch_size, 7, generator=generator)
    original = confidence.clone()
    budget = int(batch_size * 7 * fraction)
    widths, offsets = _allocation(confidence, budget)
    assert widths.tolist() == _vanilla_widths(confidence, budget)
    assert offsets.tolist() == [0] + torch.cumsum(widths, dim=0).tolist()
    assert int(widths.sum()) == batch_size + budget
    assert torch.all((widths >= 1) & (widths <= 8))
    torch.testing.assert_close(confidence, original)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_ties_prefer_position_then_request_and_never_drop_budget(value: float) -> None:
    widths, offsets = _allocation(torch.full((3, 7), value), 5)
    assert widths.tolist() == [3, 3, 2]
    assert offsets.tolist() == [0, 3, 6, 8]


def test_exact_budget_selects_closed_prefixes_across_plateaus() -> None:
    confidence = torch.tensor(
        [[1, 0.5, 1, 1, 0, 1, 1], [0.5, 1, 1, 1, 1, 0, 1], [0.5, 1, 0, 1, 1, 1, 1]], dtype=torch.float32
    )
    survival = torch.cumprod(confidence, dim=1)
    ranked = sorted(
        ((float(survival[owner, column]), column, owner) for owner in range(3) for column in range(7)),
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
    )
    for budget in range(22):
        widths, _ = _allocation(confidence, budget)
        selected = {(column, owner) for owner, width in enumerate(widths.tolist()) for column in range(width - 1)}
        assert selected == {(column, owner) for _, column, owner in ranked[:budget]}
        assert len(selected) == budget


def test_confidence_normalization_has_explicit_exact_budget_semantics() -> None:
    confidence = torch.tensor([[float("nan"), float("inf"), -float("inf"), -2, 2, 0, 1]])
    widths, offsets = _allocation(confidence, 7)
    assert widths.tolist() == [8]
    assert offsets.tolist() == [0, 8]


def _pack(widths_values: list[int], capacity: int) -> tuple[dspark_ragged.DSparkVerifyBuffers, torch.Tensor]:
    batch_size = len(widths_values)
    widths = torch.tensor(widths_values, dtype=torch.int32)
    offsets = torch.full((batch_size + 1,), -99, dtype=torch.int32)
    anchors = torch.arange(100, 100 + batch_size, dtype=torch.int64)
    proposals = torch.arange(batch_size * 7, dtype=torch.int64).view(batch_size, 7) + 1000
    positions = torch.arange(batch_size, dtype=torch.int32) * 3 + 3
    # Intentionally not position+1: slots must use cache position, while the
    # attention length follows its independently supplied anchor-inclusive base.
    kv_lengths = positions + 9
    tables = torch.arange(batch_size * 4, dtype=torch.int32).view(batch_size, 4) + 10
    outputs = create_verify_buffers(capacity, 4, "cpu")
    pack_verify_inputs(
        anchors=anchors,
        proposals=proposals,
        base_positions=positions,
        base_kv_lengths=kv_lengths,
        base_block_table=tables,
        widths=widths,
        offsets=offsets,
        logical_rows=sum(widths_values),
        block_size=16,
        outputs=outputs,
    )
    start = 0
    for owner, width in enumerate(widths_values):
        for column in range(width):
            row = start + column
            position = int(positions[owner]) + column
            expected_token = int(anchors[owner]) if column == 0 else int(proposals[owner, column - 1])
            assert int(outputs.tokens[row]) == expected_token
            assert int(outputs.positions[row]) == position
            assert int(outputs.kv_seq_lens[row]) == int(kv_lengths[owner]) + column
            assert int(outputs.slot_mapping[row]) == int(tables[owner, position // 16]) * 16 + position % 16
            assert int(outputs.owners[row]) == owner
            assert int(outputs.columns[row]) == column
            torch.testing.assert_close(outputs.block_table[row], tables[owner])
        start += width
    return outputs, offsets


@pytest.mark.parametrize("widths", [[1], [8], [8, 1], [2, 7, 4], [1, 8] * 8])
def test_pack_preserves_request_isolation_with_safe_padding(widths: list[int]) -> None:
    logical_rows = sum(widths)
    outputs, offsets = _pack(widths, logical_rows + 5)
    assert offsets.tolist() == [0] + torch.cumsum(torch.tensor(widths), dim=0).tolist()
    assert outputs.tokens[logical_rows:].tolist() == [1] * 5
    assert outputs.positions[logical_rows:].tolist() == [0] * 5
    assert outputs.kv_seq_lens[logical_rows:].tolist() == [1] * 5
    assert outputs.slot_mapping[logical_rows:].tolist() == [-1] * 5
    assert outputs.owners[logical_rows:].tolist() == [-1] * 5
    assert outputs.columns[logical_rows:].tolist() == [-1] * 5
    assert outputs.q_seq_lens.tolist() == [1] * outputs.tokens.numel()
    assert torch.count_nonzero(outputs.block_table[logical_rows:]) == 0


@pytest.mark.parametrize("capacity", [0, 4])
def test_idle_dp_pack_contains_only_safe_padding(capacity: int) -> None:
    outputs, offsets = _pack([], capacity)
    assert offsets.tolist() == [0]
    assert outputs.slot_mapping.tolist() == [-1] * capacity
    assert outputs.tokens.tolist() == [1] * capacity
    assert outputs.kv_seq_lens.tolist() == [1] * capacity


def test_pack_recomputes_offsets_after_authoritative_width_change() -> None:
    outputs, offsets = _pack([8, 1, 1], 10)
    assert offsets.tolist() == [0, 8, 9, 10]
    assert outputs.owners.tolist() == [0] * 8 + [1, 2]


@pytest.mark.parametrize("budget", [-1, 15])
def test_allocation_rejects_invalid_host_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        _allocation(torch.ones(2, 7), budget)


def test_allocation_rejects_tensor_budget_without_reading_it() -> None:
    with pytest.raises(ValueError, match="host integer"):
        _allocation(torch.ones(2, 7), torch.tensor(2))


def test_pack_rejects_capacity_smaller_than_known_logical_rows() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _pack([8, 8], 15)


@pytest.mark.parametrize("alias", ["confidence_widths", "confidence_offsets", "widths_offsets"])
def test_allocation_rejects_overlapping_spans_before_any_writes(alias: str) -> None:
    confidence = torch.ones(2, 7)
    widths = torch.full((2,), -77, dtype=torch.int32)
    offsets = torch.full((3,), -99, dtype=torch.int32)
    if alias == "confidence_widths":
        widths = confidence.view(torch.int32).flatten()[:2]
    elif alias == "confidence_offsets":
        offsets = confidence.view(torch.int32).flatten()[:3]
    else:
        widths = offsets[:2]
    tensors = (confidence, widths, offsets)
    snapshots = tuple(tensor.clone() for tensor in tensors)
    with pytest.raises(ValueError, match="overlap"):
        allocate_verify_prefixes(confidence, 5, widths, offsets)
    for actual, original in zip(tensors, snapshots):
        torch.testing.assert_close(actual, original)


def test_offsets_reject_alias_with_widths_before_initializing_zero() -> None:
    offsets = torch.tensor([2, 3, -99], dtype=torch.int32)
    original = offsets.clone()
    with pytest.raises(ValueError, match="overlap"):
        dspark_ragged.build_verify_offsets(offsets[:2], offsets)
    torch.testing.assert_close(offsets, original)


def _output_tensors(outputs: dspark_ragged.DSparkVerifyBuffers) -> tuple[torch.Tensor, ...]:
    return (
        outputs.tokens,
        outputs.positions,
        outputs.kv_seq_lens,
        outputs.slot_mapping,
        outputs.block_table,
        outputs.q_seq_lens,
        outputs.owners,
        outputs.columns,
    )


def _reference_pack(
    inputs: dict[str, torch.Tensor],
    widths: torch.Tensor,
    offsets: torch.Tensor,
    outputs: dspark_ragged.DSparkVerifyBuffers,
    logical_rows: int,
) -> None:
    pack_verify_inputs(
        **inputs, widths=widths, offsets=offsets, logical_rows=logical_rows, block_size=16, outputs=outputs
    )


@pytest.mark.parametrize(
    "alias",
    [
        "tokens_positions_input",
        "positions_lengths",
        "offsets_widths",
        "offsets_tokens",
        "tables_input",
        "proposals_tokens",
    ],
)
def test_pack_rejects_overlapping_spans_before_any_writes(alias: str) -> None:
    inputs = _native_inputs(2, "cpu")
    widths = torch.ones(2, dtype=torch.int32)
    offsets = torch.full((3,), -99, dtype=torch.int32)
    outputs = create_verify_buffers(3 if alias == "offsets_tokens" else 2, 256, "cpu")
    for tensor in _output_tensors(outputs):
        tensor.fill_(-77)
    if alias == "tokens_positions_input":
        outputs.tokens = inputs["base_positions"]
    elif alias == "positions_lengths":
        outputs.positions = outputs.kv_seq_lens
    elif alias == "offsets_widths":
        widths = offsets[:2]
    elif alias == "offsets_tokens":
        offsets = outputs.tokens
    elif alias == "tables_input":
        outputs.block_table = inputs["base_block_table"]
    else:
        outputs.tokens = inputs["proposals"].view(torch.int32).flatten()[:2]
    tensors = (*inputs.values(), widths, offsets, *_output_tensors(outputs))
    snapshots = tuple(tensor.clone() for tensor in tensors)
    with pytest.raises(ValueError, match="overlap"):
        _reference_pack(inputs, widths, offsets, outputs, 2)
    for actual, original in zip(tensors, snapshots):
        torch.testing.assert_close(actual, original)


def _arena_outputs(
    capacity: int, table_width: int, prefix_elements: int, device: str
) -> tuple[torch.Tensor, dspark_ragged.DSparkVerifyBuffers]:
    arena = torch.full((prefix_elements + capacity * (7 + table_width),), -77, dtype=torch.int32, device=device)
    vectors = [
        arena[prefix_elements + index * capacity : prefix_elements + (index + 1) * capacity] for index in range(7)
    ]
    tables = arena[prefix_elements + 7 * capacity :].view(capacity, table_width)
    outputs = dspark_ragged.DSparkVerifyBuffers(
        tokens=vectors[0],
        positions=vectors[1],
        kv_seq_lens=vectors[2],
        slot_mapping=vectors[3],
        block_table=tables,
        q_seq_lens=vectors[4],
        owners=vectors[5],
        columns=vectors[6],
    )
    return arena, outputs


@pytest.mark.parametrize("device", ["cpu", "npu:0"])
@torch.inference_mode()
def test_disjoint_live_spans_in_shared_arenas_are_supported(device: str) -> None:
    if device != "cpu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an NPU")
    allocation_arena = torch.ones(19, dtype=torch.float32, device=device)
    confidence = allocation_arena[:14].view(2, 7)
    widths = allocation_arena[14:16].view(torch.int32)
    offsets = allocation_arena[16:19].view(torch.int32)
    if device == "cpu":
        allocate_verify_prefixes(confidence, 5, widths, offsets)
    else:
        torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, widths, 5)
    assert widths.cpu().tolist() == [4, 3]
    inputs = _native_inputs(2, device)
    arena, outputs = _arena_outputs(12, 256, 2, device)
    arena[:2].copy_(inputs["base_positions"])
    inputs["base_positions"] = arena[:2]
    expected_positions = inputs["base_positions"].clone()
    if device == "cpu":
        _reference_pack(inputs, widths, offsets, outputs, 7)
    else:
        _native_pack(inputs, widths, offsets, outputs)
    torch.testing.assert_close(inputs["base_positions"], expected_positions)
    _check_native_pack(inputs, widths, offsets, outputs, 7)


def test_reference_does_not_read_device_values_or_infer_output_shapes() -> None:
    tree = ast.parse(inspect.getsource(dspark_ragged))
    forbidden = {"item", "tolist", "cpu", "numpy", "nonzero", "masked_select"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in forbidden for node in calls)
    for node in calls:
        if isinstance(node.func, ast.Name) and node.func.id in {"int", "bool", "float"}:
            pytest.fail("reference must not convert device tensor values to host scalars")


def _native_inputs(
    batch_size: int, device: str, table_width: int = 256, draft_width: int = 7
) -> dict[str, torch.Tensor]:
    positions = (torch.arange(batch_size, dtype=torch.int32) * 17 + 15) % (table_width * 16 - 7)
    return {
        "anchors": torch.arange(100, 100 + batch_size, dtype=torch.int64).to(device),
        "proposals": (
            torch.arange(batch_size * draft_width, dtype=torch.int64).view(batch_size, draft_width) + 1000
        ).to(device),
        "base_positions": positions.to(device),
        "base_kv_lengths": (positions + 9).to(device),
        "base_block_table": (
            torch.arange(batch_size * table_width, dtype=torch.int32).view(batch_size, table_width) + 10
        ).to(device),
    }


def _native_pack(
    inputs: dict[str, torch.Tensor],
    widths: torch.Tensor,
    offsets: torch.Tensor,
    outputs: dspark_ragged.DSparkVerifyBuffers,
) -> None:
    torch.ops.xllm_ops.dspark_pack_verify_inputs(
        inputs["anchors"],
        inputs["proposals"],
        inputs["base_positions"],
        inputs["base_kv_lengths"],
        inputs["base_block_table"],
        widths,
        offsets,
        outputs.tokens,
        outputs.positions,
        outputs.kv_seq_lens,
        outputs.slot_mapping,
        outputs.block_table,
        outputs.q_seq_lens,
        outputs.owners,
        outputs.columns,
        16,
    )


def _check_native_pack(
    inputs: dict[str, torch.Tensor],
    widths: torch.Tensor,
    offsets: torch.Tensor,
    outputs: dspark_ragged.DSparkVerifyBuffers,
    logical_rows: int,
) -> None:
    reference = create_verify_buffers(outputs.tokens.numel(), inputs["base_block_table"].shape[1], "cpu")
    reference_offsets = torch.empty_like(offsets, device="cpu")
    pack_verify_inputs(
        **{name: tensor.cpu() for name, tensor in inputs.items()},
        widths=widths.cpu(),
        offsets=reference_offsets,
        logical_rows=logical_rows,
        block_size=16,
        outputs=reference,
    )
    torch.testing.assert_close(offsets.cpu(), reference_offsets)
    for name in (
        "tokens",
        "positions",
        "kv_seq_lens",
        "slot_mapping",
        "block_table",
        "q_seq_lens",
        "owners",
        "columns",
    ):
        torch.testing.assert_close(getattr(outputs, name).cpu(), getattr(reference, name))


@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_native_dspark_allocation_and_packing_match_reference(
    batch_size: int, fraction: float, draft_width: int
) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    generator = torch.Generator().manual_seed(107 + batch_size)
    confidence = torch.rand(batch_size, draft_width, generator=generator)
    budget = int(batch_size * draft_width * fraction)
    expected_widths, _ = _allocation(confidence, budget)
    widths = torch.empty(batch_size, dtype=torch.int32, device="npu:0")
    offsets = torch.empty(batch_size + 1, dtype=torch.int32, device="npu:0")
    torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence.to("npu:0"), widths, budget)
    torch.testing.assert_close(widths.cpu(), expected_widths)
    inputs = _native_inputs(batch_size, "npu:0", draft_width=draft_width)
    outputs = create_verify_buffers(min(batch_size + budget + 5, 128 * (draft_width + 1)), 256, "npu:0")
    _native_pack(inputs, widths, offsets, outputs)
    _check_native_pack(inputs, widths, offsets, outputs, batch_size + budget)


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("draft_width", [5, 7])
@torch.inference_mode()
def test_native_dspark_allocation_resolves_equal_scores_stably(value: float, draft_width: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    confidence = torch.full((3, draft_width), value, device="npu:0")
    widths = torch.empty(3, dtype=torch.int32, device="npu:0")
    torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, widths, 5)
    assert widths.cpu().tolist() == [3, 3, 2]


@pytest.mark.parametrize("budget", [1, 5, 13, 20])
@torch.inference_mode()
def test_native_dspark_allocation_normalizes_extremes_before_ranking(budget: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    confidence = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), -2, 2, 0, 1],
            [float("inf"), 1, 1, 1, float("nan"), 0, 1],
            [-2, 2, 0, 0.5, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    expected, _ = _allocation(confidence, budget)
    device_confidence = confidence.to("npu:0")
    widths = torch.empty(3, dtype=torch.int32, device="npu:0")
    torch.ops.xllm_ops.dspark_allocate_verify_prefixes(device_confidence, widths, budget)
    torch.testing.assert_close(widths.cpu(), expected)
    torch.testing.assert_close(device_confidence.cpu(), confidence, equal_nan=True)


@pytest.mark.parametrize("batch_size,capacity", [(0, 0), (0, 32), (1, 1024), (3, 64)])
@torch.inference_mode()
def test_native_dspark_pack_handles_empty_and_large_physical_padding(batch_size: int, capacity: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    widths = torch.full((batch_size,), 8, dtype=torch.int32, device="npu:0")
    offsets = torch.full((batch_size + 1,), -99, dtype=torch.int32, device="npu:0")
    inputs = _native_inputs(batch_size, "npu:0")
    outputs = create_verify_buffers(capacity, 256, "npu:0")
    _native_pack(inputs, widths, offsets, outputs)
    _check_native_pack(inputs, widths, offsets, outputs, batch_size * 8)


@pytest.mark.parametrize("table_width", [1, 3, 4095, 4096, 4097, 8192, 8195])
@pytest.mark.parametrize("capacity", [31, 32, 33])
@torch.inference_mode()
def test_native_dspark_pack_handles_table_and_row_tile_boundaries(table_width: int, capacity: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    widths = torch.full((3,), 8, dtype=torch.int32, device="npu:0")
    offsets = torch.full((4,), -99, dtype=torch.int32, device="npu:0")
    inputs = _native_inputs(3, "npu:0", table_width)
    # Exercise both sides of the UB tile boundary and the last page. A short
    # prompt in a wide table would only test page lookup in the first tile.
    if table_width > 4096:
        positions = torch.tensor([4096 * 16 - 4, 4096 * 16, table_width * 16 - 8], dtype=torch.int32)
        inputs["base_positions"].copy_(positions)
        inputs["base_kv_lengths"].copy_(positions + 1)
    outputs = create_verify_buffers(capacity, table_width, "npu:0")
    _native_pack(inputs, widths, offsets, outputs)
    _check_native_pack(inputs, widths, offsets, outputs, 24)


@pytest.mark.parametrize("batch_size,table_width", [(2, 256), (3, 256), (16, 256), (3, 8192), (3, 8195)])
@torch.inference_mode()
def test_native_dspark_graph_replays_changed_prefixes_across_slots(batch_size: int, table_width: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    budget = batch_size * 3
    logical_rows = batch_size + budget
    stream = torch.npu.Stream(device="npu:0")
    slots = []
    for _ in range(2):
        confidence = torch.ones(batch_size, 7, device="npu:0")
        widths = torch.empty(batch_size, dtype=torch.int32, device="npu:0")
        offsets = torch.empty(batch_size + 1, dtype=torch.int32, device="npu:0")
        inputs = _native_inputs(batch_size, "npu:0", table_width)
        if table_width > 4096:
            inputs["base_positions"].fill_(4096 * 16 - 3)
            inputs["base_kv_lengths"].copy_(inputs["base_positions"] + 1)
        outputs = create_verify_buffers(logical_rows + 5, table_width, "npu:0")
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, widths, budget)
            _native_pack(inputs, widths, offsets, outputs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, widths, budget)
            _native_pack(inputs, widths, offsets, outputs)
        slots.append((graph, confidence, widths, offsets, inputs, outputs))
    addresses = [
        tuple(
            tensor.data_ptr() for tensor in (confidence, widths, offsets, *inputs.values(), *_output_tensors(outputs))
        )
        for _, confidence, widths, offsets, inputs, outputs in slots
    ]
    for first, second in zip(_output_tensors(slots[0][-1]), _output_tensors(slots[1][-1])):
        assert first.data_ptr() != second.data_ptr()
    for round_id, slot_id in enumerate((0, 1, 0, 1, 1, 0)):
        graph, confidence, widths, offsets, inputs, outputs = slots[slot_id]
        probabilities = torch.full((batch_size, 7), 0.05)
        probabilities[round_id % batch_size].fill_(0.99)
        expected_widths, _ = _allocation(probabilities, budget)
        confidence.copy_(probabilities)
        inputs["anchors"].add_(1)
        inputs["proposals"].add_(2)
        inputs["base_positions"].add_(1)
        inputs["base_kv_lengths"].add_(3)
        inputs["base_block_table"].add_(7)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.current_stream().wait_stream(stream)
        torch.testing.assert_close(widths.cpu(), expected_widths)
        _check_native_pack(inputs, widths, offsets, outputs, logical_rows)
        assert addresses[slot_id] == tuple(
            tensor.data_ptr() for tensor in (confidence, widths, offsets, *inputs.values(), *_output_tensors(outputs))
        )

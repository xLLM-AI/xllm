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

"""Real-device coverage for ragged page packing and per-layer translation."""

from __future__ import annotations

import pytest
import torch

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from solution import prepare  # noqa: E402

_DEVICE = "npu:0"


def _reference(x: torch.Tensor, m: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """CPU oracle that follows the operator definition literally."""
    single = m.dim() == 1
    layers, blocks = (1, m.numel()) if single else m.shape
    counts = cu.tolist()
    shape = (counts[-1],) if single else (layers, counts[-1])
    out = torch.full(shape, -1, dtype=torch.int32)
    x_cpu = x.detach().cpu()
    m_cpu = m.detach().cpu()
    for row in range(x.shape[0]):
        start, end = counts[row], counts[row + 1]
        for offset in range(end - start):
            page = int(x_cpu[row, offset])
            if page < 0:
                continue
            if single:
                out[start + offset] = m_cpu[page]
            else:
                out[:, start + offset] = m_cpu[:, page]
    return out


def _cu(counts: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        device=_DEVICE,
        dtype=torch.int32,
    )


def _x(rows: int, columns: int, counts: list[int], blocks: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randint(-2, blocks, (rows, columns), device=_DEVICE, dtype=torch.int32)
    for row, count in enumerate(counts):
        x[row, count:] = 999999
    return x


def _m(layers: int, blocks: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(blocks, device=_DEVICE, dtype=torch.int32) for _ in range(layers)]
    )
    return permutations[0] if layers == 1 else permutations


def _check(x: torch.Tensor, m: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    saved = (x.clone(), m.clone(), cu.clone())
    plan = prepare([x, m, cu], {})
    actual = plan.run()
    expected = _reference(x, m, cu).to(_DEVICE)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.int32
    assert actual.device == x.device
    assert actual.is_contiguous()
    for before, after in zip(saved, (x, m, cu)):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    return actual


def test_literal_case() -> None:
    x = torch.tensor(
        [[0, 2, -7], [1, -7, -7], [3, -7, -7]], device=_DEVICE, dtype=torch.int32
    )
    m = torch.arange(8, device=_DEVICE, dtype=torch.int32).reshape(2, 4)
    cu = torch.tensor([0, 2, 2, 3], device=_DEVICE, dtype=torch.int32)
    actual = _check(x, m, cu)
    assert tuple(actual.shape) == (2, 3)
    assert actual[0].tolist() == [0, 2, 3]
    assert actual[1].tolist() == [4, 6, 7]


def test_prefix_padding_ignores_invalid_tail() -> None:
    counts = [2, 3]
    x = _x(2, 3, counts, 4, seed=11)
    m = _m(2, 4, seed=12)
    _check(x, m, _cu(counts))
    single = _x(1, 4, [4], 7, seed=13)
    _check(single, _m(1, 7, seed=14), _cu([4]))


def test_empty_requests() -> None:
    x = torch.empty(0, 3, device=_DEVICE, dtype=torch.int32)
    m = _m(2, 31, seed=15)
    actual = _check(x, m, torch.tensor([0], device=_DEVICE, dtype=torch.int32))
    assert tuple(actual.shape) == (2, 0)


def test_zero_counts_and_empty_columns() -> None:
    x = torch.randint(0, 2, (2, 2), device=_DEVICE, dtype=torch.int32)
    actual = _check(x, _m(2, 4, seed=16), _cu([0, 0]))
    assert tuple(actual.shape) == (2, 0)

    x = torch.empty(3, 0, device=_DEVICE, dtype=torch.int32)
    actual = _check(x, _m(2, 31, seed=17), _cu([0, 0, 0]))
    assert tuple(actual.shape) == (2, 0)


def test_storage_offsets() -> None:
    counts = [13, 0, 33, 1, 7, 24, 6]
    torch.manual_seed(18)
    x_backing = torch.randint(-2, 129, (7 * 33 + 1,), device=_DEVICE, dtype=torch.int32)
    x = x_backing[1:].view(7, 33)
    for row, count in enumerate(counts):
        x[row, count:] = 999999
    m_backing = torch.zeros(3 * 129 + 1, device=_DEVICE, dtype=torch.int32)
    m_backing[1:] = torch.randperm(3 * 129, device=_DEVICE, dtype=torch.int32)
    m = m_backing[1:].view(3, 129)
    cu_backing = torch.zeros(9, device=_DEVICE, dtype=torch.int32)
    cu = cu_backing[1:]
    cu.copy_(_cu(counts))
    assert x.storage_offset() == 1 and m.storage_offset() == 1 and cu.storage_offset() == 1
    actual = _check(x, m, cu)
    assert tuple(actual.shape) == (3, 84)


def test_boundary_257() -> None:
    counts = [243, 0, 0]
    x = _x(3, 257, counts, 257, seed=19)
    torch.manual_seed(20)
    m = torch.randperm(257, device=_DEVICE, dtype=torch.int32).reshape(1, 257)
    actual = _check(x, m, _cu(counts))
    assert tuple(actual.shape) == (1, 243)


def test_performance_shapes() -> None:
    small_counts = [64, 64, 64, 64, 31, 0, 0, 0]
    x = _x(8, 64, small_counts, 1024, seed=21)
    m = _m(8, 1024, seed=22)
    actual = _check(x, m, _cu(small_counts))
    assert tuple(actual.shape) == (8, 287)

    large_counts = [128] * 60 + [55, 55, 54, 54]
    x = _x(64, 256, large_counts, 4096, seed=23)
    m = _m(32, 4096, seed=24)
    actual = _check(x, m, _cu(large_counts))
    assert tuple(actual.shape) == (32, 7898)


def test_skewed_valid_prefix() -> None:
    counts = [3, 0, 2, 1, 4, 0]
    x = _x(6, 9, counts, 4, seed=25)
    m = _m(3, 4, seed=26)
    actual = _check(x, m, _cu(counts))
    assert tuple(actual.shape) == (3, 10)


@pytest.mark.parametrize("total", [1023, 1024])
def test_route_shapes(total: int) -> None:
    head = [385, 385]
    counts = [*head, total - sum(head), 0, 0, 0, 0]
    x = _x(7, 385, counts, 67, seed=27 + total)
    m = _m(32, 67, seed=29 + total)
    actual = _check(x, m, _cu(counts))
    assert tuple(actual.shape) == (32, total)


def test_single_layer_long_rows() -> None:
    counts = [1025] * 4 + [120, 120, 120, 119, 119]
    x = _x(9, 1025, counts, 67, seed=31)
    m = _m(1, 67, seed=32)
    actual = _check(x, m, _cu(counts))
    assert tuple(actual.shape) == (4698,)


def test_dynamic_counts_with_same_total() -> None:
    counts = [64, 64, 64, 64, 31, 0, 0, 0]
    x = _x(8, 64, counts, 1024, seed=33)
    m = _m(8, 1024, seed=34)
    cu = _cu(counts)
    plan = prepare([x, m, cu], {})
    first = plan.run()
    torch.testing.assert_close(first, _reference(x, m, cu).to(_DEVICE), rtol=0, atol=0)

    shifted = [64, 64, 64, 64, 0, 31, 0, 0]
    torch.manual_seed(35)
    x_new = torch.randint(-2, 1024, (8, 64), device=_DEVICE, dtype=torch.int32)
    for row, count in enumerate(shifted):
        x_new[row, count:] = 999999
    x.copy_(x_new)
    torch.manual_seed(36)
    m_new = torch.stack([torch.randperm(1024, device=_DEVICE, dtype=torch.int32) for _ in range(8)])
    m.copy_(m_new)
    cu.copy_(_cu(shifted))
    second = plan.run()
    torch.testing.assert_close(second, _reference(x, m, cu).to(_DEVICE), rtol=0, atol=0)
    assert second.data_ptr() != first.data_ptr()


def test_storage_rebinding_keeps_prepared_plan() -> None:
    counts = [64, 64, 64, 64, 31, 0, 0, 0]
    x = _x(8, 64, counts, 1024, seed=37)
    m = _m(8, 1024, seed=38)
    cu = _cu(counts)
    plan = prepare([x, m, cu], {})
    plan.run()

    torch.manual_seed(39)
    x_backing = torch.randint(-2, 1024, (8 * 64 + 1,), device=_DEVICE, dtype=torch.int32)
    x_new = x_backing[1:].view(8, 64)
    for row, count in enumerate(counts):
        x_new[row, count:] = 999999
    m_backing = torch.zeros(8 * 1024 + 1, device=_DEVICE, dtype=torch.int32)
    m_backing[1:] = torch.stack(
        [torch.randperm(1024, device=_DEVICE, dtype=torch.int32) for _ in range(8)]
    ).reshape(-1)
    m_new = m_backing[1:].view(8, 1024)
    cu_backing = torch.zeros(10, device=_DEVICE, dtype=torch.int32)
    cu_new = cu_backing[1:]
    cu_new.copy_(_cu(counts))
    x.set_(x_new)
    m.set_(m_new)
    cu.set_(cu_new)
    actual = plan.run()
    torch.testing.assert_close(actual, _reference(x_new, m_new, cu_new).to(_DEVICE), rtol=0, atol=0)


def test_independent_outputs_and_untouched_inputs() -> None:
    counts = [2, 3]
    x = _x(2, 3, counts, 4, seed=41)
    m = _m(2, 4, seed=42)
    cu = _cu(counts)
    saved = (x.clone(), m.clone(), cu.clone())
    plan = prepare([x, m, cu], {})
    first, second = plan.run(), plan.run()
    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() != x.data_ptr() and first.data_ptr() != m.data_ptr()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for before, after in zip(saved, (x, m, cu)):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_current_stream_is_used() -> None:
    counts = [2, 3]
    x = _x(2, 3, counts, 4, seed=43)
    m = _m(2, 4, seed=44)
    cu = _cu(counts)
    plan = prepare([x, m, cu], {})
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        actual = plan.run()
    stream.synchronize()
    torch.testing.assert_close(actual, _reference(x, m, cu).to(_DEVICE), rtol=0, atol=0)


def test_large_physical_pages_are_exact_at_prepare_time() -> None:
    x = torch.tensor([[0, 1]], device=_DEVICE, dtype=torch.int32)
    m = torch.tensor([1 << 24, (1 << 24) + 1], device=_DEVICE, dtype=torch.int32)
    cu = torch.tensor([0, 2], device=_DEVICE, dtype=torch.int32)
    actual = prepare([x, m, cu], {}).run()
    torch.testing.assert_close(
        actual,
        torch.tensor([1 << 24, (1 << 24) + 1], device=_DEVICE, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_mapping_mutation_above_float_mantissa_is_exact() -> None:
    counts = [64, 64, 64, 64, 31, 0, 0, 0]
    x = _x(8, 64, counts, 1024, seed=47)
    m = _m(8, 1024, seed=48)
    cu = _cu(counts)
    plan = prepare([x, m, cu], {})
    page = (1 << 24) + 1
    m.fill_(page)
    actual = plan.run()
    expected = torch.where(
        _reference(x, torch.zeros_like(m), cu).to(_DEVICE) >= 0,
        torch.tensor(page, device=_DEVICE, dtype=torch.int32),
        torch.tensor(-1, device=_DEVICE, dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wide_mapping_uses_exact_int32_path() -> None:
    blocks = (1 << 20) + 1
    m = torch.zeros(blocks, device=_DEVICE, dtype=torch.int32)
    m[0] = 1 << 20
    m[1] = (1 << 20) - 1
    x = torch.tensor([[0, 1]], device=_DEVICE, dtype=torch.int32)
    cu = torch.tensor([0, 2], device=_DEVICE, dtype=torch.int32)
    actual = prepare([x, m, cu], {}).run()
    torch.testing.assert_close(
        actual,
        torch.tensor([1 << 20, (1 << 20) - 1], device=_DEVICE, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def _base_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = [64, 64, 64, 64, 31, 0, 0, 0]
    return _x(8, 64, counts, 1024, seed=45), _m(8, 1024, seed=46), _cu(counts)


@pytest.mark.parametrize(
    "mutation",
    [
        "input_count",
        "unknown_attribute",
        "cpu_input",
        "dtype",
        "noncontiguous",
        "negative_mapping",
        "logical_address_out_of_range",
        "nonmonotonic_cu",
    ],
)
def test_invalid_inputs_are_rejected(mutation: str) -> None:
    x, m, cu = _base_inputs()
    attributes: dict[str, int] = {}
    if mutation == "input_count":
        inputs = [x, m]
    elif mutation == "unknown_attribute":
        inputs = [x, m, cu]
        attributes = {"unknown": 1}
    elif mutation == "cpu_input":
        inputs = [x.cpu(), m, cu]
    elif mutation == "dtype":
        inputs = [x.to(torch.int64), m, cu]
    elif mutation == "noncontiguous":
        inputs = [x[:, ::2], m, cu]
    elif mutation == "negative_mapping":
        m[0] = -1
        inputs = [x, m, cu]
    elif mutation == "logical_address_out_of_range":
        x[0, 0] = 2147483647
        inputs = [x, m, cu]
    elif mutation == "nonmonotonic_cu":
        cu[1] = -1
        inputs = [x, m, cu]
    else:
        raise AssertionError(mutation)
    with pytest.raises((TypeError, ValueError)):
        prepare(inputs, attributes)


def test_non_nd_storage_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch_npu

    x, m, cu = _base_inputs()
    monkeypatch.setattr(torch_npu, "get_npu_format", lambda tensor: "FRACTAL_NZ")
    with pytest.raises(ValueError, match="ND storage"):
        prepare([x, m, cu], {})


def test_non_dict_attributes_are_rejected() -> None:
    x, m, cu = _base_inputs()
    with pytest.raises(TypeError):
        prepare([x, m, cu], None)  # type: ignore[arg-type]


def test_run_equals_prepare_and_run() -> None:
    x, m, cu = _base_inputs()
    from solution import run

    torch.testing.assert_close(
        run([x, m, cu], {}),
        prepare([x, m, cu], {}).run(),
        rtol=0,
        atol=0,
    )

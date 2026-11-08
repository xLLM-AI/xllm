# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Tensor-tree validation checks structure, values, mutations and raw aliases."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from xllm_kernel.numerics.tensor_case import Example, compare
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import KernelRegistry, KernelSpec, OpContract, PreparedKernel


def _plan(function: Callable[..., Any], mutates: tuple[str, ...]) -> PreparedKernel:
    contract = OpContract("state", 1, mutates, False, "test")
    registry = KernelRegistry()
    registry.register(KernelSpec("cpu.torch.state", contract, PlatformEnum.CPU, ("eager",), "CPU tensors"), function)
    registry.freeze()
    return registry.prepare(contract, device="cpu", execution_mode="eager")


def test_compare_rejects_wrong_integer_and_metadata() -> None:
    want = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert compare(want.clone(), want, {})["status"] == "passed"
    assert compare(want + 1, want, {})["status"] == "failed"
    assert compare(want.long(), want, {})["status"] == "failed"
    assert compare(want.reshape(1, 3), want, {})["status"] == "failed"


def test_compare_rejects_nan_even_with_tolerance() -> None:
    assert compare(torch.tensor([float("nan")]), torch.ones(1), {"": (1.0, 1.0)})["status"] == "failed"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (None, (None,)),
        ((torch.ones(1),), (torch.ones(1), None)),
        ([torch.ones(1)], (torch.ones(1),)),
        ({"value": torch.ones(1)}, {"value": torch.ones(1), "optional": None}),
        ({}, {"optional": None}),
        ([], ()),
    ],
)
def test_compare_preserves_container_structure(actual: Any, expected: Any) -> None:
    assert compare(actual, expected, {})["status"] == "failed"


def test_compare_accepts_matching_optional_branches_and_reordered_dict_keys() -> None:
    expected = {"state": [torch.ones(1), None], "empty": {}, "tail": ()}
    actual = {"tail": (), "empty": {}, "state": [torch.ones(1), None]}
    assert compare(actual, expected, {})["status"] == "passed"


@pytest.mark.parametrize("expected,difference", [(2**53, 1), (-(2**53), -1), (2**63 - 2, 1)])
def test_compare_rejects_large_integer_differences(expected: int, difference: int) -> None:
    reference = torch.tensor([expected], dtype=torch.int64)
    assert compare(reference.clone(), reference, {})["status"] == "passed"
    assert compare(torch.tensor([expected + difference], dtype=torch.int64), reference, {})["status"] == "failed"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("actual,tolerance,passed", [(1.000001, (0.0, 0.0), False), (1.05, (0.1, 0.0), True)])
def test_example_uses_declared_mutation_tolerances(
    nested: bool, actual: float, tolerance: tuple[float, float], passed: bool
) -> None:
    state = torch.zeros(1)
    expected = torch.ones(1)
    path = "/state/value" if nested else "/state"

    def update(value: torch.Tensor) -> None:
        value.fill_(actual)

    example = Example(
        inputs={"state": {"value": state} if nested else state},
        invoke=lambda function: function(state),
        expected=None,
        expected_mutations={"state": {"value": expected} if nested else expected},
        tolerances={path: tolerance},
    )
    prepared = example.bind(_plan(update, ("state",)))
    report = prepared.validate(prepared.function)
    assert (report["status"] == "passed") is passed
    result = report["mutations"]["outputs"][path]
    assert (result["rtol"], result["atol"]) == tolerance


def test_example_checks_large_integer_mutations_exactly() -> None:
    state = torch.zeros(1, dtype=torch.int64)

    def update(value: torch.Tensor) -> None:
        value.fill_(2**53 + 1)

    example = Example(
        inputs={"state": state},
        invoke=lambda function: function(state),
        expected=None,
        expected_mutations={"state": torch.tensor([2**53], dtype=torch.int64)},
    )
    prepared = example.bind(_plan(update, ("state",)))
    report = prepared.validate(prepared.function)
    assert report["status"] == report["mutations"]["status"] == "failed"


@pytest.mark.parametrize("captured", [False, True])
def test_example_observes_void_kernel_state_without_output_aliasing(captured: bool) -> None:
    state = torch.zeros(1)
    calls = 0

    def update(value: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        value.add_(1)

    example = Example(
        inputs={"state": state},
        invoke=lambda function: function(state),
        expected=torch.ones(1),
        observe=lambda _: state,
        expected_mutations={"state": torch.ones(1)},
    )
    prepared = example.bind(_plan(update, ("state",)))
    if captured:
        output = prepared.function()
        assert output is None
        report = prepared.validate(lambda: output)
    else:
        report = prepared.validate(prepared.function)
    assert report["status"] == "passed"
    assert report["aliases_inputs"] is False
    assert calls == 1


def test_example_observation_cannot_hide_a_kernel_output_alias() -> None:
    state = torch.ones(1)
    example = Example(
        inputs={"state": state},
        invoke=lambda function: function(state),
        expected=state.clone(),
        observe=lambda output: output.clone(),
    )
    prepared = example.bind(_plan(lambda value: value, ()))
    report = prepared.validate(prepared.function)
    assert report["aliases_inputs"] is True
    assert report["status"] == "failed"

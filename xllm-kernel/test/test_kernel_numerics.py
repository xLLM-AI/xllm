# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise numerical-report failures with deterministic CPU tensors."""

import torch
from xllm_kernel.numerics.rms_norm import NumericsCase, reference, verify


def test_reference_and_report_pass() -> None:
    value = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
    weight = torch.ones(4)
    report = verify(reference, NumericsCase(1, 4, "float32"), value, weight)
    assert report["status"] == "passed"
    assert report["max_absolute_error"] == 0


def test_numerical_error_and_nan_are_failures() -> None:
    value = torch.ones(1, 4)
    weight = torch.ones(4)
    for bad_value in (0.0, float("nan"), float("inf")):

        def broken(input: torch.Tensor, weight: torch.Tensor, eps: float, bad_value: float = bad_value) -> torch.Tensor:
            return torch.full_like(input, bad_value)

        report = verify(broken, NumericsCase(1, 4, "float32"), value, weight)
        assert report["status"] == "failed"
        assert report["failing_elements"] == 4


def test_alias_is_rejected_even_when_values_match() -> None:
    value = torch.zeros(1, 4)
    weight = torch.ones(4)
    report = verify(lambda input, *_: input, NumericsCase(1, 4, "float32"), value, weight)
    assert report["status"] == "failed"
    assert report["aliases_inputs"]


def test_mutation_is_reported() -> None:
    value = torch.ones(1, 4)
    weight = torch.ones(4)

    def mutating(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        output = reference(input, weight, eps)
        input.zero_()
        return output

    report = verify(mutating, NumericsCase(1, 4, "float32"), value, weight)
    assert report["status"] == "failed"
    assert report["input_mutation"]

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Numerical comparison rejects metadata errors, nonfinite values and wrong integer writes."""

import torch
from xllm_kernel.numerics.tensor_case import compare


def test_compare_rejects_wrong_integer_and_metadata() -> None:
    want = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert compare(want.clone(), want, {})["status"] == "passed"
    assert compare(want + 1, want, {})["status"] == "failed"
    assert compare(want.long(), want, {})["status"] == "failed"
    assert compare(want.reshape(1, 3), want, {})["status"] == "failed"


def test_compare_rejects_nan_even_with_tolerance() -> None:
    assert compare(torch.tensor([float("nan")]), torch.ones(1), {"": (1.0, 1.0)})["status"] == "failed"

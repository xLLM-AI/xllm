# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch registration contracts for Python kernels and extension-owned schemas."""

import pytest
import torch
from xllm_kernel.custom_ops import register_custom_op


def _add_one(input: torch.Tensor) -> torch.Tensor:
    return input + 1


def test_python_operator_executes_real_kernel() -> None:
    op = register_custom_op("test_add_one", _add_one, mutates_args=(), dispatch_key="CPU")
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(op(value), value + 1)


def test_existing_operator_preserves_real_kernel() -> None:
    op = register_custom_op("test_existing_add", _add_one, mutates_args=(), dispatch_key="CPU")
    assert register_custom_op("test_existing_add") is op
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))
    with pytest.raises(RuntimeError, match="same name and overload name"):
        register_custom_op("test_existing_add", _add_one, mutates_args=(), dispatch_key="CPU")
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))


@pytest.mark.parametrize("metadata", [{"dispatch_key": "CPU"}, {"mutates_args": ()}])
def test_python_operator_requires_explicit_registration_metadata(metadata: dict) -> None:
    with pytest.raises(ValueError, match="explicit mutates_args and dispatch_key"):
        register_custom_op("test_missing_metadata", _add_one, **metadata)


def test_existing_operator_does_not_ignore_registration_metadata() -> None:
    op = register_custom_op("test_existing_metadata", _add_one, mutates_args=(), dispatch_key="CPU")
    with pytest.raises(ValueError, match="retain their registered mutation and dispatch metadata"):
        register_custom_op("test_existing_metadata", mutates_args=("input",))
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))

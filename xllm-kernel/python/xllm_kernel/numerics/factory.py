# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Assemble operation-owned shape matrices for the numerical/benchmark CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from itertools import product

from xllm_kernel.catalog import Operation
from xllm_kernel.numerics.suite import OperationSuite
from xllm_kernel.numerics.tensor_case import Example, TensorCase
from xllm_kernel.registry import PreparedKernel
from xllm_kernel.runtime import prepare_operation


def tensor_suite(
    operator: Operation,
    dimensions: dict[str, tuple[int, ...]],
    build: Callable[[tuple[int, ...], str, int], Example],
    *,
    dtypes: tuple[str, ...] = ("bfloat16", "float16"),
    restrict: Callable[[PreparedKernel, tuple[int, ...], str], str | None] = lambda *_: None,
) -> OperationSuite:
    """Keep dimensions and eligibility in the calling operation, not in shared runners."""

    def arguments(parser: argparse.ArgumentParser) -> None:
        for name, values in dimensions.items():
            parser.add_argument(f"--{name.replace('_', '-')}", type=int, nargs="+", default=list(values))
        parser.add_argument("--dtypes", nargs="+", choices=dtypes, default=list(dtypes))
        parser.add_argument("--seed", type=int, default=5309)

    def cases(args: argparse.Namespace) -> tuple[TensorCase, ...]:
        values = [getattr(args, name) for name in dimensions]
        if any(min(items) <= 0 for items in values):
            raise ValueError("all dimensions must be positive")
        return tuple(
            TensorCase(operator.contract.operation, shape, dtype, build, args.seed, restrict)
            for dtype in args.dtypes
            for shape in product(*values)
        )

    return OperationSuite(
        operator.contract.operation,
        operator.implementations,
        arguments,
        cases,
        partial(prepare_operation, operator.contract.operation),
    )

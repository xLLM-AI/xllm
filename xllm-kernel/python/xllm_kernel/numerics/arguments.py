# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Shared CLI options and dispatch to operation-owned argument definitions."""

import argparse
from pathlib import Path

from xllm_kernel.numerics.catalog import load_suites
from xllm_kernel.numerics.suite import KernelCase, OperationSuite


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, OperationSuite, tuple[KernelCase, ...]]:
    suites = load_suites()
    parser.add_argument(
        "--operation", choices=[suite.operation for suite in suites], help="Inferred from --implementation when omitted"
    )
    parser.add_argument("--device", required=True, choices=["npu"])
    parser.add_argument("--mode", required=True, choices=["eager", "aclgraph"])
    parser.add_argument(
        "--implementation", required=True, choices=[name for suite in suites for name in suite.implementations]
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--revision", required=True, help="Exact source revision or an explicitly labeled dirty snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.epilog = "Use --operation NAME --help for operation-specific options."

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--operation")
    selector.add_argument("--implementation")
    selected, _ = selector.parse_known_args()
    matches = [
        suite
        for suite in suites
        if (selected.operation is None or suite.operation == selected.operation)
        and (selected.implementation is None or selected.implementation in suite.implementations)
    ]
    if not matches:
        parser.error("no operation suite matches the requested operation and implementation")
    if len(matches) == 1:
        matches[0].add_arguments(parser)
    args = parser.parse_args()
    if len(matches) != 1:
        parser.error("select exactly one operation suite")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    suite = matches[0]
    args.operation = suite.operation
    try:
        cases = tuple(suite.cases(args))
    except ValueError as error:
        parser.error(str(error))
    return args, suite, cases

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Run RMSNorm numerical and graph checks without performance measurement."""

import argparse
import time
from pathlib import Path

from xllm_kernel.numerics.verify import generate_cases, validation_session, verify_case, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="RMSNorm numerical checks; requires idle NPUs")
    parser.add_argument("--device", required=True, choices=["npu"])
    parser.add_argument("--mode", required=True, choices=["eager", "aclgraph"])
    parser.add_argument("--implementation", required=True, choices=["npu.xllm_native.rms_norm", "npu.xlite.rms_norm"])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 4096])
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048, 6144])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bfloat16", "float16", "float32"], default=["bfloat16", "float16"]
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--revision", required=True, help="Exact source revision or an explicitly labeled dirty snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.widths) <= 0 or min(args.rows + [args.warmup]) < 0:
        parser.error("widths must be positive; rows/warmup must be nonnegative")

    with validation_session(
        device=args.device,
        mode=args.mode,
        implementation=args.implementation,
        revision=args.revision,
        output=args.output,
        report_writer=write_report,
    ) as (plan, report):
        import torch

        from xllm_kernel.numerics.rms_norm import verify

        report["warmup"] = args.warmup
        for case in generate_cases(args.rows, args.widths, args.dtypes):
            row, value, weight = verify_case(plan, case, report)
            if row["status"] != "passed":
                continue
            function = lambda value=value, weight=weight, eps=case.eps: plan.function(value, weight, eps)
            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                function()
            torch.npu.synchronize()
            row["warmup_s"] = time.perf_counter() - warmup_start
            if args.mode == "aclgraph":
                capture_start = time.perf_counter()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = function()
                graph.replay()
                torch.npu.synchronize()
                row["graph_accuracy"] = verify(lambda *_, output=output: output, case, value, weight)
                row["status"] = row["graph_accuracy"]["status"]
                row["capture_s"] = time.perf_counter() - capture_start
            write_report(args.output, report)

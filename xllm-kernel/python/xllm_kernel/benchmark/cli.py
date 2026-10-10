# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Validate RMSNorm cases before measuring eager or ACLGraph performance."""

import argparse
import time
from pathlib import Path

from xllm_kernel.benchmark.report import write_report
from xllm_kernel.numerics.verify import generate_cases, validation_session, verify_case


def main() -> None:
    parser = argparse.ArgumentParser(description="RMSNorm validation and NPU timing; requires idle NPUs")
    parser.add_argument("--device", required=True, choices=["npu"])
    parser.add_argument("--mode", required=True, choices=["eager", "aclgraph"])
    parser.add_argument("--implementation", required=True, choices=["npu.xllm_native.rms_norm", "npu.xlite.rms_norm"])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 4096])
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048, 6144])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bfloat16", "float16", "float32"], default=["bfloat16", "float16"]
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--revision", required=True, help="Exact source revision or an explicitly labeled dirty snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.widths + [args.iterations, args.repeats]) <= 0 or min(args.rows + [args.warmup]) < 0:
        parser.error("widths/iterations/repeats must be positive; rows/warmup must be nonnegative")

    with validation_session(
        device=args.device,
        mode=args.mode,
        implementation=args.implementation,
        revision=args.revision,
        output=args.output,
        report_writer=write_report,
    ) as (plan, report):
        import torch

        from xllm_kernel.benchmark.npu import measure
        from xllm_kernel.numerics.rms_norm import verify

        report.update(
            warmup=args.warmup,
            cache_policy="hot_repeated_inputs_no_explicit_flush",
            note="Device-event intervals include host submission gaps; graph replay amortizes launch overhead.",
            measurement_target="prepared_callable",
        )
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
            divisor = 1
            if args.mode == "aclgraph":
                capture_start = time.perf_counter()
                graph = torch.npu.NPUGraph()
                divisor = 64
                with torch.npu.graph(graph):
                    outputs = [function() for _ in range(divisor)]
                graph.replay()
                torch.npu.synchronize()
                row["graph_accuracy"] = verify(lambda *_, output=outputs[-1]: output, case, value, weight)
                row["status"] = row["graph_accuracy"]["status"]
                row["capture_s"] = time.perf_counter() - capture_start
                function = graph.replay
            if row["status"] == "passed":
                row["timing"] = measure(function, iterations=args.iterations, repeats=args.repeats, divisor=divisor)
            write_report(args.output, report)

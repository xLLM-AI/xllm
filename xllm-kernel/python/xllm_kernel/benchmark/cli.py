# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Validate operation-owned cases before eager or ACLGraph measurement."""

import argparse
import time

from xllm_kernel.benchmark.report import write_report
from xllm_kernel.numerics.arguments import parse_arguments
from xllm_kernel.numerics.distributed import rank_max_samples
from xllm_kernel.numerics.verify import validation_session, verify_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel validation and NPU timing; requires idle NPUs")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=11)
    args, suite, cases = parse_arguments(parser)
    if min(args.iterations, args.repeats) <= 0:
        parser.error("iterations/repeats must be positive")

    with validation_session(
        suite=suite,
        device=args.device,
        mode=args.mode,
        implementation=args.implementation,
        revision=args.revision,
        output=args.output,
        report_writer=write_report,
        distributed=args.distributed,
    ) as (plan, report):
        import torch

        from xllm_kernel.benchmark.npu import measure, measure_graph

        report.update(
            warmup=args.warmup,
            cache_policy="hot_repeated_inputs_no_explicit_flush",
            note="Device-event intervals include host submission gaps; graph replay amortizes launch overhead.",
            measurement_target="prepared_callable",
        )
        for case in cases:
            row, prepared = verify_case(plan, case, report)
            if row["status"] != "passed" or prepared is None:
                continue
            function = prepared.function
            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                if prepared.reset is not None:
                    prepared.reset()
                function()
            torch.npu.synchronize()
            row["warmup_s"] = time.perf_counter() - warmup_start
            if args.mode == "aclgraph":
                measure_graph(prepared, row, iterations=args.iterations, repeats=args.repeats)
            else:
                row["timing"] = measure(
                    function, iterations=args.iterations, repeats=args.repeats, divisor=1, reset=prepared.reset
                )
            if row["status"] == "passed":
                row["timing"]["rank_max_samples"] = rank_max_samples(row["timing"])
            write_report(args.output, report)

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Run operation-owned numerical and graph checks without timing."""

import argparse
import time

from xllm_kernel.numerics.arguments import parse_arguments
from xllm_kernel.numerics.report import write_report
from xllm_kernel.numerics.verify import validation_session, verify_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel numerical checks; requires idle NPUs")
    args, suite, cases = parse_arguments(parser)

    with validation_session(
        suite=suite,
        device=args.device,
        mode=args.mode,
        implementation=args.implementation,
        revision=args.revision,
        output=args.output,
        report_writer=write_report,
    ) as (plan, report):
        import torch

        report["warmup"] = args.warmup
        for case in cases:
            row, prepared = verify_case(plan, case, report)
            if row["status"] != "passed" or prepared is None:
                continue
            function = prepared.function
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
                row["graph_accuracy"] = prepared.validate(lambda output=output: output)
                row["status"] = row["graph_accuracy"]["status"]
                row["capture_s"] = time.perf_counter() - capture_start
            write_report(args.output, report)

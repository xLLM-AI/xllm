# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Run operation-owned numerical and graph checks without timing."""

import argparse
import time

from xllm_kernel.numerics.arguments import parse_arguments
from xllm_kernel.numerics.distributed import agree, rank_max_samples
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
        distributed=args.distributed,
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
                if prepared.reset is not None:
                    prepared.reset()
                function()
            torch.npu.synchronize()
            row["warmup_s"] = time.perf_counter() - warmup_start
            if args.mode == "aclgraph":
                capture_start = time.perf_counter()
                graph = torch.npu.NPUGraph()
                if prepared.reset is not None:
                    prepared.reset()
                with torch.npu.graph(graph):
                    output = function()
                if prepared.reset is not None:
                    prepared.reset()
                graph.replay()
                torch.npu.synchronize()
                row["graph_accuracy"] = prepared.validate(lambda output=output: output)
                row["status"] = "passed" if agree(row["graph_accuracy"]["status"] == "passed") else "failed"
                row["capture_s"] = time.perf_counter() - capture_start
            write_report(args.output, report)

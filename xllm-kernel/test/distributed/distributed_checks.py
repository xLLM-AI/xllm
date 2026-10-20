# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""All-rank eager and graph accuracy checks with failure consensus."""

from __future__ import annotations

import argparse

from xllm_kernel.numerics.distributed import agree
from xllm_kernel.numerics.suite import OperationSuite


def check_collective(suite: OperationSuite, implementation: str, mode: str) -> None:
    import torch

    parser = argparse.ArgumentParser()
    suite.add_arguments(parser)
    args = parser.parse_args([])
    args.rows = [1, 17]
    plan = suite.prepare("npu", mode, implementation)
    for case in suite.cases(args):
        prepared = case.prepare(plan)
        if prepared.reset is not None:
            prepared.reset()
        assert agree(prepared.validate(prepared.function)["status"] == "passed"), case.case_id
        if mode == "aclgraph":
            if prepared.reset is not None:
                prepared.reset()
            prepared.function()
            torch.npu.synchronize()
            if prepared.reset is not None:
                prepared.reset()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = prepared.function()
            for _ in range(2):
                if prepared.reset is not None:
                    prepared.reset()
                graph.replay()
                torch.npu.synchronize()
                assert agree(prepared.validate(lambda output=output: output)["status"] == "passed"), case.case_id

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Shared device test lifecycle; suites own shapes, references and tolerances."""

import argparse

import torch
from xllm_kernel.numerics.suite import OperationSuite


def check_suite(suite: OperationSuite, implementation: str, mode: str) -> None:
    """Validate eager output, fresh graph inputs, and repeated replay."""
    parser = argparse.ArgumentParser()
    suite.add_arguments(parser)
    arguments = parser.parse_args([])
    arguments.rows = [1, 17]
    plan = suite.prepare("npu", mode, implementation)
    checked = 0
    for case in suite.cases(arguments):
        if case.unsupported_reason(plan) is not None:
            continue
        checked += 1
        prepared = case.prepare(plan)
        if prepared.reset is not None:
            prepared.reset()
        assert prepared.validate(prepared.function)["status"] == "passed", case.case_id
        if plan.spec.solution == "xlite":
            from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
            from torch.utils._pytree import tree_flatten

            actual, structure = tree_flatten(prepared.function())
            with FakeTensorMode(allow_non_fake_inputs=True):
                fake, fake_structure = tree_flatten(prepared.function())
            assert structure == fake_structure
            assert all(isinstance(item, FakeTensor) for item in fake)
            assert [(item.shape, item.dtype, item.device) for item in actual] == [
                (item.shape, item.dtype, item.device) for item in fake
            ]
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
                assert prepared.validate(lambda output=output: output)["status"] == "passed", case.case_id

    assert checked > 0, "no supported cases exercised this backend"

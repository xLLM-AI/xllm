# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free catalog and public backend declaration invariants."""

from importlib import import_module
from importlib.util import find_spec

import pytest
from xllm_kernel.numerics.catalog import load_suites
from xllm_kernel.numerics.suite import OperationSuite


@pytest.mark.parametrize("suite", load_suites(), ids=lambda suite: suite.operation)
def test_operator_case_matrix_is_nonempty(suite: OperationSuite) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    suite.add_arguments(parser)
    cases = suite.cases(parser.parse_args([]))
    assert cases
    assert all(case.case_id for case in cases)
    assert len(suite.implementations) == len(set(suite.implementations))


@pytest.mark.parametrize("suite", load_suites(), ids=lambda suite: suite.operation)
def test_backend_metadata_and_loader_modules(suite: OperationSuite) -> None:
    module = import_module(f"xllm_kernel.ops.{suite.operation}")
    operator = module.OPERATION
    assert operator.contract.operation == suite.operation
    assert operator.implementations == suite.implementations
    for backend in operator.backends:
        assert find_spec(backend.module) is not None
        assert backend.device == "npu"
        assert backend.requires
        assert backend.implementations

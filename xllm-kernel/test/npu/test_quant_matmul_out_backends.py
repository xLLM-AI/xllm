# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Independent quant_matmul_out backend correctness and graph replay checks."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.quant_matmul_out import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_quant_matmul_out_backends(implementation: str, mode: str) -> None:
    check_suite(SUITE, implementation, mode)

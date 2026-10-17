# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Independent silu_and_mul backend correctness and graph replay checks."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.silu_and_mul import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_silu_and_mul_backends(implementation: str, mode: str) -> None:
    check_suite(SUITE, implementation, mode)

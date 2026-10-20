# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Independent inplace_partial_rotary_mul backend correctness and graph replay checks."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.inplace_partial_rotary_mul import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_inplace_partial_rotary_mul_backends(implementation: str, mode: str) -> None:
    check_suite(SUITE, implementation, mode)

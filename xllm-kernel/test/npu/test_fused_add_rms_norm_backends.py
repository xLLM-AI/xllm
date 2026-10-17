# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Independent fused_add_rms_norm backend correctness and graph replay checks."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.fused_add_rms_norm import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_fused_add_rms_norm_backends(implementation: str, mode: str) -> None:
    check_suite(SUITE, implementation, mode)

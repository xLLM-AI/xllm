# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Independent moe_token_dispatch backend correctness and graph replay checks."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.moe_token_dispatch import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_moe_token_dispatch_backends(implementation: str, mode: str) -> None:
    check_suite(SUITE, implementation, mode)

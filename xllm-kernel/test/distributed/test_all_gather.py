# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Run under torchrun; no single-rank or CPU substitute."""

import pytest
from distributed_checks import check_collective
from xllm_kernel.numerics.all_gather import SUITE


@pytest.mark.parametrize("implementation", SUITE.implementations)
@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_all_gather(implementation: str, mode: str) -> None:
    if ".cann." in implementation and mode == "aclgraph":
        pytest.skip("ProcessGroup provider is explicitly eager-only")
    check_collective(SUITE, implementation, mode)

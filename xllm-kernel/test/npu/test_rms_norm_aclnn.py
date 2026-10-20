# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""ACLNN binding variants use the same independent RMSNorm reference."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.rms_norm import SUITE


@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
@pytest.mark.parametrize("variant", ("xllm", "torch_npu"))
def test_aclnn_rms_norm(mode: str, variant: str) -> None:
    check_suite(SUITE, f"npu.aclnn.rms_norm.{variant}", mode)

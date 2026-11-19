# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the migrated NPU operators directly from the kernel package."""

import pytest

pytest.importorskip("torch_npu", reason="NPU integration tests require torch_npu")

from xllm_kernel import initialize


@pytest.fixture(scope="session", autouse=True)
def _prepare_kernels() -> None:
    initialize(
        device="npu",
        operations=(
            "l2_norm",
            "fused_add_gemma_rms_norm",
            "rms_norm_gated",
            "rms_norm_dynamic_quant",
            "fused_recurrent_gated_delta_rule",
            "fused_sigmoid_gating_delta_rule_update",
            "fused_qkvzba_split_reshape_cat",
            "causal_conv1d_update_v2",
            "matmul_ein_sum",
            "matmul_ein_sum_out",
        ),
    )

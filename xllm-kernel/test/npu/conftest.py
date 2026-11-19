# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the migrated NPU operators directly from the kernel package."""

import pytest

pytest.importorskip("torch_npu", reason="NPU integration tests require torch_npu")

from xllm_kernel import initialize
from xllm_kernel.ops.attention.mla import has_mla_preprocess_v2
from xllm_kernel.ops.moe import FUSED_MOE_GMM1_OPERATIONS, has_fused_moe_gmm1


@pytest.fixture
def require_fused_moe_gmm1() -> None:
    if not has_fused_moe_gmm1():
        pytest.skip("Routing V3 and fused GMM1 are unavailable")


@pytest.fixture(scope="session", autouse=True)
def _prepare_kernels() -> None:
    initialize(
        device="npu",
        operations=(
            "l2_norm",
            "fused_add_gemma_rms_norm",
            "rms_norm_gated",
            "inplace_partial_rotary_mul",
            "rms_norm_dynamic_quant",
            "hc_pre",
            "hc_post",
            "fused_recurrent_gated_delta_rule",
            "fused_sigmoid_gating_delta_rule_update",
            "fused_qkvzba_split_reshape_cat",
            "causal_conv1d",
            "causal_conv1d_qkv_prefill",
            "chunk_gated_delta_rule",
            "causal_conv1d_update_v2",
            "matmul_ein_sum",
            "matmul_ein_sum_out",
            "quant_matmul_out",
            *(("mla_preprocess_v2",) if has_mla_preprocess_v2() else ()),
            "dequant_swiglu_quant",
            "moe_gating_top_k_hash",
            *(FUSED_MOE_GMM1_OPERATIONS if has_fused_moe_gmm1() else ()),
        ),
    )

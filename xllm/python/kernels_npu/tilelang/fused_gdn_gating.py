# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Source-tree reference checks and AOT settings for GDN gating."""

import tilelang
from xllm_kernel.ops.attention.gdn._tilelang._fused_gdn_gating import (
    GDN_GATING_PASS_CONFIGS,
    SUPPORTED_NUM_HEADS,
    build_fused_gdn_gating_kernel,
)

from scripts.logger import logger

from .utils import detect_vec_core_num

DEFAULT_NUM_HEADS = 32
DEFAULT_DTYPE = "bf16"
DEFAULT_MAX_BATCH = 262144
DEFAULT_MAX_HEADS = 128
REF_CHECK_NUM_BATCHES = 16
REF_CHECK_NUM_HEADS = (1, 16, 32, 48, 64, 128)
MAX_VEC_CORE_NUM = detect_vec_core_num()
BATCH_SIZE_SPECIALIZATIONS = tuple(range(2, 49, 2))


@tilelang.jit(pass_configs=GDN_GATING_PASS_CONFIGS)
def fused_gdn_gating_kernel_jit(
    num_batches: int,
    compile_max_batch: int,
    num_heads: int,
):
    return build_fused_gdn_gating_kernel(
        batch_size=num_batches,
        compile_max_batch=compile_max_batch,
        num_heads=num_heads,
        vec_core_num=MAX_VEC_CORE_NUM,
    )


def _torch_fused_gdn_gating(
    A_log: "torch.Tensor",
    a: "torch.Tensor",
    b: "torch.Tensor",
    dt_bias: "torch.Tensor",
    softplus_beta: float,
    softplus_threshold: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch
    import torch.nn.functional as F

    softplus_out = F.softplus(
        a.to(torch.float32) + dt_bias,
        beta=softplus_beta,
        threshold=softplus_threshold,
    )
    g_ref = -A_log.exp() * softplus_out
    beta_ref = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
    return g_ref, beta_ref


def _run_ref_check(
    *,
    num_batches: int,
    num_heads: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning("Skip fused_gdn_gating reference check: NPU is not available")
        return

    if num_batches <= 0:
        raise ValueError(f"num_batches({num_batches}) must be > 0")
    if num_batches > compile_max_batch:
        raise ValueError(f"num_batches({num_batches}) must be <= compile_max_batch({compile_max_batch})")

    torch.manual_seed(42)
    device = torch.device("npu")

    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((num_batches, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty((num_batches, num_heads), device=device, dtype=torch.bfloat16)

    kernel = fused_gdn_gating_kernel_jit(
        num_batches=num_batches,
        compile_max_batch=num_batches,
        num_heads=num_heads,
    )
    kernel(
        A_log,
        a,
        b,
        dt_bias,
        g_out,
        beta_out,
        num_batches,
        softplus_beta,
        softplus_threshold,
    )
    torch.npu.synchronize()

    g_ref, beta_ref = _torch_fused_gdn_gating(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    torch.testing.assert_close(g_out, g_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        beta_out.to(torch.float32),
        beta_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    logger.info("fused_gdn_gating output matches torch reference for num_heads=%s", num_heads)


def _run_ref_suite(
    *,
    num_batches: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
    ref_num_heads_list: list[int],
) -> None:
    for num_heads in ref_num_heads_list:
        _run_ref_check(
            num_batches=num_batches,
            num_heads=num_heads,
            compile_max_batch=compile_max_batch,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )

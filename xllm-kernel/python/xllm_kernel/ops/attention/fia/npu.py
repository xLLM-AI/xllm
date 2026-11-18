# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind TorchNPU FIA operators at worker startup."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

from xllm_kernel.ops.attention.fia import FUSED_ATTENTION_CONTRACT, FusedAttention
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

_Result = TypeVar("_Result")
_SPLIT_ROPE_ARGUMENTS = {
    "actual_seq_lengths": "actual_seq_qlen",
    "actual_seq_lengths_kv": "actual_seq_kvlen",
    "num_heads": "num_query_heads",
    "scale": "softmax_scale",
    "softmax_lse_flag": "return_softmax_lse",
}


def _split_rope_adapter(operator: Callable[..., _Result]) -> Callable[..., _Result]:
    def call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **options: object) -> _Result:
        return operator(
            query, key, value, **{_SPLIT_ROPE_ARGUMENTS.get(name, name): value for name, value in options.items()}
        )

    return call


if current_platform.is_npu():
    import torch_npu

    _fia = torch.ops.npu.npu_fused_infer_attention_score
    _split_rope = None
    if hasattr(torch.ops.npu, "npu_fused_infer_attention_score_v2") and hasattr(
        torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace"
    ):
        _fia_v2 = torch.ops.npu.npu_fused_infer_attention_score_v2
        _split_rope = FusedAttention(
            run=_split_rope_adapter(_fia_v2.default),
            out=_split_rope_adapter(_fia_v2.out),
            workspace=_split_rope_adapter(torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace),
        )

    register_kernel(
        name="npu.torch_npu.fused_attention",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Dense, paged and split-RoPE FIA with TND/BNSD tensors and optional caller-owned results.",
        solution="torch_npu",
        binding="torch_npu",
        contract=FUSED_ATTENTION_CONTRACT,
    )(
        FusedAttention(
            run=_fia.default,
            out=_fia.out,
            workspace=torch_npu._npu_fused_infer_attention_score_get_max_workspace,
            with_separate_rope=_split_rope,
        )
    )

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepared fused attention and its graph execution contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

FUSED_ATTENTION_CONTRACT = OpContract("fused_attention", 1, ("workspace", "out"), True, "fused_infer_attention")
FUSED_ATTENTION_OPERATION = Operation(
    FUSED_ATTENTION_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.attention.fia.npu",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch_npu.fused_attention",),
        ),
    ),
)
OPERATIONS = (FUSED_ATTENTION_OPERATION,)


@dataclass(frozen=True, slots=True)
class FusedAttention:
    """One fixed attention binding for eager execution and ACLGraph.

    Q uses packed TND or batched BNSD layout. Paged K/V use the caller's
    flattened page views and int32 block table. Sequence lengths are host
    lists or device tensors as supported by the selected layout. Split-RoPE
    bindings accept separate query/key rotary tensors in TND layout.

    The functional form returns attention and optional softmax LSE. The out
    form writes the caller-owned output and LSE and may use mutable scratch.
    The workspace query allocates scratch for the supplied descriptors. Capture
    and task updates use the same out buffers, workspace and block table.
    """

    run: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    out: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    workspace: Callable[..., torch.Tensor]
    with_separate_rope: FusedAttention | None = None

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **options: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.run(query, key, value, **options)


def get_fused_attention() -> FusedAttention:
    """Get the startup binding and its optional split-RoPE capability."""
    return cast(FusedAttention, FUSED_ATTENTION_OPERATION.get_prepared().function)

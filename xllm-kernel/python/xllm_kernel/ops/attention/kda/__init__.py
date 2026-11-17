# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free KDA convolution contract and startup-bound semantic API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

CAUSAL_CONV1D_UPDATE_V2_CONTRACT = OpContract(
    "causal_conv1d_update_v2", 1, ("conv_state",), False, "fp16_history_fp32_accumulation_bf16_output"
)
CAUSAL_CONV1D_UPDATE_V2_OPERATION = Operation(
    CAUSAL_CONV1D_UPDATE_V2_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.attention.kda.triton",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "triton"),
            implementations=("npu.triton.causal_conv1d_update_v2",),
        ),
    ),
)
OPERATIONS = (CAUSAL_CONV1D_UPDATE_V2_OPERATION,)


def causal_conv1d_update_v2(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = 1,
    num_accepted_tokens: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: bool = True,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Convolve BF16 tokens and mutate framework [slot, state, dim] storage.

    ``weight`` is BF16 [dim, width], with width in [1, 6]; bias is optional
    BF16 [dim]. Arbitrary input strides and state views are supported. A
    contiguous state pool is written directly; a strided view is staged and
    copied back through that same view, preserving its backing allocation.

    Packed input is [tokens, dim] with int32/int64 cumulative lengths
    [sequences + 1] and a positive ``max_query_len`` bounding every sequence.
    Without lengths, dense input is [B, dim] or [B, dim, T]. As in the native
    V2 bridge, output is always fresh contiguous BF16 [total_tokens, dim],
    flattened in batch/token order. Output does not alias any input.

    Int32/int64 state indices [sequences] select distinct mutable rows;
    omitted indices select consecutive rows. A two-dimensional slot table
    [sequences, >=2] selects column zero, matching the bridge without APC
    metadata. The host owns valid lengths, slots and nonoverlapping rows.
    Empty sequences and ``pad_slot_id`` leave state intact; padded output
    tokens are unspecified. Slot zero remains valid with the default -1 pad.

    Ordinary convolution keeps width - 1 history rows. With accepted counts
    [sequences], read history starting at accepted - 1, then retain width - 1
    + sequence_length - 1 rows for speculative verification. Counts are in
    [1, max_query_len]; the pool needs width - 1 + max_query_len - 1 rows.
    This preserves the V2 checkpoint convention, including untouched rows.
    FP16 history/token intermediates, FP32 accumulation, BF16 rounding and
    optional SiLU match the legacy implementation. Eager and ACLGraph use
    one generic kernel compiled and loaded during startup preparation.
    """
    return CAUSAL_CONV1D_UPDATE_V2_OPERATION.get_prepared().function(
        value,
        weight,
        conv_state,
        state_indices,
        query_start_loc,
        max_query_len,
        num_accepted_tokens,
        bias,
        activation,
        pad_slot_id,
    )

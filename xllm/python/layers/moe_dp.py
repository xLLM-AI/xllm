# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-parallel token gather/scatter shared by the sparse-MoE forward paths.

Every routed-MoE forward (dense ``FusedMoE``, NPU Qwen3.5, DeepSeek family) runs
the experts over the whole DP group's tokens, then slices the result back to the
rank-local rows. That gather prologue and scatter epilogue are identical across
backends; only the expert compute between them differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xllm.python import distributed
from xllm.python.model_executor.forward_context import get_forward_context


@dataclass(frozen=True, slots=True)
class DpScatterState:
    """Slices a DP-gathered MoE output back to this rank's local tokens."""

    output_offset: int
    local_tokens: int
    enabled: bool

    def scatter(self, output: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return output.narrow(0, self.output_offset, self.local_tokens)
        return output


# Shared no-op state for the single-DP path: its scatter returns the output
# unchanged, so one frozen instance is reused instead of allocating per forward.
_NO_SCATTER = DpScatterState(0, 0, False)


def dp_gather_tokens(
    hidden: torch.Tensor,
    dp_size: int,
    dp_rank: int,
) -> tuple[torch.Tensor, DpScatterState]:
    """All-gather this rank's tokens across the DP group for expert compute.

    Returns the gathered hidden states and the :class:`DpScatterState` that
    slices the computed output back to the local execution rows. Metadata
    already accounts for the dummy row materialized by an empty DP rank. The
    collective is fixed exactly when every rank executes the same row count;
    otherwise it is variable. ``dp_size <= 1`` is a no-op whose ``scatter``
    returns the output unchanged.
    """
    if dp_size <= 1:
        return hidden, _NO_SCATTER
    ctx = get_forward_context()
    execution_token_counts = tuple(ctx.metadata.dp_execution_token_counts)
    if len(execution_token_counts) != dp_size:
        raise RuntimeError(f"expected {dp_size} DP execution token counts, got {execution_token_counts}")
    local_tokens = hidden.shape[0]
    gathered, output_offset = distributed.gather_dp_execution_tokens(
        hidden,
        execution_token_counts,
        dp_rank,
    )
    return gathered, DpScatterState(output_offset, local_tokens, True)


def reduce_and_scatter(
    output: torch.Tensor,
    scatter_state: DpScatterState,
    *,
    reduce_results: bool,
    moe_tp_size: int,
    ep_size: int,
) -> torch.Tensor:
    """Reduce a routed-MoE output across its parallel axes, then slice back to local tokens.

    The reduce-then-scatter epilogue shared by the routed-MoE forwards (dense
    ``FusedMoE``, NPU Qwen3.5 experts): when ``reduce_results``, TP-reduce across the
    MoE tensor-parallel group and EP-reduce across the expert-parallel group, then
    slice the DP-gathered rows back to this rank via ``scatter_state``.
    """
    if reduce_results:
        if moe_tp_size > 1:
            distributed.moe_tp_all_reduce(output)
        if ep_size > 1:
            distributed.moe_ep_all_reduce(output)
    return scatter_state.scatter(output)

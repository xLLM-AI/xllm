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

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from xllm.python import distributed
from xllm.python.model_executor.forward_context import get_forward_context


@dataclass(frozen=True, slots=True)
class DpScatterState:
    """Slices a DP-gathered MoE output back to this rank's local tokens."""

    dp_rank: int
    local_tokens: int
    padded_tokens: int
    token_counts: Sequence[int]
    compact: bool

    def scatter(self, output: torch.Tensor) -> torch.Tensor:
        if self.compact:
            offset = sum(self.token_counts[: self.dp_rank])
            return output.narrow(0, offset, self.local_tokens)
        if self.padded_tokens > 0:
            start = self.dp_rank * self.padded_tokens
            return output.narrow(0, start, self.local_tokens)
        return output


# Shared no-op state for the single-DP path: its scatter returns the output
# unchanged, so one frozen instance is reused instead of allocating per forward.
_NO_SCATTER = DpScatterState(0, 0, 0, (), False)


def dp_gather_tokens(hidden: torch.Tensor, dp_size: int, dp_rank: int) -> tuple[torch.Tensor, DpScatterState]:
    """All-gather this rank's tokens across the DP group for expert compute.

    Returns the gathered hidden states and the :class:`DpScatterState` that slices
    the computed output back to the local rows. Graph, prefill, or mixed
    prefill/decode batches pad to the max count and gather a dense ``[dp_size * pad]``
    tensor; an all-decode eager batch gathers the compact variable-length layout.
    ``dp_size <= 1`` is a no-op whose ``scatter`` returns the output unchanged.
    """
    if dp_size <= 1:
        return hidden, _NO_SCATTER
    ctx = get_forward_context()
    token_counts = list(ctx.metadata.dp_token_counts)
    if len(token_counts) != dp_size:
        raise RuntimeError(f"expected {dp_size} DP token counts, got {token_counts}")
    local_tokens = hidden.shape[0]
    is_graph = ctx.execution_state is not None
    is_prefill = ctx.metadata.is_prefill or ctx.metadata.is_chunked_prefill
    dp_is_decode = getattr(ctx.metadata, "dp_is_decode", None)
    all_decode = dp_is_decode is not None and all(dp_is_decode)
    if is_graph or is_prefill or not all_decode:
        padded_tokens = max(token_counts)
        pad_size = padded_tokens - local_tokens
        if pad_size > 0:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_size))
        gathered = distributed.all_gather(hidden, dim=0, world_size=dp_size, group_name="dp")
        return gathered, DpScatterState(dp_rank, local_tokens, padded_tokens, token_counts, False)
    gathered = distributed.all_gather_variable(hidden, token_counts, dp_rank, "dp")
    return gathered, DpScatterState(dp_rank, local_tokens, 0, token_counts, True)


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

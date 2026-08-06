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

"""Graph interfaces for mixture-of-experts operators."""

from __future__ import annotations

import torch


@torch.library.custom_op("xllm_triton::fused_moe", mutates_args=())
def fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Run unquantized CUDA Triton experts as one graph node."""
    from xllm.python.kernels.triton.cuda.fused_moe import (
        fused_moe as triton_fused_moe,
    )

    return triton_fused_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w2,
    )


@fused_moe.register_fake
def _fused_moe_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    del topk_ids, topk_weights, w13, w2
    return torch.empty_like(hidden_states)

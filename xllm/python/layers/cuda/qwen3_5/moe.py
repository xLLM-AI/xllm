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

"""CUDA-owned Qwen3.5 sparse MoE composition."""

from __future__ import annotations

import torch

from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.qwen3_5_common import Qwen3_5MoEConfig
from xllm.python.layers.qwen3_5_moe import Qwen3_5SparseMoEBlockBase


class CudaQwen3_5SparseMoEBlock(Qwen3_5SparseMoEBlockBase):
    """CUDA expert graph with CUTLASS/Triton-native weight ordering."""

    def __init__(
        self,
        cfg: Qwen3_5MoEConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, dtype, device)
        self.experts = FusedMoE(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            renormalize=cfg.norm_topk_prob,
            moe_tp_size=cfg.moe_tp_size,
            moe_tp_rank=cfg.moe_tp_rank,
            ep_size=cfg.ep_size,
            ep_rank=cfg.ep_rank,
            dp_size=cfg.dp_size,
            dp_rank=cfg.dp_rank,
            dtype=dtype,
            device=device,
            reduce_results=not self.fuse_reductions,
        )

    def _pack_gate_up(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((up, gate), dim=1)

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

"""Parallel topology used while materializing checkpoint weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _ParallelTopologyConfig(Protocol):
    """Config contract for the topology :meth:`ParallelLoadContext.from_config` reads."""

    dp_rank: int
    dp_size: int
    moe_tp_rank: int
    moe_tp_size: int
    ep_rank: int
    ep_size: int


@dataclass(frozen=True, slots=True)
class ParallelLoadContext:
    """Rank-local parallel topology for model-owned weight loading."""

    tp_rank: int
    tp_size: int
    dp_rank: int = 0
    dp_size: int = 1
    moe_tp_rank: int = 0
    moe_tp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1

    @classmethod
    def from_config(
        cls,
        cfg: _ParallelTopologyConfig,
        tp_rank: int,
        tp_size: int,
    ) -> ParallelLoadContext:
        """Build from a model config: ``tp_rank``/``tp_size`` from the caller,
        DP/MoE-TP/EP from ``cfg``."""
        return cls(
            tp_rank=tp_rank,
            tp_size=tp_size,
            dp_rank=cfg.dp_rank,
            dp_size=cfg.dp_size,
            moe_tp_rank=cfg.moe_tp_rank,
            moe_tp_size=cfg.moe_tp_size,
            ep_rank=cfg.ep_rank,
            ep_size=cfg.ep_size,
        )

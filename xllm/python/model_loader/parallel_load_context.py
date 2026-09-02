# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel topology used while materializing checkpoint weights."""

from __future__ import annotations

from dataclasses import dataclass


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

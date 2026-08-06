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

"""Distributed execution for the Python model executor."""

from __future__ import annotations

from xllm.python.distributed.collectives import (
    all_gather,
    all_reduce_,
    init_tp_group,
    tp_rank,
)

__all__ = ["init_tp_group", "tp_rank", "all_reduce_", "all_gather"]

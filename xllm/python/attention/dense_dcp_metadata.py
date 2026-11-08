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

"""Execution inputs shared by dense DCP attention layers."""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DenseDcpMetadata:
    """Layer-shared attention inputs, independent of eager/graph execution.

    Graph entries own their instances. The builder updates host lists in place;
    captured FIA tasks read those lists again during task update. The execution
    adapter binds the runner's existing persistent block-table view before capture.
    """

    local_slot_mapping: torch.Tensor
    query_seq_ends: list[int]
    local_kv_lengths: list[int]
    empty_kv_shards: torch.Tensor
    block_table: torch.Tensor | None
    is_prefill: bool
    is_chunked_prefill: bool
    has_context: bool
    has_local_context: bool

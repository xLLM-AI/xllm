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

"""Forward-scoped inputs for NPU Qwen3.5 token-owner MegaMoe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from xllm.python.layers.npu.mega_moe_metadata import MEGA_MOE_MAX_TOKENS, MegaMoeMetadata
from xllm.python.layers.qwen3_5.common import Qwen3_5MoEConfig

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache


class TokenOwnerMegaMoeMetadataBuilder:
    """Build eager inputs or views of the graph runner's persistent token mask."""

    metadata_type: type[object] = MegaMoeMetadata

    def __init__(self, cfg: Qwen3_5MoEConfig) -> None:
        self._is_token_owner = cfg.tp_rank == 0
        self._dp_size = cfg.dp_size
        self._dp_rank = cfg.dp_rank
        self._hidden_size = cfg.hidden_size
        self._top_k = cfg.num_experts_per_tok
        self._token_limit = cfg.mega_moe_num_max_tokens_per_rank

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        del layer_caches

    def build(self, metadata: AttentionMetadata) -> MegaMoeMetadata:
        local_token_count = int(metadata.slot_mapping.numel())
        execution_counts = tuple(int(count) for count in metadata.dp_execution_token_counts)
        if not execution_counts and self._dp_size == 1:
            execution_counts = (local_token_count,)
        if len(execution_counts) != self._dp_size or any(count <= 0 for count in execution_counts):
            raise RuntimeError(f"MegaMoe requires positive DP execution token counts, got {execution_counts}")
        if execution_counts[self._dp_rank] != local_token_count:
            raise RuntimeError(
                "DP execution token count does not match the local input: "
                f"rank={self._dp_rank}, rows={local_token_count}, token_counts={execution_counts}"
            )
        token_capacity = max(execution_counts)
        device = metadata.slot_mapping.device
        graph_mask = getattr(metadata, "mega_moe_token_mask", None)
        if graph_mask is not None and (
            graph_mask.shape != (self._dp_size * token_capacity,)
            or graph_mask.dtype != torch.int8
            or graph_mask.device != device
        ):
            raise RuntimeError("MegaMoe graph token mask has an invalid shape, dtype or device")

        if token_capacity > min(MEGA_MOE_MAX_TOKENS, self._token_limit):
            return MegaMoeMetadata(None)

        if self._is_token_owner and graph_mask is not None:
            # The runner refreshes this fixed storage before graph replay.
            active_token_mask = graph_mask.narrow(0, self._dp_rank * token_capacity, token_capacity)
        else:
            active_token_mask = torch.zeros(token_capacity, dtype=torch.int8, device=device)
            if self._is_token_owner:
                active_token_mask[:local_token_count].fill_(1)
            else:
                # Every non-owner TP rank participates with one immutable dummy.
                active_token_mask[0] = 1

        if self._is_token_owner:
            return MegaMoeMetadata(active_token_mask)

        return MegaMoeMetadata(
            active_token_mask=active_token_mask,
            dummy_input=torch.zeros(token_capacity, self._hidden_size, dtype=torch.bfloat16, device=device),
            dummy_topk_weights=torch.full(
                (token_capacity, self._top_k), 1.0 / self._top_k, dtype=torch.float32, device=device
            ),
            dummy_topk_ids=(
                torch.arange(self._top_k, dtype=torch.int32, device=device)
                .expand(token_capacity, self._top_k)
                .contiguous()
            ),
        )

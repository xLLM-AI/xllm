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

"""Per-forward token layout for FlashComm1 sequence parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.model_executor.forward_context import get_forward_context_or_none

if TYPE_CHECKING:
    from xllm.python.attention.backend import ExecutionMetadataSource, LayerCache
    from xllm.python.model_executor.input_batch import InputBatch


@dataclass(frozen=True)
class FlashComm1Context:
    num_tokens: int
    local_num_tokens: int
    tp_size: int
    tp_rank: int
    enable_mmrs_fusion: bool = False
    mmrs_comm_mode: str = "aiv"

    def _pad(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[0] != self.num_tokens:
            raise ValueError("FlashComm1 requires the full packed token rows before sharding")
        padding = self.local_num_tokens * self.tp_size - self.num_tokens
        return torch.nn.functional.pad(hidden, (0, 0, 0, padding)) if padding else hidden.contiguous()

    def shard(self, hidden: torch.Tensor) -> torch.Tensor:
        padded = self._pad(hidden)
        return padded.narrow(0, self.tp_rank * self.local_num_tokens, self.local_num_tokens).contiguous()

    def gather(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[0] != self.local_num_tokens:
            raise ValueError("FlashComm1 requires the padded rank-local token rows before gathering")
        gathered = distributed.tp_all_gather(hidden, dim=0, world_size=self.tp_size)
        # Attention and GDN always receive real packed rows: padding must never
        # acquire KV slots or update recurrent state.
        return gathered[: self.num_tokens]

    def reduce_scatter(self, hidden: torch.Tensor) -> torch.Tensor:
        return distributed.reduce_scatter(self._pad(hidden), self.tp_size)

    def matmul_reduce_scatter(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return distributed.matmul_reduce_scatter(self._pad(inputs), weight, self.tp_size, self.mmrs_comm_mode)


def get_flash_comm1_context() -> FlashComm1Context | None:
    context = get_forward_context_or_none()
    return (
        None if context is None else cast(FlashComm1Context | None, context.layer_shared_cache.get(FlashComm1Context))
    )


class FlashComm1MetadataBuilder:
    metadata_type: type[object] = FlashComm1Context

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        min_prefill_tokens: int,
        *,
        enable_mmrs_fusion: bool = False,
        mmrs_comm_mode: str = "aiv",
    ) -> None:
        if tp_size <= 1 or not 0 <= tp_rank < tp_size:
            raise ValueError("FlashComm1 requires a valid rank in a TP group larger than one")
        if min_prefill_tokens < 0:
            raise ValueError("FlashComm1 minimum prefill tokens must be nonnegative")
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._min_prefill_tokens = min_prefill_tokens
        self._enable_mmrs_fusion = enable_mmrs_fusion
        self._mmrs_comm_mode = mmrs_comm_mode
        self._activation_logged = False

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        pass

    def allocate_persistent(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> None:
        self.update_persistent(None, input_batch, metadata)

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: ExecutionMetadataSource
    ) -> None:
        if persistent_metadata is not None or self.build(input_batch, metadata) is not None:
            raise NotImplementedError("FlashComm1 does not support persistent graph metadata")

    def build(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> FlashComm1Context | None:
        num_tokens = input_batch.num_tokens
        if (
            not (metadata.is_prefill or metadata.is_chunked_prefill)
            or metadata.is_mixed
            or metadata.is_spec_verify
            or metadata.is_dummy
            or num_tokens == 0
            or num_tokens < self._min_prefill_tokens
        ):
            return None
        # Match native FC1's 16-token alignment on each TP rank.
        alignment = 16 * self._tp_size
        padded_tokens = (num_tokens + alignment - 1) // alignment * alignment
        if not self._activation_logged:
            logger.info(
                "Qwen Python FlashComm1 prefill activated: tokens=%s, TP=%s, local_rows=%s, threshold=%s, MMRS=%s, comm_mode=%s",
                num_tokens,
                self._tp_size,
                padded_tokens // self._tp_size,
                self._min_prefill_tokens,
                self._enable_mmrs_fusion,
                self._mmrs_comm_mode,
            )
            self._activation_logged = True
        return FlashComm1Context(
            num_tokens,
            padded_tokens // self._tp_size,
            self._tp_size,
            self._tp_rank,
            enable_mmrs_fusion=self._enable_mmrs_fusion,
            mmrs_comm_mode=self._mmrs_comm_mode,
        )

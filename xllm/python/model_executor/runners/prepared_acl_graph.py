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

"""Decode graphs bound to final Slot views; never allocates model inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.forward_context import AclGraphExecutionState, LayerSynchronizer
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.base import ModelExecutionOutput


class PreparedAclGraphRunner(AclGraphRunner):
    """Owns captured graphs; input storage is borrowed from native Slots.

    Regular warmup Tasks populate the cache before serving admits a bucket.
    Native Slots own bucket selection and padding. Launch looks up the captured
    binding, uses the caller's task stream and returns a borrowed output whose
    readers must retire before reuse. A missing binding never captures at runtime.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        dp_size: int = 1,
        dp_rank: int = 0,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self._prepared_graphs: dict[tuple[object, ...], AclGraphEntry] = {}
        self.prepared_replays = 0

    @staticmethod
    def _prepared_binding(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> tuple[object, ...]:
        tensors = (
            input_ids,
            positions,
            metadata.slot_mapping,
            metadata.block_table,
            metadata.q_seq_lens,
            getattr(metadata, "q_cu_seq_lens", None),
            metadata.kv_seq_lens,
            input_embedding,
            mtp_topk_indices,
        )
        addresses = tuple(
            None
            if tensor is None
            else (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            for tensor in tensors
        )

        counts = tuple(getattr(metadata, "dp_execution_token_counts", ()))
        phases = tuple(getattr(metadata, "dp_is_decode", ()))
        return addresses + ((counts, phases) if len(counts) > 1 else ())

    def _validate_prepared_dp(self, input_ids: torch.Tensor, metadata: AttentionMetadata) -> None:
        if self.dp_size == 1:
            return
        counts = tuple(metadata.dp_execution_token_counts)
        phases = tuple(metadata.dp_is_decode)
        if len(counts) != self.dp_size or len(phases) != self.dp_size or any(count <= 0 for count in counts):
            raise ValueError("prepared DP requires complete positive execution counts and phases")
        if counts[self.dp_rank] != input_ids.numel():
            raise ValueError("prepared DP execution counts do not match local physical rows")
        if not all(phases) or len(set(counts)) != 1:
            raise ValueError("prepared DP graphs require the same decode batch size on every rank")

    def _validate_prepared_input(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> None:
        batch_size = input_ids.numel()
        if not 0 < batch_size <= self.max_batch:
            raise ValueError("prepared ACL graph batch size exceeds capacity or is empty")
        self._validate_prepared_dp(input_ids, metadata)
        if metadata.is_prefill or metadata.is_chunked_prefill or metadata.is_spec_verify:
            raise ValueError("prepared ACL graph requires ordinary decode")
        if getattr(metadata, "prepared_attention_state", None) is None:
            raise ValueError("prepared ACL graph requires prepared metadata")
        block_table = metadata.block_table
        if block_table is None or block_table.dim() != 2:
            raise ValueError("prepared ACL graph requires a block table per metadata row")
        self._validate_decode_token_layout(input_ids, positions, metadata.slot_mapping, block_table.shape[0])
        if metadata.q_seq_lens.numel() != batch_size or metadata.kv_seq_lens.numel() != batch_size:
            raise ValueError("prepared ACL graph requires query and KV lengths per metadata row")
        for name, tensor in (("hidden state", input_embedding), ("MTP top-k", mtp_topk_indices)):
            if tensor is not None and (
                tensor.dim() < 2 or tensor.shape[0] != batch_size or tensor.device != input_ids.device
            ):
                raise ValueError(f"prepared ACL graph requires {name} on the input device with one row per token")
        if mtp_topk_indices is not None and input_embedding is None:
            raise ValueError("prepared MTP top-k requires hidden state")

    @torch.inference_mode()
    def warmup_prepared(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> None:
        self._validate_prepared_input(input_ids, positions, metadata, input_embedding, mtp_topk_indices)
        binding = self._prepared_binding(input_ids, positions, metadata, input_embedding, mtp_topk_indices)
        if binding in self._prepared_graphs:
            return
        self._initialize_task_updates()
        entry = AclGraphEntry()
        entry.batch_size = input_ids.numel()
        entry.graph = None
        entry.static_output = None
        entry.static_input_ids = input_ids
        entry.static_positions = positions
        entry.static_input_embedding = input_embedding
        entry.static_mtp_topk_indices = mtp_topk_indices
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        # Retain the exact views independently of the mutable native Slot
        # metadata. Only the original model's inputs are captured here.
        entry.static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=metadata.slot_mapping,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            block_table=metadata.block_table,
            q_seq_lens=metadata.q_seq_lens,
            q_cu_seq_lens=metadata.q_cu_seq_lens,
            kv_seq_lens=metadata.kv_seq_lens,
            kv_seq_lens_host_values=list(metadata.kv_seq_lens_host_values),
            dp_execution_token_counts=tuple(getattr(metadata, "dp_execution_token_counts", ())),
            dp_is_decode=tuple(getattr(metadata, "dp_is_decode", ())),
            prepared_attention_state=(
                metadata.prepared_attention_state if getattr(self.attention_backend, "is_mla", False) else None
            ),
        )
        self._prepare_attention(entry, entry.static_metadata)
        self._capture(entry, torch.npu.current_stream(self.device))
        self._prepared_graphs[binding] = entry

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> ModelExecutionOutput:
        if layer_synchronizer is not None:
            raise ValueError("prepared ACL replay does not support layer synchronization")
        self._validate_prepared_input(input_ids, positions, metadata, input_embedding, mtp_topk_indices)
        entry = self._prepared_graphs.get(
            self._prepared_binding(input_ids, positions, metadata, input_embedding, mtp_topk_indices)
        )
        if entry is None:
            # Native admission already selected a captured bucket for every
            # rank. A missing Slot binding is an error, never a local fallback.
            raise RuntimeError("prepared ACL replay requires a warmed Slot binding")
        # Only serialized Launch installs shared backend state. This path
        # selects entry-owned buffers and refreshes its captured Host lists.
        self._prepare_attention(entry, metadata, replay=True)
        stream = torch.npu.current_stream(self.device)
        entry.graph.replay()
        self._update_after_replay(entry, stream)
        self.prepared_replays += 1
        if not entry.replay_logged:
            logger.info(
                "Python prepared ACL graph first replay: model=%s bucket=%d physical_rows=%d "
                "mtp_topk=%s hidden_shape=%s topk_shape=%s",
                type(self.model).__name__,
                entry.batch_size,
                input_ids.numel(),
                mtp_topk_indices is not None,
                None if input_embedding is None else tuple(input_embedding.shape),
                None if mtp_topk_indices is None else tuple(mtp_topk_indices.shape),
            )
            entry.replay_logged = True
        return entry.static_output

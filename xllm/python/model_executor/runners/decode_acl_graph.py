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

"""NPU (Ascend) ACL graph runner for the Python model executor.

Captures and replays decode-step graphs using ``torch.npu.NPUGraph``.
Mirrors the structure of ``decode_cuda_graph.py`` but adds NPU-specific
logic:

* ``torch.npu.graph_task_group_begin/end`` around FIA ``.out`` calls during
  capture.
* ``torch.npu.graph_task_update_begin/end`` to refresh FIA host params after
  replay is queued; captured external events gate each FIA task until its
  current parameters are ready.
* Static ``block_table`` and ``slot_mapping`` tensors so the graph records
  fixed addresses whose *contents* are updated via ``_fill_entry`` each step.
* C++ ACLNN ops (RMSNorm, SiLU, reshape_paged_cache) are used in both eager
  and capture modes — no PyTorch fallbacks needed.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadata,
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
)
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import (
    _decode_bucket,
)

_GraphKey = tuple[
    int,
    bool,
    torch.dtype | None,
    torch.device | None,
    tuple[int, ...] | None,
    torch.dtype | None,
    torch.device | None,
    tuple[int, ...] | None,
    tuple[int, ...],
]


class DecodeAclGraphRunner(AclGraphRunner):
    """Owns static inputs and lazily captures decode graphs on a private stream.

    Inputs are copied into entry-owned storage; returned outputs are detached
    from replay buffers. Slot-bound execution uses PreparedAclGraphRunner.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        decode_batch_size_limit: int | None = None,
        num_decoding_tokens: int = 1,
        enable_mega_moe_token_mask: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self.max_model_len = max_model_len
        self.decode_batch_size_limit = None if decode_batch_size_limit is None else max(1, int(decode_batch_size_limit))
        self.num_decoding_tokens = max(1, int(num_decoding_tokens))
        self._batch_limit_warning_logged = False
        self._graphs: dict[_GraphKey, AclGraphEntry] = {}
        self._dp_graph_variants: dict[_GraphKey, int] = {}
        self._enable_mega_moe_token_mask = enable_mega_moe_token_mask
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._stream: torch.npu.Stream | None = None
        self._update_done_event: torch.npu.Event | None = None
        self._update_done_recorded = False

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> bool:
        if input_ids.dim() != 1:
            return False
        is_expanded_spec_verify = resolve_expanded_decode_metadata(metadata) is not None
        if (metadata.is_prefill or metadata.is_chunked_prefill) and not is_expanded_spec_verify:
            return False

        # KDA verify needs an explicit recurrent-state protocol. Preserve
        # the eager fallback while both opt-in protocols are disabled.
        linear_indices = getattr(metadata, "linear_state_indices", None)
        if (
            is_expanded_spec_verify
            and not self.attention_backend.is_mla
            and linear_indices is not None
            and linear_indices.numel() > 0
            and not self.attention_backend.supports_linear_spec_verify_graph
        ):
            return False

        batch_size = input_ids.numel()
        if mtp_topk_indices is not None:
            if mtp_topk_indices.dim() < 2 or mtp_topk_indices.shape[0] != batch_size:
                raise ValueError(
                    "MTP top-k indices must have one leading row per decode token "
                    f"(got shape={tuple(mtp_topk_indices.shape)}, rows={batch_size})"
                )
        if not (
            self._has_compatible_decode_metadata(input_ids, metadata)
            and (input_embedding is None or input_embedding.shape[0] == batch_size)
        ):
            return False

        local_batch_size, global_batch_size = self._decode_batch_sizes(
            input_ids,
            metadata,
        )
        if self.decode_batch_size_limit is not None and global_batch_size > self.decode_batch_size_limit:
            if not self._batch_limit_warning_logged:
                logger.warning(
                    f"Falling back to eager mode because decode batch_size "
                    f"(global={global_batch_size}, local={local_batch_size}) > "
                    f"{self.decode_batch_size_limit}; ACL graph is disabled for "
                    f"this request size to avoid OOM. This message is logged "
                    f"only once."
                )
                self._batch_limit_warning_logged = True
            return False

        if self.dp_size > 1:
            # DP ranks share one graph shape, so a missing or malformed
            # Execution counts cannot silently fall back to eager: divergent
            # execution paths across ranks would deadlock HCCL collectives.
            execution_counts = getattr(
                metadata,
                "dp_execution_token_counts",
                None,
            )
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, "
                    f"expected length {self.dp_size}). All DP ranks must use the same graph shape."
                )
            if any(count <= 0 for count in execution_counts):
                raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")
            dp_is_decode = getattr(metadata, "dp_is_decode", None)
            if dp_is_decode is not None and not all(dp_is_decode):
                return False
            if execution_counts[self.dp_rank] != input_ids.shape[0]:
                raise RuntimeError(
                    "DP execution token count does not match the local input: "
                    f"rank={self.dp_rank}, rows={input_ids.shape[0]}, "
                    f"counts={execution_counts}"
                )
            global_batch = max(execution_counts)
            return global_batch <= self.max_batch
        return batch_size <= self.max_batch

    def _padded_batch_size(self, batch_size: int, metadata: AttentionMetadata) -> int:
        # All DP ranks must use the same shape for captured MoE collectives.
        if self.dp_size > 1:
            execution_counts = metadata.dp_execution_token_counts
            if any(count <= 0 for count in execution_counts):
                raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")
            batch_size = max(execution_counts)
        if batch_size > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")
        # Keep a final partial bucket instead of falling back to eager or
        # padding beyond the backend's declared row capacity.
        return min(_decode_bucket(batch_size), self.max_batch)

    def _decode_batch_sizes(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> tuple[int, int]:
        logical_sequence_counts = getattr(metadata, "dp_global_sequence_nums", ())
        if logical_sequence_counts:
            if len(logical_sequence_counts) != self.dp_size:
                raise RuntimeError(
                    "dp_global_sequence_nums must contain one value per DP rank "
                    f"(got {logical_sequence_counts!r}, expected {self.dp_size})"
                )
            if any(count < 0 for count in logical_sequence_counts):
                raise RuntimeError(f"dp_global_sequence_nums must be nonnegative, got {logical_sequence_counts!r}")
            local_batch_size = logical_sequence_counts[self.dp_rank]
            global_batch_size = max(logical_sequence_counts, default=0)
            return local_batch_size, global_batch_size

        local_num_tokens = input_ids.numel()
        global_num_tokens = local_num_tokens
        execution_counts = getattr(
            metadata,
            "dp_execution_token_counts",
            (),
        )
        if isinstance(execution_counts, (list, tuple)) and len(execution_counts) > 1:
            if any(count <= 0 for count in execution_counts):
                raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")
            global_num_tokens = max(execution_counts)
        return (
            local_num_tokens // self.num_decoding_tokens,
            global_num_tokens // self.num_decoding_tokens,
        )

    @property
    def _logical_page_size(self) -> int:
        return int(getattr(self.attention_backend, "logical_page_size", self.attention_backend.page_size))

    def _decode_metadata(
        self, metadata: AttentionMetadata
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int] | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return per-row KV and paging metadata for decode graph replay."""
        expanded = resolve_expanded_decode_metadata(metadata, block_size=self._logical_page_size)
        block_table = expanded.block_table if expanded is not None else metadata.block_table
        if block_table is None:
            multi_block_tables = getattr(metadata, "multi_block_tables", ()) or ()
            block_table = multi_block_tables[0] if multi_block_tables else None
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        if block_table is None or kv_seq_lens is None:
            raise RuntimeError("decode graph requires block and KV metadata")
        block_table = block_table.to(device=kv_seq_lens.device, dtype=torch.int32).contiguous()
        kv_seq_lens = kv_seq_lens.to(torch.int32)
        is_mla = getattr(self.attention_backend, "is_mla", False)
        requires_host_kv_lengths = not is_mla or getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        )
        if not requires_host_kv_lengths:
            # The C++ engine sizes kv_seq_lens_host_values by the global DP
            # batch, while block_table holds only this rank's local rows. When
            # the backend does not consume host KV lengths (e.g. sparse MLA),
            # drop them so the mismatched global length never reaches shape
            # validation.
            kv_seq_lens_host_values = None
        else:
            kv_seq_lens_host_values = (
                expanded.kv_seq_lens_host_values
                if expanded is not None
                else getattr(metadata, "kv_seq_lens_host_values", None)
            )
            if not kv_seq_lens_host_values:
                raise RuntimeError("decode graph requires scheduler-provided host KV lengths")

        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        if paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None:
            raise RuntimeError("decode graph requires paged KV metadata")
        if expanded is None and not self._has_row_aligned_paged_kv_metadata(
            block_table,
            paged_kv_indptr,
            paged_kv_last_page_len,
        ):
            (
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
            ) = self._build_row_aligned_paged_kv_metadata(
                block_table,
                kv_seq_lens,
            )
        self._validate_decode_metadata_shapes(
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        return (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    @staticmethod
    def _has_row_aligned_paged_kv_metadata(
        block_table: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> bool:
        row_count = block_table.shape[0]
        return (
            paged_kv_indptr.dim() == 1
            and paged_kv_indptr.numel() == row_count + 1
            and paged_kv_last_page_len.dim() == 1
            and paged_kv_last_page_len.numel() == row_count
        )

    def _build_row_aligned_paged_kv_metadata(
        self,
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token-row paging metadata like the C++ graph input builder."""
        page_size = self._logical_page_size
        if page_size <= 0:
            raise RuntimeError("decode graph page size must be positive")

        effective_kv_seq_lens = torch.clamp(kv_seq_lens, min=1)
        page_counts = torch.div(
            effective_kv_seq_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        paged_kv_indptr = torch.cat(
            (
                torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=block_table.device,
                ),
                torch.cumsum(page_counts, dim=0, dtype=torch.int32),
            )
        )
        page_offsets = torch.arange(
            block_table.shape[1],
            dtype=torch.int32,
            device=block_table.device,
        )
        valid_pages = page_offsets.unsqueeze(0) < page_counts.unsqueeze(1)
        paged_kv_indices = block_table.masked_select(valid_pages).contiguous()
        paged_kv_last_page_len = ((effective_kv_seq_lens - 1) % page_size + 1).to(torch.int32)
        return (
            paged_kv_indptr.contiguous(),
            paged_kv_indices,
            paged_kv_last_page_len.contiguous(),
        )

    @staticmethod
    def _validate_decode_metadata_shapes(
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_seq_lens_host_values: list[int] | None,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        if block_table.dim() != 2:
            raise RuntimeError("decode block_table must be two-dimensional")
        row_count = block_table.shape[0]
        per_row_tensors = (
            ("kv_seq_lens", kv_seq_lens),
            ("paged_kv_last_page_len", paged_kv_last_page_len),
        )
        for name, tensor in per_row_tensors:
            if tensor.dim() != 1 or tensor.numel() != row_count:
                raise RuntimeError(f"decode {name} must contain one value per metadata row")
        if kv_seq_lens_host_values is not None and len(kv_seq_lens_host_values) != row_count:
            raise RuntimeError("decode kv_seq_lens_host_values must contain one value per metadata row")
        if paged_kv_indptr.dim() != 1 or paged_kv_indptr.numel() != row_count + 1:
            raise RuntimeError(
                "decode paged_kv_indptr must contain one offset per metadata row plus the terminal offset"
            )
        if paged_kv_indices.dim() != 1 or paged_kv_indices.numel() == 0:
            raise RuntimeError("decode paged_kv_indices must be a non-empty flat page list")

    def _has_compatible_decode_metadata(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> bool:
        """Check that token tensors and decode metadata use the same row layout."""
        (
            block_table,
            _,
            _,
            _,
            _,
            _,
        ) = self._decode_metadata(metadata)
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is None or slot_mapping.numel() == 0:
            # Admission only needs the host layout; defer slot upload until fill.
            host_slots = getattr(metadata, "new_cache_slots_host_values", None)
            if (
                input_ids.dim() != 1
                or input_ids.numel() != block_table.shape[0]
                or host_slots is None
                or len(host_slots) != input_ids.numel()
            ):
                return False
        else:
            self._validate_decode_token_layout(input_ids, None, slot_mapping, block_table.shape[0])
        batch_size = input_ids.numel()
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        if not is_expanded and metadata.kv_cu_seq_lens is not None:
            if metadata.kv_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        if metadata.q_cu_seq_lens is not None and not is_expanded:
            if metadata.q_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        has_initial_state = getattr(metadata, "has_initial_state", None)
        if has_initial_state is not None:
            state_count = has_initial_state.numel()
            if state_count != batch_size and not (is_expanded and state_count > 0 and batch_size % state_count == 0):
                return False
        linear_idx = getattr(metadata, "linear_state_indices", None)
        return not (
            linear_idx is not None
            and linear_idx.numel() != batch_size
            and not (is_expanded and linear_idx.numel() > 0 and batch_size % linear_idx.numel() == 0)
        )

    @staticmethod
    def _cumulative_lengths(
        sequence_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize NPU sequence ends to a cumulative tensor with a zero."""
        batch_size = sequence_lengths.numel()
        if cumulative_lengths is None:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=sequence_lengths.device,
                    ),
                    torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32),
                )
            )
        cumulative_lengths = cumulative_lengths.to(torch.int32)
        if cumulative_lengths.numel() == batch_size + 1:
            return cumulative_lengths
        if cumulative_lengths.numel() == batch_size:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=cumulative_lengths.device,
                    ),
                    cumulative_lengths,
                )
            )
        raise RuntimeError(
            "cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> _GraphKey:
        batch_size = input_ids.shape[0]
        padded_batch_size = self._padded_batch_size(batch_size, metadata)

        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        graph_key = self._graph_key(
            padded_batch_size,
            is_expanded,
            input_embedding,
            mtp_topk_indices,
        )
        graph_key = self._synchronize_dp_graph_key(graph_key, input_ids)
        if not self._synchronize_dp_graph_presence(graph_key, input_ids):
            return graph_key

        self._graphs.pop(graph_key, None)

        if mtp_topk_indices is None:
            self._prepare_graph_entry(input_ids, positions, metadata, input_embedding, graph_key=graph_key)
        else:
            self._prepare_graph_entry(
                input_ids, positions, metadata, input_embedding, mtp_topk_indices, graph_key=graph_key
            )
        return graph_key

    def _synchronize_dp_graph_key(
        self,
        local_key: _GraphKey,
        input_ids: torch.Tensor,
    ) -> _GraphKey:
        """Keep graphs captured with different peer variants in separate entries.

        Empty DP ranks omit MTP embeddings and top-k inputs. A local variant
        can therefore participate in several distinct collective captures.
        Local cache presence alone cannot establish that the peers are about
        to replay graphs captured together: both entries can exist but belong
        to different captures. Include every DP rank's variant in the key.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return local_key
        variant = self._dp_graph_variants.setdefault(local_key, len(self._dp_graph_variants) + 1)
        local_variant = torch.tensor([variant], dtype=torch.int32, device=input_ids.device)
        from xllm.python import distributed

        variants = distributed.all_gather(local_variant, dim=0, world_size=self.dp_size, group_name="dp")
        return (*local_key[:-1], tuple(variants.cpu().tolist()))

    def _synchronize_dp_graph_presence(
        self,
        graph_key: _GraphKey,
        input_ids: torch.Tensor,
    ) -> bool:
        """Make DP ranks capture their local graph variants together.

        A rank can enter a request with a warmup graph left over from startup,
        while another rank saw an empty shard and has to capture lazily. Letting
        the former replay while the latter captures makes HCCL graph collectives
        diverge. Synchronize the cache-presence bit before deciding whether to
        reuse the entry; if any rank is missing, all ranks recapture it. Local
        signatures can differ for empty ranks (embedding/top-k are absent),
        so a cached local key must still participate on every call.
        """
        if graph_key in self._graphs and self.dp_size <= 1:
            return False
        if self.dp_size <= 1 or not dist.is_initialized():
            return graph_key not in self._graphs
        presence = torch.tensor(
            [int(graph_key in self._graphs)],
            dtype=torch.int32,
            device=input_ids.device,
        )
        from xllm.python import distributed

        distributed.all_reduce_(presence, group_name="dp")
        return int(presence.item()) != self.dp_size

    def _synchronize_dp_graph_execute(self, input_ids: torch.Tensor) -> None:
        """Keep graph replay starts ordered after the warmup collective.

        ``warmup`` has already resolved the paired graph key. Repeating the
        variant all-gather in ``execute`` would copy the same NPU tensor back
        to the host on every decode step. A one-value reduction keeps the
        cross-rank launch barrier without rebuilding the Python cache key.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return
        ready = torch.ones(1, dtype=torch.int32, device=input_ids.device)
        from xllm.python import distributed

        distributed.all_reduce_(ready, group_name="dp")

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _GraphKey | None = None,
    ) -> ModelExecutionOutput:
        batch_size = input_ids.shape[0]
        if graph_key is not None:
            self._synchronize_dp_graph_execute(input_ids)
        entry = self._prepare_graph_entry(
            input_ids,
            positions,
            metadata,
            input_embedding,
            mtp_topk_indices,
            graph_key=graph_key,
        )

        assert self._stream is not None
        assert self._update_stream is not None
        assert self._replay_done_event is not None

        self._stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._stream):
            entry.graph.replay()
            output = self._slice_output(entry.static_output, batch_size)

        if not getattr(entry, "replay_logged", False):
            logger.info(
                "Python ACL graph first replay: model=%s bucket=%d rows=%d mtp_topk=%s",
                type(self.model).__name__,
                entry.batch_size,
                batch_size,
                mtp_topk_indices is not None,
            )
            entry.replay_logged = True

        assert self._update_done_event is not None
        self._update_after_replay(entry, self._stream, self._update_done_event)
        self._update_done_recorded = True

        torch.npu.current_stream().wait_stream(self._stream)
        # The graph replay waits on each task's external event, but the task
        # update itself runs on a separate stream.  Schedule-overlap returns
        # to C++ without a device-wide synchronize, so the next MTP draft or
        # target graph can otherwise start while this runner is still updating
        # FIA/MLA task parameters.  Order the caller's compute stream after
        # those updates before exposing the output to the next model stage.
        assert self._update_done_event is not None
        torch.npu.current_stream().wait_event(self._update_done_event)
        return output

    @staticmethod
    def _slice_output(
        output: ModelExecutionOutput,
        batch_size: int,
    ) -> ModelExecutionOutput:
        """Slice graph outputs and detach them from the replay buffers.

        Graph replay writes the same persistent output allocation on every
        step.  The MTP schedule-overlap path can retain both hidden states and
        top-k indices until a later step, so returning views allows a replay to
        overwrite data that an in-flight worker still consumes.  Clone every
        tensor output while it is still ordered on the graph stream.
        """
        if not isinstance(output, tuple):
            return output[:batch_size].clone()
        hidden, aux_hidden = output[:2]
        hidden = hidden[:batch_size].clone()
        if aux_hidden is not None:
            aux_hidden = aux_hidden[:batch_size].clone()
        if len(output) == 2:
            return hidden, aux_hidden
        topk = output[2]
        if topk is not None:
            topk = topk[:batch_size].clone()
        return hidden, aux_hidden, topk

    def _capture(self, entry: AclGraphEntry, stream: torch.npu.Stream) -> None:
        # Graph warmup and capture execute the model before the first replay.
        # KDA advances recurrent cache and speculative state on each forward,
        # so restore the touched slots before serving the decode step.
        idx = entry.static_metadata.linear_state_indices
        linear_snapshot = []
        v2_snapshot = None
        v3_snapshot = None
        # A first capture may create backend-owned KDA state after snapshots.
        # Record the existing objects so only new states have their global
        # ever_armed flag reset when the synthetic capture is discarded.
        backend_state = getattr(self.attention_backend, "__dict__", {})
        v2_before = {id(st) for st in backend_state.get("_kda_v2", {}).values()}
        v3_before = {id(st) for st in backend_state.get("_kda_v3", {}).values()}
        if idx is not None:
            for cache in self.layer_caches:
                conv = getattr(cache, "conv", None)
                ssm = getattr(cache, "ssm", None)
                if conv is not None and ssm is not None:
                    linear_snapshot.append(
                        (conv, ssm, conv.index_select(0, idx).clone(), ssm.index_select(0, idx).clone())
                    )
            v2_snapshot_fn = getattr(self.attention_backend, "snapshot_kda_v2_state", None)
            if v2_snapshot_fn is not None:
                v2_snapshot = v2_snapshot_fn(idx)
            v3_snapshot_fn = getattr(self.attention_backend, "snapshot_kda_v3_state", None)
            if v3_snapshot_fn is not None:
                v3_snapshot = v3_snapshot_fn(idx)

        try:
            super()._capture(entry, stream)
        finally:
            # Snapshot reads precede capture via the parent stream wait; wait
            # for capture writes before restoring on the caller's stream.
            torch.npu.current_stream().wait_stream(stream)
            if idx is not None:
                for conv, ssm, conv_rows, ssm_rows in linear_snapshot:
                    conv.index_copy_(0, idx, conv_rows)
                    ssm.index_copy_(0, idx, ssm_rows)
            if idx is not None:
                # Disarm all touched slots first; snapshots then restore the
                # exact previous contents of states that existed before capture.
                for version, prior in (("v2", v2_before), ("v3", v3_before)):
                    disarm = getattr(self.attention_backend, f"disarm_kda_{version}_slots", None)
                    if disarm is not None:
                        disarm(idx)
                    for st in backend_state.get(f"_kda_{version}", {}).values():
                        if id(st) not in prior:
                            st["ever_armed"] = False
            if v2_snapshot is not None:
                self.attention_backend.restore_kda_v2_state(v2_snapshot)
            if v3_snapshot is not None:
                self.attention_backend.restore_kda_v3_state(v3_snapshot)

    def _prepare_graph_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _GraphKey | None = None,
    ) -> AclGraphEntry:
        batch_size = input_ids.shape[0]
        padded_batch_size = self._padded_batch_size(batch_size, metadata)

        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        if graph_key is None:
            graph_key = self._graph_key(
                padded_batch_size,
                is_expanded,
                input_embedding,
                mtp_topk_indices,
            )
            graph_key = self._synchronize_dp_graph_key(graph_key, input_ids)
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(
                padded_batch_size,
                input_ids,
                positions,
                metadata,
                mtp_topk_indices,
            )
            self._graphs[graph_key] = entry

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._initialize_task_updates()
            self._update_done_event = torch.npu.Event()

        # The previous replay may still be reading the capture buffers on the
        # graph stream while the scheduler prepares the next step on the
        # current stream.  Wait before mutating static inputs/metadata; the
        # later graph-stream wait only orders the new replay after these
        # writes and cannot protect this earlier update.
        if self._replay_done_event is not None:
            # All buckets share the paged-KV index buffer, including a newly
            # allocated bucket whose graph has not been captured yet.
            torch.npu.current_stream().wait_event(self._replay_done_event)
        if self._update_done_recorded:
            # Task updates run on a separate stream and read the same static
            # metadata tensors that _fill_entry mutates.  Waiting only for the
            # graph replay leaves that read/write pair unordered when the
            # scheduler advances quickly under overlap.
            assert self._update_done_event is not None
            torch.npu.current_stream().wait_event(self._update_done_event)

        self._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size,
            input_embedding,
            mtp_topk_indices,
        )

        self._prepare_attention(entry, entry.static_metadata, replay=not first_capture)

        if first_capture:
            self._capture(entry, self._stream)
        return entry

    @staticmethod
    def _graph_key(
        padded_batch_size: int,
        is_expanded: bool,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> _GraphKey:
        """Return the key for a shape- and metadata-specific graph."""
        input_signature = (None, None, None)
        if input_embedding is not None:
            input_signature = (
                input_embedding.dtype,
                input_embedding.device,
                tuple(input_embedding.shape[1:]),
            )
        topk_signature = (None, None, None)
        if mtp_topk_indices is not None:
            topk_signature = (
                mtp_topk_indices.dtype,
                mtp_topk_indices.device,
                tuple(mtp_topk_indices.shape[1:]),
            )
        return (
            padded_batch_size,
            is_expanded,
            *input_signature,
            *topk_signature,
            (),
        )

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> AclGraphEntry:
        device = input_ids.device
        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if self._paged_kv_indices_buffer is None:
            page_size = self._logical_page_size
            # Reserve the scheduler's lookahead page, including for the MTP
            # draft executor, which reports one decoding token per request.
            max_blocks_per_sequence = (self.max_model_len + page_size - 1) // page_size + 1
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * max_blocks_per_sequence,
                dtype=paged_kv_indices.dtype,
                device=device,
            )
            self._max_blocks_per_sequence = max_blocks_per_sequence

        static_block_table = torch.zeros(
            padded_batch_size,
            self._max_blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        entry = AclGraphEntry()
        entry.batch_size = padded_batch_size
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        entry.static_input_ids = torch.zeros(padded_batch_size, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(padded_batch_size, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        entry.static_mtp_topk_indices = None
        if mtp_topk_indices is not None:
            entry.static_mtp_topk_indices = torch.zeros(
                (padded_batch_size, *mtp_topk_indices.shape[1:]),
                dtype=mtp_topk_indices.dtype,
                device=mtp_topk_indices.device,
            )
        entry.static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            linear_state_indices=torch.zeros(
                padded_batch_size,
                dtype=torch.int32,
                device=device,
            ),
            kv_seq_lens_host_values=[1] * padded_batch_size,
            block_table=static_block_table,
            dp_execution_token_counts=(padded_batch_size,) * self.dp_size if self.dp_size > 1 else (),
            dp_global_sequence_nums=(),
            dp_is_decode=tuple([1] * self.dp_size) if self.dp_size > 1 else (),
            # One mask slot per padded row across all DP ranks. aclnnMegaMoe
            # dispatches the fixed graph shape over EP, so padded lanes must be
            # marked inactive or they are routed as real tokens.
            mega_moe_token_mask=(
                torch.zeros(
                    padded_batch_size * self.dp_size,
                    dtype=torch.int8,
                    device=device,
                )
                if self._enable_mega_moe_token_mask
                else None
            ),
        )
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        entry.kv_seq_lens_delta = torch.empty(padded_batch_size, dtype=torch.int32, device=device)
        # The graph metadata update writes per-sequence KV lengths into this
        # buffer.  MLA/SFA consumes the same stable buffer as its key lengths.
        entry.static_metadata.kv_seq_lens = entry.kv_seq_lens_delta
        if is_expanded:
            entry.static_metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
                kv_seq_lens=entry.kv_seq_lens_delta,
                block_table=entry.static_metadata.block_table,
                paged_kv_indptr=entry.static_metadata.paged_kv_indptr,
                paged_kv_indices=entry.static_metadata.paged_kv_indices,
                paged_kv_last_page_len=(entry.static_metadata.paged_kv_last_page_len),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=(
                    entry.static_metadata.kv_seq_lens_host_values
                    if getattr(
                        self.attention_backend,
                        "requires_host_kv_lengths",
                        False,
                    )
                    or not getattr(self.attention_backend, "is_mla", False)
                    else None
                ),
            )
        entry.static_metadata.multi_block_tables = self.attention_backend.create_graph_block_tables(
            padded_batch_size,
            self.max_model_len,
            self._max_blocks_per_sequence,
        )
        if entry.static_metadata.multi_block_tables:
            entry.static_metadata.new_cache_slots_host_values = []
            entry.static_metadata.dsa_graph_block_table_cols = self._max_blocks_per_sequence
            entry.static_metadata.dsa_graph_mode = True
        return entry

    def _fill_entry(
        self,
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_metadata.is_dummy = bool(getattr(metadata, "is_dummy", False))
        slot_mapping = self._effective_slot_mapping(metadata, batch_size, input_ids.device)
        (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if entry.static_metadata.multi_block_tables:
            self._fill_graph_dsa_positions(entry, positions)
        if batch_size != block_table.shape[0]:
            raise RuntimeError("ACL graph decode batch size must match metadata sequences")
        self._validate_decode_token_layout(
            input_ids,
            positions,
            slot_mapping,
            block_table.shape[0],
        )
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        cumulative_kv_seq_lens = self._cumulative_lengths(
            kv_seq_lens,
            None if is_expanded else metadata.kv_cu_seq_lens,
        )
        graph_positions = positions.to(torch.int32).contiguous()
        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            slot_mapping,
            cumulative_kv_seq_lens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            entry.static_input_ids,
            entry.static_positions,
            static_metadata.slot_mapping,
            static_metadata.kv_cu_seq_lens,
            entry.kv_seq_lens_delta,
            static_metadata.paged_kv_indptr,
            static_metadata.paged_kv_indices,
            static_metadata.paged_kv_last_page_len,
            padded_batch_size,
        )
        # MLA does not consume recurrent state. The scheduler can still carry
        # one linear-state index per request while GLM MTP adds a nonuniform
        # number of repair rows; those unrelated indices need no expansion.
        linear_state_indices = (
            None
            if getattr(self.attention_backend, "is_mla", False)
            else getattr(metadata, "linear_state_indices", None)
        )
        if linear_state_indices is not None:
            # MTP metadata carries one state index per sequence while the
            # decode graph input is expanded to one row per speculative token.
            # Repeat each sequence index across its token rows before copying
            # into the token-shaped persistent buffer.
            if linear_state_indices.numel() < batch_size:
                if batch_size % linear_state_indices.numel() != 0:
                    raise RuntimeError(
                        "ACL graph MTP linear_state_indices must divide the "
                        f"expanded token count (tokens={batch_size}, "
                        f"sequences={linear_state_indices.numel()})"
                    )
                repeat_count = batch_size // linear_state_indices.numel()
                linear_state_indices = linear_state_indices.repeat_interleave(repeat_count)
            if linear_state_indices.numel() < batch_size:
                raise RuntimeError(
                    "ACL graph linear_state_indices must contain one value "
                    f"per decode row (rows={batch_size}, "
                    f"values={linear_state_indices.numel()})"
                )
            static_metadata.linear_state_indices[:batch_size].copy_(linear_state_indices[:batch_size])
        if padded_batch_size > batch_size:
            static_metadata.linear_state_indices[batch_size:].zero_()
        self._fill_host_metadata(entry, kv_seq_lens_host_values, batch_size)
        self._fill_new_cache_slots_host_metadata(entry, metadata, batch_size)

        if input_embedding is not None:
            if input_embedding.shape[0] != batch_size:
                raise ValueError("ACL graph input_embedding token count must match input_ids")
            if entry.static_input_embedding is None:
                entry.static_input_embedding = torch.zeros(
                    entry.batch_size,
                    input_embedding.shape[-1],
                    dtype=input_embedding.dtype,
                    device=input_embedding.device,
                )
            elif entry.static_input_embedding.shape[1:] != input_embedding.shape[1:]:
                raise ValueError("ACL graph input_embedding shape changed for a graph bucket")
            entry.static_input_embedding[:batch_size].copy_(input_embedding)
            if entry.batch_size > batch_size:
                entry.static_input_embedding[batch_size:].zero_()
        elif entry.static_input_embedding is not None:
            entry.static_input_embedding.zero_()

        if mtp_topk_indices is not None:
            if entry.static_mtp_topk_indices is None:
                raise RuntimeError("MTP graph entry was captured without top-k input")
            if mtp_topk_indices.shape[0] != batch_size:
                raise ValueError("MTP top-k input row count must match decode batch")
            if mtp_topk_indices.shape[1:] != entry.static_mtp_topk_indices.shape[1:]:
                raise ValueError(
                    "MTP top-k input shape changed for an ACL graph bucket: "
                    f"got {tuple(mtp_topk_indices.shape[1:])}, "
                    f"expected {tuple(entry.static_mtp_topk_indices.shape[1:])}"
                )
            entry.static_mtp_topk_indices[:batch_size].copy_(mtp_topk_indices)
            if padded_batch_size > batch_size:
                entry.static_mtp_topk_indices[batch_size:].zero_()

        if static_metadata.block_table is not None:
            src_bt = block_table
            copy_cols = min(
                src_bt.shape[1],
                static_metadata.block_table.shape[1],
            )
            static_metadata.block_table[:batch_size, :copy_cols].copy_(src_bt[:batch_size, :copy_cols])
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()

        # Padded lanes must remain valid inputs for sparse MLA tiling.  Their
        # token and slot mapping are dummy values, so one KV token is safe.
        if padded_batch_size > batch_size:
            static_metadata.slot_mapping[batch_size:].fill_(-1)
            entry.kv_seq_lens_delta[batch_size:].fill_(1)

        self._fill_dsa_block_tables(
            static_metadata,
            metadata,
            batch_size,
        )
        if static_metadata.mega_moe_token_mask is not None:
            self._fill_mega_moe_token_mask(
                entry, (batch_size,) if self.dp_size == 1 else metadata.dp_execution_token_counts
            )

    def _fill_dsa_block_tables(
        self,
        static_metadata: StaticGraphAttentionMetadata,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Refresh every stable DSA manager table for the current decode."""
        static_tables = static_metadata.multi_block_tables
        if not static_tables:
            return
        source_tables = getattr(metadata, "multi_block_tables", ()) or ()
        if not source_tables:
            if getattr(metadata, "is_dummy", False):
                for table in static_tables:
                    table.zero_()
                return
            raise RuntimeError("ACL graph requires all captured manager block tables")
        if len(source_tables) < len(static_tables):
            raise RuntimeError(f"ACL graph DSA manager count changed: {len(source_tables)} < {len(static_tables)}")
        # MTP draft layers consume a prefix of the target's cache managers.
        source_tables = source_tables[: len(static_tables)]

        for manager_id, (target, source) in enumerate(zip(static_tables, source_tables, strict=True)):
            target.zero_()
            if source is None:
                continue
            if source.dim() != 2:
                raise RuntimeError(f"ACL graph DSA manager {manager_id} block table must be two-dimensional")
            source_rows = int(source.shape[0])
            if source_rows != batch_size:
                if source_rows <= 0 or batch_size % source_rows != 0:
                    raise RuntimeError(
                        f"ACL graph DSA manager {manager_id} row count "
                        f"{source_rows} does not match batch size {batch_size}"
                    )
                source = source.repeat_interleave(batch_size // source_rows, dim=0)
            if source.shape[1] > target.shape[1]:
                raise RuntimeError(
                    f"ACL graph DSA manager {manager_id} requires {source.shape[1]} "
                    f"columns, capacity is {target.shape[1]}"
                )
            target[:batch_size, : source.shape[1]].copy_(source[:batch_size])

    def _fill_graph_dsa_positions(
        self,
        entry: AclGraphEntry,
        positions: torch.Tensor,
    ) -> None:
        """Keep a stable DSA position tensor synced with the static bucket."""
        static_metadata = entry.static_metadata
        padded_batch_size = entry.batch_size
        if static_metadata.dsa_positions is None:
            static_metadata.dsa_positions = torch.zeros(
                padded_batch_size,
                dtype=torch.int64,
                device=entry.static_positions.device,
            )
        graph_positions = positions.to(device=entry.static_positions.device, dtype=torch.int64)
        copy_count = min(int(graph_positions.numel()), padded_batch_size)
        static_metadata.dsa_positions[:copy_count].copy_(graph_positions[:copy_count])
        if padded_batch_size > copy_count:
            static_metadata.dsa_positions[copy_count:].zero_()

    def _fill_host_metadata(
        self,
        entry: AclGraphEntry,
        kv_seq_lens: list[int] | None,
        batch_size: int,
    ) -> None:
        if getattr(self.attention_backend, "is_mla", False) and not getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        ):
            return
        if kv_seq_lens is None:
            raise RuntimeError("decode ACL graph requires scheduler-provided host KV lengths")
        if len(kv_seq_lens) != batch_size:
            raise RuntimeError("decode ACL graph requires per-sequence host KV lengths")

        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_kv_seq_lens = static_metadata.kv_seq_lens_host_values
        if static_kv_seq_lens is None:
            raise RuntimeError("decode ACL graph host KV buffer is missing")
        static_kv_seq_lens[:batch_size] = kv_seq_lens
        if padded_batch_size > batch_size:
            static_kv_seq_lens[batch_size:] = [1] * (padded_batch_size - batch_size)

    @staticmethod
    def _effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve scheduler cache slots when DSV4 omits the top-level tensor."""
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return slot_mapping.to(device=device, dtype=torch.int32).contiguous()
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        if host_slots is None or len(host_slots) != token_count:
            raise RuntimeError("ACL graph decode requires one scheduler cache slot per token")
        return torch.tensor(host_slots, dtype=torch.int32, device=device)

    @staticmethod
    def _fill_new_cache_slots_host_metadata(
        entry: AclGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Copy scheduler-resolved physical DSA slots into graph metadata."""
        source_slots = getattr(metadata, "new_cache_slots_host_values", None)
        target_slots = entry.static_metadata.new_cache_slots_host_values
        if target_slots is None:
            return
        if source_slots is None or (not source_slots and getattr(metadata, "is_dummy", False)):
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if len(source_slots) < batch_size:
            raise RuntimeError("decode ACL graph requires one host cache slot per token")
        real_slots = [int(slot) for slot in source_slots[:batch_size]]
        padding_slots = [0] * (entry.batch_size - batch_size)
        entry.static_metadata.new_cache_slots_host_values = real_slots + padding_slots

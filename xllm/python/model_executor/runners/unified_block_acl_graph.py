# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-width and adaptive greedy block rounds over native task Slot buffers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import fields, replace
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

import torch

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.forward_context import AclGraphCaptureContext, AclGraphExecutionState
from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner

if TYPE_CHECKING:
    from xllm.python.model_executor.executor import ModelExecutor

Binding = tuple[torch.Tensor, torch.Tensor, AttentionMetadata]


class TensorDesc(NamedTuple):
    """Hashable storage and layout metadata for one borrowed tensor."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def _tensor_binding(tensor: torch.Tensor) -> TensorDesc:
    return TensorDesc(tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _metadata_view(metadata: AttentionMetadata) -> StaticGraphAttentionMetadata:
    return StaticGraphAttentionMetadata(
        **{
            field.name: getattr(metadata, field.name)
            for field in fields(StaticGraphAttentionMetadata)
            if hasattr(metadata, field.name)
        }
    )


def _graph_buckets(maximum: int) -> tuple[int, ...]:
    # Match native decode_graph_batch_sizes, including its truncated last bucket.
    return tuple(sorted({min(size, maximum) for size in (1, 2, 4, 8, *range(16, maximum + 16, 16))}))


def _verify_buckets(batch: int, maximum: int, draft_width: int) -> tuple[int, ...]:
    # The unified planner and startup enumerate the same four useful tiers.
    # Keeping every 16-row bucket would need quadratically many whole graphs.
    buckets = _graph_buckets(maximum)
    return tuple(
        dict.fromkeys(
            next((rows for rows in buckets if rows >= batch * scale), maximum) for scale in (1, 2, 4, draft_width + 1)
        )
    )


def _accepted_prefix_lengths(proposal: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    # aclnnCumprod requires a numeric input even when its output dtype is set.
    matches = (proposal == target_tokens[:, : proposal.shape[1]]).to(torch.int32)
    return matches.cumprod(dim=1, dtype=torch.int32).sum(dim=1, dtype=torch.int32)


def _greedy_dflash2_path(candidate_ids: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
    # Structurally match the C++ greedy selector walk (sample_dflash2_path
    # in runtime/dflash2_worker_impl): argmax the raw FP32 edge logits
    # directly — log_softmax is strictly monotone per row, so normalizing
    # first would draw the same path at extra cost.
    batch, steps, top_k = candidate_ids.shape
    edge_scores = edge_logits.float()
    previous = torch.zeros((batch, 1), dtype=torch.int64, device=candidate_ids.device)
    proposals = []
    for step in range(steps):
        row = edge_scores[:, step].gather(1, previous.unsqueeze(-1).expand(-1, 1, top_k)).squeeze(1)
        previous = row.argmax(dim=-1, keepdim=True)
        proposals.append(candidate_ids[:, step].gather(1, previous).squeeze(1))
    return torch.stack(proposals, dim=1)


def _static_metadata(metadata: AttentionMetadata, pages: int, page_size: int) -> StaticGraphAttentionMetadata:
    # Detach the Host object from the mutable native Slot, retaining its final
    # device views. Bounded tables avoid planning FIA for the full model limit.
    result = _metadata_view(metadata)
    result.block_table = torch.empty(
        (metadata.block_table.shape[0], pages), dtype=torch.int32, device=metadata.block_table.device
    )
    prepared = result.prepared_attention_state
    capacity = pages * page_size
    if metadata.is_chunked_prefill:
        result.kv_seq_lens_host_values = [capacity] * result.block_table.shape[0]
        result.prepared_attention_state = replace(
            prepared,
            block_table=result.block_table,
            actual_seq_kv=list(result.kv_seq_lens_host_values),
            kv_capacity=capacity,
        )
    else:
        result.prepared_attention_state = replace(prepared, block_table=result.block_table)
    return result


class _BlockGraphLifecycle(ABC):
    """Capture, replay and retire one graph without owning its borrowed Slot."""

    def __init__(self, tokens: torch.Tensor, stream: torch.npu.Stream, pool: tuple[int, int]) -> None:
        self.tokens = tokens
        self.stream = stream
        self.pool = pool
        self.graph = None
        self.context: AclGraphCaptureContext | None = None
        self.hidden: torch.Tensor | None = None

    @abstractmethod
    def _run(self) -> torch.Tensor:
        """Enqueue the fixed-width or adaptive round on the active stream."""
        ...

    @torch.inference_mode()
    def capture(self) -> None:
        current = torch.npu.current_stream(self.tokens.device)
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            for _ in range(2):
                self._run()
        self.stream.synchronize()
        current.wait_stream(self.stream)
        self.graph = torch.npu.NPUGraph()
        self.context = AclGraphCaptureContext(self.stream, [])
        # Native task preparation can enqueue another Slot on its own stream.
        # Only this Launch thread participates in the composite capture.
        with (
            torch.npu.stream(self.stream),
            torch.npu.graph(self.graph, pool=self.pool, stream=self.stream, capture_error_mode="thread_local"),
        ):
            self.hidden = self._run()

    @torch.inference_mode()
    def execute(self) -> torch.Tensor:
        current = torch.npu.current_stream(self.tokens.device)
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            for task in self.context.tasks:
                torch.npu.graph_task_update_begin(self.stream, task.handle)
                task.update()
                torch.npu.graph_task_update_end(self.stream)
                task.event.record(self.stream)
            self.graph.replay()
        current.wait_stream(self.stream)
        return self.hidden

    def close(self) -> None:
        self.stream.synchronize()
        torch.npu.current_stream(self.tokens.device).synchronize()
        self.graph.reset()


class UnifiedBlockAclGraph(_BlockGraphLifecycle):
    """One shape/binding; task retirement protects borrowed output buffers."""

    def __init__(
        self,
        target: ModelExecutor,
        draft: ModelExecutor,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark: bool,
        pages: int,
        activate_target: Callable[[], None],
        activate_draft: Callable[[], None],
        synchronize_tokens: Callable[[torch.Tensor], None],
        stream: torch.npu.Stream,
        pool: tuple[int, int],
    ) -> None:
        super().__init__(tokens, stream, pool)
        self.target = target
        self.draft = draft
        self.target_binding = target_binding
        self.draft_binding = draft_binding
        self.lengths = lengths
        self.dspark = dspark
        self._dflash2_candidates = getattr(draft.model, "dflash2_candidates", None)
        self.activate_target = activate_target
        self.activate_draft = activate_draft
        self.synchronize_tokens = synchronize_tokens
        self.target_backend = target.create_mtp_role_backends(1)[0]
        self.draft_backend = draft.create_mtp_role_backends(1)[0]
        self.target_metadata = _static_metadata(target_binding[2], pages, self.target_backend.page_size)
        self.draft_metadata = _static_metadata(draft_binding[2], pages, self.draft_backend.page_size)
        self.target_state = AclGraphExecutionState({})
        self.draft_state = AclGraphExecutionState({})
        self.draft_caches = [(cache.key, cache.value) for cache in draft.eager_runner.layer_caches]
        self.offsets = torch.arange(tokens.shape[1], device=tokens.device).view(1, -1)

    def _run(self) -> torch.Tensor:
        batch, width = self.tokens.shape
        steps = width - 1
        target_ids, target_positions, target_metadata = self.target_binding
        draft_ids, draft_positions, draft_metadata = self.draft_binding
        for static, source in (
            (self.target_metadata, target_metadata),
            (self.draft_metadata, draft_metadata),
        ):
            static.block_table.copy_(source.block_table[:, : static.block_table.shape[1]])

        self.activate_draft()
        draft_output = self.draft.execute_mtp_role(
            draft_ids,
            draft_positions,
            self.draft_metadata,
            acl_graph=self.context,
            execution_state=self.draft_state,
            attention_backend=self.draft_backend,
        )
        draft_hidden = draft_output[0] if isinstance(draft_output, tuple) else draft_output
        draft_width = steps if self.dspark else width
        draft_hidden = draft_hidden.view(batch, draft_width, -1)
        if not self.dspark:
            draft_hidden = draft_hidden[:, 1:]
        logits = self.draft.model.compute_logits(draft_hidden.reshape(batch * steps, -1), None)
        logits = logits.view(batch, steps, -1)
        if self.dspark:
            previous = draft_ids.view(batch, steps)[:, 0].to(torch.int64)
            proposals = []
            for step in range(steps):
                bias = self.draft.model.dspark_markov_bias(previous)
                previous = (logits[:, step] + bias).argmax(dim=-1)
                proposals.append(previous)
            proposal = torch.stack(proposals, dim=1)
        elif self._dflash2_candidates is not None:
            candidates, edge_logits = self._dflash2_candidates(
                draft_hidden, logits, draft_ids.view(batch, width)[:, 0].to(torch.int64)
            )
            proposal = _greedy_dflash2_path(candidates, edge_logits)
        else:
            proposal = logits.argmax(dim=-1)
        target_ids[: batch * width].view(batch, width)[:, 1:].copy_(proposal)

        self.activate_target()
        target_output = self.target.execute_mtp_role(
            target_ids,
            target_positions,
            self.target_metadata,
            acl_graph=self.context,
            execution_state=self.target_state,
            attention_backend=self.target_backend,
        )
        if not isinstance(target_output, tuple) or len(target_output) < 2:
            raise ValueError("Unified block draft requires target auxiliary hidden states")
        target_hidden, context_hidden = target_output[:2]
        context_hidden = context_hidden[: batch * width]
        target_tokens = self.target.model.compute_greedy_tokens(target_hidden[: batch * width]).view(batch, width)
        matched = _accepted_prefix_lengths(proposal, target_tokens)
        self.tokens.copy_(target_tokens.masked_fill(self.offsets > matched[:, None], -1))
        # Match block task consensus before publishing accepted context on any rank.
        self.synchronize_tokens(self.tokens)
        accepted = self.tokens >= 0
        self.lengths.copy_(accepted.sum(dim=1, dtype=torch.int32))
        slots = self.target_metadata.slot_mapping[: batch * width].masked_fill(~accepted.flatten(), -1)
        self.activate_draft()
        self.draft.model.write_context_kv(
            context_hidden, target_positions[: batch * width], slots, self.draft_caches, None
        )
        return context_hidden


class UnifiedBlockGraphRegistry:
    """Bounded graph lifetime, shared by the target/draft executor pair."""

    _MAX_VARIANTS = 8

    def __init__(
        self,
        target: ModelExecutor,
        draft: ModelExecutor,
        activate_target: Callable[[], None],
        activate_draft: Callable[[], None],
        synchronize_tokens: Callable[[torch.Tensor], None],
        capture: Callable[[UnifiedBlockAclGraph], None],
    ) -> None:
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._capture_seconds = 0.0
        self.target = target
        self.draft = draft
        self.activate_target = activate_target
        self.activate_draft = activate_draft
        self.synchronize_tokens = synchronize_tokens
        self.capture = capture
        # Launch serializes variants and consumes their outputs before reuse.
        # A shared stream/pool avoids an expandable allocator segment per variant.
        self.stream = torch.npu.Stream(device=target.eager_runner.device)
        self.pool = torch.npu.graph_pool_handle()
        self.variants: OrderedDict[tuple[object, ...], UnifiedBlockAclGraph] = OrderedDict()

    def execute(
        self,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark: bool,
    ) -> torch.Tensor:
        page_size = self.draft.attention_backend.page_size
        upper = max(target_binding[2].kv_seq_lens_host_values)
        pages = min(1 << ((upper + page_size - 1) // page_size - 1).bit_length(), draft_binding[2].block_table.shape[1])
        key = (
            PreparedAclGraphRunner._prepared_binding(*target_binding),
            PreparedAclGraphRunner._prepared_binding(*draft_binding),
            tokens.data_ptr(),
            tuple(tokens.shape),
            lengths.data_ptr(),
            dspark,
            pages,
        )
        runner = self.variants.get(key)
        if runner is None:
            self._misses += 1
            if len(self.variants) >= self._MAX_VARIANTS:
                _, retired = self.variants.popitem(last=False)
                retired.close()
                del retired
                self._evictions += 1
            runner = UnifiedBlockAclGraph(
                self.target,
                self.draft,
                target_binding,
                draft_binding,
                tokens,
                lengths,
                dspark,
                pages,
                self.activate_target,
                self.activate_draft,
                self.synchronize_tokens,
                self.stream,
                self.pool,
            )
            started = perf_counter()
            self.capture(runner)
            elapsed = perf_counter() - started
            self._capture_seconds += elapsed
            self.variants[key] = runner
            logger.info(
                "Unified block graph capture: algorithm=%s batch=%d K=%d pages=%d seconds=%.3f cache=%s",
                "DSpark" if dspark else "DFlash2" if runner._dflash2_candidates is not None else "DFlash",
                tokens.shape[0],
                tokens.shape[1] - 1,
                pages,
                elapsed,
                self.cache_info(),
            )
        else:
            self._hits += 1
            self.variants.move_to_end(key)
        return runner.execute()

    def cache_info(self) -> dict[str, int | float]:
        return {
            "capacity": self._MAX_VARIANTS,
            "size": len(self.variants),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "capture_seconds": self._capture_seconds,
        }

    def close(self) -> None:
        logger.info("Unified block graph cache retired: %s", self.cache_info())
        for runner in self.variants.values():
            runner.close()
        self.variants.clear()


class _AdaptiveUnifiedBlockAclGraph(_BlockGraphLifecycle):
    """One actual B, physical Draft/Target bucket and borrowed native Slot."""

    def __init__(
        self,
        target: ModelExecutor,
        draft: ModelExecutor,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark_adaptive: DSparkAdaptiveGraphInputs,
        context_slots: torch.Tensor,
        pages: int,
        activate_target: Callable[[], None],
        activate_draft: Callable[[], None],
        synchronize_tokens: Callable[[torch.Tensor], None],
        stream: torch.npu.Stream,
        pool: tuple[int, int],
    ) -> None:
        super().__init__(tokens, stream, pool)
        self.target = target
        self.draft = draft
        self.target_binding = target_binding
        self.draft_binding = draft_binding
        self.lengths = lengths
        self.inputs = dspark_adaptive
        self.context_slots = context_slots
        self.batch = tokens.shape[0]
        self.draft_width = dspark_adaptive.draft_width
        self.graph_rows = target_binding[0].numel()
        self.pages = pages
        self.target_kv_bound = max(target_binding[2].kv_seq_lens_host_values)
        self.draft_kv_bound = pages * draft.attention_backend.page_size
        self.activate_target = activate_target
        self.activate_draft = activate_draft
        self.synchronize_tokens = synchronize_tokens
        self.target_backend = target.create_mtp_role_backends(1)[0]
        self.draft_backend = draft.create_mtp_role_backends(1)[0]
        # Packing writes the final Target table directly. A compact shadow here
        # would either retain stale rows or require another table copy.
        self.target_metadata = _metadata_view(target_binding[2])
        self.target_metadata.kv_seq_lens_host_values = list(target_binding[2].kv_seq_lens_host_values)
        self.draft_metadata = _metadata_view(draft_binding[2])
        self.draft_metadata.block_table = torch.empty(
            (draft_binding[2].block_table.shape[0], pages), dtype=torch.int32, device=draft_binding[0].device
        )
        self.draft_metadata.q_cu_seq_lens_host_values = list(draft_binding[2].q_cu_seq_lens_host_values)
        self.draft_metadata.kv_seq_lens_host_values = [self.draft_kv_bound] * self.draft_metadata.block_table.shape[0]
        # Let the backend retain its own typed state: MLA uses Device KV
        # tensors, while paged FIA also stores its conservative Host capacity.
        self.draft_metadata.prepared_attention_state = self.draft_backend.prepare_metadata(
            self.draft_metadata, device_kv_lengths=True
        )
        self.target_state = AclGraphExecutionState({})
        self.draft_state = AclGraphExecutionState({})
        self.draft_caches = [(cache.key, cache.value) for cache in draft.eager_runner.layer_caches]
        arena = self.inputs.arena
        proposal_bytes = 8 * self.draft_width * self.batch
        consensus_bytes = 12 * self.draft_width * self.batch
        self.proposals = arena[:proposal_bytes].view(torch.int64).view(self.batch, self.draft_width)
        self.confidence = arena[proposal_bytes:consensus_bytes].view(torch.float32).view(self.batch, self.draft_width)
        self.widths = arena[consensus_bytes : consensus_bytes + 4 * self.batch].view(torch.int32)
        self.consensus = arena[:consensus_bytes]
        self.offsets = self.inputs.offsets[: self.batch + 1]

    def _run(self) -> torch.Tensor:
        target_ids, target_positions, _ = self.target_binding
        draft_ids, draft_positions, draft_metadata = self.draft_binding
        self.draft_metadata.block_table.copy_(draft_metadata.block_table[:, : self.pages])
        # Native patches fresh Draft slots before every replay. The same graph
        # uses the device control's high word for harmless startup warmup.
        self.draft_metadata.slot_mapping.masked_fill_(self.inputs.control >= 2**32, -1)
        self.activate_draft()
        output = self.draft.execute_mtp_role(
            draft_ids,
            draft_positions,
            self.draft_metadata,
            acl_graph=self.context,
            execution_state=self.draft_state,
            attention_backend=self.draft_backend,
        )
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden[: self.batch * self.draft_width].view(self.batch, self.draft_width, -1)
        logits = self.draft.model.compute_logits(hidden.reshape(self.batch * self.draft_width, -1), None)
        logits = logits.view(self.batch, self.draft_width, -1)
        anchor = draft_ids[: self.batch * self.draft_width].view(self.batch, self.draft_width)[:, 0].to(torch.int64)
        previous = anchor
        proposals = []
        for step in range(self.draft_width):
            previous = (logits[:, step] + self.draft.model.dspark_markov_bias(previous)).argmax(dim=-1)
            proposals.append(previous)
        self.proposals.copy_(torch.stack(proposals, dim=1))
        previous_matrix = torch.cat((anchor[:, None], self.proposals[:, : self.draft_width - 1]), dim=1)
        self.confidence.copy_(self.draft.model.dspark_confidence_probs(hidden, previous_matrix))
        # Confidence is part of the consensus payload, not a Host decision.
        self.synchronize_tokens(self.consensus)

        inputs = self.inputs
        torch.ops.xllm_ops.dspark_allocate_verify_prefixes_graph(
            inputs.arena, inputs.control, self.graph_rows, self.draft_width
        )
        torch.ops.xllm_ops.dspark_pack_verify_inputs_graph(
            inputs.anchors,
            inputs.arena,
            inputs.base_positions,
            inputs.base_kv_lengths,
            inputs.base_block_table,
            inputs.control,
            inputs.offsets,
            target_ids,
            target_positions,
            self.target_metadata.kv_seq_lens,
            self.target_metadata.slot_mapping,
            self.target_metadata.block_table,
            self.target_metadata.q_seq_lens,
            inputs.owners,
            inputs.columns,
            inputs.block_size,
            self.draft_width,
        )
        self.activate_target()
        target_output = self.target.execute_mtp_role(
            target_ids,
            target_positions,
            self.target_metadata,
            acl_graph=self.context,
            execution_state=self.target_state,
            attention_backend=self.target_backend,
        )
        if not isinstance(target_output, tuple) or len(target_output) < 2:
            raise ValueError("Adaptive unified DSpark requires Target auxiliary hidden states")
        target_hidden, context_hidden = target_output[:2]
        context_hidden = context_hidden[: self.graph_rows]
        target_tokens = self.target.model.compute_greedy_tokens(target_hidden[: self.graph_rows])
        torch.ops.xllm_ops.dspark_greedy_verify_commit(
            self.proposals,
            target_tokens,
            self.widths,
            self.offsets,
            inputs.owners,
            inputs.columns,
            self.target_metadata.slot_mapping,
            self.tokens,
            self.lengths,
            self.context_slots,
        )
        self.activate_draft()
        self.draft.model.write_context_kv(context_hidden, target_positions, self.context_slots, self.draft_caches, None)
        return context_hidden


class AdaptiveUnifiedBlockGraphRegistry:
    """Capture finite Slot/B/G bindings at startup; serving only replays them."""

    _MAX_SLOTS = 2

    def __init__(
        self,
        target: ModelExecutor,
        draft: ModelExecutor,
        activate_target: Callable[[], None],
        activate_draft: Callable[[], None],
        synchronize_tokens: Callable[[torch.Tensor], None],
        capture: Callable[[_AdaptiveUnifiedBlockAclGraph], None],
    ) -> None:
        self.target = target
        self.draft = draft
        self.activate_target = activate_target
        self.activate_draft = activate_draft
        self.synchronize_tokens = synchronize_tokens
        self.capture = capture
        self.stream = torch.npu.Stream(device=target.eager_runner.device)
        self.pool = torch.npu.graph_pool_handle()
        self.variants: dict[tuple[object, ...], _AdaptiveUnifiedBlockAclGraph] = {}
        self._slots: dict[int, tuple[TensorDesc, ...]] = {}
        self._capacity = 0
        self._hits = 0
        self._misses = 0
        self._capture_seconds = 0.0

    def _binding(
        self,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark_adaptive: DSparkAdaptiveGraphInputs,
        context_slots: torch.Tensor,
    ) -> tuple[object, ...]:
        target_runner = self.target.prepared_graph_runner
        draft_runner = self.draft.prepared_graph_runner
        if target_runner is None or draft_runner is None:
            raise ValueError("Adaptive unified DSpark requires prepared Target and Draft graph runners")
        target_runner._validate_prepared_input(*target_binding, dspark_adaptive=dspark_adaptive)
        draft_runner._validate_prepared_input(*draft_binding)
        batch = tokens.shape[0] if tokens.ndim == 2 else 0
        draft_width = dspark_adaptive.draft_width
        capacity = dspark_adaptive.anchors.numel()
        graph_rows = target_binding[0].numel()
        maximum = target_runner.max_batch
        if not 1 <= batch <= capacity:
            raise ValueError("Adaptive unified DSpark requires a supported actual batch and Target graph bucket")
        if graph_rows not in _verify_buckets(batch, maximum, draft_width) or graph_rows < batch:
            raise ValueError("Adaptive unified DSpark Target bucket must cover anchors within its configured width")
        if (
            draft_binding[0].numel() < batch * draft_width
            or draft_binding[0].numel() != draft_binding[2].block_table.shape[0] * draft_width
        ):
            raise ValueError("Adaptive unified DSpark Draft query rows must match its configured width")
        if draft_runner.dp_size != 1 or not draft_runner._block_draft or not draft_binding[2].is_chunked_prefill:
            raise ValueError("Adaptive unified DSpark requires a prepared block Draft with DP=1")
        for name, tensor, shape, dtype in (
            ("result tokens", tokens, (batch, draft_width + 1), torch.int64),
            ("result lengths", lengths, (batch,), torch.int32),
            ("context slots", context_slots, (graph_rows,), torch.int32),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != target_binding[0].device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"Adaptive unified DSpark requires contiguous {dtype} {name} with shape {shape}")
        return (
            PreparedAclGraphRunner._prepared_binding(*target_binding, dspark_adaptive=dspark_adaptive),
            PreparedAclGraphRunner._prepared_binding(*draft_binding),
            _tensor_binding(tokens),
            _tensor_binding(lengths),
            _tensor_binding(context_slots),
        )

    @staticmethod
    def _validate_kv_bounds(
        runner: _AdaptiveUnifiedBlockAclGraph, target_binding: Binding, draft_binding: Binding
    ) -> None:
        if max(target_binding[2].kv_seq_lens_host_values) > runner.target_kv_bound:
            raise ValueError("Adaptive unified DSpark Target KV bound exceeds its startup binding")
        if max(draft_binding[2].kv_seq_lens_host_values) > runner.draft_kv_bound:
            raise ValueError("Adaptive unified DSpark Draft KV bound exceeds its startup binding")

    def warmup(
        self,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark_adaptive: DSparkAdaptiveGraphInputs,
        context_slots: torch.Tensor,
    ) -> None:
        key = self._binding(target_binding, draft_binding, tokens, lengths, dspark_adaptive, context_slots)
        if key in self.variants:
            self._validate_kv_bounds(self.variants[key], target_binding, draft_binding)
            return
        capacity = dspark_adaptive.anchors.numel()
        maximum = self.target.prepared_graph_runner.max_batch
        slot = dspark_adaptive.arena.data_ptr()
        identity = tuple(
            _tensor_binding(tensor)
            for tensor in (dspark_adaptive.arena, dspark_adaptive.control, dspark_adaptive.anchors)
        )
        if slot not in self._slots and len(self._slots) == self._MAX_SLOTS:
            raise ValueError("Adaptive unified DSpark exceeds its native Slot capacity")
        if slot in self._slots and self._slots[slot] != identity:
            raise ValueError("Adaptive unified DSpark Slot storage changed after startup binding")
        limit = self._MAX_SLOTS * sum(
            len(_verify_buckets(batch, maximum, dspark_adaptive.draft_width)) for batch in range(1, capacity + 1)
        )
        if self._capacity and self._capacity != limit:
            raise ValueError("Adaptive unified DSpark request capacity changed after startup")
        if len(self.variants) >= limit:
            raise ValueError("Adaptive unified DSpark exceeds its finite startup variant capacity")
        page_size = self.draft.attention_backend.page_size
        upper = max(draft_binding[2].kv_seq_lens_host_values)
        required_pages = (upper + page_size - 1) // page_size
        pages = min(max(32, 1 << (required_pages - 1).bit_length()), draft_binding[2].block_table.shape[1])
        runner = _AdaptiveUnifiedBlockAclGraph(
            self.target,
            self.draft,
            target_binding,
            draft_binding,
            tokens,
            lengths,
            dspark_adaptive,
            context_slots,
            pages,
            self.activate_target,
            self.activate_draft,
            self.synchronize_tokens,
            self.stream,
            self.pool,
        )
        started = perf_counter()
        self.capture(runner)
        self._capture_seconds += perf_counter() - started
        self._slots[slot] = identity
        self._capacity = limit
        self.variants[key] = runner
        logger.info(
            "Adaptive DSpark whole-round graph captured: B=%d Draft=%d G=%d pages=%d cache=%s",
            tokens.shape[0],
            draft_binding[0].numel(),
            target_binding[0].numel(),
            pages,
            self.cache_info(),
        )

    def execute(
        self,
        target_binding: Binding,
        draft_binding: Binding,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        dspark_adaptive: DSparkAdaptiveGraphInputs,
        context_slots: torch.Tensor,
    ) -> torch.Tensor:
        key = self._binding(target_binding, draft_binding, tokens, lengths, dspark_adaptive, context_slots)
        runner = self.variants.get(key)
        if runner is None:
            self._misses += 1
            raise ValueError("Adaptive unified DSpark binding was not captured during startup warmup")
        self._validate_kv_bounds(runner, target_binding, draft_binding)
        self._hits += 1
        return runner.execute()

    def cache_info(self) -> dict[str, int | float]:
        return {
            "capacity": self._capacity,
            "size": len(self.variants),
            "slots": len(self._slots),
            "hits": self._hits,
            "misses": self._misses,
            "capture_seconds": self._capture_seconds,
        }

    def close(self) -> None:
        logger.info("Adaptive DSpark whole-round graphs retired: %s", self.cache_info())
        for runner in self.variants.values():
            runner.close()
        self.variants.clear()
        self._slots.clear()

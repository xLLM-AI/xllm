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

"""A complete greedy DFlash/DFlash2/DSpark round over native task Slot buffers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import fields, replace
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.forward_context import AclGraphCaptureContext, AclGraphExecutionState
from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner

if TYPE_CHECKING:
    from xllm.python.model_executor.executor import ModelExecutor

Binding = tuple[torch.Tensor, torch.Tensor, AttentionMetadata]


def _accepted_prefix_lengths(proposal: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    # aclnnCumprod requires a numeric input even when its output dtype is set.
    matches = (proposal == target_tokens[:, : proposal.shape[1]]).to(torch.int32)
    return matches.cumprod(dim=1, dtype=torch.int32).sum(dim=1, dtype=torch.int32)


def _greedy_dflash2_path(candidate_ids: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
    # Match sample_dflash2_path's FP32 normalization and predecessor ordering.
    batch, steps, top_k = candidate_ids.shape
    edge_log_probs = edge_logits.float().log_softmax(dim=-1)
    previous = torch.zeros((batch, 1), dtype=torch.int64, device=candidate_ids.device)
    proposals = []
    for step in range(steps):
        row = edge_log_probs[:, step].gather(1, previous.unsqueeze(-1).expand(-1, 1, top_k)).squeeze(1)
        previous = row.argmax(dim=-1, keepdim=True)
        proposals.append(candidate_ids[:, step].gather(1, previous).squeeze(1))
    return torch.stack(proposals, dim=1)


def _static_metadata(metadata: AttentionMetadata, pages: int, page_size: int) -> StaticGraphAttentionMetadata:
    # Detach the Host object from the mutable native Slot, retaining its final
    # device views. Bounded tables avoid planning FIA for the full model limit.
    values = {
        field.name: getattr(metadata, field.name)
        for field in fields(StaticGraphAttentionMetadata)
        if hasattr(metadata, field.name)
    }
    result = StaticGraphAttentionMetadata(**values)
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


class UnifiedBlockAclGraph:
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
        self.target = target
        self.draft = draft
        self.target_binding = target_binding
        self.draft_binding = draft_binding
        self.tokens = tokens
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
        self.stream = stream
        self.pool = pool
        self.graph = None
        self.context: AclGraphCaptureContext | None = None
        self.hidden: torch.Tensor | None = None

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

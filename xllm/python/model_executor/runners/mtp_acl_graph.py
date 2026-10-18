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

"""Single-replay MTP graph recipe.

The regular model runners deliberately stop after the transformer body.  MTP
needs a different lifetime: K draft body/head/sample steps feed one target
verification body/head/sample, followed by acceptance and next-state update.
This module owns that fixed-shape recipe without putting speculative control
flow in a model's ``forward`` method.

The model-specific adapter is supplied as callables.  It is responsible for
installing the correct role-scoped ``ForwardContext`` and for translating the
fixed row layout into the target/draft attention metadata.  The recipe itself
only uses fixed-shape tensors and device operations, so it can be captured by
``torch.npu.NPUGraph`` once the adapter is graph-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
)
from xllm.python.model_executor.runners.base import SpeculativeRuntimeOutput
from xllm.python.model_executor.runners.mtp_sparse_metadata import (
    MtpSparseMetadataBinding,
    MtpSparseMetadataStorage,
    MtpSparsePositionStorage,
    SparseMetadataFactory,
)

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.model_executor.forward_context import LayerSynchronizer

GraphBackend = Literal["eager", "aclgraph"]
ForwardFn = Callable[
    [torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None],
    torch.Tensor | tuple[torch.Tensor, ...],
]
LogitsFn = Callable[[torch.Tensor], torch.Tensor]
ActivateFn = Callable[[], None]


@dataclass(frozen=True)
class MtpSamplingPlan:
    batch_size: int
    logprobs: bool = False
    max_top_logprobs: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("MTP sampling batch_size must be positive")
        if self.max_top_logprobs < 0:
            raise ValueError("max_top_logprobs must be nonnegative")


class MtpRoleAdapter:
    """Role-scoped body adapter for one composite MTP graph.

    Each role owns a separate ``ModelExecutor`` and therefore a separate
    attention backend, layer-cache list, and ``ForwardContext``.  The adapter
    deliberately calls that executor's eager body runner instead of its decode
    graph runner: the outer MTP runner is the graph being captured, so nesting
    another graph runner here would capture the wrong ownership boundary.
    """

    def __init__(
        self,
        executor: object,
        metadata_by_step: Sequence[AttentionMetadata],
        *,
        speculative_tokens: int,
        target: bool = False,
        step_major_layout: bool = False,
        layer_synchronizer: LayerSynchronizer | None = None,
        repair_token_ids: torch.Tensor | None = None,
        metadata_storage: MtpSparseMetadataStorage | None = None,
        repair_positions: torch.Tensor | None = None,
    ) -> None:
        if not metadata_by_step:
            raise ValueError("MTP role adapter requires at least one metadata plan")
        if not hasattr(executor, "eager_runner"):
            raise TypeError("MTP role adapter requires a ModelExecutor-like object")
        if speculative_tokens <= 0:
            raise ValueError("MTP role adapter requires positive fixed K")
        if not target and len(metadata_by_step) != speculative_tokens:
            raise ValueError("draft metadata plan must contain one entry per fixed K step")
        if target and len(metadata_by_step) != 1:
            raise ValueError("target metadata plan must contain one K+1-row entry")
        self._executor = executor
        if metadata_storage is not None:
            if len(metadata_by_step) != len(metadata_storage.metadata) or any(
                actual is not owned for actual, owned in zip(metadata_by_step, metadata_storage.metadata)
            ):
                raise ValueError("sparse MTP role metadata must belong to the supplied storage")
            self._metadata_by_step = metadata_storage.metadata
            self._metadata_arenas = (metadata_storage.arena,)
        else:
            self._metadata_by_step = tuple(metadata_by_step)
            self._metadata_arenas = ()
        self._target = target
        self._step_major_layout = target and step_major_layout
        self._target_width = speculative_tokens + 1
        self._layer_synchronizer = layer_synchronizer
        self._repair_positions = repair_positions
        # Worker may supply a strided int32 view of its two-row fused output.
        # Cast while copying into the final graph-owned int64 destination.
        self._repair_token_ids = (
            torch.empty(repair_token_ids.shape, dtype=torch.long, device=repair_token_ids.device)
            if repair_token_ids is not None
            else None
        )
        if self._repair_token_ids is not None:
            self._repair_token_ids.copy_(repair_token_ids)
        self._repair_selected_rows = (
            None
            if self._repair_token_ids is None or self._repair_token_ids.numel() <= 1
            else torch.arange(
                1,
                self._repair_token_ids.numel() * 2,
                2,
                dtype=torch.long,
                device=self._repair_token_ids.device,
            )
        )
        self._acl_graph: AclGraphCaptureContext | None = None
        self._execution_state: AclGraphExecutionState | None = None
        create_backends = getattr(executor, "create_mtp_role_backends", None)
        self._attention_backends = (
            tuple(create_backends(len(self._metadata_by_step))) if create_backends is not None else ()
        )
        if self._attention_backends and len(self._attention_backends) != len(self._metadata_by_step):
            raise RuntimeError("MTP role backend count does not match metadata plan")
        self._graph_prepared = False
        self._step_execution_states: tuple[AclGraphExecutionState | None, ...] = ()
        self._shared_execution_buffers: dict[tuple[object, ...], object] | None = None

    def bind_graph_context(
        self,
        acl_graph: AclGraphCaptureContext | None,
        execution_state: AclGraphExecutionState | None,
    ) -> None:
        """Bind the role-local context owned by the enclosing graph entry."""
        if acl_graph is not None and not self._attention_backends:
            raise RuntimeError("MTP ACL graph requires one pre-prepared attention backend per fixed role metadata plan")
        self._acl_graph = acl_graph
        self._execution_state = execution_state
        if execution_state is None:
            self._step_execution_states = tuple(None for _ in self._metadata_by_step)
        elif not self._step_execution_states:
            # Keep outputs and graph-visible metadata private to each step,
            # but share the role-owned temporary attention workspace. Draft
            # steps execute serially on the capture stream, so a workspace is
            # dead before the next step consumes the same slice.
            self._shared_execution_buffers = {}
            self._step_execution_states = tuple(
                AclGraphExecutionState({}, shared_persistent_buffers=self._shared_execution_buffers)
                for _ in self._metadata_by_step
            )
        elif len(self._step_execution_states) != len(self._metadata_by_step):
            raise RuntimeError("MTP role execution-state plan changed after warmup")

    def finish_warmup(self) -> None:
        """Mark fixed role backends ready for graph capture and replay."""
        if self._attention_backends:
            self._graph_prepared = True

    def require_direct_metadata_update(self) -> None:
        """Check once that every captured consumer borrows the owned tensors."""
        if not self._graph_prepared or not self._attention_backends:
            raise RuntimeError("sparse MTP binding requires prepared attention backends")
        if not all(getattr(backend, "graph_metadata_updated_in_place", False) for backend in self._attention_backends):
            raise RuntimeError("sparse MTP binding requires in-place graph metadata consumers")

    def update_repair_token_ids(self, repair_token_ids: torch.Tensor | None) -> None:
        if (self._repair_token_ids is None) != (repair_token_ids is None):
            raise RuntimeError("MTP repair-token presence changed after capture")
        if self._repair_token_ids is not None:
            assert repair_token_ids is not None
            if repair_token_ids.shape != self._repair_token_ids.shape:
                raise RuntimeError("MTP repair-token shape changed after capture")
            self._repair_token_ids.copy_(repair_token_ids)

    def __call__(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        metadata_index = 0 if self._target else step
        if metadata_index < 0 or metadata_index >= len(self._metadata_by_step):
            raise ValueError(f"MTP role adapter received invalid step={step}")
        call_token_ids = token_ids
        call_positions = positions
        call_embedding = input_embedding
        selected_rows: torch.Tensor | None = None
        select_current_rows = False
        target_batch = token_ids.numel() // self._target_width if self._step_major_layout else 0
        if target_batch > 1:
            call_token_ids = token_ids.view(target_batch, self._target_width).transpose(0, 1).reshape(-1)
            call_positions = positions.view(target_batch, self._target_width).transpose(0, 1).reshape(-1)
        if not self._target and step == 0 and self._repair_token_ids is not None:
            if self._repair_token_ids.shape != token_ids.shape:
                raise ValueError("MTP repair token ids must match the draft batch")
            if input_embedding is None or input_embedding.shape[0] != token_ids.numel() * 2:
                raise ValueError("MTP first draft repair layout requires 2B embeddings")
            call_token_ids = torch.stack((self._repair_token_ids, token_ids), dim=1).reshape(-1)
            call_positions = (
                self._repair_positions
                if self._repair_positions is not None
                else torch.stack((positions - 1, positions), dim=1).reshape(-1)
            )
            select_current_rows = True
            if token_ids.numel() > 1:
                if self._repair_selected_rows is None or self._repair_selected_rows.device != token_ids.device:
                    raise RuntimeError("MTP repair row selection was not prepared for this device")
                selected_rows = self._repair_selected_rows
        execute_role = getattr(self._executor, "execute_mtp_role", None)
        if execute_role is None:
            if self._acl_graph is not None or self._execution_state is not None:
                raise TypeError("MTP graph adapter executor lacks execute_mtp_role")
            output = self._executor.eager_runner.execute(
                call_token_ids,
                call_positions,
                self._metadata_by_step[metadata_index],
                call_embedding,
                self._layer_synchronizer,
                topk_indices,
            )
        else:
            execution_state = (
                self._step_execution_states[metadata_index] if self._step_execution_states else self._execution_state
            )
            output = execute_role(
                call_token_ids,
                call_positions,
                self._metadata_by_step[metadata_index],
                call_embedding,
                self._layer_synchronizer,
                topk_indices,
                acl_graph=self._acl_graph,
                execution_state=execution_state,
                attention_backend=(self._attention_backends[metadata_index] if self._attention_backends else None),
                skip_prepare=self._graph_prepared,
            )
        if not select_current_rows and target_batch <= 1:
            return output

        def select_rows(value: torch.Tensor) -> torch.Tensor:
            if target_batch > 1:
                # Acceptance and recurrent state use request-major rows.
                return (
                    value.reshape(self._target_width, target_batch, *value.shape[1:])
                    .transpose(0, 1)
                    .reshape(value.shape)
                )
            # B1's current row is a single contiguous view. Avoid materializing
            # an index tensor and launching a gather for this common path;
            # batches larger than one still need the strided row selection.
            return value.narrow(0, 1, 1) if selected_rows is None else value.index_select(0, selected_rows)

        if isinstance(output, tuple):
            hidden = select_rows(output[0])
            if len(output) >= 3 and isinstance(output[2], torch.Tensor):
                return hidden, output[1], select_rows(output[2])
            return hidden, output[1]
        return select_rows(output)


@dataclass(frozen=True)
class MtpBodyOutput:
    """Body result needed by the next speculative invocation."""

    hidden: torch.Tensor
    topk_indices: torch.Tensor | None = None


def _body_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> MtpBodyOutput:
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("MTP model adapter must return a tensor or a tensor-first tuple")
        topk_indices = output[2] if len(output) >= 3 and isinstance(output[2], torch.Tensor) else None
        return MtpBodyOutput(output[0], topk_indices)
    return MtpBodyOutput(output)


def _runtime_token_views(token_state: torch.Tensor, batch: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_bytes = batch * (steps + 1) * 8
    tokens = token_state[:token_bytes].view(torch.int64).view(batch, steps + 1)
    counts = token_state[token_bytes:].view(torch.int32)
    return tokens, counts


class MtpGraphRecipe(nn.Module):
    """K-unrolled draft/target recipe used by eager oracle and ACL graph."""

    def __init__(
        self,
        draft_forward: ForwardFn,
        draft_logits: LogitsFn,
        target_forward: ForwardFn,
        target_logits: LogitsFn,
        *,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        device: torch.device,
        kv_seq_lens: torch.Tensor | None = None,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        target_sampling: MtpSamplingPlan | None = None,
        draft_greedy: LogitsFn | None = None,
        target_greedy: LogitsFn | None = None,
        position_storage: MtpSparsePositionStorage | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if speculative_tokens <= 0:
            raise ValueError("speculative_tokens must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if kv_seq_lens is not None and tuple(kv_seq_lens.shape) != (batch_size,):
            raise ValueError("kv_seq_lens must have one value per request")
        self.batch_size = batch_size
        self.speculative_tokens = speculative_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.draft_forward = draft_forward
        self.draft_logits = draft_logits
        self.target_forward = target_forward
        self.target_logits = target_logits
        self.draft_greedy = draft_greedy
        self.target_greedy = target_greedy
        self.draft_activate = draft_activate
        self.target_activate = target_activate
        self._position_storage = position_storage
        self.requires_kv_seq_lens = kv_seq_lens is not None
        if target_sampling is not None and target_sampling.batch_size != batch_size:
            raise ValueError("target sampling plan batch does not match MTP graph batch")
        self.target_sampling = target_sampling
        self.register_buffer(
            "_position_arena",
            position_storage.positions
            if position_storage is not None
            else torch.empty((2 * speculative_tokens + 1) * batch_size, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_target_position_offsets",
            torch.arange(speculative_tokens + 1, dtype=torch.long, device=device).unsqueeze(0)
            if position_storage is None
            else torch.empty(0, dtype=torch.long, device=device),
            persistent=False,
        )

    def bind_graph_contexts(
        self,
        acl_graph: AclGraphCaptureContext | None,
        draft_state: AclGraphExecutionState | None,
        target_state: AclGraphExecutionState | None,
    ) -> None:
        """Bind role-local graph state without assuming adapter internals."""
        for forward, state in (
            (self.draft_forward, draft_state),
            (self.target_forward, target_state),
        ):
            bind_context = getattr(forward, "bind_graph_context", None)
            if bind_context is not None:
                bind_context(acl_graph, state)

    def finish_warmup(self) -> None:
        """Freeze role backend metadata before entering ACL graph capture."""
        for forward in (self.draft_forward, self.target_forward):
            finish_warmup = getattr(forward, "finish_warmup", None)
            if finish_warmup is not None:
                finish_warmup()

    @torch.inference_mode()
    def forward(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> SpeculativeRuntimeOutput:
        draft_tokens: list[torch.Tensor] = []
        current_ids = seed_token_ids
        current_embedding = draft_input_embedding
        current_topk_indices = draft_topk_indices
        use_draft_greedy = self.draft_greedy is not None
        for step in range(self.speculative_tokens):
            if self.draft_activate is not None:
                self.draft_activate()
            step_positions = self._position_arena[step * self.batch_size : (step + 1) * self.batch_size]
            if self._position_storage is None:
                torch.add(base_positions, step, out=step_positions)
            draft_output = _body_output(
                self.draft_forward(
                    current_ids,
                    step_positions,
                    step,
                    current_embedding,
                    current_topk_indices,
                )
            )
            draft_hidden = draft_output.hidden
            step_logits = None if use_draft_greedy else self.draft_logits(draft_hidden)
            if use_draft_greedy:
                current_ids = self.draft_greedy(draft_hidden)
            elif step_logits.shape != (self.batch_size, self.vocab_size):
                raise ValueError("draft logits must have shape [batch, vocab]")
            else:
                current_ids = step_logits.argmax(dim=-1)
            # GLM MTP consumes the preceding body hidden as the next-step
            # embedding and optionally reuses the preceding DSA top-k state.
            current_embedding = draft_hidden
            current_topk_indices = draft_output.topk_indices
            draft_tokens.append(current_ids)
            # Release logits before the next body allocates its
            # workspace. The capture pool can reuse storage once it is dead.
            del step_logits, draft_hidden, draft_output
        del current_embedding, current_topk_indices
        draft_matrix = torch.stack(draft_tokens, dim=1)

        target_token_matrix = torch.cat((seed_token_ids.unsqueeze(1), draft_matrix), dim=1)
        target_ids = target_token_matrix.reshape(-1)
        target_positions = self._position_arena[self.speculative_tokens * self.batch_size :]
        if self._position_storage is None:
            torch.add(
                base_positions.unsqueeze(1),
                self._target_position_offsets,
                out=target_positions.view(self.batch_size, self.speculative_tokens + 1),
            )
        if self.target_activate is not None:
            self.target_activate()
        target_output = _body_output(self.target_forward(target_ids, target_positions, -1, None, None))
        target_hidden = target_output.hidden
        if target_hidden.ndim != 2 or target_hidden.shape[0] != self.batch_size * (self.speculative_tokens + 1):
            raise ValueError("target hidden must have one row per target verification token")
        use_target_greedy = self.target_greedy is not None and (
            self.target_sampling is None or not (self.target_sampling.logprobs or self.target_sampling.max_top_logprobs)
        )
        target_logits = None if use_target_greedy else self.target_logits(target_hidden)
        if target_logits is not None and target_logits.shape != (
            self.batch_size * (self.speculative_tokens + 1),
            self.vocab_size,
        ):
            raise ValueError("target logits must have shape [batch*(K+1), vocab]")
        target_log_normalizer = None
        if use_target_greedy:
            target_tokens = self.target_greedy(target_hidden).reshape(self.batch_size, self.speculative_tokens + 1)
        else:
            target_tokens = target_logits.argmax(dim=-1).reshape(self.batch_size, self.speculative_tokens + 1)
            if self.target_sampling is not None and (
                self.target_sampling.logprobs or self.target_sampling.max_top_logprobs
            ):
                # Only committed and top-k scores leave the graph. A scalar
                # normalizer per row avoids a full-vocabulary log-prob tensor.
                target_log_normalizer = torch.logsumexp(target_logits.float(), dim=-1).reshape(
                    self.batch_size, self.speculative_tokens + 1, 1
                )

        token_state, compact_hidden = torch.ops.xllm_ops.mtp_greedy_commit(draft_matrix, target_tokens, target_hidden)
        committed_tokens, accepted_count = _runtime_token_views(token_state, self.batch_size, self.speculative_tokens)
        committed_log_probs = None
        target_top_log_probs = None
        target_top_tokens = None
        score_rows = (
            target_logits.view(self.batch_size, self.speculative_tokens + 1, self.vocab_size)
            if target_log_normalizer is not None
            else None
        )
        if score_rows is not None and self.target_sampling is not None and self.target_sampling.logprobs:
            committed_indices = committed_tokens.clamp_min(0).unsqueeze(-1)
            selected_scores = score_rows.gather(-1, committed_indices).float() - target_log_normalizer
            committed_log_probs = selected_scores.squeeze(-1)
            committed_log_probs = torch.where(
                committed_tokens.ge(0), committed_log_probs, torch.zeros_like(committed_log_probs)
            )
        if score_rows is not None and self.target_sampling is not None and self.target_sampling.max_top_logprobs > 0:
            top_width = min(self.target_sampling.max_top_logprobs, self.vocab_size)
            target_top_log_probs, target_top_tokens = score_rows.topk(top_width, dim=-1)
            target_top_log_probs = target_top_log_probs.float() - target_log_normalizer
        return SpeculativeRuntimeOutput(
            committed_tokens=committed_tokens,
            accepted_count=accepted_count,
            target_embeddings=compact_hidden,
            token_state=token_state,
            logprobs=committed_log_probs,
            top_logprobs=target_top_log_probs,
            top_tokens=target_top_tokens,
        )


class MtpAclGraphRunner:
    """Capture and replay one fixed-shape MTP recipe.

    The runner intentionally requires a graph-safe model adapter.  It does
    not catch capture errors and silently switch to eager execution; callers
    choose ``backend='eager'`` explicitly for the oracle/fallback path.
    """

    def __init__(
        self,
        recipe: MtpGraphRecipe,
        *,
        backend: GraphBackend = "aclgraph",
        warmup_steps: int = 2,
    ) -> None:
        if backend not in ("eager", "aclgraph"):
            raise ValueError(f"unknown MTP graph backend: {backend!r}")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.recipe = recipe
        self.backend = backend
        self.warmup_steps = warmup_steps
        self._closed = False
        self._captured = False
        self._graph = None
        self._capture_stream = None
        self._graph_tasks: list[AclGraphTask] = []
        self._execution_states = (
            (AclGraphExecutionState({}), AclGraphExecutionState({})) if backend == "aclgraph" else (None, None)
        )
        self._static_seed_token_ids = torch.zeros(recipe.batch_size, dtype=torch.long, device=recipe.device)
        self._static_base_positions = torch.zeros(recipe.batch_size, dtype=torch.long, device=recipe.device)
        self._static_draft_input_embedding: torch.Tensor | None = None
        self._static_draft_topk_indices: torch.Tensor | None = None
        self._static_output: SpeculativeRuntimeOutput | None = None

    def close(self) -> None:
        """Drain replay and output copies before releasing graph storage."""
        if self._closed:
            return
        if self._capture_stream is not None:
            self._capture_stream.synchronize()
            # Output snapshots are enqueued on the caller stream after it
            # waits for replay. They must finish before graph.reset frees it.
            torch.npu.current_stream(self.recipe.device).synchronize()
        if self._graph is not None:
            self._graph.reset()
        self._graph = None
        self._captured = False
        self._static_output = None
        self._graph_tasks.clear()
        self.recipe.bind_graph_contexts(None, None, None)
        self._execution_states = (None, None)
        self._capture_stream = None
        self._closed = True

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def draft_embedding_destination(self) -> torch.Tensor | None:
        """Borrow the final input buffer; producer must follow replay's read event."""
        return self._static_draft_input_embedding

    def _validate_inputs(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None,
    ) -> None:
        expected_device = self.recipe.device
        for name, tensor in (("seed_token_ids", seed_token_ids), ("base_positions", base_positions)):
            if tensor.device != expected_device or tensor.shape != (self.recipe.batch_size,):
                raise ValueError(f"{name} must be a [{self.recipe.batch_size}] tensor on {expected_device}")
        token_dtypes = (torch.int32, torch.int64) if self.backend == "aclgraph" else (torch.int64,)
        if seed_token_ids.dtype not in token_dtypes or base_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("MTP graph inputs require integer ids/positions; eager token ids must be int64")
        if self.recipe.requires_kv_seq_lens:
            if kv_seq_lens is None or kv_seq_lens.device != expected_device:
                raise ValueError("kv_seq_lens is required for this graph variant")
            if kv_seq_lens.shape != (self.recipe.batch_size,) or kv_seq_lens.dtype != torch.int32:
                raise ValueError("kv_seq_lens must be an int32 vector matching the graph batch")
        elif kv_seq_lens is not None:
            raise ValueError("this graph variant was captured without kv_seq_lens")

    def _validate_recurrent_inputs(
        self,
        draft_input_embedding: torch.Tensor | None,
        draft_topk_indices: torch.Tensor | None,
    ) -> None:
        if self.backend == "eager":
            return
        expected_device = self.recipe.device
        for name, value, static in (
            ("draft_input_embedding", draft_input_embedding, self._static_draft_input_embedding),
            ("draft_topk_indices", draft_topk_indices, self._static_draft_topk_indices),
        ):
            if static is None:
                if value is not None:
                    raise ValueError(f"{name} was not part of the captured graph variant")
                continue
            if value is None:
                raise ValueError(f"{name} is required for this graph variant")
            if value.device != expected_device or value.shape != static.shape or value.dtype != static.dtype:
                raise ValueError(f"{name} shape, dtype, or device changed after graph capture")

    def _run_static(self) -> SpeculativeRuntimeOutput:
        return self.recipe(
            self._static_seed_token_ids,
            self._static_base_positions,
            self._static_draft_input_embedding,
            self._static_draft_topk_indices,
        )

    @staticmethod
    def _clone_graph_output(
        output: SpeculativeRuntimeOutput,
    ) -> SpeculativeRuntimeOutput:
        """Detach one replay result from the persistent ACL graph buffers.

        ACL graph replay writes the same output allocations on every call.
        Schedule overlap can keep an earlier MTP result alive while the next
        draft or target invocation starts, so returning those allocations
        directly allows a later replay to overwrite data still in use.
        """

        def clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.clone()

        token_state = output.token_state.clone()
        batch, width = output.committed_tokens.shape
        tokens, counts = _runtime_token_views(token_state, batch, width - 1)
        return SpeculativeRuntimeOutput(
            committed_tokens=tokens,
            accepted_count=counts,
            target_embeddings=output.target_embeddings.clone(),
            token_state=token_state,
            logprobs=clone(output.logprobs),
            top_logprobs=clone(output.top_logprobs),
            top_tokens=clone(output.top_tokens),
        )

    def capture(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("MTP graph runner is closed")
        if self.backend != "aclgraph":
            raise RuntimeError("capture requires the ACL graph backend")
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        if draft_input_embedding is not None:
            self._static_draft_input_embedding = torch.empty_like(draft_input_embedding)
        if draft_topk_indices is not None:
            self._static_draft_topk_indices = torch.empty_like(draft_topk_indices)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            raise RuntimeError("ACL graph backend requires torch.npu.NPUGraph")
        self._static_seed_token_ids.copy_(seed_token_ids)
        self._static_base_positions.copy_(base_positions)
        if self._static_draft_input_embedding is not None:
            self._static_draft_input_embedding.copy_(draft_input_embedding)
        if self._static_draft_topk_indices is not None:
            self._static_draft_topk_indices.copy_(draft_topk_indices)
        self._capture_stream = torch.npu.Stream(device=self.recipe.device)
        self._capture_stream.wait_stream(torch.npu.current_stream())
        self.recipe.bind_graph_contexts(None, *self._execution_states)
        with torch.npu.stream(self._capture_stream):
            for _ in range(self.warmup_steps):
                self._run_static()
        self.recipe.finish_warmup()
        self._capture_stream.synchronize()
        torch.npu.current_stream().wait_stream(self._capture_stream)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(self._capture_stream, [])
        self.recipe.bind_graph_contexts(capture_context, *self._execution_states)
        with torch.npu.stream(self._capture_stream), torch.npu.graph(self._graph, stream=self._capture_stream):
            self._static_output = self._run_static()
        self._graph_tasks = capture_context.tasks
        self._captured = True

    def execute(
        self,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        draft_input_embedding: torch.Tensor | None = None,
        draft_topk_indices: torch.Tensor | None = None,
    ) -> SpeculativeRuntimeOutput:
        if self._closed:
            raise RuntimeError("MTP graph runner is closed")
        self._validate_inputs(seed_token_ids, base_positions, kv_seq_lens)
        self._validate_recurrent_inputs(draft_input_embedding, draft_topk_indices)
        if self.backend == "eager":
            output = self.recipe(
                seed_token_ids,
                base_positions,
                draft_input_embedding,
                draft_topk_indices,
            )
        else:
            if not self._captured or self._graph is None or self._capture_stream is None:
                raise RuntimeError("MTP ACL graph must be captured before replay")
            self._static_seed_token_ids.copy_(seed_token_ids)
            # Sparse replay reads the binding's positions directly.
            if self.recipe._position_storage is None:
                self._static_base_positions.copy_(base_positions)
            if self._static_draft_input_embedding is not None:
                if (
                    draft_input_embedding.data_ptr() != self._static_draft_input_embedding.data_ptr()
                    or draft_input_embedding.stride() != self._static_draft_input_embedding.stride()
                ):
                    self._static_draft_input_embedding.copy_(draft_input_embedding)
            if self._static_draft_topk_indices is not None:
                self._static_draft_topk_indices.copy_(draft_topk_indices)
            self._capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self._capture_stream):
                self._update_graph_tasks(self._capture_stream)
                self._graph.replay()
            torch.npu.current_stream().wait_stream(self._capture_stream)
            assert self._static_output is not None
            output = self._clone_graph_output(self._static_output)

        return output

    def _update_graph_tasks(self, stream: object) -> None:
        """Refresh captured attention arguments before serial replay."""
        for task in self._graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            task.update()
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)


@dataclass
class _SparseVariant:
    runner: MtpAclGraphRunner
    binding: MtpSparseMetadataBinding


class MtpGraphVariantRegistry:
    """Own captured MTP recipes and their FIFO lifetime in Python.

    The C++ worker supplies sparse inputs and output options. Each FIFO entry
    owns both its graph runner and the final metadata binding used by replay.
    """

    def __init__(
        self,
        target_executor: object,
        draft_executor: object,
        *,
        max_variants: int = 8,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        capture_runner: Callable[..., None] | None = None,
        draft_metadata_factory: SparseMetadataFactory | None = None,
        target_metadata_factory: SparseMetadataFactory | None = None,
    ) -> None:
        if max_variants <= 0:
            raise ValueError("MTP graph variant capacity must be positive")
        if not hasattr(target_executor, "create_mtp_graph_runner_from_metadata"):
            raise TypeError("target executor cannot create MTP graph runners")
        self._target_executor = target_executor
        self._draft_executor = draft_executor
        self._max_variants = max_variants
        self._draft_activate = draft_activate
        self._target_activate = target_activate
        self._capture_runner = capture_runner
        self._draft_metadata_factory = draft_metadata_factory
        self._target_metadata_factory = target_metadata_factory
        self._variants: dict[tuple[object, ...], _SparseVariant] = {}
        self.draft_embedding_destination: torch.Tensor | None = None

    @property
    def variant_count(self) -> int:
        return len(self._variants)

    def _evict_if_full(self) -> None:
        if len(self._variants) >= self._max_variants:
            logger.warning(
                "MTP Python graph variant capacity reached; evicting oldest: capacity=%d", self._max_variants
            )
            oldest_key = next(iter(self._variants))
            self._retire_variant(oldest_key)

    def _retire_variant(self, key: tuple[object, ...]) -> None:
        self._variants[key].runner.close()
        del self._variants[key]

    @torch.inference_mode()
    def execute_sparse(
        self,
        block_table: torch.Tensor,
        first_kv_seq_lens: torch.Tensor,
        first_slots: torch.Tensor,
        repair_token_ids: torch.Tensor,
        seed_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        draft_input_embedding: torch.Tensor,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        block_size: int,
        target_step_major_layout: bool = False,
        logprobs: bool = False,
        max_top_logprobs: int = 0,
    ) -> SpeculativeRuntimeOutput:
        """Execute the admitted sparse/greedy path using final graph storage.

        Sampling flags are immutable for a variant. The Worker has already
        rejected per-request distribution transforms before this entry point.
        """
        capacity = 1 << (max(32, block_table.shape[1]) - 1).bit_length()
        key = (
            "sparse",
            batch_size,
            speculative_tokens,
            vocab_size,
            block_size,
            target_step_major_layout,
            # Token/position destinations have fixed integer dtypes; their
            # caller dtype/stride does not change the captured graph. Reuse
            # the same variant across bootstrap and fused continuation.
            draft_input_embedding.shape,
            draft_input_embedding.dtype,
            draft_input_embedding.device,
            logprobs,
            max_top_logprobs,
        )
        entry = self._variants.get(key)
        if entry is not None and entry.binding.table_capacity < capacity:
            # A single capacity per batch/output contract: grow before capture,
            # reuse it on shrink, and clear unused columns in the fused update.
            # Retirement waits for replay/snapshot consumers before allocation.
            self._retire_variant(key)
            entry = None
        if entry is None:
            self._evict_if_full()
            draft_factory = self._draft_metadata_factory
            target_factory = self._target_metadata_factory
            if draft_factory is None or target_factory is None:
                raise RuntimeError("sparse MTP registry requires native metadata factories")
            target_plan = MtpSamplingPlan(batch_size=batch_size, logprobs=logprobs, max_top_logprobs=max_top_logprobs)

            draft_storage = MtpSparseMetadataStorage(
                draft_factory,
                [batch_size * (2 if step == 0 else 1) for step in range(speculative_tokens)],
                capacity,
                block_table.device,
                share_block_tables=True,
            )
            target_storage = MtpSparseMetadataStorage(
                target_factory, [batch_size * (speculative_tokens + 1)], capacity, block_table.device
            )
            position_storage = MtpSparsePositionStorage(batch_size, speculative_tokens, block_table.device)
            variant = self._target_executor.create_mtp_graph_runner_from_metadata(
                self._draft_executor,
                draft_storage.metadata,
                target_storage.metadata[0],
                repair_token_ids=repair_token_ids,
                batch_size=batch_size,
                speculative_tokens=speculative_tokens,
                vocab_size=vocab_size,
                kv_seq_lens=kv_seq_lens,
                draft_activate=self._draft_activate,
                target_activate=self._target_activate,
                target_sampling=target_plan,
                draft_metadata_storage=draft_storage,
                target_metadata_storage=target_storage,
                position_storage=position_storage,
                target_step_major_layout=target_step_major_layout,
            )
            draft_role = variant.recipe.draft_forward
            target_role = variant.recipe.target_forward
            binding = MtpSparseMetadataBinding(
                draft_role._metadata_by_step,
                draft_role._metadata_arenas,
                target_role._metadata_by_step,
                target_role._metadata_arenas,
                position_storage=position_storage,
                batch_size=batch_size,
                speculative_tokens=speculative_tokens,
                table_capacity=capacity,
                block_size=block_size,
                target_step_major=target_step_major_layout,
            )
            binding.update(block_table, base_positions, kv_seq_lens, first_kv_seq_lens, first_slots)
            logger.info("MTP Python graph variant capture: index=%d", len(self._variants))
            inputs = (seed_token_ids, base_positions, kv_seq_lens, draft_input_embedding, None)
            if self._capture_runner is None:
                variant.capture(*inputs)
            else:
                self._capture_runner(variant, *inputs)
            draft_role.require_direct_metadata_update()
            target_role.require_direct_metadata_update()
            self._variants[key] = _SparseVariant(variant, binding)
        else:
            entry.binding.update(block_table, base_positions, kv_seq_lens, first_kv_seq_lens, first_slots)
            variant = entry.runner
            variant.recipe.draft_forward.update_repair_token_ids(repair_token_ids)
            logger.debug("MTP Python graph variant replay: count=%d", len(self._variants))
        output = variant.execute(seed_token_ids, base_positions, kv_seq_lens, draft_input_embedding)
        # Export the destination for the next C++ fused prepare. A capacity or
        # output-layout transition copies once into the newly resolved variant
        # and publishes its replacement destination after execution.
        self.draft_embedding_destination = variant.draft_embedding_destination
        return output

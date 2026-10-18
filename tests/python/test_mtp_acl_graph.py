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

from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.python.mtp_graph_test_utils import commit_reference, make_recipe, output_tensors, scalar_reference
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
    get_execution_buffer,
)
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpRoleAdapter,
    MtpSamplingPlan,
)
from xllm.python.model_executor.runners.mtp_sparse_metadata import MtpSparsePositionStorage


@pytest.fixture(autouse=True)
def cpu_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.ops.xllm_ops, "mtp_greedy_commit", commit_reference, raising=False)


@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parametrize("prepared_positions", [False, True])
def test_eager_recipe_matches_scalar_oracle_for_each_fixed_k(steps: int, prepared_positions: bool) -> None:
    rejection_steps = torch.arange(steps + 1)
    storage = MtpSparsePositionStorage(steps + 1, steps, torch.device("cpu")) if prepared_positions else None
    recipe = make_recipe(rejection_steps, steps, position_storage=storage)
    activation_calls: list[str] = []
    recipe.draft_activate = lambda: activation_calls.append("draft")
    recipe.target_activate = lambda: activation_calls.append("target")
    runner = MtpAclGraphRunner(recipe, backend="eager")
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    if storage is not None:
        storage.positions.copy_(
            torch.tensor(
                [position + step for step in range(steps) for position in positions]
                + [position + step for position in positions for step in range(steps + 1)]
            )
        )
    kv_lengths = [position + 1 for position in positions]
    expected = scalar_reference(seeds, positions, rejection_steps.tolist(), steps)
    output = runner.execute(torch.tensor(seeds), torch.tensor(positions), torch.tensor(kv_lengths, dtype=torch.int32))
    for name, actual in output_tensors(output).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert activation_calls == ["draft"] * steps + ["target"]
    if storage is not None:
        # Prepared positions, rather than the unused caller base, feed replay.
        changed_base = torch.tensor(positions) + 100
        replay = runner.execute(torch.tensor(seeds), changed_base, torch.tensor(kv_lengths, dtype=torch.int32))
        torch.testing.assert_close(replay.committed_tokens, expected["committed_tokens"], rtol=0, atol=0)
        torch.testing.assert_close(replay.target_embeddings, expected["target_embeddings"], rtol=0, atol=0)


def test_eager_recipe_carries_mtp_embedding_and_topk_state() -> None:
    batch_size = 2
    speculative_tokens = 3
    seen: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] = []

    def draft_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        seen.append(
            (
                step,
                None if input_embedding is None else input_embedding.clone(),
                None if topk_indices is None else topk_indices.clone(),
            )
        )
        hidden = (ids + positions + step + 1).unsqueeze(-1)
        next_topk = (ids + step + 11).unsqueeze(-1)
        return hidden, None, next_topk

    def target_body(
        ids: torch.Tensor,
        positions: torch.Tensor,
        step: int,
        input_embedding: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        del positions, step, input_embedding, topk_indices
        hidden = ids.unsqueeze(-1)
        topk = (ids + 100).unsqueeze(-1)
        return hidden, None, topk

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((hidden.shape[0], 64), -100.0)
        return logits.scatter_(1, hidden.remainder(64).to(torch.long), 1.0)

    runner = MtpAclGraphRunner(
        MtpGraphRecipe(
            draft_body,
            head,
            target_body,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=64,
            device=torch.device("cpu"),
        ),
        backend="eager",
    )
    initial_embedding = torch.tensor([[500.0], [600.0]])
    initial_topk = torch.tensor([[7], [8]], dtype=torch.long)
    output = runner.execute(
        torch.tensor([1, 2]),
        torch.tensor([10, 20]),
        draft_input_embedding=initial_embedding,
        draft_topk_indices=initial_topk,
    )

    assert [step for step, _, _ in seen] == [0, 1, 2]
    assert torch.equal(seen[0][1], initial_embedding)
    assert torch.equal(seen[0][2], initial_topk)
    assert torch.equal(seen[1][1], torch.tensor([[12], [23]]))
    assert torch.equal(seen[1][2], torch.tensor([[12], [13]]))
    assert torch.equal(seen[2][1], torch.tensor([[25], [46]]))
    assert torch.equal(seen[2][2], torch.tensor([[24], [35]]))
    assert output.target_embeddings.shape == (2 * batch_size, 1)


def test_role_adapter_keeps_draft_and_target_metadata_scoped() -> None:
    calls: list[tuple[str, int, object, torch.Tensor | None, torch.Tensor | None]] = []

    class FakeRunner:
        def __init__(self, role: str) -> None:
            self.role = role

        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> torch.Tensor:
            del synchronizer
            calls.append((self.role, int(ids.numel()), metadata, embedding, topk))
            return ids.unsqueeze(-1)

    class FakeExecutor:
        def __init__(self, role: str) -> None:
            self.eager_runner = FakeRunner(role)

    draft = MtpRoleAdapter(FakeExecutor("draft"), ("d0", "d1", "d2"), speculative_tokens=3)
    target = MtpRoleAdapter(FakeExecutor("target"), ("target-k+1",), speculative_tokens=3, target=True)
    ids = torch.tensor([1, 2])
    positions = torch.tensor([10, 20])
    embedding = torch.ones(2, 1)
    topk = torch.ones(2, 1, dtype=torch.long)

    draft(ids, positions, 1, embedding, topk)
    target(torch.arange(8), torch.arange(8), -1, None, None)

    assert calls[0][0:3] == ("draft", 2, "d1")
    assert calls[1][0:3] == ("target", 8, "target-k+1")


@pytest.mark.parametrize("batch,steps", [(1, 1), (2, 1), (3, 3)])
@pytest.mark.parametrize("step_major", [False, True])
@pytest.mark.parametrize("with_topk", [False, True])
def test_target_role_matches_metadata_order_and_restores_outputs(
    batch: int, steps: int, step_major: bool, with_topk: bool
) -> None:
    request_rows = [(request, step) for request in range(batch) for step in range(steps + 1)]
    metadata_rows = sorted(request_rows, key=lambda row: (row[1], row[0])) if step_major else request_rows
    ids = torch.tensor([request * 10 + step for request, step in request_rows])
    positions = torch.tensor([request * 100 + step for request, step in request_rows])
    body_outputs = []

    def execute(call_ids: torch.Tensor, call_positions: torch.Tensor, metadata: object, *args: object) -> object:
        assert metadata is metadata_rows
        assert call_ids.tolist() == [request * 10 + step for request, step in metadata_rows]
        assert call_positions.tolist() == [request * 100 + step for request, step in metadata_rows]
        if batch == 1 or not step_major:
            assert call_ids is ids and call_positions is positions
        hidden = torch.stack((call_ids, call_positions), dim=1)
        output = (hidden, None, (call_ids + 1000).unsqueeze(-1)) if with_topk else hidden
        body_outputs.append(output)
        return output

    adapter = MtpRoleAdapter(
        SimpleNamespace(eager_runner=SimpleNamespace(execute=execute)),
        (metadata_rows,),
        speculative_tokens=steps,
        target=True,
        step_major_layout=step_major,
    )
    output = adapter(ids, positions, -1, None, None)
    hidden = output[0] if with_topk else output
    torch.testing.assert_close(hidden, torch.stack((ids, positions), dim=1))
    if with_topk:
        torch.testing.assert_close(output[2], (ids + 1000).unsqueeze(-1))
    if batch == 1 or not step_major:
        assert output is body_outputs[0]


def test_serial_mtp_attention_workspace_uses_role_shared_pool() -> None:
    shared: dict[tuple[object, ...], object] = {}
    first_state = AclGraphExecutionState({}, shared_persistent_buffers=shared)
    second_state = AclGraphExecutionState({}, shared_persistent_buffers=shared)
    device = torch.device("cpu")

    class Backend:
        pass

    metadata = object()
    context = ForwardContext(Backend(), device, metadata, [], execution_state=first_state)
    with forward_context(context):
        first = get_execution_buffer(("FIA_WORKSPACE", 2), lambda: torch.zeros(4), shared=True)
    context = ForwardContext(Backend(), device, metadata, [], execution_state=second_state)
    with forward_context(context):
        second = get_execution_buffer(("FIA_WORKSPACE", 2), lambda: torch.ones(4), shared=True)

    assert first.data_ptr() == second.data_ptr()
    assert first_state.persistent_buffers == {}
    assert second_state.persistent_buffers == {}


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64])
def test_role_adapter_selects_current_rows_from_first_draft_repair_layout(source_dtype: torch.dtype) -> None:
    seen: list[tuple[torch.Tensor, torch.Tensor, object, torch.Tensor]] = []

    class FakeRunner:
        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> tuple[torch.Tensor, None, torch.Tensor]:
            del synchronizer, topk
            assert embedding is not None
            seen.append((ids, positions, metadata, embedding))
            return ids.unsqueeze(-1), None, (ids + 100).unsqueeze(-1)

    class FakeExecutor:
        eager_runner = FakeRunner()

    prepared_positions = torch.zeros(4, dtype=torch.long) if source_dtype == torch.int32 else None
    adapter = MtpRoleAdapter(
        FakeExecutor(),
        ("d0", "d1"),
        speculative_tokens=2,
        repair_token_ids=torch.tensor([[4, -1], [5, -1]], dtype=source_dtype)[:, 0],
        repair_positions=prepared_positions,
    )
    if prepared_positions is not None:
        prepared_positions.copy_(torch.tensor([19, 20, 29, 30]))
    output = adapter(
        torch.tensor([9, 10]),
        torch.tensor([20, 30]),
        0,
        torch.arange(4, dtype=torch.float32).reshape(4, 1),
        None,
    )

    assert seen[0][0].tolist() == [4, 9, 5, 10]
    assert seen[0][0].dtype == torch.long
    assert seen[0][1].tolist() == [19, 20, 29, 30]
    assert seen[0][2] == "d0"
    assert seen[0][3].shape == (4, 1)
    assert isinstance(output, tuple)
    assert output[0].flatten().tolist() == [9, 10]


def test_role_adapter_reuses_prepared_repair_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRunner:
        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> torch.Tensor:
            del positions, metadata, embedding, synchronizer, topk
            return ids.unsqueeze(-1)

    class FakeExecutor:
        eager_runner = FakeRunner()

    adapter = MtpRoleAdapter(
        FakeExecutor(),
        ("d0",),
        speculative_tokens=1,
        repair_token_ids=torch.tensor([4, 5]),
    )
    embedding = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    def unexpected_arange(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("repair row indices must be prepared before replay")

    monkeypatch.setattr(torch, "arange", unexpected_arange)
    output = adapter(
        torch.tensor([9, 10]),
        torch.tensor([20, 30]),
        0,
        embedding,
        None,
    )
    assert output.flatten().tolist() == [9, 10]


def test_role_adapter_b1_uses_a_single_row_view(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRunner:
        def execute(
            self,
            ids: torch.Tensor,
            positions: torch.Tensor,
            metadata: object,
            embedding: torch.Tensor | None,
            synchronizer: object,
            topk: torch.Tensor | None,
        ) -> torch.Tensor:
            del positions, metadata, embedding, synchronizer, topk
            return ids.unsqueeze(-1)

    class FakeExecutor:
        eager_runner = FakeRunner()

    adapter = MtpRoleAdapter(
        FakeExecutor(),
        ("d0",),
        speculative_tokens=1,
        repair_token_ids=torch.tensor([4]),
    )
    embedding = torch.tensor([[0.0], [1.0]])

    def unexpected_arange(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("B1 should not allocate selected row indices")

    monkeypatch.setattr(torch, "arange", unexpected_arange)
    output = adapter(
        torch.tensor([9]),
        torch.tensor([20]),
        0,
        embedding,
        None,
    )

    assert isinstance(output, torch.Tensor)
    assert output.tolist() == [[9]]
    assert output.storage_offset() == 1


def test_draft_temporaries_live_only_as_long_as_their_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_recipe(torch.tensor([1]), 3)
    recipe.target_sampling = MtpSamplingPlan(batch_size=1)
    draft_forward, draft_head, target_forward = recipe.draft_forward, recipe.draft_logits, recipe.target_forward
    logits_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    hidden_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def draft(*args: object) -> torch.Tensor:
        hidden = draft_forward(*args)
        hidden_refs.append(weakref.ref(hidden))
        return hidden

    def head(hidden: torch.Tensor) -> torch.Tensor:
        logits = draft_head(hidden)
        logits_refs.append(weakref.ref(logits))
        return logits

    def target(*args: object) -> torch.Tensor:
        assert len(logits_refs) == len(hidden_refs) == 3
        assert all(value() is None for value in logits_refs + hidden_refs)
        return target_forward(*args)

    recipe.draft_forward, recipe.draft_logits, recipe.target_forward = draft, head, target
    normalize = Mock(side_effect=AssertionError("greedy draft has no probability/logprob consumer"))
    monkeypatch.setattr(torch, "log_softmax", normalize)
    actual = MtpAclGraphRunner(recipe, backend="eager").execute(
        torch.tensor([2]), torch.tensor([10]), torch.tensor([11], dtype=torch.int32)
    )
    expected = scalar_reference([2], [10], [1], 3)
    for name, value in output_tensors(actual).items():
        torch.testing.assert_close(value, expected[name])


def test_runner_close_waits_for_replay_and_output_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    recipe = make_recipe(torch.tensor([1]), 3)
    runner = MtpAclGraphRunner(recipe)
    runner._capture_stream = SimpleNamespace(synchronize=lambda: order.append("replay-complete"))
    runner._graph = SimpleNamespace(reset=lambda: order.append("graph-freed"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: order.append("snapshots-complete"))
        ),
        raising=False,
    )
    runner.close()
    runner.close()
    assert order == ["replay-complete", "snapshots-complete", "graph-freed"]
    inputs = (torch.tensor([2]), torch.tensor([10]), torch.tensor([11], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="closed"):
        runner.execute(*inputs)
    with pytest.raises(RuntimeError, match="closed"):
        runner.capture(*inputs)


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("logprobs", [False, True])
def test_runtime_output_uses_only_consumed_snapshots_and_greedy_heads(steps: int, logprobs: bool) -> None:
    from dataclasses import fields

    from xllm.python.model_executor.runners.base import SpeculativeRuntimeOutput

    rejects = torch.arange(steps + 1)
    fast, reference = make_recipe(rejects, steps), make_recipe(rejects, steps)
    for recipe in (fast, reference):
        recipe.target_sampling = MtpSamplingPlan(batch_size=steps + 1, logprobs=logprobs)
    draft_head, target_head = fast.draft_logits, fast.target_logits
    fast.draft_greedy = lambda hidden: draft_head(hidden).argmax(-1)
    fast.target_greedy = lambda hidden: target_head(hidden).argmax(-1)
    fast.draft_logits = Mock(side_effect=AssertionError("greedy draft must not gather full logits"))
    if not logprobs:
        fast.target_logits = Mock(side_effect=AssertionError("plain greedy target must not gather full logits"))
    inputs = (
        torch.arange(steps + 1),
        torch.arange(steps + 1, dtype=torch.int32) + 7,
        torch.arange(steps + 1, dtype=torch.int32) + 8,
    )
    actual = MtpAclGraphRunner(fast, backend="eager").execute(*inputs)
    expected = MtpAclGraphRunner(reference, backend="eager").execute(*inputs)
    assert isinstance(actual, SpeculativeRuntimeOutput)
    oracle = scalar_reference(inputs[0].tolist(), inputs[1].tolist(), rejects.tolist(), steps)
    expected_compact = oracle["target_embeddings"]
    torch.testing.assert_close(actual.target_embeddings, expected_compact)
    for field in fields(actual):
        if field.name not in ("token_state", "target_embeddings"):
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name))
    assert actual.token_state is not None
    assert actual.committed_tokens.untyped_storage().data_ptr() == actual.token_state.data_ptr()
    assert actual.accepted_count.untyped_storage().data_ptr() == actual.token_state.data_ptr()
    assert actual.token_state.numel() == (steps + 1) * ((steps + 1) * 8 + 4)
    snapshot = MtpAclGraphRunner._clone_graph_output(actual)
    with torch.inference_mode():
        actual.committed_tokens.fill_(-99)
        actual.target_embeddings.fill_(-99)
    torch.testing.assert_close(snapshot.committed_tokens, expected.committed_tokens)
    torch.testing.assert_close(snapshot.accepted_count, expected.accepted_count)
    assert snapshot.token_state.data_ptr() != actual.token_state.data_ptr()
    assert snapshot.committed_tokens.untyped_storage().data_ptr() == snapshot.token_state.data_ptr()
    assert snapshot.accepted_count.untyped_storage().data_ptr() == snapshot.token_state.data_ptr()
    torch.testing.assert_close(snapshot.target_embeddings, expected_compact)

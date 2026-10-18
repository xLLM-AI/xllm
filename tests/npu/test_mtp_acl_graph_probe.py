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

"""Real-NPU probes of the composite runner for several independent K variants.

Run with the built native library and an explicitly assigned NPU.
Deterministic bodies exercise graph mechanics and acceptance, not real GLM
attention.
"""

from __future__ import annotations

import os
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from tests.python.mtp_graph_test_utils import make_recipe, output_tensors, scalar_reference
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpGraphVariantRegistry,
    MtpRoleAdapter,
    MtpSamplingPlan,
    _runtime_token_views,
)


@pytest.mark.parametrize("steps", [3, 5])
def test_owned_sparse_metadata_updates_reach_captured_consumers(npu_device: torch.device, steps: int) -> None:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if not library:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY for the real attention backend")
    import xllm.python as runtime

    torch.ops.load_library(library)
    runtime.initialize_runtime()
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.runners.mtp_sparse_metadata import MtpSparseMetadataStorage

    variants = []
    cache = torch.empty(4, 128, 1, 16, device=npu_device)
    for batch in (1, 3):

        def native_metadata(table: torch.Tensor, lengths: torch.Tensor, slots: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                slot_mapping=slots,
                block_table=table,
                kv_seq_lens=lengths,
                q_cu_seq_lens=None,
                q_seq_lens=None,
                expanded_decode_metadata=None,
                is_prefill=False,
                is_chunked_prefill=False,
            )

        rows = [batch * 2, *([batch] * (steps - 1)), batch * (steps + 1)]
        storage = MtpSparseMetadataStorage(native_metadata, rows, 32, npu_device)
        metadata, backends = storage.metadata, []
        for item in metadata:
            item.kv_seq_lens.fill_(128)
            item.slot_mapping.fill_(127)
            item.block_table.zero_()
        arenas = (storage.arena,)
        for item in metadata:
            backend = NpuPagedAttentionBackend(
                num_heads=1,
                num_kv_heads=1,
                head_dim=16,
                scale=0.25,
                sliding_window=0,
                is_mla=True,
                device=npu_device,
                dtype=torch.bfloat16,
            )
            backend.bind_kv_caches([LayerCache(key=cache, value=cache, index=cache)])
            backend.prepare_owned_graph_metadata(item)
            backends.append(backend)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = torch.cat(
                [
                    backend._mla_actual_seq_kv + backend._block_table_i32[:, 0] + item.slot_mapping
                    for backend, item in zip(backends, metadata)
                ]
            )
        torch.npu.current_stream().wait_stream(stream)
        variants.append((metadata, backends, arenas, stream, graph, output))

    retained = []
    for generation in range(4):
        for metadata, backends, arenas, stream, graph, output in variants:
            expected = []
            for step, (item, backend) in enumerate(zip(metadata, backends)):
                rows = item.kv_seq_lens.numel()
                # Cross a page boundary, reorder rows and alternate graph
                # variants without changing their independently owned storage.
                lengths = torch.arange(rows, dtype=torch.int32).flip(0) + 129 + generation + step
                table = torch.full((rows, 32), generation + step, dtype=torch.int32)
                slots = lengths + 127
                item.kv_seq_lens.copy_(lengths.to(npu_device))
                item.block_table.copy_(table.to(npu_device))
                item.slot_mapping.copy_(slots.to(npu_device))
                expected.append(lengths + table[:, 0] + slots)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.current_stream().wait_stream(stream)
            snapshot = output.clone()
            reference = torch.cat(expected)
            torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
            retained.append((snapshot, reference))
    for snapshot, reference in retained:
        torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("batch,step_major", [(1, False), (2, False), (2, True)])
def test_sparse_binding_reaches_prepared_position_runtime_runner(
    npu_device: torch.device, batch: int, step_major: bool
) -> None:
    """Bind native metadata and prepared positions through one captured replay."""

    class RoleExecutor:
        def __init__(self) -> None:
            self.eager_runner = self

        def create_mtp_role_backends(self, count: int) -> tuple[SimpleNamespace, ...]:
            return tuple(SimpleNamespace(graph_metadata_updated_in_place=True) for _ in range(count))

        def execute(self, ids, positions, metadata, embedding, synchronizer, topk):
            del embedding, synchronizer, topk
            return (
                ids.to(torch.float32) + positions.to(torch.float32) + metadata.kv_seq_lens.to(torch.float32)
            ).unsqueeze(-1)

        def execute_mtp_role(self, ids, positions, metadata, embedding, synchronizer, topk, **kwargs):
            del kwargs
            return self.execute(ids, positions, metadata, embedding, synchronizer, topk)

    target_executor = RoleExecutor()
    draft_executor = RoleExecutor()

    def metadata(table: torch.Tensor, lengths: torch.Tensor, slots: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(slot_mapping=slots, kv_seq_lens=lengths, block_table=table)

    def create_runner(
        draft_executor: RoleExecutor,
        draft_metadata: tuple[object, ...],
        target_metadata: object,
        *,
        repair_token_ids: torch.Tensor,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        kv_seq_lens: torch.Tensor,
        draft_metadata_storage: object,
        target_metadata_storage: object,
        position_storage: object,
        target_step_major_layout: bool,
        **kwargs: object,
    ) -> MtpAclGraphRunner:
        del kwargs
        draft_role = MtpRoleAdapter(
            draft_executor,
            draft_metadata,
            speculative_tokens=speculative_tokens,
            repair_token_ids=repair_token_ids,
            repair_positions=position_storage.first_draft,
            metadata_storage=draft_metadata_storage,
        )
        target_role = MtpRoleAdapter(
            target_executor,
            (target_metadata,),
            speculative_tokens=speculative_tokens,
            target=True,
            step_major_layout=target_step_major_layout,
            metadata_storage=target_metadata_storage,
        )

        def head(hidden: torch.Tensor) -> torch.Tensor:
            logits = torch.full((hidden.shape[0], vocab_size), -100.0, device=hidden.device)
            return logits.scatter_(1, hidden[:, 0].to(torch.long).remainder(vocab_size).unsqueeze(1), 1.0)

        recipe = MtpGraphRecipe(
            draft_role,
            head,
            target_role,
            head,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            device=npu_device,
            kv_seq_lens=kv_seq_lens,
            target_sampling=MtpSamplingPlan(batch_size=batch_size),
            position_storage=position_storage,
        )
        return MtpAclGraphRunner(recipe)

    target_executor.create_mtp_graph_runner_from_metadata = create_runner
    registry = MtpGraphVariantRegistry(
        target_executor,
        draft_executor,
        draft_metadata_factory=metadata,
        target_metadata_factory=metadata,
    )
    block_table = torch.zeros((batch, 32), dtype=torch.int32, device=npu_device)
    first_kv = torch.tensor(
        [6 + row * 20 + step for row in range(batch) for step in range(2)], device=npu_device, dtype=torch.int32
    )
    first_slots = first_kv - 2
    repair = torch.tensor([13 + row * 10 for row in range(batch)], dtype=torch.int64, device=npu_device)
    seed = repair + 4
    base = torch.tensor([8 + row * 20 for row in range(batch)], dtype=torch.int64, device=npu_device)
    kv = (base + 1).to(torch.int32)
    embedding = torch.ones((2 * batch, 1), dtype=torch.float32, device=npu_device)

    first = registry.execute_sparse(
        block_table,
        first_kv,
        first_slots,
        repair,
        seed,
        base,
        kv,
        embedding,
        batch_size=batch,
        speculative_tokens=3,
        vocab_size=256,
        block_size=128,
        target_step_major_layout=step_major,
    )
    second_base = base + 10
    second_kv = kv + 20
    second = registry.execute_sparse(
        block_table,
        first_kv + 20,
        first_slots + 20,
        repair,
        seed,
        second_base,
        second_kv,
        embedding,
        batch_size=batch,
        speculative_tokens=3,
        vocab_size=256,
        block_size=128,
        target_step_major_layout=step_major,
    )
    assert first.target_embeddings.shape == second.target_embeddings.shape == (2 * batch, 1)
    assert first.accepted_count.cpu().tolist() == [0] * batch
    assert first.target_embeddings.cpu().flatten().tolist() == [34 + row * 50 for row in range(batch) for _ in range(2)]
    torch.testing.assert_close(
        second.target_embeddings - first.target_embeddings,
        torch.full_like(first.target_embeddings, 30),
        rtol=0,
        atol=0,
    )
    registry._retire_variant(next(iter(registry._variants)))


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if device_index is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE for the real MTP ACL graph probe")
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(device_index))
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    if library:
        torch.ops.load_library(library)
    return torch.device(f"npu:{device_index}")


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_mtp_recipe_replay_every_rejection_position(npu_device: torch.device, steps: int) -> None:
    rejects_device = torch.arange(steps + 1, device=npu_device)
    runner = MtpAclGraphRunner(make_recipe(rejects_device, steps))
    eager = MtpAclGraphRunner(make_recipe(rejects_device, steps), backend="eager")
    seeds = [2 + row for row in range(steps + 1)]
    positions = [10 + row * 3 for row in range(steps + 1)]
    kv_lengths = [position + 1 for position in positions]
    runner.capture(
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    rejects = list(range(steps + 1))
    expected = scalar_reference(seeds, positions, rejects, steps)
    inputs = (
        torch.tensor(seeds, device=npu_device),
        torch.tensor(positions, device=npu_device),
        torch.tensor(kv_lengths, dtype=torch.int32, device=npu_device),
    )
    eager_output = output_tensors(eager.execute(*inputs))
    actual = runner.execute(*inputs)
    for name, tensor in output_tensors(actual).items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
        torch.testing.assert_close(tensor, eager_output[name], rtol=0, atol=0)
    # C++ Unified now passes the current column of its fused int32 [B,2]
    # output directly; the graph-owned destination performs the only cast.
    token_rows = torch.tensor([[seed - 1, seed] for seed in seeds], dtype=torch.int32, device=npu_device)
    replayed = runner.execute(token_rows[:, 1], inputs[1], inputs[2])
    for name, tensor in output_tensors(replayed).items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
    runner.close()


@pytest.mark.parametrize("steps", [3, 5])
@pytest.mark.parametrize("logprobs", [False, True])
def test_mtp_greedy_bfloat16_scores_match_probability_reference(
    npu_device: torch.device, steps: int, logprobs: bool
) -> None:
    rejects = torch.arange(steps + 1, device=npu_device)
    plan = MtpSamplingPlan(
        batch_size=steps + 1,
        logprobs=logprobs,
        max_top_logprobs=3 if logprobs else 0,
    )
    fast = make_recipe(rejects, steps, logits_dtype=torch.bfloat16)
    reference = make_recipe(rejects, steps, logits_dtype=torch.bfloat16)
    fast.target_sampling = plan
    reference.target_sampling = plan
    inputs = (
        torch.arange(steps + 1, device=npu_device),
        torch.arange(steps + 1, device=npu_device) + 127,
        torch.arange(steps + 1, dtype=torch.int32, device=npu_device) + 128,
    )
    runner = MtpAclGraphRunner(fast)
    eager = MtpAclGraphRunner(reference, backend="eager")
    runner.capture(*inputs)
    for offset in (0, 7):
        updated_inputs = (inputs[0] + offset, inputs[1] + offset, inputs[2] + offset)
        expected = eager.execute(*updated_inputs)
        actual = runner.execute(*updated_inputs)
        torch.testing.assert_close(actual.target_embeddings, expected.target_embeddings, rtol=0, atol=0)
        for name in ("committed_tokens", "accepted_count"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        assert actual.token_state is not None
        assert actual.token_state.dtype == torch.uint8
        assert actual.committed_tokens.untyped_storage().data_ptr() == actual.token_state.data_ptr()
        assert actual.accepted_count.untyped_storage().data_ptr() == actual.token_state.data_ptr()
        for name in ("logprobs", "top_logprobs", "top_tokens"):
            value, expected_value = getattr(actual, name), getattr(expected, name)
            if value is None or expected_value is None:
                assert value is expected_value
            else:
                torch.testing.assert_close(value.cpu(), expected_value.cpu(), rtol=1e-5, atol=1e-5)
    runner.close()


@pytest.mark.parametrize("steps", [3, 5])
def test_mtp_replay_output_survives_the_next_replay(npu_device: torch.device, steps: int) -> None:
    """A returned graph result must not alias the next replay's buffers."""
    recipe = make_recipe(torch.tensor([1], device=npu_device), steps)
    runner = MtpAclGraphRunner(recipe)

    def tensors(output: object) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(output, field.name)
            for field in fields(output)
            if isinstance(getattr(output, field.name), torch.Tensor)
        }

    first_inputs = (
        torch.tensor([2], device=npu_device),
        torch.tensor([10], dtype=torch.int32, device=npu_device),
        torch.tensor([11], dtype=torch.int32, device=npu_device),
    )
    second_inputs = (
        torch.tensor([9], device=npu_device),
        torch.tensor([20], dtype=torch.int32, device=npu_device),
        torch.tensor([21], dtype=torch.int32, device=npu_device),
    )
    runner.capture(*first_inputs)
    first = runner.execute(*first_inputs)
    first_snapshot = {name: tensor.clone() for name, tensor in tensors(first).items()}
    second = runner.execute(*second_inputs)
    assert not torch.equal(first_snapshot["committed_tokens"], tensors(second)["committed_tokens"])

    runner.close()
    for name, expected in first_snapshot.items():
        torch.testing.assert_close(tensors(first)[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [16, 8193])
def test_compact_commit_hidden_tiles_and_old_output_lifetime(
    npu_device: torch.device, dtype: torch.dtype, hidden_size: int
) -> None:
    # More than 32 requests exercises the count tile boundary; H=8193
    # exercises the hidden tile and unaligned tail. Every rejection position
    # and all-accepted are present, with independently recognizable rows.
    batch, steps = 33, 5
    target_cpu = torch.arange(batch * (steps + 1)).view(batch, steps + 1)
    draft_cpu = target_cpu[:, :steps].clone()
    counts = [row % (steps + 1) for row in range(batch)]
    for row, count in enumerate(counts):
        if count < steps:
            draft_cpu[row, count] = -7
    hidden_cpu = (torch.arange(batch * (steps + 1) * hidden_size) % 997).view(-1, hidden_size).to(dtype)
    draft, target, hidden = [value.to(npu_device) for value in (draft_cpu, target_cpu, hidden_cpu)]
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        state, compact = torch.ops.xllm_ops.mtp_greedy_commit(draft, target, hidden)
    torch.npu.current_stream().wait_stream(stream)
    expected = torch.stack(
        [
            hidden_cpu[row * (steps + 1) + index]
            for row, count in enumerate(counts)
            for index in (max(count - 1, 0), count)
        ]
    )
    graph.replay()
    retained = compact.clone()
    torch.testing.assert_close(retained.cpu(), expected, rtol=0, atol=0)
    hidden.add_(1)
    graph.replay()
    torch.testing.assert_close(compact.cpu(), expected + 1, rtol=0, atol=0)
    torch.testing.assert_close(retained.cpu(), expected, rtol=0, atol=0)
    torch.npu.synchronize()
    graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_b1_k3_h6144_replays_all_acceptance_counts(npu_device: torch.device, dtype: torch.dtype) -> None:
    """Exercise the current AOT shape and every accepted-prefix width."""
    batch, steps, hidden_size = 1, 3, 6144
    draft = torch.tensor([[10, 11, 12]], dtype=torch.int64, device=npu_device)
    target = torch.empty((batch, steps + 1), dtype=torch.int64, device=npu_device)
    hidden = torch.empty((batch * (steps + 1), hidden_size), dtype=dtype, device=npu_device)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream(device=npu_device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        token_state, compact = torch.ops.xllm_ops.mtp_greedy_commit(draft, target, hidden)
    torch.npu.current_stream().wait_stream(stream)
    token_state_address = token_state.data_ptr()
    compact_address = compact.data_ptr()

    first_raw: torch.Tensor | None = None
    first_compact: torch.Tensor | None = None
    first_compact_expected: torch.Tensor | None = None
    for accepted in range(steps + 1):
        target_host = [99, 98, 97, 77]
        for index in range(accepted):
            target_host[index] = int(draft[0, index].item())
        target.copy_(torch.tensor([target_host], dtype=torch.int64, device=npu_device))
        hidden_host = (
            torch.arange(batch * (steps + 1) * hidden_size, dtype=torch.float32)
            .reshape(batch * (steps + 1), hidden_size)
            .add_(accepted * 10000)
            .to(dtype)
        )
        hidden.copy_(hidden_host)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.current_stream().wait_stream(stream)

        raw = token_state.clone()
        compact_snapshot = compact.clone()
        actual_tokens, actual_count = _runtime_token_views(raw, batch, steps)
        expected_tokens = torch.tensor(
            [[*draft[0, :accepted].tolist(), target_host[accepted], *([-1] * (steps - accepted))]],
            dtype=torch.int64,
        )
        expected_hidden = hidden_host[[max(accepted - 1, 0), accepted]]
        torch.testing.assert_close(actual_tokens.cpu(), expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(actual_count.cpu(), torch.tensor([accepted], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(compact_snapshot.cpu(), expected_hidden.cpu(), rtol=0, atol=0)
        assert token_state.dtype == torch.uint8
        assert token_state.data_ptr() == token_state_address
        assert compact.data_ptr() == compact_address
        if first_raw is None:
            first_raw, first_compact = raw, compact_snapshot
            first_compact_expected = expected_hidden.cpu().clone()

    assert first_raw is not None and first_compact is not None and first_compact_expected is not None
    # The first replay was cloned before later replays and must remain stable.
    first_tokens, first_count = _runtime_token_views(first_raw, batch, steps)
    torch.testing.assert_close(first_tokens.cpu(), torch.tensor([[99, -1, -1, -1]]), rtol=0, atol=0)
    torch.testing.assert_close(first_count.cpu(), torch.tensor([0], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(first_compact.cpu(), first_compact_expected, rtol=0, atol=0)
    torch.npu.synchronize()
    graph.reset()


def test_next_prepare_writes_graph_embedding_destination(npu_device: torch.device) -> None:
    def body(
        ids: torch.Tensor, positions: torch.Tensor, step: int, embedding: torch.Tensor | None, topk: torch.Tensor | None
    ) -> torch.Tensor:
        del positions, step, topk
        hidden = ids.float().unsqueeze(1)
        return hidden if embedding is None else hidden + embedding

    def head(hidden: torch.Tensor) -> torch.Tensor:
        words = torch.arange(32, device=npu_device).float()
        return -(words.unsqueeze(0) - hidden.remainder(32)).abs()

    def recipe() -> MtpGraphRecipe:
        return MtpGraphRecipe(
            body,
            head,
            body,
            head,
            batch_size=1,
            speculative_tokens=3,
            vocab_size=32,
            device=npu_device,
            kv_seq_lens=torch.ones(1, dtype=torch.int32, device=npu_device),
        )

    runner, reference = MtpAclGraphRunner(recipe()), MtpAclGraphRunner(recipe(), backend="eager")
    inputs = (
        torch.tensor([2], device=npu_device),
        torch.tensor([7], device=npu_device),
        torch.tensor([8], dtype=torch.int32, device=npu_device),
    )
    embedding = torch.ones(1, 1, device=npu_device)
    runner.capture(*inputs, embedding)
    destination = runner.draft_embedding_destination
    assert destination is not None and destination.data_ptr() != embedding.data_ptr()
    retained = runner.execute(*inputs, embedding)
    retained_tokens = retained.committed_tokens.cpu().clone()
    for value in (3, 5, 9):
        # Models C++ MtpPrepareNextDraft writing the final destination on the
        # producer stream after the previous replay's reads have completed.
        destination.fill_(value)
        actual = runner.execute(*inputs, destination)
        expected = reference.execute(*inputs, torch.full_like(embedding, value))
        for name, tensor in output_tensors(actual).items():
            torch.testing.assert_close(tensor, output_tensors(expected)[name], rtol=0, atol=0)
    torch.testing.assert_close(retained.committed_tokens.cpu(), retained_tokens, rtol=0, atol=0)
    runner.close()

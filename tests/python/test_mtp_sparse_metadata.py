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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpGraphVariantRegistry,
    MtpRoleAdapter,
)
from xllm.python.model_executor.runners.mtp_sparse_metadata import (
    MtpSparseMetadataBinding,
    MtpSparseMetadataStorage,
    MtpSparsePositionStorage,
)


def _metadata(table: torch.Tensor, lengths: torch.Tensor, slots: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(block_table=table, kv_seq_lens=lengths, slot_mapping=slots)


@pytest.mark.parametrize("step_major", [False, True])
def test_sparse_registry_owns_final_destinations_and_reuses_capacity(
    monkeypatch: pytest.MonkeyPatch, step_major: bool
) -> None:
    update = Mock()
    monkeypatch.setattr(torch.ops.xllm_ops, "mtp_sparse_metadata_update", update, raising=False)
    runners = []
    plans = []

    class Runner:
        def __init__(self, draft: MtpSparseMetadataStorage, target: MtpSparseMetadataStorage) -> None:
            def role(storage: MtpSparseMetadataStorage) -> SimpleNamespace:
                return SimpleNamespace(
                    _metadata_by_step=storage.metadata,
                    _metadata_arenas=(storage.arena,),
                    require_direct_metadata_update=Mock(),
                    update_repair_token_ids=Mock(),
                )

            self.recipe = SimpleNamespace(draft_forward=role(draft), target_forward=role(target))
            self.capture = Mock()
            self.closed = False
            self.draft_embedding_destination = None

        def execute(self, seed: torch.Tensor, *args: object) -> torch.Tensor:
            return seed.clone()

        def close(self) -> None:
            self.closed = True

    def create(
        _executor: object,
        _draft: tuple[object, ...],
        _target: object,
        **kwargs: object,
    ) -> Runner:
        assert kwargs["target_step_major_layout"] == step_major
        runner = Runner(kwargs["draft_metadata_storage"], kwargs["target_metadata_storage"])
        runners.append(runner)
        plans.append(kwargs["target_sampling"])
        return runner

    factory = Mock(side_effect=_metadata)
    registry = MtpGraphVariantRegistry(
        SimpleNamespace(create_mtp_graph_runner_from_metadata=create),
        object(),
        max_variants=2,
        draft_metadata_factory=factory,
        target_metadata_factory=factory,
    )

    def execute(
        columns: int, seed: int = 3, *, token_dtype: torch.dtype = torch.int64, **sampling: object
    ) -> torch.Tensor:
        return registry.execute_sparse(
            torch.ones(1, columns, dtype=torch.int32),
            torch.tensor([7, 8], dtype=torch.int32),
            torch.tensor([6, 7], dtype=torch.int32),
            repair_token_ids=torch.tensor([seed - 1], dtype=token_dtype),
            seed_token_ids=torch.tensor([seed], dtype=token_dtype),
            base_positions=torch.tensor([7]),
            kv_seq_lens=torch.tensor([8], dtype=torch.int32),
            draft_input_embedding=torch.ones(2, 4),
            batch_size=1,
            speculative_tokens=3,
            vocab_size=37,
            block_size=128,
            target_step_major_layout=step_major,
            **sampling,
        )

    first_output = execute(1)
    assert update.call_args.args[-1] == step_major
    first_arena = update.call_args.args[6]
    for columns in (2, 9, 32, 1):
        execute(columns, 4)
        assert update.call_args.args[6] is first_arena
    execute(1, token_dtype=torch.int32)
    assert len(runners) == 1
    assert factory.call_count == 4  # K draft views + one target, cold path only.
    assert runners[0].capture.call_count == 1
    execute(41)
    second_arena = update.call_args.args[6]
    execute(33)
    assert update.call_args.args[6] is second_arena
    execute(65)
    assert runners[0].closed
    assert runners[1].closed
    assert registry.variant_count == 1
    assert first_output.item() == 3
    execute(1)
    assert len(runners) == 3
    assert factory.call_count == 12
    assert all(runner.capture.call_count == 1 for runner in runners)
    execute(1, logprobs=True, max_top_logprobs=3)
    assert len(runners) == 4
    assert plans[-1].logprobs and plans[-1].max_top_logprobs == 3
    execute(1)
    assert len(runners) == 4  # Plain greedy reuses its distinct output contract.


def test_sparse_role_adopts_final_storage_and_rejects_foreign_views() -> None:
    storage = MtpSparseMetadataStorage(_metadata, [6, 3, 3], 64, torch.device("cpu"))
    executor = SimpleNamespace(eager_runner=object())
    role = MtpRoleAdapter(executor, storage.metadata, speculative_tokens=3, metadata_storage=storage)
    assert role._metadata_arenas == (storage.arena,)
    assert role._metadata_by_step == storage.metadata
    original = storage.metadata[0]
    foreign = _metadata(original.block_table.clone(), original.kv_seq_lens.clone(), original.slot_mapping.clone())
    with pytest.raises(ValueError, match="belong to the supplied storage"):
        MtpRoleAdapter(executor, (foreign, *storage.metadata[1:]), speculative_tokens=3, metadata_storage=storage)
    del storage
    role._metadata_arenas[0].fill_(17)
    for metadata in role._metadata_by_step:
        for field in ("slot_mapping", "kv_seq_lens", "block_table"):
            assert torch.all(getattr(metadata, field) == 17)
    # Step destinations share an allocation but never overlap.
    role._metadata_by_step[0].block_table.fill_(29)
    assert torch.all(role._metadata_by_step[1].block_table == 17)


@pytest.mark.parametrize("invalid", ["overlap", "foreign_storage", "shape"])
def test_sparse_binding_rejects_invalid_final_destinations(invalid: str) -> None:
    draft_storage = MtpSparseMetadataStorage(_metadata, [2], 32, torch.device("cpu"))
    target_storage = MtpSparseMetadataStorage(_metadata, [2], 32, torch.device("cpu"))
    draft = draft_storage.metadata[0]
    if invalid == "overlap":
        draft.kv_seq_lens = draft.slot_mapping
    elif invalid == "foreign_storage":
        draft.kv_seq_lens = draft.kv_seq_lens.clone()
    else:
        draft.kv_seq_lens = draft.kv_seq_lens[:1]
    with pytest.raises(ValueError, match="sparse MTP"):
        MtpSparseMetadataBinding(
            (draft,),
            (draft_storage.arena,),
            target_storage.metadata,
            (target_storage.arena,),
            position_storage=MtpSparsePositionStorage(1, 1, torch.device("cpu")),
            batch_size=1,
            speculative_tokens=1,
            table_capacity=32,
            block_size=128,
            target_step_major=False,
        )


@pytest.mark.parametrize("steps", [3, 5])
def test_sparse_tables_share_only_identical_read_only_rows(steps: int) -> None:
    device = torch.device("cpu")
    batch, capacity = 3, 64
    rows = [2 * batch] + [batch] * (steps - 1)
    independent = MtpSparseMetadataStorage(_metadata, rows, capacity, device)
    shared = MtpSparseMetadataStorage(_metadata, rows, capacity, device, share_block_tables=True)
    target = MtpSparseMetadataStorage(_metadata, [batch * (steps + 1)], capacity, device)
    shared.arena.zero_()
    first, *later = shared.metadata
    assert len({item.block_table.data_ptr() for item in later}) == 1
    assert first.block_table.data_ptr() != later[0].block_table.data_ptr()
    assert len({item.kv_seq_lens.data_ptr() for item in shared.metadata}) == steps
    assert len({item.slot_mapping.data_ptr() for item in shared.metadata}) == steps
    later[0].block_table.fill_(41)
    assert all(torch.all(item.block_table == 41) for item in later)
    assert torch.all(first.block_table == 0)
    later[0].kv_seq_lens.fill_(99)
    assert torch.all(later[1].kv_seq_lens == 0)
    aligned_table_words = ((batch * capacity + 127) // 128) * 128
    assert independent.arena.numel() - shared.arena.numel() == (steps - 2) * aligned_table_words
    MtpSparseMetadataBinding(
        shared.metadata,
        (shared.arena,),
        target.metadata,
        (target.arena,),
        position_storage=MtpSparsePositionStorage(batch, steps, device),
        batch_size=batch,
        speculative_tokens=steps,
        table_capacity=capacity,
        block_size=128,
        target_step_major=False,
    )
    # Sharing applies only to identical table regions. Partial overlap with
    # another table, or aliasing a field whose value changes by step, is invalid.
    later[1].kv_seq_lens = later[0].kv_seq_lens
    with pytest.raises(ValueError, match="identical block-table"):
        MtpSparseMetadataBinding(
            shared.metadata,
            (shared.arena,),
            target.metadata,
            (target.arena,),
            position_storage=MtpSparsePositionStorage(batch, steps, device),
            batch_size=batch,
            speculative_tokens=steps,
            table_capacity=capacity,
            block_size=128,
            target_step_major=False,
        )

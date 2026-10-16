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

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.runners.unified_block_acl_graph import (
    UnifiedBlockAclGraph,
    UnifiedBlockGraphRegistry,
    _greedy_dflash2_path,
)


def test_unified_dflash2_path_tracks_predecessors_per_sequence() -> None:
    candidates = torch.tensor([[[10, 11], [20, 21], [30, 31]], [[40, 41], [50, 51], [60, 61]]])
    edges = torch.tensor(
        [
            [[[0.0, 2.0], [0.0, 2.0]], [[0.0, 3.0], [4.0, 0.0]], [[1.0, 1.0], [0.0, 5.0]]],
            [[[3.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [4.0, 0.0]], [[4.0, 0.0], [0.0, 3.0]]],
        ]
    )
    # Batch 0 follows 1 -> 0 -> 0 (tie); batch 1 follows 0 -> 1 -> 1.
    torch.testing.assert_close(_greedy_dflash2_path(candidates, edges), torch.tensor([[11, 20, 30], [40, 51, 61]]))


def _binding(rows: int, width: int, device: str, *, draft: bool) -> tuple:
    from xllm.python.attention.npu_paged_attention import _PreparedPagedAttention
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata

    ids = torch.zeros(rows * width, dtype=torch.int32, device=device)
    positions = (torch.arange(rows, device=device)[:, None] * 16 + torch.arange(width, device=device)).to(torch.int32)
    empty = torch.empty(0, dtype=torch.int32, device=device)
    table = torch.zeros((rows if draft else rows * width, 8), dtype=torch.int32, device=device)
    slots = positions.flatten() + 1
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=slots,
        paged_kv_indptr=empty,
        paged_kv_indices=empty,
        paged_kv_last_page_len=empty,
        block_table=table,
        q_seq_lens=torch.full((table.shape[0],), width if draft else 1, dtype=torch.int32, device=device),
        q_cu_seq_lens=torch.arange(1, table.shape[0] + 1, dtype=torch.int32, device=device),
        kv_seq_lens=torch.full((table.shape[0],), 32, dtype=torch.int32, device=device),
        kv_seq_lens_host_values=[32] * table.shape[0],
        is_chunked_prefill=draft,
        prepared_attention_state=_PreparedPagedAttention(table, [], [], [], kv_capacity=32),
    )
    return ids, positions.flatten(), metadata


class _ToyExecutor:
    """Deterministic tensor computation through the real composite graph recipe."""

    def __init__(self, algorithm: str, device: str, *, target: bool = False) -> None:
        self.target = target
        self.algorithm = algorithm
        self.model = self
        self.attention_backend = SimpleNamespace(page_size=16)
        self.eager_runner = SimpleNamespace(device=torch.device(device), layer_caches=[])
        self.cache = torch.full((128, 2), -50.0, device=device)
        self.last_slots = torch.zeros(6, dtype=torch.int32, device=device)
        self._target_mismatch = torch.tensor([[0, 1, 0], [0, 0, 0]], device=device)
        if algorithm == "dflash2" and not target:
            self.dflash2_candidates = self._candidates

    def create_mtp_role_backends(self, count: int) -> tuple:
        return tuple(SimpleNamespace(page_size=16) for _ in range(count))

    def execute_mtp_role(
        self, ids: torch.Tensor, positions: torch.Tensor, metadata: object, **kwargs: object
    ) -> object:
        if self.target:
            tokens = ids.view(2, 3).to(torch.int64) + 1
            # First row accepts one draft token, second accepts both plus bonus.
            hidden = (tokens + self._target_mismatch).flatten()
            context = torch.stack((positions.float(), ids.float()), dim=1)
            return hidden, context
        return positions.float().view(-1, 1)

    def compute_logits(self, hidden: torch.Tensor, selected: object) -> torch.Tensor:
        logits = torch.zeros((hidden.shape[0], 64), device=hidden.device)
        if self.algorithm != "dspark":
            logits.scatter_(1, hidden.to(torch.int64) % 64, 20.0)
        return logits

    def dspark_markov_bias(self, previous: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros((previous.numel(), 64), device=previous.device)
        return bias.scatter_(1, ((previous + 1) % 64).view(-1, 1), 20.0)

    def _candidates(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor) -> tuple:
        ids = logits.argmax(-1, keepdim=True)
        return ids, torch.zeros((*ids.shape, 1), device=ids.device)

    def compute_greedy_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.to(torch.int64)

    def write_context_kv(
        self, hidden: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, caches: object, params: object
    ) -> None:
        self.last_slots.copy_(slots)
        indices = slots.clamp_min(0).to(torch.int64)
        values = torch.where((slots >= 0)[:, None], hidden, self.cache[0])
        self.cache.index_copy_(0, indices, values)


@pytest.mark.parametrize("algorithm", ["dspark", "dflash", "dflash2"])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_composite_round_updates_only_accepted_context_and_reuses_bindings(algorithm: str, device: str) -> None:
    if device == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("NPU is unavailable")
    target = _ToyExecutor(algorithm, device, target=True)
    draft = _ToyExecutor(algorithm, device)
    target_binding = _binding(2, 3, device, draft=False)
    draft_width = 2 if algorithm == "dspark" else 3
    draft_binding = _binding(2, draft_width, device, draft=True)
    tokens = torch.empty((2, 3), dtype=torch.int64, device=device)
    lengths = torch.empty(2, dtype=torch.int32, device=device)
    stream = torch.npu.Stream() if device == "npu" else None
    pool = torch.npu.graph_pool_handle() if device == "npu" else None
    runner = UnifiedBlockAclGraph(
        target,
        draft,
        target_binding,
        draft_binding,
        tokens,
        lengths,
        algorithm == "dspark",
        2,
        lambda: None,
        lambda: None,
        lambda _: None,
        stream,
        pool,
    )
    for binding in (target_binding, draft_binding):
        binding[0].view(2, -1)[:, 0].copy_(torch.tensor([0, 16], device=device))
    if device == "npu":
        runner.capture()
        runner.execute()
    else:
        runner._run()
    torch.testing.assert_close(tokens.cpu(), torch.tensor([[1, 3, -1], [17, 18, 19]]))
    torch.testing.assert_close(lengths.cpu(), torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(draft.last_slots.cpu(), torch.tensor([1, 2, -1, 17, 18, 19], dtype=torch.int32))
    torch.testing.assert_close(draft.cache[3].cpu(), torch.tensor([-50.0, -50.0]))
    # Native context advancement publishes the last accepted token and advances
    # positions by the accepted length, not by K or by the scheduler placeholder.
    previous_tokens = tokens.gather(1, lengths.to(torch.int64)[:, None] - 1).flatten()
    saved = tokens.clone()
    for binding, width in ((target_binding, 3), (draft_binding, draft_width)):
        binding[0].view(2, width)[:, 0].copy_(previous_tokens)
        binding[1].view(2, width).add_(lengths[:, None])
        binding[2].slot_mapping.view(2, width).add_(lengths[:, None])
    # Position-based DFlash draft now intentionally disagrees with row 0's
    # corrected token. DSpark must instead condition on that corrected token.
    if device == "npu":
        runner.execute()
    else:
        runner._run()
    expected = [[4, 6, -1], [20, 21, 22]] if algorithm == "dspark" else [[4, -1, -1], [20, 21, 22]]
    torch.testing.assert_close(tokens.cpu(), torch.tensor(expected))
    torch.testing.assert_close(saved.cpu(), torch.tensor([[1, 3, -1], [17, 18, 19]]))
    torch.testing.assert_close(draft.cache[0].cpu(), torch.tensor([-50.0, -50.0]))
    if device == "npu":
        runner.close()


def test_registry_replays_mutable_slots_across_eviction_and_page_buckets_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    device = "npu"
    monkeypatch.setattr(UnifiedBlockGraphRegistry, "_MAX_VARIANTS", 2)
    captured_capacities = []

    def capture(runner: UnifiedBlockAclGraph) -> None:
        assert runner.stream is registry.stream
        assert runner.pool is registry.pool
        captured_capacities.append(runner.draft_metadata.prepared_attention_state.kv_capacity)
        runner.capture()

    target = _ToyExecutor("dspark", device, target=True)
    draft = _ToyExecutor("dspark", device)
    registry = UnifiedBlockGraphRegistry(target, draft, lambda: None, lambda: None, lambda _: None, capture)
    bindings = []
    for _ in range(3):
        bindings.append(
            (
                _binding(2, 3, device, draft=False),
                _binding(2, 2, device, draft=True),
                torch.empty((2, 3), dtype=torch.int64, device=device),
                torch.empty(2, dtype=torch.int32, device=device),
            )
        )
    # Reuse the hot Slot, evict/recapture another, then cross the 32-token
    # capacity boundary without changing any native buffer address.
    rounds = [(0, 31), (1, 31), (0, 32), (2, 31), (0, 32), (1, 32), (1, 33), (1, 34)]
    expected_slots = {}
    try:
        for step, (index, kv_length) in enumerate(rounds):
            target_binding, draft_binding, tokens, lengths = bindings[index]
            anchors = torch.tensor([2 + 4 * step, 18 + 4 * step], device=device)
            starts = torch.tensor([index * 32 + step, index * 32 + step + 16], device=device)
            for ids, positions, metadata in (target_binding, draft_binding):
                width = ids.numel() // 2
                ids.view(2, width)[:, 0].copy_(anchors)
                positions.view(2, width).copy_(starts[:, None] + torch.arange(width, device=device))
                metadata.slot_mapping.copy_(positions + 1)
                metadata.kv_seq_lens.fill_(kv_length)
                metadata.kv_seq_lens_host_values[:] = [kv_length] * metadata.block_table.shape[0]
                metadata.block_table.fill_(step + index)
            # A cache hit must overwrite the outputs, not return the last round.
            tokens.fill_(-77)
            lengths.fill_(-77)
            hidden = registry.execute(target_binding, draft_binding, tokens, lengths, True)

            first, second = anchors.cpu().tolist()
            expected_tokens = torch.tensor([[first + 1, first + 3, -1], [second + 1, second + 2, second + 3]])
            expected_lengths = torch.tensor([2, 3], dtype=torch.int32)
            expected_ids = (anchors[:, None] + torch.arange(3, device=device)).flatten()
            expected_hidden = torch.stack((target_binding[1].float(), expected_ids.float()), dim=1)
            torch.testing.assert_close(hidden, expected_hidden)
            expected_slots[index] = (expected_tokens, expected_lengths)
            # Executing one Slot must not overwrite another Slot's output buffers.
            for slot, (slot_tokens, slot_lengths) in expected_slots.items():
                torch.testing.assert_close(bindings[slot][2].cpu(), slot_tokens)
                torch.testing.assert_close(bindings[slot][3].cpu(), slot_lengths)

        assert captured_capacities == [32, 32, 32, 32, 64]
    finally:
        registry.close()

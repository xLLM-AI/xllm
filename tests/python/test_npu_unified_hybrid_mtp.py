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

"""Real unified ACL replay versus dense eager Full/GDN execution."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu", reason="requires the real NPU backend")

from tests.python.qwen3_5_test_utils import make_linear_config
from tests.python.test_qwen3_5_npu import _shared_gdn_caches
from xllm.python.attention.backend import LayerCache, detect_shared_kv_cache_layout
from xllm.python.layers.attention import Attention
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.runners.mtp_acl_graph import (
    MtpAclGraphRunner,
    MtpGraphVariantRegistry,
    _MtpPagedAttentionMetadata,
)
from xllm.python.models.qwen3_5_mtp import Qwen3_5MtpModel

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)


class _NumericalBody(nn.Module):
    def __init__(self, *, target: bool, device: torch.device) -> None:
        super().__init__()
        self.embedding = nn.Embedding(37, 128, device=device, dtype=torch.bfloat16)
        self.attention = Attention(1, 1, 128, 128**-0.5, 0, 1 if target else 0)
        self.gdn = None
        if target:
            config = make_linear_config(
                hidden_size=128,
                linear_num_key_heads=1,
                linear_num_value_heads=1,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
            )
            self.gdn = NpuQwen3_5GatedDeltaNet(config, 0, torch.bfloat16, device)
            self.execution_metadata_builders = (Qwen3_5GdnMetadataBuilder(config),)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, input_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.embedding(input_ids) + positions[:, None].to(torch.bfloat16) * 0.001
        if input_embedding is not None:
            hidden = hidden + input_embedding * 0.1
        if self.gdn is not None:
            hidden = hidden + self.gdn(hidden)
        query = hidden.view(-1, 1, 128)
        return self.attention(query, query * 0.5, query * 0.25)


class _NumericalModel(nn.Module):
    def __init__(self, *, target: bool, device: torch.device, width: int = 5) -> None:
        super().__init__()
        self._target = target
        self._width = width
        self.observed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if target:
            self.model = _NumericalBody(target=True, device=device)
        else:
            config = make_linear_config(
                hidden_size=128,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=128,
                layer_types=["full_attention"],
                linear_num_key_heads=1,
                linear_num_value_heads=1,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
            )
            self.model = Qwen3_5MtpModel(config, torch.bfloat16, device)
        self.lm_head = nn.Linear(128, 37, bias=False, device=device, dtype=torch.bfloat16)
        self.register_buffer("rejection_step", torch.tensor([width - 1], dtype=torch.long, device=device))

    def compute_greedy_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.compute_logits(hidden, None)
        tokens = logits.argmax(dim=-1)
        if not self._target:
            self.observed.append((hidden, logits, tokens))
            return tokens
        # Exercise specified accept lengths without forcing the draft's real
        # LM head. An ordinary token-by-token oracle checks every proposal.
        proposals = torch.stack([record[2] for record in self._draft.observed[-(self._width - 1) :]], dim=1)
        target_tokens = torch.cat((proposals, tokens.view(-1, self._width)[:, -1:]), dim=1)
        offsets = torch.arange(self._width, device=hidden.device)
        mismatch = offsets.eq(self.rejection_step) & offsets.lt(self._width - 1)
        return (target_tokens + mismatch.to(torch.long)).remainder(37).flatten()

    def compute_logits(self, hidden: torch.Tensor, _indices: object) -> torch.Tensor:
        return self.lm_head(hidden)


def _metadata(table: torch.Tensor, lengths: torch.Tensor, slots: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        block_table=table,
        kv_seq_lens=lengths,
        slot_mapping=slots,
        is_dummy=False,
        has_kv_shard=False,
        expanded_decode_metadata=None,
        linear_verify_graph=None,
    )


def _eager_capture(runner: MtpAclGraphRunner, *inputs: object) -> None:
    """Keep the same fixed role recipe while running the oracle eagerly."""
    runner.backend = "eager"
    runner.recipe.bind_graph_contexts(None, *runner._execution_states)
    snapshots = runner.recipe.target_forward.snapshot_cache_state()
    runner.execute(*inputs)
    for snapshot in snapshots:
        snapshot.restore()
    runner.recipe.finish_warmup()


@torch.inference_mode()
def test_unified_shared_gdn_full_target_matches_dense_eager_across_variants() -> None:
    torch.manual_seed(66301)
    device = torch.device("npu:0")
    width, speculative_tokens = 5, 4
    target = _NumericalModel(target=True, device=device, width=width)
    draft = _NumericalModel(target=False, device=device, width=width)
    for model in (target, draft):
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    target.model.gdn.norm_weight.fill_(1)
    models = [(copy.deepcopy(target), copy.deepcopy(draft)), (target, draft)]
    oracle_model = copy.deepcopy(draft)
    for target_model, draft_model in models:
        object.__setattr__(target_model, "_draft", draft_model)
    shared = _shared_gdn_caches(width, device)
    layout = detect_shared_kv_cache_layout(shared)
    assert layout is not None and layout.conv_row_stride is None
    dense = [
        LayerCache(
            key=None,
            value=None,
            conv=torch.randn(8, width + 2, 384, device=device, dtype=torch.bfloat16) * 0.05,
            ssm=torch.randn(8 * width, 1, 128, 128, device=device) * 0.05,
        ),
        LayerCache(
            key=torch.randn(8, 128, 1, 128, device=device, dtype=torch.bfloat16) * 0.05,
            value=torch.randn(8, 128, 1, 128, device=device, dtype=torch.bfloat16) * 0.05,
        ),
    ]
    state_ids = torch.arange(1, 5, device=device, dtype=torch.int64)
    conv_rows = layout.linear_state_indices(state_ids, role="conv")
    ssm_rows = (
        layout.linear_state_indices(state_ids, role="ssm")[:, None] + torch.arange(width, device=device)
    ).flatten()
    dense_ssm_rows = (state_ids[:, None] * width + torch.arange(width, device=device)).flatten()
    shared[0].conv.index_copy_(0, conv_rows, dense[0].conv[1:5])
    shared[0].ssm.index_copy_(0, ssm_rows, dense[0].ssm.index_select(0, dense_ssm_rows))
    full_ids = torch.tensor([5, 6], device=device, dtype=torch.int64)
    full_rows = layout.full_block_table(full_ids)
    for field in ("key", "value"):
        getattr(shared[1], field).index_copy_(0, full_rows, getattr(dense[1], field).index_select(0, full_ids))
    draft_cache = LayerCache(
        key=torch.randn(8, 128, 1, 128, device=device, dtype=torch.bfloat16) * 0.05,
        value=torch.randn(8, 128, 1, 128, device=device, dtype=torch.bfloat16) * 0.05,
    )
    draft_caches = [copy.deepcopy(draft_cache), draft_cache]
    oracle_cache = copy.deepcopy(draft_cache)
    oracle_executor = ModelExecutor(oracle_model, {"model_type": "qwen3_5_mtp", "python_graph_backend": "off"}, 2)
    oracle_executor.bind_kv_caches([oracle_cache])
    registries, executors = [], []
    for index, ((target_model, draft_model), target_caches) in enumerate(zip(models, (dense, shared))):
        target_executor = ModelExecutor(
            target_model,
            {"model_type": "qwen3_5_text", "num_speculative_tokens": speculative_tokens, "python_graph_backend": "off"},
            2,
            num_decoding_tokens=width,
        )
        draft_executor = ModelExecutor(draft_model, {"model_type": "qwen3_5_mtp", "python_graph_backend": "off"}, 2)
        target_executor.bind_kv_caches(target_caches)
        draft_executor.bind_kv_caches([draft_caches[index]])
        executors.append((target_executor, draft_executor))
        registries.append(
            MtpGraphVariantRegistry(
                target_executor,
                draft_executor,
                draft_metadata_factory=_metadata,
                target_metadata_factory=_metadata,
                capture_runner=_eager_capture if index == 0 else None,
            )
        )
    ordinary_shared = MtpGraphVariantRegistry(
        *executors[1],
        draft_metadata_factory=_metadata,
        target_metadata_factory=_metadata,
        capture_runner=_eager_capture,
    )
    contexts = {}
    graph_draft_records = {}
    # L=1 bootstrap -> L=4 -> L=5 -> L=1, then an ordinary request between
    # two replays and L=3. Request IDs reorder and the B1 variant captures live
    # state. Position coordinates deliberately differ from KV length by 70+.
    cases = (
        ([1, 3], [5, 6], 3, False),
        ([3, 1], [6, 5], 4, False),
        ([1], [5], 0, False),
        ([3, 1], [6, 5], 2, True),
        ([1, 3], [5, 6], 4, False),
        ([2, 4], [6, 5], 0, False),
    )

    def ordinary_token(
        token: torch.Tensor, hidden: torch.Tensor, position: int, length: int, page: int
    ) -> torch.Tensor:
        source = _metadata(
            torch.tensor([[page]], device=device, dtype=torch.int32),
            torch.tensor([length], device=device, dtype=torch.int32),
            torch.tensor([page * 128 + length - 1], device=device, dtype=torch.int32),
        )
        metadata = _MtpPagedAttentionMetadata(source, batch_size=1, query_width=1, block_size=128)
        metadata.prepared_attention_state = oracle_executor.attention_backend.prepare_metadata(
            metadata, device_kv_lengths=True
        )
        return oracle_executor.execute_mtp_role(
            token.reshape(1), torch.tensor([position], device=device), metadata, hidden.reshape(1, -1)
        )

    try:
        for logical_states, pages, accepted_drafts, ordinary_route in cases:
            batch = len(logical_states)
            rows = []
            for state in logical_states:
                if state not in contexts:
                    tokens = torch.full((width,), -1, device=device, dtype=torch.long)
                    tokens[0] = 4
                    hidden = torch.zeros(width, 128, device=device, dtype=torch.bfloat16)
                    hidden[0].normal_(std=0.1)
                    contexts[state] = (tokens, hidden, 99 + state * 7, 7 + state)
                rows.append(contexts[state])
            prefix_tokens = torch.stack([row[0] for row in rows])
            embedding = torch.stack([row[1] for row in rows]).flatten(0, 1)
            accepted = prefix_tokens.ge(0).sum(1).long()
            lengths = [row[3] + int(length) for row, length in zip(rows, accepted.cpu().tolist())]
            positions = [row[2] + int(length) for row, length in zip(rows, accepted.cpu().tolist())]
            offsets = torch.arange(width, device=device)
            first_positions = torch.tensor([row[2] for row in rows], device=device)[:, None] + offsets
            first_lengths = torch.tensor([row[3] for row in rows], device=device)[:, None] + offsets
            first_slots = torch.tensor(pages, device=device)[:, None] * 128 + first_lengths - 1
            selected_rows = torch.arange(batch, device=device) * width + accepted - 1
            seed = prefix_tokens.flatten().index_select(0, selected_rows)
            outputs = []
            for index, ((target_model, _draft_model), registry) in enumerate(zip(models, registries)):
                target_model.rejection_step.fill_(accepted_drafts)
                if index == 1 and ordinary_route:
                    registry = ordinary_shared
                outputs.append(
                    registry.execute_sparse(
                        torch.tensor(pages, device=device, dtype=torch.int32)[:, None],
                        first_lengths.int().flatten(),
                        first_slots.int().flatten(),
                        prefix_tokens[:, 0],
                        seed,
                        torch.tensor(positions, device=device),
                        torch.tensor(lengths, device=device, dtype=torch.int32),
                        embedding,
                        batch,
                        speculative_tokens,
                        37,
                        128,
                        linear_state_indices=torch.tensor(logical_states, device=device, dtype=torch.int32),
                        num_accepted_tokens=accepted.int(),
                        first_draft_token_ids=prefix_tokens.clamp_min(0).flatten(),
                        first_draft_positions=first_positions.flatten(),
                        first_draft_selected_rows=selected_rows,
                    )
                )
            if ordinary_route:
                actual_draft = tuple(draft.observed[-speculative_tokens:])
            else:
                graph_draft_records.setdefault(batch, tuple(draft.observed[-speculative_tokens:]))
                actual_draft = graph_draft_records[batch]
            # Independent oracle: only accepted target-conditioned rows run,
            # one real Qwen MTP token at a time. It never calls the recipe,
            # registry, fixed-W metadata binding, or acceptance kernel.
            for request, (row, length, page) in enumerate(zip(rows, accepted.cpu().tolist(), pages)):
                tokens, target_hidden, old_position, old_length = row
                hidden = None
                for prefix_row in range(length):
                    hidden = ordinary_token(
                        tokens[prefix_row],
                        target_hidden[prefix_row],
                        old_position + prefix_row,
                        old_length + prefix_row,
                        page,
                    )
                for step in range(speculative_tokens):
                    if step:
                        hidden = ordinary_token(
                            proposal, hidden, positions[request] + step - 1, lengths[request] + step - 1, page
                        )
                    logits = oracle_model.compute_logits(hidden, None)
                    proposal = logits.argmax(-1)
                    observed_hidden, observed_logits, observed_tokens = actual_draft[step]
                    torch.testing.assert_close(observed_hidden[request : request + 1], hidden, rtol=0.02, atol=0.002)
                    torch.testing.assert_close(observed_logits[request : request + 1], logits, rtol=0.02, atol=0.002)
                    torch.testing.assert_close(observed_tokens[request : request + 1], proposal, rtol=0, atol=0)
            expected, actual = outputs
            assert actual.accepted_count.cpu().tolist() == [accepted_drafts] * batch
            torch.testing.assert_close(actual.committed_tokens, expected.committed_tokens, rtol=0, atol=0)
            torch.testing.assert_close(actual.target_embeddings, expected.target_embeddings, rtol=0.02, atol=0.002)
            torch.testing.assert_close(
                shared[0].conv.index_select(0, conv_rows), dense[0].conv[1:5], rtol=0.02, atol=0.002
            )
            torch.testing.assert_close(
                shared[0].ssm.index_select(0, ssm_rows),
                dense[0].ssm.index_select(0, dense_ssm_rows),
                rtol=0.01,
                atol=5e-5,
            )
            for field in ("key", "value"):
                torch.testing.assert_close(
                    getattr(shared[1], field).index_select(0, full_rows),
                    getattr(dense[1], field).index_select(0, full_ids),
                    rtol=0.02,
                    atol=0.002,
                )
                torch.testing.assert_close(
                    getattr(draft_caches[1], field), getattr(draft_caches[0], field), rtol=0.02, atol=0.002
                )
            for request, state in enumerate(logical_states):
                contexts[state] = (
                    actual.committed_tokens[request].clone(),
                    actual.target_prefix_embeddings[request].clone(),
                    positions[request],
                    lengths[request],
                )
        assert registries[1].variant_count == 2
    finally:
        for registry in [*registries, ordinary_shared]:
            for entry in tuple(registry._variants.values()):
                entry.runner.close()

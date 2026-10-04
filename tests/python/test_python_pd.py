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

"""Python PD cache reconstruction and per-layer completion contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.attention.backend import AttentionMetadata, MlaIndexContext
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
    in_acl_graph,
)
from xllm.python.models.deepseek_v32 import DeepseekV3Model
from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextIndexer, Glm5NextModel
from xllm.python.models.glm5_next_kpool import alloc_pool_cache, compress_completed_pools
from xllm.python.models.glm5_next_mtp import Glm5NextMtpModel


class _IndexHistory:
    def __init__(self, cache: torch.Tensor, block_table: torch.Tensor) -> None:
        self.cache = cache
        self.block_table = block_table
        self.graph_index_history_max_kv = block_table.shape[1] * cache.shape[1]
        self.length = 0

    def write(self, rows: torch.Tensor, positions: torch.Tensor) -> None:
        page_size = self.cache.shape[1]
        slots = self.block_table[0, positions // page_size] * page_size + positions % page_size
        self.cache.flatten(0, 2).index_copy_(0, slots, rows)

    def gather_index_history(self, layer: Attention | None) -> torch.Tensor:
        del layer
        history = self.cache[self.block_table[0]].reshape(1, -1, self.cache.shape[-1])
        if in_acl_graph():
            # Mirror fixed-capacity graph history and live-length masking;
            # this fixture exercises tensor semantics, not hardware replay.
            row_valid = torch.arange(history.shape[1]) < self.length
            history = history * row_valid[None, :, None].to(history.dtype)
        else:
            history = history[:, : self.length]
        return history

    def context(self, positions: torch.Tensor, length: int) -> MlaIndexContext:
        self.length = length

        def update(rows: torch.Tensor, scales: torch.Tensor | None) -> None:
            assert scales is None
            self.write(rows, positions)

        return MlaIndexContext(
            index_cache=self.cache,
            slot_mapping=positions,
            block_table=self.block_table,
            actual_seq_q=torch.tensor([positions.numel()]),
            actual_seq_kv=torch.tensor([length]),
            index_cache_scale=None,
            get_quant_indexer_metadata=MagicMock(),
            update_index_cache=update,
            materialize_index_cache=MagicMock(),
        )


@pytest.mark.parametrize("prompt_length", [11, 12])
@pytest.mark.parametrize("stale_pool", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_pd_kpool_decode_uses_transferred_history(prompt_length: int, stale_pool: bool, graph_mode: bool) -> None:
    cfg = Glm5NextConfig(
        hidden_size=8,
        q_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        index_kpool_compress=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        indexer = Glm5NextIndexer(cfg, 0, torch.bfloat16, torch.device("cpu")).to(torch.bfloat16)
        # Linear/LayerNorm constructors initialize their weights and APE is
        # zero-filled. The compression gate is the only empty Parameter.
        nn.init.normal_(indexer.index_kpool_compress_gate)
        hidden = torch.randn(1, prompt_length + 2, cfg.hidden_size, dtype=torch.bfloat16)
        queries = torch.randn(1, 2, cfg.q_lora_rank, dtype=torch.bfloat16)
    shape = (5, 4, 1, 2 * cfg.index_head_dim + 1)
    source = _IndexHistory(torch.zeros(shape, dtype=torch.bfloat16), torch.tensor([[3, 1, 4, 2]]))
    # Reused blocks can retain keys, gates and valid=1 beyond the live prefix.
    target = _IndexHistory(torch.ones(shape, dtype=torch.bfloat16), torch.tensor([[2, 4, 1, 3]]))
    prompt_positions = torch.arange(prompt_length)
    prompt_rows = indexer.get_packed_states(
        hidden[:, :prompt_length], torch.ones(1, prompt_length, dtype=torch.bool)
    ).reshape(prompt_length, -1)
    source.write(prompt_rows, prompt_positions)
    # Transfer only framework-owned raw rows, remapping physical block IDs.
    source.length = prompt_length
    target.write(source.gather_index_history(None).reshape(prompt_length, -1), prompt_positions)
    local_pool = alloc_pool_cache(source.cache, cfg.index_kpool)
    compress_completed_pools(
        source.cache,
        local_pool,
        source.block_table,
        prompt_positions,
        indexer.index_kpool_compress_ape,
        cfg.index_head_dim,
        cfg.index_kpool,
    )
    layer = Attention(2, 1, 4, 0.5, 0, 0)

    def _select(history: _IndexHistory, step: int) -> torch.Tensor:
        position = torch.tensor([prompt_length + step])
        context = ForwardContext(
            attention_backend=history,
            device=torch.device("cpu"),
            metadata=MagicMock(spec=AttentionMetadata),
            layer_caches=[],
            execution_state=AclGraphExecutionState({}) if graph_mode else None,
        )
        with forward_context(context):
            return indexer.select_qli(
                hidden[:, prompt_length + step : prompt_length + step + 1],
                queries[:, step : step + 1].reshape(1, -1),
                position,
                torch.ones(1, 1, dtype=torch.bool),
                history.context(position, prompt_length + step + 1),
                layer,
                history,
            )

    for step in range(2):
        layer.cache_transfer_enabled = False
        indexer._pool_caches[0] = local_pool
        expected = _select(source, step)
        layer.cache_transfer_enabled = True
        indexer._pool_caches.clear()
        if stale_pool:
            indexer._pool_caches[0] = torch.full_like(local_pool, float("nan"))
        actual = _select(target, step)
        # Compare ordered indices: candidate width can change top-k tie ordering.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bool((expected >= 0).any())


class _RecordingSynchronizer:
    def __init__(self, trace: list[tuple[str, int]]) -> None:
        self.trace = trace

    def record_event(self, layer_id: int) -> bool:
        self.trace.append(("event", layer_id))
        return True


class _Decoder(nn.Module):
    def __init__(self, trace: list[tuple[str, int]], layer_id: int, model_type: str) -> None:
        super().__init__()
        self.trace = trace
        self.layer_id = layer_id
        self.model_type = model_type

    def forward(self, hidden: torch.Tensor, *args: object) -> object:
        self.trace.append(("compute", self.layer_id))
        if self.model_type == "mtp":
            return hidden + 1
        return hidden + 1, None


class _HyperHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(2)


@pytest.mark.parametrize("model_type", ["glm", "mtp"])
def test_model_records_each_event_after_its_layer(model_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _concat_hidden(embed: torch.Tensor, carried: torch.Tensor, *norm_args: object) -> torch.Tensor:
        return torch.cat((embed, carried), dim=-1)

    # This test tracks layer completion; normalization stays outside its scope.
    monkeypatch.setattr("xllm.python.models.glm5_next_mtp.kernels.fused_eh_norm", _concat_hidden)
    model_class = Glm5NextMtpModel if model_type == "mtp" else Glm5NextModel
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    trace: list[tuple[str, int]] = []
    layer_count = 1 if model_type == "mtp" else 3
    model.layers = nn.ModuleList(_Decoder(trace, i, model_type) for i in range(layer_count))
    model.cfg = SimpleNamespace(hidden_size=4, hc_mult=2)
    model.embed_tokens = nn.Embedding(8, 4)
    model.norm = nn.Identity()
    model.hc_head = _HyperHead()
    model._inputs_embeds = None
    model.enorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    model.hnorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    model.eh_proj = nn.Linear(8, 4)
    context = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(spec=AttentionMetadata),
        layer_caches=[],
        layer_synchronizer=_RecordingSynchronizer(trace),
    )
    with forward_context(context):
        model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert trace == [(action, i) for i in range(layer_count) for action in ("compute", "event")]


def test_deepseek_records_transfer_event() -> None:
    trace: list[tuple[str, int]] = []
    context = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(spec=AttentionMetadata),
        layer_caches=[],
        layer_synchronizer=_RecordingSynchronizer(trace),
    )
    with forward_context(context):
        DeepseekV3Model._record_layer_event(None, 2)
    assert trace == [("event", 2)]

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

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.attention.mla_index_context import KPoolIndexContext, KPoolMlaIndexContext
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.deepseek_v32 import DeepseekV3Model
from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextIndexer, Glm5NextModel
from xllm.python.models.glm5_next_kpool import update_compressed_kpool
from xllm.python.models.glm5_next_mtp import Glm5NextMtpModel


class _IndexHistory:
    def __init__(self, cache: torch.Tensor, tail: torch.Tensor, block_table: torch.Tensor) -> None:
        self.cache = cache
        self.kpool_tail = tail
        self.block_table = block_table

    def context(self, positions: torch.Tensor, length: int) -> KPoolMlaIndexContext:
        return KPoolMlaIndexContext(
            index_cache=self.cache,
            slot_mapping=positions,
            block_table=self.block_table,
            actual_seq_q=torch.tensor([positions.numel()], device=positions.device),
            actual_seq_kv=torch.tensor([length], device=positions.device),
            index_cache_scale=None,
            get_quant_indexer_metadata=MagicMock(),
            update_index_cache=lambda rows, scales: None,
            materialize_index_cache=MagicMock(),
            kpool=KPoolIndexContext(
                tail=self.kpool_tail,
                tail_read_indices=torch.tensor([1], dtype=torch.int32, device=positions.device),
                tail_write_indices=torch.tensor([1], dtype=torch.int32, device=positions.device),
                query_lens=[1],
                query_lens_device=torch.tensor([1], dtype=torch.int32, device=positions.device),
            ),
        )


@pytest.mark.parametrize("prompt_length", [11, 12])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_pd_kpool_decode_uses_transferred_history(prompt_length: int, graph_mode: bool) -> None:
    cfg = Glm5NextConfig(
        hidden_size=8,
        q_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        index_kpool_compress=True,
        requires_kpool_tail_cache=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        indexer = Glm5NextIndexer(cfg, 0, torch.bfloat16, torch.device("cpu")).to(torch.bfloat16)
        # Linear/LayerNorm constructors initialize their weights and APE is
        # zero-filled. The compression gate is the only empty Parameter.
        nn.init.normal_(indexer.index_kpool_compress_gate)
        hidden = torch.randn(1, prompt_length + 2, cfg.hidden_size, dtype=torch.bfloat16)
        queries = torch.randn(1, 2, cfg.q_lora_rank, dtype=torch.bfloat16)
    source = _IndexHistory(
        torch.zeros(5, 4, 1, cfg.index_head_dim, dtype=torch.bfloat16),
        torch.full((2, 2, 4, cfg.index_head_dim), float("-inf"), dtype=torch.bfloat16),
        torch.tensor([[3, 1, 4, 2]]),
    )
    prompt_positions = torch.arange(prompt_length)
    prompt_rows = indexer.get_packed_states(
        hidden[:, :prompt_length], torch.ones(1, prompt_length, dtype=torch.bool)
    ).reshape(prompt_length, -1)
    raw_k, raw_gate, valid = torch.split(prompt_rows, [cfg.index_head_dim, cfg.index_head_dim, 1], dim=-1)
    update_compressed_kpool(
        raw_k,
        raw_gate,
        valid.reshape(-1).bool(),
        prompt_positions,
        source.cache,
        source.kpool_tail,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        source.block_table,
        [prompt_length],
        indexer.index_kpool_compress_ape,
        cfg.index_kpool,
    )
    target = _IndexHistory(source.cache.clone(), source.kpool_tail.clone(), source.block_table)
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
        expected = _select(source, step)
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
    model.enorm = SimpleNamespace(weight=torch.ones(4), eps=1e-6)
    model.hnorm = SimpleNamespace(weight=torch.ones(4), eps=1e-6)
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

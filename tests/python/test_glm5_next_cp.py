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
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.models import glm5_next


class _Embedding(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._events.append("embedding")
        return input_ids.to(torch.float32).unsqueeze(-1)


class _DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, events: list[str]) -> None:
        super().__init__()
        self._layer_id = layer_id
        self._events = events
        self.positions: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.positions = positions
        self.attention_mask = attention_mask
        self.prev_topk = prev_topk
        self.output_topk = hidden[:, :1, 0].clone()
        self._events.append(f"layer_{self._layer_id}")
        return hidden + self._layer_id + 1, self.output_topk


class _Norm(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._events.append("norm")
        return hidden


class _HyperHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=2)


def _make_model(events: list[str]) -> tuple[glm5_next.Glm5NextModel, list[_DecoderLayer]]:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hidden_size=1, hc_mult=4)
    model.embed_tokens = _Embedding(events)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm(events)
    model.hc_head = _HyperHead()
    model._inputs_embeds = None
    return model, layers


def test_cp_model_loop_shards_rows_and_merges_final_hidden() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = SimpleNamespace(shard_valid_mask=torch.tensor([True, False]))
    merged_output = torch.tensor([[13.0], [23.0], [33.0], [43.0]])

    def shard_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_rows")
        return hidden.index_select(0, torch.tensor([3, 0]))

    def shard_positions(positions: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_positions")
        return positions.index_select(0, torch.tensor([3, 0]))

    def merge_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        torch.testing.assert_close(hidden, torch.tensor([[43.0], [13.0]]))
        events.append("merge_rows")
        return merged_output

    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(glm5_next, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_next, "cp_shard_positions", side_effect=shard_positions),
        patch.object(glm5_next, "cp_merge_rows", side_effect=merge_rows),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert events == [
        "embedding",
        "shard_rows",
        "shard_positions",
        "layer_0",
        "layer_1",
        "norm",
        "merge_rows",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[3, 0]], dtype=torch.int32))
    torch.testing.assert_close(layers[0].attention_mask, torch.tensor([[True, False]]))
    assert layers[1].prev_topk is layers[0].output_topk
    torch.testing.assert_close(output, merged_output)


def _cp2_context(rank: int) -> SimpleNamespace:
    shard_index = torch.tensor([0, 3], dtype=torch.int64) if rank == 0 else torch.tensor([1, 2], dtype=torch.int64)
    return SimpleNamespace(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=shard_index,
        shard_gather_index=shard_index,
        shard_valid_mask=torch.ones(2, dtype=torch.bool),
        restore_index=torch.tensor([0, 2, 3, 1], dtype=torch.int64),
    )


def _patch_cp_gather(monkeypatch, remote_by_local: dict[tuple[object, ...], torch.Tensor]) -> MagicMock:
    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        key = tuple(value.reshape(-1).tolist())
        return torch.cat([value, remote_by_local[key]], dim=0)

    gather = MagicMock(side_effect=all_gather)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    return gather


def _make_kda_attention() -> glm5_next.Glm5NextKdaAttention:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 1
    attention.layer_id = 0
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.conv_weight_t = torch.zeros(3, 2)
    attention.activation = "silu"
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()
    return attention


def test_kda_cp_materializes_global_rows_and_reshards_output(monkeypatch) -> None:
    attention = _make_kda_attention()

    cp_context = _cp2_context(0)
    local_hidden = torch.tensor([[[1.0], [4.0]]])
    remote_hidden = torch.tensor([[2.0], [3.0]])
    remote_by_local = {
        (1.0, 1.0, 1.0, 4.0, 4.0, 4.0): remote_hidden.expand(-1, 3),
        (10.0, 40.0): remote_hidden.mul(10).unsqueeze(-1),
        tuple(torch.sigmoid(local_hidden).reshape(-1).tolist()): torch.sigmoid(remote_hidden),
    }
    gather = _patch_cp_gather(monkeypatch, remote_by_local)

    backend = MagicMock()
    backend.execute_linear.side_effect = (
        lambda mixed_qkv, _beta, _layer, _weight, _gate, _activation, *, raw_gate_proj: (
            mixed_qkv[:, :1].transpose(1, 2).unsqueeze(2)
        )
    )
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
    ):
        output = attention(local_hidden, torch.tensor([[0, 3]], dtype=torch.int32), torch.tensor([[True, True]]))

    mixed_qkv, beta, layer, conv_weight, forget_gate, activation = backend.execute_linear.call_args.args
    assert conv_weight is attention.conv_weight_t
    assert forget_gate is attention.forget_gate
    assert activation == attention.activation
    raw_gate_proj = backend.execute_linear.call_args.kwargs["raw_gate_proj"]
    assert layer == attention.layer_id
    expected_hidden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(mixed_qkv, expected_hidden.view(1, 1, 4).expand(1, 3, 4))
    torch.testing.assert_close(raw_gate_proj, expected_hidden.view(1, 4, 1, 1) * 10)
    torch.testing.assert_close(beta, torch.sigmoid(expected_hidden).view(1, 4, 1))
    assert gather.call_count == 3
    torch.testing.assert_close(attention.o_norm.call_args.args[1], local_hidden.unsqueeze(-1) * 100)
    torch.testing.assert_close(output, torch.tensor([[[101.0], [404.0]]]))


def test_kda_cp_one_keeps_full_postprocessing() -> None:
    attention = _make_kda_attention()
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.execute_linear.return_value = hidden.unsqueeze(2)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=None),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output = attention(hidden, torch.tensor([[0, 1, 2]], dtype=torch.int32), torch.ones(1, 3, dtype=torch.bool))

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden * 101)


def _cp2_padded_context() -> SimpleNamespace:
    return SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        total_local=2,
        shard_index=torch.tensor([0, -1], dtype=torch.int64),
        shard_gather_index=torch.tensor([0, 0], dtype=torch.int64),
        shard_valid_mask=torch.tensor([True, False]),
        restore_index=torch.tensor([0, 2, 3], dtype=torch.int64),
    )


class _KdaNorm(nn.Module):
    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        assert core.shape == (1, 2, 2, 2)
        assert gate.shape == (1, 2, 2, 2)
        return core


class _OutputProjection(nn.Module):
    """Capture the input shape and emit nonfinite values to prove zero-filling."""

    def __init__(self, expected_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.expected_shape = expected_shape
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        assert hidden.shape == self.expected_shape
        rows = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]], device=hidden.device)
        return rows.expand(self.expected_shape[:-1] + (3,))


def test_kda_cp_projects_only_local_rows_and_overwrites_padding(monkeypatch) -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 3
    attention.head_dim = 2
    attention.qkv_dim = 4
    attention.conv_dim = 12
    attention.layer_id = 0
    attention.conv_weight_t = torch.zeros(12, 2)
    attention.forget_gate = SimpleNamespace()
    attention.activation = "silu"
    attention.num_heads_local = 2
    attention.tp = 2
    attention.input_projection_sizes = (12, 2, 4)
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (
            hidden[..., :1].expand(-1, -1, 12),
            hidden[..., :1].expand(-1, -1, 2),
            hidden[..., :1].expand(-1, -1, 4),
        ),
        dim=-1,
    )
    attention._project_fg = lambda latents: (latents.view(1, 2, 2, 2), latents.view(1, 2, 2, 2))
    attention.o_norm = _KdaNorm()
    attention.o_proj = _OutputProjection((1, 2, 4))

    cp_context = _cp2_padded_context()
    local_hidden = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    global_core = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    backend = MagicMock()
    backend.execute_linear.return_value = global_core
    merge = MagicMock(
        side_effect=[
            torch.ones(3, 12),
            torch.ones(3, 2, 2),
            torch.ones(3, 2),
        ]
    )

    def compensate(output: torch.Tensor) -> None:
        output.add_(7)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=compensate, create=True) as all_reduce,
    ):
        output = attention(local_hidden, torch.tensor([[0, 0]], dtype=torch.int32), torch.tensor([[True, False]]))

    assert merge.call_count == 3
    assert backend.execute_linear.call_args.args[0].shape == (1, 12, 3)
    assert backend.execute_linear.call_args.args[1].shape == (1, 3, 2)
    assert backend.execute_linear.call_args.kwargs["raw_gate_proj"].shape == (1, 3, 2, 2)
    assert attention.o_proj.input_shape == (1, 2, 4)
    all_reduce.assert_called_once()
    torch.testing.assert_close(output[0, 0], torch.tensor([8.0, 9.0, 10.0]))
    assert torch.equal(output[0, 1], torch.zeros(3))


def _make_mla_attention(indexer: object | None) -> glm5_next.Glm5NextMlaAttention:
    attention = glm5_next.Glm5NextMlaAttention.__new__(glm5_next.Glm5NextMlaAttention)
    nn.Module.__init__(attention)
    attention.hidden_size = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 0
    attention.qk_head_dim = 1
    attention.kv_lora_rank = 1
    attention.num_heads_local = 1
    attention.v_head_dim = 1
    attention.cfg = SimpleNamespace(tp_size=1)
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention._qkv_a_weight = None
    attention.o_proj = nn.Identity()
    attention.register_buffer("W_UK", torch.ones(1, 1, 1), persistent=False)
    attention.register_buffer("W_UV", torch.ones(1, 1, 1), persistent=False)
    attention.indexer = indexer
    return attention


def test_mla_cp_one_keeps_full_postprocessing() -> None:
    attention = _make_mla_attention(MagicMock())
    attention.indexer.select_qli.return_value = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32)
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = hidden.reshape(3, 1, 1)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output, topk = attention(
            hidden,
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.ones(1, 3, dtype=torch.bool),
        )

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden.reshape(3, 1))
    torch.testing.assert_close(topk, torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32))


def test_mla_cp_projects_only_local_rows_and_overwrites_padding(monkeypatch) -> None:
    indexer = MagicMock()
    global_topk = torch.arange(6, dtype=torch.int32).view(3, 1, 2)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    attention.hidden_size = 3
    attention.qk_nope_head_dim = 2
    attention.qk_head_dim = 2
    attention.kv_lora_rank = 3
    attention.num_heads_local = 2
    attention.v_head_dim = 3
    attention.cfg = SimpleNamespace(tp_size=2)
    attention.q_b_proj = nn.Linear(3, 4, bias=False)
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.o_proj = _OutputProjection((2, 6))
    attention.W_UK = torch.ones(2, 2, 3)
    attention.W_UV = torch.ones(2, 3, 3)

    cp_context = _cp2_padded_context()
    global_hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    global_positions = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    global_attn_out = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = global_attn_out
    merge = MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)])

    def compensate(output: torch.Tensor) -> None:
        output.add_(5)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend, layer_shared_cache={}),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=compensate, create=True) as all_reduce,
    ):
        output, topk = attention(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
        )

    assert merge.call_count == 2
    assert backend.execute_mla.call_args.args[0].shape == (3, 2, 3)
    assert backend.execute_mla.call_args.args[2].shape == (3, 1, 3)
    assert attention.o_proj.input_shape == (2, 6)
    all_reduce.assert_called_once()
    torch.testing.assert_close(output[0], torch.tensor([6.0, 7.0, 8.0]))
    assert torch.equal(output[1], torch.zeros(3))
    torch.testing.assert_close(topk[0], global_topk[0])
    torch.testing.assert_close(topk[1], torch.zeros((1, 2), dtype=torch.int32))


@pytest.mark.parametrize("num_layers", [1, 2])
def test_dsa_cp_full_indexer_ignores_previous_topk_and_reshards_output(
    monkeypatch: pytest.MonkeyPatch, num_layers: int
) -> None:
    indexer = MagicMock()
    global_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    indexer.select_qli.return_value = global_topk
    attentions = [_make_mla_attention(indexer) for _ in range(num_layers)]
    cp_context = _cp2_context(0)
    previous_topk = torch.tensor([[[50]], [[80]]], dtype=torch.int32)
    remote_by_local = {
        (1.0, 4.0): torch.tensor([[2.0], [3.0]]),
        (0, 3): torch.tensor([[1], [2]], dtype=torch.int32),
    }
    gather = _patch_cp_gather(monkeypatch, remote_by_local)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with patch.object(
        glm5_next,
        "get_forward_context",
        return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend, layer_shared_cache={}),
    ):
        outputs = [
            attention(
                torch.tensor([[[1.0], [4.0]]]),
                torch.tensor([[0, 3]], dtype=torch.int32),
                torch.tensor([[True, False]]),
                previous_topk,
            )
            for attention in attentions
        ]

    assert gather.call_count == num_layers + 1
    assert indexer.select_qli.call_count == num_layers
    first_index_args = indexer.select_qli.call_args_list[0].args
    for call in indexer.select_qli.call_args_list:
        assert call.args[2] is first_index_args[2]
        assert call.args[3] is first_index_args[3]
    index_args = indexer.select_qli.call_args.args
    torch.testing.assert_close(index_args[0], torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]))
    torch.testing.assert_close(index_args[2], torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
    torch.testing.assert_close(index_args[3], torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    for output, topk in outputs:
        torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
        torch.testing.assert_close(topk, torch.tensor([[[10]], [[40]]], dtype=torch.int32))


def test_dsa_cp_gathers_shared_topk_before_attention(monkeypatch) -> None:
    attention = _make_mla_attention(None)
    cp_context = _cp2_context(0)
    local_topk = torch.tensor([[[10]], [[40]]], dtype=torch.int32)
    remote_by_local = {
        (1.0, 4.0): torch.tensor([[2.0], [3.0]]),
        (0, 3): torch.tensor([[1], [2]], dtype=torch.int32),
        (10, 40): torch.tensor([[20], [30]], dtype=torch.int32),
    }
    gather = _patch_cp_gather(monkeypatch, remote_by_local)
    backend = MagicMock()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with patch.object(
        glm5_next,
        "get_forward_context",
        return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend, layer_shared_cache={}),
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
            local_topk,
        )

    expected_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    assert gather.call_count == 3
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], expected_topk)
    torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
    torch.testing.assert_close(topk, local_topk)


def test_cp_one_preserves_full_rows_without_shard_or_merge() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
        patch.object(glm5_next, "cp_shard_positions") as shard_positions,
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    shard_rows.assert_not_called()
    shard_positions.assert_not_called()
    merge_rows.assert_not_called()
    assert events == ["embedding", "layer_0", "layer_1", "norm"]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
    torch.testing.assert_close(output, torch.tensor([[13.0], [23.0], [33.0], [43.0]]))

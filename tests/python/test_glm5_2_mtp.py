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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import deepseek_v32, glm5_2_mtp


def _config(**overrides) -> dict:
    config = {
        "model_type": "glm_moe_dsa_mtp",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "indexer_types": ["full"],
        "index_topk_pattern": "S",
        "index_share_for_mtp_iteration": True,
        "mlp_layer_types": ["dense"],
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 1,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("rank", range(4))
def test_glm_mtp_dp_experts_use_cross_dp_moe_tp(rank: int) -> None:
    draft = glm5_2_mtp.Glm52MtpForCausalLM(
        _config(
            tp_size=2,
            tp_rank=rank % 2,
            dp_size=2,
            dp_rank=rank // 2,
            world_size=4,
            moe_tp_size=4,
            moe_tp_rank=rank,
            n_routed_experts=8,
            n_shared_experts=1,
            moe_intermediate_size=8,
            mlp_layer_types=["moe"],
            first_k_dense_replace=0,
        )
    )
    moe = draft.model.layers[0].mlp
    assert moe.num_local_experts == 8
    assert moe.inter_local == 2
    assert moe.shared_experts.tp == 4
    assert moe.shared_experts.down_proj.in_features == 2
    assert draft.cfg.attention_weight_shard() == (2, rank % 2)


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack((input_ids.float(), input_ids.float()), dim=-1)


class _SelectTokenEmbedding(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, :2]


class _ScaleNorm(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 2


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reuse_flags: list[bool] = []
        self.ropes: list[tuple[torch.Tensor, ...]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor | None,
        reuse_topk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del residual
        self.ropes.append((half_rope_cos, half_rope_sin, rope_cos, rope_sin))
        assert query_cos_sin[0] is rope_cos and query_cos_sin[1] is rope_sin
        self.reuse_flags.append(reuse_topk)
        if not reuse_topk:
            topk_indices = torch.full((hidden.shape[0], 1, 2), 7, dtype=torch.int32)
        assert topk_indices is not None
        return hidden, torch.zeros_like(hidden), topk_indices


def _mtp_body() -> tuple[glm5_2_mtp.Glm52MtpModel, _DecoderLayer]:
    body = glm5_2_mtp.Glm52MtpModel.__new__(glm5_2_mtp.Glm52MtpModel)
    nn.Module.__init__(body)
    body.cfg = SimpleNamespace(
        index_share_for_mtp_iteration=True, indexer_rope_interleave=True, enable_attn_dp_weight_sharding=False
    )
    body.embed_tokens = _Embedding()
    body.eh_proj = _SelectTokenEmbedding()
    body.rot = nn.Identity()
    body.enorm = nn.Identity()
    body.hnorm = nn.Identity()
    body.norm = nn.Identity()
    layer = _DecoderLayer()
    body.layers = nn.ModuleList([layer])
    body.rotary = deepseek_v32.DeepseekYarnRotaryEmbedding(
        4, 16, 1.0, 10000.0, 32, 1, 1.0, 1.0, dtype=torch.float32, device=torch.device("cpu")
    )
    body.enable_rot = False
    body._reuse_topk_by_layer = (True,)
    return body, layer


def test_mtp_constructor_defers_shared_target_modules() -> None:
    draft = glm5_2_mtp.Glm52MtpForCausalLM(_config())

    assert draft.lm_head is None
    assert draft.model.embed_tokens is None


def test_mtp_builds_fallback_indexer_for_pattern_shared_layers() -> None:
    draft = glm5_2_mtp.Glm52MtpForCausalLM(_config(indexer_types=None, index_topk_pattern="S"))

    assert draft.cfg.indexer_types == ["shared"]
    assert draft.model.layers[0].self_attn.indexer is not None


def test_glm53_reads_rope_theta_from_rope_parameters() -> None:
    cfg = glm5_2_mtp.Glm52Config.from_dict(
        _config(
            rope_parameters={"rope_theta": 8_000_000, "rope_type": "default"},
        )
    )

    assert cfg.rope_theta == 8_000_000


def test_mtp_ignores_target_layer_metadata_with_mismatched_depth() -> None:
    cfg = glm5_2_mtp.Glm52Config.from_dict(
        _config(
            first_k_dense_replace=0,
            indexer_types=["full", "shared"],
            mlp_layer_types=["dense", "dense"],
            index_topk_pattern="",
        )
    )

    assert cfg.indexer_types == ["full"]
    assert cfg.mlp_layer_types == ["sparse"]


def test_mtp_reuses_external_topk_and_emits_fallback_topk() -> None:
    body, layer = _mtp_body()
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([1, 2])
    external_topk = torch.tensor([[[3, 4]], [[5, 6]]], dtype=torch.int32)

    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
        patch.object(glm5_2_mtp, "record_layer_event"),
    ):
        _, _, reused_topk = body(input_ids, positions, mtp_topk_indices=external_topk)
        _, _, fallback_topk = body(input_ids, positions)

    assert layer.reuse_flags == [True, False]
    assert reused_topk is external_topk
    assert fallback_topk is not None
    assert torch.equal(fallback_topk, torch.full((2, 1, 2), 7, dtype=torch.int32))


def test_mtp_masks_position_zero_token_embedding() -> None:
    body, _ = _mtp_body()
    input_ids = torch.tensor([5, 6])
    positions = torch.tensor([0, 1])

    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
        patch.object(glm5_2_mtp, "record_layer_event"),
    ):
        hidden, _, _ = body(input_ids, positions)

    assert torch.equal(hidden[0], torch.zeros(2))
    assert torch.equal(hidden[1], torch.full((2,), 6.0))


def test_mtp_body_normalizes_recurrent_hidden_once() -> None:
    body, _ = _mtp_body()
    body.norm = _ScaleNorm()

    class PreviousHidden(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden[:, 2:]

    body.eh_proj = PreviousHidden()
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
        patch.object(glm5_2_mtp, "record_layer_event"),
        patch.object(body.norm, "forward", wraps=body.norm.forward) as norm,
    ):
        first, _, _ = body(torch.tensor([1, 2]), torch.tensor([5, 6]), previous)
        second, _, _ = body(torch.tensor([3, 4]), torch.tensor([6, 7]), first)
        assert norm.call_count == 2
    assert torch.equal(first, previous * 2)
    assert torch.equal(second, previous * 4)
    assert torch.equal(previous, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("S", (True,)),
        ("F", (False,)),
        (["shared"], (True,)),
        (["full"], (False,)),
    ],
)
def test_mtp_topk_reuse_plan(pattern, expected: tuple[bool, ...]) -> None:
    cfg = SimpleNamespace(
        index_share_for_mtp_iteration=True,
        index_topk_pattern=pattern,
        index_topk_freq=4,
        index_skip_topk_offset=2,
        n_layers=1,
    )

    assert glm5_2_mtp._resolve_mtp_topk_reuse(cfg) == expected


def test_mtp_prepares_fresh_coefficients_for_full_and_reuse_steps() -> None:
    body, layer = _mtp_body()
    input_ids = torch.tensor([3, 4])
    topk = None
    positions_by_step = [torch.tensor([0, 1]), torch.tensor([7, 7]), torch.tensor([15, 2])]
    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
        patch.object(glm5_2_mtp, "record_layer_event"),
        patch.object(body.rotary, "forward", wraps=body.rotary.forward) as prepare,
    ):
        for positions in positions_by_step:
            _, _, topk = body(input_ids, positions, mtp_topk_indices=topk)
    assert prepare.call_count == 3
    assert layer.reuse_flags == [False, True, True]
    for positions, rope in zip(positions_by_step, layer.ropes):
        expected = body.rotary.cos_sin_cache[positions]
        torch.testing.assert_close(rope[0], expected[:, :2])
        torch.testing.assert_close(rope[1], expected[:, 2:])
        torch.testing.assert_close(rope[2].view(2, 4), torch.cat((expected[:, :2], expected[:, :2]), dim=-1))
    assert layer.ropes[0][2].data_ptr() != layer.ropes[1][2].data_ptr()
    assert not torch.equal(layer.ropes[0][2], layer.ropes[1][2])


def test_attn_dp_mtp_keeps_reused_topk_local_and_returns_owner_hidden() -> None:
    body, layer = _mtp_body()
    body.cfg.enable_attn_dp_weight_sharding = True
    body.cfg.dp_size = 2
    body.cfg.dp_rank = 1
    counts = (1, 2)
    external_topk = torch.ones(2, 1, 2, dtype=torch.int32)
    gathers = []

    def gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert (dim, world_size, group_name) == (0, 2, "dp")
        gathers.append(value)
        return torch.cat((value + 10, value), dim=0)

    with (
        forward_context(
            ForwardContext(
                attention_backend=None,
                device=torch.device("cpu"),
                metadata=SimpleNamespace(dp_execution_token_counts=counts),
                layer_caches=[],
            )
        ),
        patch.object(glm5_2_mtp, "record_layer_event"),
        patch("xllm.python.models.glm5_2.distributed.all_gather", side_effect=gather),
    ):
        hidden, _, topk = body(torch.tensor([1, 2]), torch.tensor([1, 2]), mtp_topk_indices=external_topk)
    assert len(gathers) == 2
    assert hidden.shape == (2, 2)
    assert topk is external_topk
    assert layer.reuse_flags == [True]
    torch.testing.assert_close(hidden, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_mtp_rope_is_shared_across_layers_and_refreshes_positions() -> None:
    body, first = _mtp_body()
    second = _DecoderLayer()
    body.layers.append(second)
    body._reuse_topk_by_layer = (False, False)
    with (
        patch.object(glm5_2_mtp, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_2_mtp, "record_layer_event"),
        patch.object(body.rotary, "forward", wraps=body.rotary.forward) as prepare,
    ):
        for positions in (torch.tensor([1, 2]), torch.tensor([7, 9])):
            body(torch.tensor([3, 4]), positions)
    assert prepare.call_count == 2
    for generation in range(2):
        for first_coefficient, second_coefficient in zip(first.ropes[generation], second.ropes[generation]):
            assert first_coefficient is second_coefficient
    assert not torch.equal(first.ropes[0][2], first.ropes[1][2])

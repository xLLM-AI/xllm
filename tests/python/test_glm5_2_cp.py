# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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

from xllm.python.models import glm5_2


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
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del cos_sin_cache
        self.positions = positions
        self.prev_topk = prev_topk
        self.output_topk = hidden[:, :1].clone()
        if residual is None:
            residual = hidden
        self._events.append(f"cache_write_{self._layer_id}")
        return hidden + self._layer_id + 1, residual, self.output_topk


class _Norm(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events
        self.input_rows: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._events.append("norm")
        self.input_rows = hidden
        return hidden + residual, residual


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)


def _make_model(events: list[str]) -> tuple[glm5_2.Glm52Model, list[_DecoderLayer]]:
    model = glm5_2.Glm52Model.__new__(glm5_2.Glm52Model)
    nn.Module.__init__(model)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.embed_tokens = _Embedding(events)
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm(events)
    model.rotary = _Rotary()
    model.aux_hidden_capture = glm5_2.AuxHiddenCapture(())
    return model, layers


def test_cp_model_loop_shards_local_rows_and_merges_after_norm() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = object()
    merged_output = torch.tensor([[23.0], [43.0], [63.0], [83.0]])

    def shard_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_rows")
        return hidden.index_select(0, torch.tensor([3, 0]))

    def shard_positions(positions: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        assert positions.dtype == torch.int64
        events.append("shard_positions")
        return positions.index_select(0, torch.tensor([3, 0]))

    def merge_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        torch.testing.assert_close(hidden, torch.tensor([[83.0], [23.0]]))
        events.append("merge_rows")
        return merged_output

    def record_event(layer_id: int) -> None:
        events.append(f"event_{layer_id}")

    with (
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(glm5_2, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_2, "cp_shard_positions", side_effect=shard_positions),
        patch.object(glm5_2, "cp_merge_rows", side_effect=merge_rows),
        patch.object(glm5_2, "record_layer_event", side_effect=record_event),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert events == [
        "embedding",
        "shard_rows",
        "shard_positions",
        "cache_write_0",
        "event_0",
        "cache_write_1",
        "event_1",
        "norm",
        "merge_rows",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([3, 0]))
    assert layers[1].prev_topk is layers[0].output_topk
    torch.testing.assert_close(layers[1].prev_topk, torch.tensor([[40.0], [10.0]]))
    torch.testing.assert_close(output, merged_output)


def test_cp_one_preserves_full_rows_without_shard_or_merge() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    shard_rows = MagicMock()
    shard_positions = MagicMock()
    merge_rows = MagicMock()

    def record_event(layer_id: int) -> None:
        events.append(f"event_{layer_id}")

    with (
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_2, "cp_shard_rows", shard_rows),
        patch.object(glm5_2, "cp_shard_positions", shard_positions),
        patch.object(glm5_2, "cp_merge_rows", merge_rows),
        patch.object(glm5_2, "record_layer_event", side_effect=record_event),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    shard_rows.assert_not_called()
    shard_positions.assert_not_called()
    merge_rows.assert_not_called()
    assert events == [
        "embedding",
        "cache_write_0",
        "event_0",
        "cache_write_1",
        "event_1",
        "norm",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(output, torch.tensor([[23.0], [43.0], [63.0], [83.0]]))


def test_cp_one_preserves_aux_hidden_capture() -> None:
    events: list[str] = []
    model, _ = _make_model(events)
    model.aux_hidden_capture = glm5_2.AuxHiddenCapture((0, 1))

    with (
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_2, "record_layer_event"),
    ):
        output, aux_hidden = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    torch.testing.assert_close(output, torch.tensor([[23.0], [43.0], [63.0], [83.0]]))
    torch.testing.assert_close(
        aux_hidden,
        torch.tensor(
            [
                [21.0, 23.0],
                [41.0, 43.0],
                [61.0, 63.0],
                [81.0, 83.0],
            ]
        ),
    )


def test_cp_ep_moe_materializes_global_rows_before_expert_reduction() -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 4
    cp_context = object()
    local_hidden = torch.tensor([[30.0], [10.0]])
    global_hidden = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    global_output = global_hidden + 100.0

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch.object(glm5_2, "cp_gather_kv", return_value=global_hidden) as gather,
        patch.object(glm5_2.DeepseekV3MoE, "forward", return_value=global_output) as ep_forward,
        patch.object(
            glm5_2,
            "cp_shard_rows",
            return_value=torch.tensor([[130.0], [110.0]]),
        ) as shard,
    ):
        output = moe(local_hidden)

    gather.assert_called_once_with(local_hidden, cp_context)
    ep_forward.assert_called_once_with(global_hidden)
    shard.assert_called_once_with(global_output, cp_context)
    torch.testing.assert_close(output, torch.tensor([[130.0], [110.0]]))


def test_glm_ep1_moe_reduces_only_on_ordinary_tp_group() -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 1
    moe.moe_tp_size = 4
    moe.cfg = SimpleNamespace(tp_size=2)
    routed = torch.tensor([[1.0], [2.0]])
    shared = torch.tensor([[10.0], [20.0]])

    with patch.object(glm5_2.distributed, "all_reduce_", create=True) as reduce:
        output = moe._combine_expert_outputs(routed, shared)

    reduce.assert_called_once_with(output, "tp")
    torch.testing.assert_close(output, routed + shared)


def test_glm_ep_moe_preserves_parent_combine_behavior() -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 2
    routed = torch.tensor([[1.0]])
    shared = torch.tensor([[2.0]])
    expected = torch.tensor([[3.0]])

    with patch.object(
        glm5_2.DeepseekV3MoE,
        "_combine_expert_outputs",
        return_value=expected,
    ) as parent_combine:
        output = moe._combine_expert_outputs(routed, shared)

    parent_combine.assert_called_once_with(routed, shared)
    assert output is expected


def test_glm_dense_mlp_fused_path_reduces_in_fp32() -> None:
    mlp = glm5_2.Glm52MLP.__new__(glm5_2.Glm52MLP)
    nn.Module.__init__(mlp)
    mlp.tp = 2
    mlp.skip_tp_reduce = False
    mlp.gate_up_proj = SimpleNamespace(
        weight_scale=torch.ones(1),
        forward_accumulated=MagicMock(return_value=torch.ones(2, 4)),
    )
    reduced_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    mlp.down_proj = SimpleNamespace(
        forward_quantized=MagicMock(return_value=reduced_input),
    )

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.float32
        output.add_(1.0)

    with (
        patch.object(
            glm5_2.kernels,
            "dynamic_quant",
            return_value=(torch.ones(2, 2, dtype=torch.int8), torch.ones(2)),
            create=True,
        ),
        patch.object(
            glm5_2.kernels,
            "dequant_swiglu_quant",
            return_value=(torch.ones(2, 2, dtype=torch.int8), torch.ones(2)),
            create=True,
        ),
        patch.object(
            glm5_2.distributed,
            "tp_all_reduce",
            side_effect=reduce_output,
            create=True,
        ) as reduce,
    ):
        output = mlp.forward_dequant_swiglu_quant(torch.ones(2, 2))

    reduce.assert_called_once()
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reduced_input + 1)


def test_glm_dense_mlp_unfused_path_reduces_in_fp32() -> None:
    mlp = glm5_2.Glm52MLP.__new__(glm5_2.Glm52MLP)
    nn.Module.__init__(mlp)
    mlp.tp = 2
    mlp.skip_tp_reduce = False
    mlp.swiglu_limit = 0.0
    reduced_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    mlp.down_proj = MagicMock(return_value=reduced_input)
    activation = torch.ones(2, 2, dtype=torch.bfloat16)

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.float32
        output.add_(1.0)

    with (
        patch.object(glm5_2, "_swiglu_with_clamp", return_value=activation) as swiglu,
        patch.object(
            glm5_2.distributed,
            "tp_all_reduce",
            side_effect=reduce_output,
            create=True,
        ) as reduce,
    ):
        output = mlp._forward_gate_up(torch.ones(2, 4), None)

    swiglu.assert_called_once()
    mlp.down_proj.assert_called_once_with(activation)
    reduce.assert_called_once()
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reduced_input + 1)


def test_glm_attention_reduces_o_projection_in_fp32_for_tensor_parallel() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = False
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.indexer = None
    attention.num_heads_local = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 2
    attention.layer_id = 0
    attention.cfg = SimpleNamespace(
        tp_size=2,
        layerwise_split_size=1,
        layerwise_split_rank=0,
    )
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)

    previous_topk = torch.tensor([[0], [1]])
    projected = torch.tensor(
        [[[11.0, 13.0]], [[17.0, 19.0]]],
        dtype=torch.bfloat16,
    )
    reduction_delta = torch.tensor(
        [[0.125, -0.25], [0.5, -0.75]],
        dtype=torch.float32,
    )
    backend = MagicMock()
    backend.execute_mla.return_value = torch.tensor([[[5.0]], [[7.0]]])

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.float32
        output.add_(reduction_delta)

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(
                attention_backend=backend,
                cp_context=None,
                metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            ),
        ),
        patch.object(
            glm5_2,
            "_gather_interleave_cos_sin",
            return_value=(torch.empty(0), torch.empty(0)),
        ),
        patch.object(glm5_2, "_interleave_rope_with", side_effect=lambda value, *_args: value),
        patch.object(
            glm5_2.kernels,
            "batch_matmul_transpose",
            return_value=projected,
            create=True,
        ) as project,
        patch.object(
            glm5_2.distributed,
            "all_reduce_",
            side_effect=reduce_output,
            create=True,
        ) as reduce,
    ):
        output, topk = attention(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([0, 1]),
            torch.empty(0),
            previous_topk,
        )

    project.assert_called_once()
    reduce.assert_called_once()
    assert reduce.call_args.args[0].dtype == torch.float32
    assert output.dtype == projected.dtype
    torch.testing.assert_close(
        output,
        (projected.reshape(2, 2).float() + reduction_delta).to(projected.dtype),
    )
    assert topk is previous_topk


def test_glm_attention_reuse_updates_index_cache() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = False
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.num_heads_local = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 2
    attention.layer_id = 0
    attention.cfg = SimpleNamespace(
        tp_size=1,
        layerwise_split_size=1,
        layerwise_split_rank=0,
    )
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)
    attention.indexer = MagicMock()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([0, 1])
    cos_sin_cache = torch.empty(0)
    previous_topk = torch.tensor([[0], [1]])
    projected = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    backend = MagicMock()
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = projected

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(
                attention_backend=backend,
                cp_context=None,
                metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            ),
        ),
        patch.object(
            glm5_2,
            "_gather_interleave_cos_sin",
            return_value=(torch.empty(0), torch.empty(0)),
        ),
        patch.object(glm5_2, "_interleave_rope_with", side_effect=lambda value, *_args: value),
        patch.object(glm5_2.kernels, "batch_matmul_transpose", return_value=projected, create=True),
    ):
        _, topk = attention(
            hidden,
            positions,
            cos_sin_cache,
            previous_topk,
            reuse_topk_indices=True,
        )

    attention.indexer._update_index_cache.assert_called_once()
    cache_args = attention.indexer._update_index_cache.call_args.args
    assert cache_args[0] is hidden
    assert cache_args[1] is positions
    assert cache_args[2] is backend.mla_index_context.return_value
    assert cache_args[3] is cos_sin_cache
    assert topk is previous_topk


@pytest.mark.parametrize(
    ("use_mlapo_v2", "num_tokens", "reuse_topk_indices", "expect_mlapo_v2"),
    [
        (False, 2, False, False),
        (True, 2, False, True),
        (True, 2, True, True),
        (
            True,
            glm5_2.kernels.MLA_PREPROCESS_V2_MAX_TOKENS + 1,
            False,
            False,
        ),
    ],
)
def test_glm_attention_fused_decode_preprocesses_and_writes_cache_once(
    use_mlapo_v2: bool,
    num_tokens: int,
    reuse_topk_indices: bool,
    expect_mlapo_v2: bool,
) -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = True
    attention._use_mlapo_v2 = use_mlapo_v2
    attention._dynamic_mla_ready = False
    attention.indexer = MagicMock() if reuse_topk_indices else None
    attention.num_heads_local = 1
    attention.q_lora_rank = 2
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 2
    attention.layer_id = 0
    attention.cfg = SimpleNamespace(
        tp_size=1,
        layerwise_split_size=1,
        layerwise_split_rank=0,
    )
    attention.qkv_a_proj = SimpleNamespace(
        input_scale=torch.ones(1),
        input_offset=torch.zeros(1),
        weight=torch.empty(0),
        deq_scale=torch.empty(0),
        quant_bias=torch.empty(0),
        forward_quantized=MagicMock(),
    )
    attention.q_b_proj = SimpleNamespace(
        input_scale=torch.ones(1),
        input_offset=torch.zeros(1),
        weight=torch.empty(0),
        deq_scale=torch.empty(0),
        quant_bias=torch.empty(0),
    )
    attention.q_a_layernorm = SimpleNamespace(weight=torch.ones(2), eps=1e-5)
    attention.kv_a_layernorm = SimpleNamespace(weight=torch.ones(1), eps=1e-5)
    attention._mlapo_input_norm_weight = torch.ones(2)
    attention._mlapo_input_norm_bias = torch.zeros(2)
    attention._mlapo_q_norm_bias = torch.zeros(2)
    attention._mlapo_qkv_input_offset = torch.zeros(1, dtype=torch.int8)
    attention._mlapo_qkv_weight = torch.empty(0)
    attention._mlapo_qkv_deq_scale = torch.empty(0)
    attention._mlapo_qkv_quant_bias = torch.empty(0)
    attention._mlapo_q_b_input_offset = torch.zeros(1, dtype=torch.int8)
    attention._mlapo_q_b_weight = torch.empty(0)
    attention._mlapo_q_b_deq_scale = torch.empty(0)
    attention._mlapo_q_b_quant_bias = torch.empty(0)
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)
    attention.o_proj = nn.Identity()

    hidden = torch.ones(num_tokens, 2)
    positions = torch.arange(num_tokens)
    cos_sin_cache = torch.empty(0)
    previous_topk = torch.zeros(num_tokens, 1, dtype=torch.int64)
    q_c = torch.ones(num_tokens, 2)
    q_latent = torch.ones(num_tokens, 1, 1)
    q_pe = torch.ones(num_tokens, 1, 1)
    attn_out = torch.ones(num_tokens, 1, 1)
    projected = torch.ones(num_tokens, 1, 2)
    preprocess_context = glm5_2.MlaPreprocessContext(
        kv_cache=torch.empty(2, 1, 1),
        rope_cache=torch.empty(2, 1, 1),
        slot_mapping=torch.arange(num_tokens + 1),
    )
    backend = MagicMock()
    backend.mla_preprocess_context.return_value = preprocess_context
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = attn_out

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(
                attention_backend=backend,
                cp_context=None,
                metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            ),
        ),
        patch.object(
            glm5_2,
            "_gather_interleave_cos_sin",
            return_value=(torch.empty(0), torch.empty(0)),
        ),
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode",
            return_value=(q_c, q_latent, q_pe),
            create=True,
        ) as capturable_preprocess,
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode_v2",
            return_value=(q_c, q_latent, q_pe),
            create=True,
        ) as mlapo_v2,
        patch.object(
            glm5_2.kernels,
            "batch_matmul_transpose",
            return_value=projected,
            create=True,
        ),
    ):
        output, topk = attention(
            hidden,
            positions,
            cos_sin_cache,
            previous_topk,
            reuse_topk_indices=reuse_topk_indices,
        )

    attention.qkv_a_proj.forward_quantized.assert_not_called()
    selected_preprocess = mlapo_v2 if expect_mlapo_v2 else capturable_preprocess
    unselected_preprocess = capturable_preprocess if expect_mlapo_v2 else mlapo_v2
    selected_preprocess.assert_called_once()
    unselected_preprocess.assert_not_called()
    slot_mapping_arg = 21 if expect_mlapo_v2 else 16
    torch.testing.assert_close(
        selected_preprocess.call_args.args[slot_mapping_arg],
        torch.arange(num_tokens),
    )
    backend.execute_mla.assert_called_once_with(
        q_latent,
        q_pe,
        None,
        None,
        attention,
        topk=previous_topk,
        cache_is_preprocessed=True,
    )
    if reuse_topk_indices:
        attention.indexer._update_index_cache.assert_called_once_with(
            hidden,
            positions,
            backend.mla_index_context.return_value,
            cos_sin_cache,
        )
        attention.indexer.select_qli.assert_not_called()
    torch.testing.assert_close(output, projected.reshape(num_tokens, 2))
    assert topk is previous_topk


def test_glm_attention_dynamic_fused_decode_reuses_topk_after_cache_write() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = True
    attention._use_mlapo_v2 = False
    attention._fused_mla_ready = True
    attention._dynamic_mla_ready = True
    attention.indexer = MagicMock()
    attention.num_heads_local = 1
    attention.q_lora_rank = 2
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 2
    attention.layer_id = 0
    attention.cfg = SimpleNamespace(
        tp_size=1,
        layerwise_split_size=1,
        layerwise_split_rank=0,
    )
    attention._dynamic_qkv_weight = torch.empty(0)
    attention._dynamic_qkv_weight_scale = torch.empty(0)
    attention.q_a_layernorm = SimpleNamespace(weight=torch.ones(2), eps=1e-5)
    attention.q_b_proj = SimpleNamespace(weight=torch.empty(0), weight_scale=torch.empty(0))
    attention.kv_a_layernorm = SimpleNamespace(weight=torch.ones(1), eps=1e-5)
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)
    attention.o_proj = nn.Identity()

    hidden = torch.ones(2, 2)
    positions = torch.arange(2)
    cos_sin_cache = torch.empty(0)
    previous_topk = torch.zeros(2, 1, dtype=torch.int64)
    q_c = torch.ones(2, 2)
    q_latent = torch.ones(2, 1, 1)
    q_pe = torch.ones(2, 1, 1)
    projected = torch.ones(2, 1, 2)
    preprocess_context = glm5_2.MlaPreprocessContext(
        kv_cache=torch.empty(2, 1, 1),
        rope_cache=torch.empty(2, 1, 1),
        slot_mapping=torch.arange(3),
    )
    backend = MagicMock()
    backend.mla_preprocess_context.return_value = preprocess_context
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = torch.ones(2, 1, 1)

    with (
        patch.object(
            glm5_2,
            "get_forward_context",
            return_value=SimpleNamespace(
                attention_backend=backend,
                cp_context=None,
                metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
            ),
        ),
        patch.object(
            glm5_2,
            "_gather_interleave_cos_sin",
            return_value=(torch.empty(0), torch.empty(0)),
        ),
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode_dynamic",
            return_value=(q_c, q_latent, q_pe),
            create=True,
        ) as dynamic_preprocess,
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode",
            create=True,
        ) as static_preprocess,
        patch.object(
            glm5_2.kernels,
            "batch_matmul_transpose",
            return_value=projected,
            create=True,
        ),
    ):
        output, topk = attention(
            hidden,
            positions,
            cos_sin_cache,
            previous_topk,
            reuse_topk_indices=True,
        )

    dynamic_preprocess.assert_called_once()
    static_preprocess.assert_not_called()
    torch.testing.assert_close(dynamic_preprocess.call_args.args[10], torch.arange(2))
    attention.indexer._update_index_cache.assert_called_once_with(
        hidden,
        positions,
        backend.mla_index_context.return_value,
        cos_sin_cache,
    )
    attention.indexer.select_qli.assert_not_called()
    backend.execute_mla.assert_called_once_with(
        q_latent,
        q_pe,
        None,
        None,
        attention,
        topk=previous_topk,
        cache_is_preprocessed=True,
    )
    torch.testing.assert_close(output, projected.reshape(2, 2))
    assert topk is previous_topk

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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import MlaPreprocessContext
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import deepseek_v32, glm5_2


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
        self.rope: tuple[torch.Tensor, ...] | None = None
        self.query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None
        self.self_attn = SimpleNamespace(_dynamic_mla_ready=False)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        prev_topk: torch.Tensor | None,
        *,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert slot_mapping_int64 is None
        self.rope = (half_rope_cos, half_rope_sin, rope_cos, rope_sin)
        self.query_cos_sin = query_cos_sin
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


class _Rotary(glm5_2.Glm52YarnRotaryEmbedding):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.register_buffer("cos_sin_cache", torch.arange(64, dtype=torch.float32).view(16, 4), persistent=False)
        self.positions: list[torch.Tensor] = []

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        assert positions.dtype == torch.int64 and positions.is_contiguous()
        self.positions.append(positions.clone())
        return super().forward(positions)


def _attention_rope(rows: int) -> tuple[object, ...]:
    half_cos, half_sin = torch.empty(rows, 0), torch.empty(rows, 0)
    cos, sin = torch.ones(rows, 1, 1, 1), torch.zeros(rows, 1, 1, 1)
    return half_cos, half_sin, cos, sin, (cos, sin)


def _make_model(events: list[str]) -> tuple[glm5_2.Glm52Model, list[_DecoderLayer]]:
    model = glm5_2.Glm52Model.__new__(glm5_2.Glm52Model)
    nn.Module.__init__(model)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.embed_tokens = _Embedding(events)
    model.layers = nn.ModuleList(layers)
    model.cfg = glm5_2.Glm52Config(n_layers=2, indexer_rope_interleave=True)
    model.layer_begin = 0
    model.layer_end = 2
    model.norm = _Norm(events)
    model.rotary = _Rotary()
    model.aux_hidden_capture = glm5_2.AuxHiddenCapture(())
    return model, layers


def test_cp_model_loop_shards_local_rows_and_merges_after_norm() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = SimpleNamespace(query_index=torch.tensor([0]))
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
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=cp_context)),
        patch.object(deepseek_v32, "cp_shard_rows", side_effect=shard_rows),
        patch.object(deepseek_v32, "cp_shard_positions", side_effect=shard_positions),
        patch.object(deepseek_v32, "cp_merge_rows", side_effect=merge_rows),
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
    torch.testing.assert_close(model.rotary.positions[0], torch.tensor([3, 0]))
    assert len(model.rotary.positions) == 1
    assert all(first is second for first, second in zip(layers[0].rope, layers[1].rope))
    assert layers[0].query_cos_sin is layers[1].query_cos_sin
    torch.testing.assert_close(layers[0].rope[0], model.rotary.cos_sin_cache[[3, 0], :2])
    torch.testing.assert_close(layers[0].query_cos_sin[0], layers[0].rope[2][[0]])
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
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
        patch.object(deepseek_v32, "cp_shard_rows", shard_rows),
        patch.object(deepseek_v32, "cp_shard_positions", shard_positions),
        patch.object(deepseek_v32, "cp_merge_rows", merge_rows),
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
    torch.testing.assert_close(model.rotary.positions[0], torch.tensor([0, 1, 2, 3]))
    assert len(model.rotary.positions) == 1
    assert layers[0].query_cos_sin[0] is layers[0].rope[2]
    assert layers[0].query_cos_sin[1] is layers[0].rope[3]
    torch.testing.assert_close(output, torch.tensor([[23.0], [43.0], [63.0], [83.0]]))


def test_cp_one_preserves_aux_hidden_capture() -> None:
    events: list[str] = []
    model, _ = _make_model(events)
    model.aux_hidden_capture = glm5_2.AuxHiddenCapture((0, 1))

    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=None)),
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


@pytest.mark.parametrize("ep_size", [1, 4])
def test_cp_moe_materializes_global_rows_before_expert_reduction(ep_size: int) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = ep_size
    moe.cfg = SimpleNamespace(enable_attn_dp_weight_sharding=False)
    cp_context = object()
    local_hidden = torch.tensor([[30.0], [10.0]])
    global_hidden = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    global_output = global_hidden + 100.0

    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=cp_context)),
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


@pytest.mark.parametrize(("cp_size", "group"), [(1, "tp"), (2, "moe_tp")])
def test_glm_ep1_moe_reduction_matches_weight_sharding(cp_size: int, group: str) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 1
    moe.moe_tp_size = 4
    moe.cfg = SimpleNamespace(
        ep_size=1,
        tp_size=2,
        tp_rank=0,
        moe_tp_size=4,
        moe_tp_rank=0,
        dp_size=1,
        cp_size=cp_size,
        enable_attn_dp_weight_sharding=False,
    )
    routed = torch.tensor([[1.0], [2.0]])
    shared = torch.tensor([[10.0], [20.0]])

    with patch.object(glm5_2.distributed, "all_reduce_", create=True) as reduce:
        output = moe._combine_expert_outputs(routed, shared, False)

    reduce.assert_called_once_with(output, group)
    torch.testing.assert_close(output, routed + shared)


@pytest.mark.parametrize(
    ("enabled", "ep_size", "gate_overlap", "fine_overlap", "expected"),
    [
        (True, 1, False, False, True),
        (False, 1, False, False, False),
        (True, 2, False, False, False),
        (True, 1, True, False, True),
        (True, 1, True, True, True),
        (False, 1, True, True, False),
        (True, 2, True, True, False),
    ],
)
def test_glm_moe_finalize_follows_unified_switch_contract(
    enabled: bool,
    ep_size: int,
    gate_overlap: bool,
    fine_overlap: bool,
    expected: bool,
) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = ep_size
    moe._expert_parallel_enabled = True
    moe._gate_overlap_enabled = gate_overlap
    moe._fine_overlap_enabled = fine_overlap
    moe.set_unified_mtp_graph_enabled(enabled)
    assert moe._enable_moe_finalize_routing is expected
    if ep_size == 1:
        assert moe._gate_overlap_enabled is enabled
        assert moe._fine_overlap_enabled is False
    else:
        assert moe._gate_overlap_enabled is gate_overlap
        assert moe._fine_overlap_enabled is fine_overlap


def test_glm_unified_switch_does_not_enable_overlap_on_other_devices() -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 1
    moe._expert_parallel_enabled = False
    moe._gate_overlap_enabled = False
    moe._fine_overlap_enabled = False
    moe.set_unified_mtp_graph_enabled(True)
    assert not moe._enable_moe_finalize_routing
    assert not moe._gate_overlap_enabled
    assert not moe._fine_overlap_enabled


@pytest.mark.parametrize(("enabled", "tokens"), [(False, 2), (True, 2), (True, 0)])
def test_glm_unified_switch_selects_grouped_moe_output(enabled: bool, tokens: int) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe._enable_moe_finalize_routing = enabled
    moe.cfg = SimpleNamespace(norm_topk_prob=True)
    moe.gate = MagicMock(return_value=torch.zeros(tokens, 4))
    for name in (
        "experts_w13",
        "experts_w2",
        "experts_w13_scale",
        "experts_w2_scale_compute",
        "e_score_correction_bias",
    ):
        setattr(moe, name, torch.empty(1))
    moe.topk = 2
    moe.topk_group = 1
    moe.n_group = 1
    moe.routed_scaling = 1.0
    moe.local_expert_start = 0
    moe.local_expert_end = 4
    hidden = SimpleNamespace(shape=(tokens, 4), device=SimpleNamespace(type="npu"))
    routed = torch.ones(tokens, 4)
    metadata = (routed, torch.ones(tokens, 2), torch.zeros(tokens * 2, dtype=torch.int32))
    with (
        patch.object(glm5_2.kernels, "supports_fused_moe_gmm1", return_value=True),
        patch.object(glm5_2.kernels, "grouped_moe", return_value=routed) as legacy,
        patch.object(glm5_2.kernels, "grouped_moe_with_routing", return_value=metadata) as retained,
    ):
        result = moe._run_routed_experts(hidden)
    use_finalize = enabled and tokens > 0
    assert result is (metadata if use_finalize else routed)
    assert retained.call_count == int(use_finalize)
    assert legacy.call_count == int(not use_finalize)


@pytest.mark.parametrize(
    "device_type",
    ["cpu", "cuda", "npu", "privateuseone"],
    ids=["host", "other-device", "ascend", "ascend-alias"],
)
@pytest.mark.parametrize("supports_fused", [False, True])
def test_glm_moe_finalize_device_and_capability_guard(device_type: str, supports_fused: bool) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe._enable_moe_finalize_routing = True
    hidden = SimpleNamespace(shape=(1, 4), device=SimpleNamespace(type=device_type))
    with patch.object(glm5_2.kernels, "supports_fused_moe_gmm1", return_value=supports_fused) as capability:
        assert moe._use_moe_finalize_routing(hidden) is (device_type in ("npu", "privateuseone") and supports_fused)
    if device_type in ("cpu", "cuda"):
        capability.assert_not_called()


@pytest.mark.parametrize("gate_overlap", [False, True])
def test_glm_moe_finalize_orders_stream_dependencies(gate_overlap: bool) -> None:
    class _Event:
        def __init__(self, name: str, events: list[tuple[str, object]]) -> None:
            self.name = name
            self.events = events

        def record(self, stream: object) -> None:
            self.events.append((f"record:{self.name}", stream))

    class _Stream:
        def __init__(self, name: str, events: list[tuple[str, object]]) -> None:
            self.name = name
            self.events = events

        def wait_event(self, event: object) -> None:
            self.events.append((f"wait:{self.name}", event))

    events: list[tuple[str, object]] = []
    event_ids = iter(("start", "shared_done", "gate_done"))
    current_stream = _Stream("current", events)
    gate_stream = _Stream("gate", events)
    shared_stream = _Stream("shared", events)
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 1
    moe.moe_tp_size = 1
    moe.cfg = SimpleNamespace(
        ep_size=1,
        tp_size=1,
        tp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        dp_size=1,
        cp_size=1,
        enable_attn_dp_weight_sharding=False,
        norm_topk_prob=True,
    )
    moe._gate_overlap_enabled = gate_overlap
    moe._fine_overlap_enabled = False
    moe._enable_moe_finalize_routing = True
    moe.gate = MagicMock(return_value=torch.zeros(2, 4))
    moe.e_score_correction_bias = torch.zeros(4)
    moe.topk = 2
    moe.topk_group = 1
    moe.n_group = 1
    moe.routed_scaling = 1.0
    moe.experts_w13 = torch.empty(1)
    moe.experts_w2 = torch.empty(1)
    moe.experts_w13_scale = torch.empty(1)
    moe.experts_w2_scale_compute = torch.empty(1)
    moe._shared_expert_start_event = None
    moe._gate_done_event = None
    moe._shared_expert_done_event = None
    routed = (
        torch.ones(4, 4),
        torch.tensor([[0.2, 0.8], [0.7, 0.3]]),
        torch.tensor([1, 0, 1, 0], dtype=torch.int32),
    )
    shared = torch.full((2, 4), 2.0)

    def make_event() -> _Event:
        return _Event(next(event_ids), events)

    def fake_finalize(
        routed_value: torch.Tensor,
        shared_value: torch.Tensor,
        probs_value: torch.Tensor,
        indices_value: torch.Tensor,
    ) -> torch.Tensor:
        assert routed_value is routed[0]
        assert shared_value is shared
        assert probs_value is routed[1]
        assert indices_value is routed[2]
        events.append(("finalize", None))
        return torch.full((2, 4), 3.0)

    with (
        patch.object(deepseek_v32, "_gate_stream", return_value=gate_stream),
        patch.object(glm5_2.kernels, "supports_fused_moe_gmm1", return_value=True),
        patch.object(deepseek_v32, "_shared_expert_stream", return_value=shared_stream),
        patch.object(
            torch,
            "npu",
            SimpleNamespace(
                Event=make_event,
                current_stream=lambda: current_stream,
                stream=lambda _: nullcontext(),
            ),
            create=True,
        ),
        patch.object(
            glm5_2.kernels,
            "moe_gate_routing",
            return_value=(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32)),
        ),
        patch.object(
            glm5_2.kernels,
            "moe_expert_compute_with_routing",
            side_effect=lambda *args: events.append(("routed", None)) or routed,
        ),
        patch.object(
            moe,
            "_run_shared_experts",
            side_effect=lambda hidden: events.append(("shared", None)) or shared,
        ),
        patch.object(
            moe,
            "_run_routed_experts",
            side_effect=lambda *args: events.append(("routed", None)) or routed,
        ),
        patch.object(glm5_2.kernels, "moe_finalize_routing", side_effect=fake_finalize),
    ):
        hidden = SimpleNamespace(shape=(2, 4), device=SimpleNamespace(type="npu"))
        result = moe._forward_parallel(hidden, use_mega_moe=False)

    assert torch.equal(result, torch.full((2, 4), 3.0))
    expected = ["record:start"]
    if gate_overlap:
        expected.extend(("wait:gate", "record:gate_done"))
    expected.extend(("wait:shared", "shared", "record:shared_done"))
    if gate_overlap:
        expected.append("wait:current")
    expected.extend(("routed", "wait:current", "finalize"))
    assert [event[0] for event in events] == expected


@pytest.mark.parametrize(
    ("mode", "size", "expected_group"),
    [("cp", 1, "tp"), ("cp", 2, "moe_tp"), ("dp", 1, "tp"), ("dp", 2, "moe_tp")],
)
def test_glm_ep1_moe_finalize_combines_permuted_routing_before_tp_reduce(
    mode: str, size: int, expected_group: str
) -> None:
    moe = glm5_2.Glm52MoE.__new__(glm5_2.Glm52MoE)
    nn.Module.__init__(moe)
    moe.ep_size = 1
    cp_size = size if mode == "cp" else 1
    dp_size = size if mode == "dp" else 1
    moe.moe_tp_size = size if mode == "cp" else 2 * size
    moe.cfg = SimpleNamespace(
        ep_size=1,
        tp_size=2,
        tp_rank=0,
        moe_tp_size=moe.moe_tp_size,
        moe_tp_rank=0,
        dp_size=dp_size,
        cp_size=cp_size,
        enable_attn_dp_weight_sharding=False,
    )
    moe._enable_moe_finalize_routing = True
    permuted = torch.tensor([[1.0], [2.0], [3.0]])
    probs = torch.tensor([[0.25], [0.5], [0.75]])
    row_idx = torch.tensor([1, 0, 1], dtype=torch.int32)
    shared = torch.tensor([[10.0], [20.0]])
    finalized = torch.tensor([[11.0], [22.5]])

    def fake_finalize(
        routed_value: torch.Tensor,
        shared_value: torch.Tensor,
        probs_value: torch.Tensor,
        indices_value: torch.Tensor,
    ) -> torch.Tensor:
        assert routed_value is permuted
        assert shared_value is shared
        assert probs_value is probs
        assert indices_value is row_idx
        output = shared_value.clone()
        output.index_add_(0, indices_value.to(torch.long), routed_value * probs_value)
        return output

    with (
        patch.object(glm5_2.kernels, "moe_finalize_routing", side_effect=fake_finalize) as finalize,
        patch.object(glm5_2.distributed, "all_reduce_", create=True) as reduce,
    ):
        output = moe._combine_expert_outputs((permuted, probs, row_idx), shared)

    assert finalize.call_count == 1
    finalize_args = finalize.call_args.args
    assert finalize_args[0] is permuted
    assert finalize_args[1] is shared
    assert finalize_args[2] is probs
    assert finalize_args[3] is row_idx
    assert reduce.call_count == 1
    reduce_args = reduce.call_args.args
    torch.testing.assert_close(reduce_args[0], finalized)
    assert reduce_args[1] == expected_group
    torch.testing.assert_close(output, finalized)


def test_shared_dense_mlp_fused_path_preserves_reduction_dtype() -> None:
    mlp = deepseek_v32.DeepseekV3MLP.__new__(deepseek_v32.DeepseekV3MLP)
    nn.Module.__init__(mlp)
    mlp.tp = 2
    mlp.skip_tp_reduce = False
    mlp.gate_up_proj = SimpleNamespace(
        weight_scale=torch.ones(1),
        forward_accumulated=MagicMock(return_value=torch.ones(2, 4)),
    )
    reduced_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    expected = reduced_input + 1
    mlp.down_proj = SimpleNamespace(
        forward_quantized=MagicMock(return_value=reduced_input),
    )

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.bfloat16
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
    torch.testing.assert_close(output, expected)


def test_shared_dense_mlp_unfused_path_preserves_reduction_dtype() -> None:
    mlp = deepseek_v32.DeepseekV3MLP.__new__(deepseek_v32.DeepseekV3MLP)
    nn.Module.__init__(mlp)
    mlp.tp = 2
    mlp.skip_tp_reduce = False
    mlp.swiglu_limit = 0.0
    reduced_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    expected = reduced_input + 1
    mlp.down_proj = MagicMock(return_value=reduced_input)
    activation = torch.ones(2, 2, dtype=torch.bfloat16)

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.bfloat16
        output.add_(1.0)

    with (
        patch.object(deepseek_v32, "_swiglu_with_clamp", return_value=activation) as swiglu,
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
    torch.testing.assert_close(output, expected)


def test_glm_attention_preserves_o_projection_dtype_for_tensor_parallel() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = False
    attention._combined_qkv = None
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.indexer = None
    attention._use_fused_mla_decode = False
    attention.num_heads_local = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 1
    attention.kv_lora_rank = 1
    attention.v_head_dim = 2
    attention.layer_id = 0
    attention._use_fused_mla_decode = False
    attention.cfg = SimpleNamespace(
        tp_size=2,
        layerwise_split_size=1,
        layerwise_split_rank=0,
        indexer_rope_interleave=True,
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
    expected = (projected.reshape(2, 2).float() + reduction_delta).to(projected.dtype)
    backend = MagicMock()
    backend.execute_mla.return_value = torch.tensor([[[5.0]], [[7.0]]])

    def reduce_output(output: torch.Tensor) -> None:
        assert output.dtype == torch.bfloat16
        output.add_(reduction_delta)

    with (
        forward_context(
            ForwardContext(
                backend,
                torch.device("cpu"),
                SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                [],
            )
        ),
        patch.object(deepseek_v32, "_interleave_rope_with", side_effect=lambda value, *_args: value),
        patch.object(
            glm5_2.kernels,
            "atb_matmul_ein_sum",
            side_effect=[torch.tensor([[[1.0]], [[3.0]]]), projected],
            create=True,
        ),
        patch.object(
            glm5_2.distributed,
            "all_reduce_",
            side_effect=reduce_output,
            create=True,
        ) as reduce,
    ):
        output, topk = attention(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            *_attention_rope(2),
            previous_topk,
        )

    reduce.assert_called_once()
    assert reduce.call_args.args[0].dtype == torch.bfloat16
    assert output.dtype == projected.dtype
    torch.testing.assert_close(output, expected)
    assert topk is previous_topk


def test_glm_attention_reuse_updates_index_cache() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = False
    attention._combined_qkv = None
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
        indexer_rope_interleave=True,
    )
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)
    attention.indexer = MagicMock()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rope = _attention_rope(2)
    previous_topk = torch.tensor([[0], [1]])
    projected = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    backend = MagicMock()
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = projected

    with (
        forward_context(
            ForwardContext(
                backend,
                torch.device("cpu"),
                SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                [],
            )
        ),
        patch.object(deepseek_v32, "_interleave_rope_with", side_effect=lambda value, *_args: value),
        patch.object(
            glm5_2.kernels,
            "atb_matmul_ein_sum",
            side_effect=[hidden[:, :1].unsqueeze(1), projected],
            create=True,
        ),
    ):
        _, topk = attention(
            hidden,
            *rope,
            previous_topk,
            reuse_topk_indices=True,
        )

    attention.indexer._update_index_cache.assert_called_once()
    cache_args = attention.indexer._update_index_cache.call_args.args
    assert cache_args[0] is hidden
    assert cache_args[1] is backend.mla_index_context.return_value
    assert cache_args[2][0] is rope[2]
    assert cache_args[2][1] is rope[3]
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
    attention._fused_mla_ready = True
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
        indexer_rope_interleave=True,
    )
    attention._combined_qkv = SimpleNamespace(
        _dynamic_activation=False,
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
    rope = _attention_rope(num_tokens)
    previous_topk = torch.zeros(num_tokens, 1, dtype=torch.int64)
    q_c = torch.ones(num_tokens, 2)
    q_latent = torch.ones(num_tokens, 1, 1)
    q_pe = torch.ones(num_tokens, 1, 1)
    attn_out = torch.ones(num_tokens, 1, 1)
    projected = torch.ones(num_tokens, 1, 2)
    preprocess_context = MlaPreprocessContext(
        kv_cache=torch.empty(2, 1, 1),
        rope_cache=torch.empty(2, 1, 1),
        slot_mapping=torch.arange(num_tokens + 1),
    )
    backend = MagicMock()
    backend.mla_preprocess_context.return_value = preprocess_context
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = attn_out

    with (
        forward_context(
            ForwardContext(
                backend,
                torch.device("cpu"),
                SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                [],
            )
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
            "atb_matmul_ein_sum",
            return_value=projected,
            create=True,
        ),
    ):
        output, topk = attention(
            hidden,
            *rope,
            previous_topk,
            reuse_topk_indices=reuse_topk_indices,
        )

    attention._combined_qkv.forward_quantized.assert_not_called()
    selected_preprocess = mlapo_v2 if expect_mlapo_v2 else capturable_preprocess
    unselected_preprocess = capturable_preprocess if expect_mlapo_v2 else mlapo_v2
    selected_preprocess.assert_called_once()
    unselected_preprocess.assert_not_called()
    cos_arg = 16 if expect_mlapo_v2 else 14
    assert selected_preprocess.call_args.args[cos_arg] is rope[2]
    assert selected_preprocess.call_args.args[cos_arg + 1] is rope[3]
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
            backend.mla_index_context.return_value,
            rope[4],
        )
        attention.indexer.select_qli.assert_not_called()
    torch.testing.assert_close(output, projected.reshape(num_tokens, 2))
    assert topk is previous_topk


def test_glm_attention_dynamic_fused_decode_reuses_topk_after_cache_write() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention._use_fused_mla_decode = True
    attention._fused_mla_ready = True
    attention._use_mlapo_v2 = False
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
        indexer_rope_interleave=True,
    )
    attention._combined_qkv = SimpleNamespace(
        _dynamic_activation=True, weight=torch.empty(0), weight_scale=torch.empty(0)
    )
    attention.q_a_layernorm = SimpleNamespace(weight=torch.ones(2), eps=1e-5)
    attention.q_b_proj = SimpleNamespace(weight=torch.empty(0), weight_scale=torch.empty(0))
    attention.kv_a_layernorm = SimpleNamespace(weight=torch.ones(1), eps=1e-5)
    attention.W_UK = torch.ones(1, 1, 1)
    attention.W_UV = torch.ones(1, 1, 2)
    attention.o_proj = nn.Identity()

    hidden = torch.ones(2, 2)
    rope = _attention_rope(2)
    previous_topk = torch.zeros(2, 1, dtype=torch.int64)
    q_c = torch.ones(2, 2)
    q_latent = torch.ones(2, 1, 1)
    q_pe = torch.ones(2, 1, 1)
    projected = torch.ones(2, 1, 2)
    preprocess_context = MlaPreprocessContext(
        kv_cache=torch.empty(2, 1, 1),
        rope_cache=torch.empty(2, 1, 1),
        slot_mapping=torch.arange(3),
    )
    backend = MagicMock()
    backend.mla_preprocess_context.return_value = preprocess_context
    backend.mla_index_context.return_value = MagicMock()
    backend.execute_mla.return_value = torch.ones(2, 1, 1)

    with (
        forward_context(
            ForwardContext(
                backend,
                torch.device("cpu"),
                SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
                [],
            )
        ),
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode_dynamic",
            return_value=(q_c, torch.ones(2), q_latent, q_pe),
            create=True,
        ) as dynamic_preprocess,
        patch.object(
            glm5_2.kernels,
            "deepseek_mla_preprocess_decode",
            create=True,
        ) as static_preprocess,
        patch.object(
            glm5_2.kernels,
            "atb_matmul_ein_sum",
            return_value=projected,
            create=True,
        ),
    ):
        output, topk = attention(
            hidden,
            *rope,
            previous_topk,
            reuse_topk_indices=True,
        )

    dynamic_preprocess.assert_called_once()
    assert dynamic_preprocess.call_args.args[8] is rope[2]
    assert dynamic_preprocess.call_args.args[9] is rope[3]
    static_preprocess.assert_not_called()
    torch.testing.assert_close(dynamic_preprocess.call_args.args[10], torch.arange(2))
    attention.indexer._update_index_cache.assert_called_once_with(
        hidden,
        backend.mla_index_context.return_value,
        rope[4],
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


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("query_rows", [[2, 0], [0, 2], []])
def test_model_shares_coefficients_across_layers_for_packed_and_empty_queries(
    interleaved: bool, query_rows: list[int]
) -> None:
    model, layers = _make_model([])
    model.cfg.indexer_rope_interleave = interleaved
    context = SimpleNamespace(query_index=torch.tensor(query_rows, dtype=torch.int64))
    # A packed shard includes an interior padding row; query order need not be contiguous.
    local_positions = torch.tensor([7, 0, 2, 0])
    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=context)),
        patch.object(deepseek_v32, "cp_shard_rows", side_effect=lambda hidden, _ctx: hidden),
        patch.object(deepseek_v32, "cp_shard_positions", return_value=local_positions),
        patch.object(deepseek_v32, "cp_merge_rows", side_effect=lambda hidden, _ctx: hidden),
        patch.object(glm5_2, "record_layer_event"),
        patch.object(
            deepseek_v32, "_select_indexer_query_cos_sin", wraps=deepseek_v32._select_indexer_query_cos_sin
        ) as select_query,
    ):
        model(torch.arange(4), torch.arange(4))

    assert len(model.rotary.positions) == select_query.call_count == 1
    assert layers[0].query_cos_sin is layers[1].query_cos_sin
    for first, second in zip(layers[0].rope, layers[1].rope):
        assert first is second
    expected_half = model.rotary.cos_sin_cache[local_positions]
    torch.testing.assert_close(layers[0].rope[0], expected_half[:, :2])
    torch.testing.assert_close(layers[0].rope[1], expected_half[:, 2:])
    offset = 2 if interleaved else 0
    for index, selected in enumerate(layers[0].query_cos_sin):
        torch.testing.assert_close(selected, layers[0].rope[offset + index][context.query_index])


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("invalid", ["rows", "width", "dtype", "device"])
def test_rope_contract_rejects_inconsistent_metadata(interleaved: bool, invalid: str) -> None:
    shape = [2, 1, 1, 4] if interleaved else [2, 2]
    if invalid == "rows":
        shape[0] = 1
    if invalid == "width":
        shape[-1] += 1
    dtype = torch.bfloat16 if invalid == "dtype" else torch.float32
    device = "meta" if invalid == "device" else "cpu"
    cos_sin = (torch.empty(shape, dtype=dtype, device=device),) * 2
    with pytest.raises(ValueError, match="test consumer cos: expected shape=.*got shape="):
        glm5_2._validate_rope_cos_sin(cos_sin, torch.empty(2, 8), 4, interleaved, "test consumer")


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
        output = moe._combine_expert_outputs(routed, shared, False)

    parent_combine.assert_called_once_with(routed, shared, False)
    assert output is expected

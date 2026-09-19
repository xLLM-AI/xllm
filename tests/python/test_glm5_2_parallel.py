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

"""Parallel-layout tests for the GLM-5.2 Python NPU model."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.device_stream import get_device_stream
from xllm.python.models import glm5_2
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM
from xllm.python.models.weight_utils import W8A8WeightLoader


def _config(**overrides) -> dict:
    values = {
        "model_type": "glm_moe_dsa",
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
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 2,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 4,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 4,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def test_full_world_ep_partitions_glm_experts() -> None:
    cfg = Glm52Config.from_dict(_config(ep_rank=3))
    cfg.validate()

    model = Glm52ForCausalLM(_config(ep_rank=3))
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 6
    assert moe.local_expert_end == 8
    assert moe.num_local_experts == 2

    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.shape == (2, 16, 8)


def test_proper_divisor_ep_partitions_experts_and_moe_intermediate() -> None:
    values = _config(ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    cfg = Glm52Config.from_dict(values)

    cfg.validate()
    model = Glm52ForCausalLM(values)
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 4
    assert moe.local_expert_end == 8
    assert moe.num_local_experts == 4
    assert moe.inter_local == 4

    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (4, 8, 16)
    assert moe.experts_w2.shape == (4, 16, 4)


def test_glm_ep8_moe_tp2_topology_is_valid() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            num_attention_heads=64,
            n_routed_experts=256,
            moe_intermediate_size=2048,
            tp_size=8,
            dp_size=2,
            world_size=16,
            ep_size=8,
            ep_rank=7,
            moe_tp_size=2,
            moe_tp_rank=1,
        )
    )

    cfg.validate()


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


@pytest.mark.parametrize(("configured", "expected"), [(None, True), (False, False)])
def test_glm_mlapo_config_defaults_on_and_can_be_disabled(
    configured: bool | None,
    expected: bool,
) -> None:
    values = _config()
    if configured is not None:
        values["enable_mlapo"] = configured

    cfg = Glm52Config.from_dict(values)

    assert cfg.enable_mlapo is expected


@pytest.mark.parametrize(
    ("enable_mlapo", "device_type", "runtime_supported", "expected"),
    [
        (True, "npu", True, True),
        (True, "privateuseone", True, True),
        (False, "npu", True, False),
        (True, "cpu", True, False),
        (True, "npu", False, False),
    ],
)
def test_glm_mlapo_requires_config_device_and_runtime_support(
    monkeypatch,
    enable_mlapo: bool,
    device_type: str,
    runtime_supported: bool,
    expected: bool,
) -> None:
    cfg = Glm52Config.from_dict(_config(enable_mlapo=enable_mlapo))
    supports = MagicMock(return_value=runtime_supported)
    monkeypatch.setattr(glm5_2.kernels, "supports_mla_preprocess_v2", supports, raising=False)

    actual = glm5_2._can_use_mlapo_v2(cfg, MagicMock(type=device_type))

    assert actual is expected
    if enable_mlapo and device_type in ("npu", "privateuseone"):
        supports.assert_called_once_with(cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    else:
        supports.assert_not_called()


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


def test_glm_dsa_multi_stream_config_is_opt_in() -> None:
    assert not Glm52Config.from_dict(_config()).enable_dsa_multi_stream
    assert Glm52Config.from_dict(_config(enable_dsa_multi_stream=True)).enable_dsa_multi_stream


@pytest.mark.parametrize("interleave", [False, True])
def test_glm_indexer_stream_selection(monkeypatch: pytest.MonkeyPatch, interleave: bool) -> None:
    streams = MagicMock(side_effect=lambda _device, name: name)
    monkeypatch.setattr(glm5_2, "get_device_stream", streams)
    cfg = Glm52Config.from_dict(_config(enable_dsa_multi_stream=True, indexer_rope_interleave=interleave))
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    assert indexer._weights_stream == "dsa_indexer_weights"
    assert indexer._q_stream == (None if interleave else "dsa_indexer_q")
    assert streams.call_count == (1 if interleave else 2)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_glm_projection_quantization_loads_tp_shards(dynamic: bool, rank: int) -> None:
    model = Glm52ForCausalLM(_config(tp_rank=rank))
    name = "model.layers.0.self_attn.q_b_proj"
    original = model.get_submodule(name)
    rows = original.out_features * 2
    tensors = {
        name + ".weight": torch.arange(rows * original.weight.shape[1], dtype=torch.int8).reshape(rows, -1),
    }
    if dynamic:
        tensors[name + ".weight_scale"] = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) + 1
        tensors[name + ".weight_offset"] = torch.zeros(rows, 1)
        sharded = ("weight", "weight_scale", "weight_offset")
    else:
        tensors[name + ".deq_scale"] = torch.arange(rows, dtype=torch.float32) + 1
        tensors[name + ".quant_bias"] = torch.arange(rows, dtype=torch.int32)
        tensors[name + ".input_scale"] = torch.ones(1, dtype=original.input_scale.dtype)
        tensors[name + ".input_offset"] = torch.zeros(1, dtype=original.input_offset.dtype)
        sharded = ("weight", "deq_scale", "quant_bias")
    state_dict = SimpleNamespace(has=tensors.__contains__, get_tensor=tensors.__getitem__)
    projection = model.get_submodule(name)
    loader = W8A8WeightLoader(model, [state_dict], 2, rank)
    glm5_2._load_w8a8_attention_projection(
        loader, projection, "model.layers.0.self_attn.", "q_b_proj", dict.fromkeys(sharded, 0)
    )
    assert projection._dynamic_activation is dynamic
    for suffix in sharded:
        expected = tensors[name + "." + suffix].chunk(2, dim=0)[rank]
        torch.testing.assert_close(getattr(projection, suffix), expected)
    if not dynamic:
        assert projection is original
        torch.testing.assert_close(projection.input_scale, tensors[name + ".input_scale"])
        torch.testing.assert_close(projection.input_offset, tensors[name + ".input_offset"])


@pytest.mark.parametrize("mode", ["normal", "cp", "layerwise"])
def test_glm_dsa_projections_precede_indexer(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    cfg_values = _config(
        tp_size=1,
        dp_size=1,
        world_size=1,
        ep_size=1,
        num_attention_heads=2,
    )
    model = Glm52ForCausalLM(cfg_values)
    attention = model.model.layers[0].self_attn
    call_order: list[str] = []

    class _Projection(torch.nn.Module):
        def __init__(self, output: torch.Tensor, name: str) -> None:
            super().__init__()
            self._output = output
            self._name = name

        def forward(self, _input: torch.Tensor) -> torch.Tensor:
            call_order.append(self._name)
            return self._output

    class _Indexer(torch.nn.Module):
        def select_qli(self, *_args: object, **_kwargs: object) -> torch.Tensor:
            call_order.append("indexer")
            return torch.zeros(2, 1, cfg_values["index_topk"], dtype=torch.int32)

    hidden = torch.zeros(2, cfg_values["hidden_size"])
    attention.q_a_proj = _Projection(torch.zeros(2, cfg_values["q_lora_rank"]), "q_a")
    attention.q_a_layernorm = torch.nn.Identity()
    attention.q_b_proj = _Projection(
        torch.zeros(
            2,
            cfg_values["num_attention_heads"] * (cfg_values["qk_nope_head_dim"] + cfg_values["qk_rope_head_dim"]),
        ),
        "q_b",
    )
    attention.kv_a_proj_with_mqa = _Projection(
        torch.zeros(2, cfg_values["kv_lora_rank"] + cfg_values["qk_rope_head_dim"]),
        "kv_a",
    )
    attention.kv_a_layernorm = torch.nn.Identity()
    attention.o_proj = torch.nn.Identity()
    attention.indexer = _Indexer()

    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    cp_context = SimpleNamespace(query_index=torch.arange(2)) if mode == "cp" else None
    if mode == "layerwise":
        attention.cfg.layerwise_split_size = 2
        monkeypatch.setattr(
            glm5_2.distributed,
            "broadcast_",
            lambda *_args: call_order.append("broadcast"),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_2.distributed,
            "all_gather",
            lambda tensor, **_kwargs: tensor,
            raising=False,
        )

    def _execute_mla(*_args: object, **_kwargs: object) -> torch.Tensor:
        call_order.append("attention")
        return torch.zeros(2, cfg_values["num_attention_heads"], cfg_values["kv_lora_rank"])

    backend.execute_mla.side_effect = _execute_mla
    monkeypatch.setattr(
        glm5_2,
        "get_forward_context",
        lambda: SimpleNamespace(
            attention_backend=backend,
            cp_context=cp_context,
            metadata=SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
        ),
    )
    monkeypatch.setattr(
        glm5_2,
        "_gather_interleave_cos_sin",
        lambda cache, _positions: (cache, cache),
    )
    monkeypatch.setattr(glm5_2, "_interleave_rope_with", lambda tensor, _cos, _sin: tensor)
    monkeypatch.setattr(
        glm5_2.kernels,
        "batch_matmul_transpose",
        lambda lhs, rhs: torch.bmm(lhs, rhs).transpose(0, 1),
        raising=False,
    )

    attention(hidden, torch.arange(2), model.model.rotary.cos_sin_cache)

    assert call_order.index("q_b") < call_order.index("indexer")
    assert call_order.index("kv_a") < call_order.index("indexer")
    assert call_order.index("indexer") < call_order.index("attention")


@pytest.mark.parametrize("device_type", ["cpu", "npu"])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_glm_indexer_projection_overlap_matches_serial(
    monkeypatch: pytest.MonkeyPatch, device_type: str, interleave: bool, quantized: bool, dtype: torch.dtype
) -> None:
    if device_type == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("requires an available NPU")
    device = torch.device(device_type)
    torch.manual_seed(42)
    cfg = Glm52Config.from_dict(_config(indexer_rope_interleave=interleave))
    indexer = glm5_2.Glm52Indexer(cfg, dtype, device)
    indexer.wq_b = torch.nn.Linear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, device=device, dtype=dtype)
    hidden = torch.randn(3, cfg.hidden_size, device=device, dtype=dtype)
    qr = torch.randn(3, cfg.q_lora_rank, device=device, dtype=dtype)
    positions = torch.arange(3, device=device)
    angles = torch.randn(3, cfg.qk_rope_head_dim // 2, device=device, dtype=dtype)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    cache = torch.zeros(1, 3, 1, cfg.index_head_dim, device=device, dtype=torch.int8 if quantized else dtype)
    scales = torch.ones(1, 3, 1, 1, device=device, dtype=torch.float16) if quantized else None
    order: list[str] = []
    active = ["main"]

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_for_current(self) -> None:
            order.append(self.name + "_fork")

        @contextmanager
        def activate(self) -> Iterator[None]:
            active[0] = self.name
            try:
                yield
            finally:
                active[0] = "main"

        def join(self) -> None:
            order.append(self.name + "_join")

        def record_on_current(self, tensor: torch.Tensor) -> None:
            pass

    for name, module in (("q", indexer.wq_b), ("weights", indexer.weights_proj), ("k", indexer.wk)):
        module.register_forward_hook(lambda _module, _args, _out, name=name: order.append(name + "_" + active[0]))

    def _update(k: torch.Tensor, scale: torch.Tensor | None) -> None:
        order.append("cache")
        cache.copy_(k.reshape_as(cache))
        if scales is not None:
            scales.copy_(scale.reshape_as(scales))

    ctx = SimpleNamespace(
        actual_seq_q=[3],
        actual_seq_kv=[3],
        cp_context=None,
        index_cache=cache,
        index_cache_scale=scales,
        update_index_cache=_update,
        materialize_index_cache=lambda: (cache, scales, None),
        get_quant_indexer_metadata=lambda *_args: None,
    )

    def _select(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, *_args: object) -> torch.Tensor:
        order.append("select")
        # Consume every branch so stale side-stream outputs change the result.
        return q.float().sum(dim=(1, 2)) + weights.float().sum(dim=1) + k.float().sum()

    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        c, s = cos[:, 0, :, :half], sin[:, 0, :, :half]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * c - odd * s, odd * c + even * s), dim=-1).flatten(-2)

    monkeypatch.setattr(glm5_2.kernels, "interleaved_rotary_embedding", _rope, raising=False)
    monkeypatch.setattr(glm5_2.kernels, "lightning_indexer", _select, raising=False)
    monkeypatch.setattr(glm5_2.kernels, "quant_lightning_indexer", _select, raising=False)
    monkeypatch.setattr(
        glm5_2.kernels,
        "dynamic_quant",
        lambda x: (x.round().to(torch.int8), torch.ones(x.shape[:-1], device=device)),
        raising=False,
    )

    def _run() -> torch.Tensor:
        return indexer.select_qli(hidden, qr, positions, ctx, cos_sin)

    expected = _run()
    if device_type == "npu":
        torch.npu.synchronize()
    expected_cache = cache.clone()
    weights_stream = get_device_stream(device, "test_indexer_weights") if device_type == "npu" else _Stream("weights")
    q_stream = get_device_stream(device, "test_indexer_q") if device_type == "npu" else _Stream("q")
    indexer._weights_stream = weights_stream
    indexer._q_stream = None if interleave else q_stream
    order.clear()
    actual = _run()
    if device_type == "npu":
        torch.npu.synchronize()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache, expected_cache)
    if device_type == "cpu":
        assert "weights_weights" in order
        assert order.index("cache") < order.index("weights_join") < order.index("select")
        assert ("q_main" if interleave else "q_q") in order
        if not interleave:
            assert order.index("cache") < order.index("q_join") < order.index("select")
        return

    capture_stream = torch.npu.Stream(device=device)
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        for _ in range(2):
            _run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        graph_output = _run()
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        qr.copy_(torch.randn_like(qr))
        indexer._weights_stream = None
        indexer._q_stream = None
        expected = _run()
        expected_cache = cache.clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, expected)
        torch.testing.assert_close(cache, expected_cache)


def test_glm_layerwise_split_rank_is_validated() -> None:
    cfg = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=1))
    cfg.validate()
    assert cfg.layerwise_split_rank == 1

    invalid = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=2))
    with pytest.raises(ValueError, match="layerwise_split_rank"):
        invalid.validate()


def test_glm_layerwise_split_cannot_overlap_context_parallel() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            cp_size=2,
            cp_rank=0,
            world_size=8,
            ep_size=8,
            layerwise_split_size=2,
        )
    )
    with pytest.raises(ValueError, match="CP and layerwise"):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 3}, "ep_size must divide world_size"),
        ({"ep_size": 2}, r"moe_tp_size \* ep_size"),
        ({"world_size": 8}, r"world_size must equal tp_size \* dp_size \* cp_size"),
        ({"cp_rank": 2}, "cp_rank must be in"),
        ({"n_routed_experts": 10}, "n_routed_experts must be divisible by ep_size"),
        ({"moe_tp_size": 2}, r"moe_tp_size \* ep_size"),
        ({"ep_rank": 4}, "ep_rank must be in"),
    ],
)
def test_invalid_glm_parallel_topology_is_rejected(overrides: dict, message: str) -> None:
    cfg = Glm52Config.from_dict(_config(**overrides))

    with pytest.raises(ValueError, match=message):
        cfg.validate()


class _RecordingLoader(W8A8WeightLoader):
    latest: _RecordingLoader | None = None
    dynamic_activation = False

    def __init__(self, model, state_dicts, tp_size: int, tp_rank: int) -> None:
        super().__init__(model, state_dicts, tp_size, tp_rank)
        self.loaded: list[str] = []
        self.fused_projections: list[tuple[str, str, tuple[str, ...]]] = []
        self.shared_shards: list[tuple[str, int, int]] = []
        type(self).latest = self

    def load_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        if ".mlp.experts." not in name:
            return torch.zeros(32, 32)
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return torch.zeros(8, 16, dtype=torch.int8)
        if name.endswith(("gate_proj.weight_scale", "up_proj.weight_scale")):
            return torch.zeros(8, 1)
        if name.endswith(("gate_proj.weight_offset", "up_proj.weight_offset")):
            return torch.zeros(8, 1)
        if name.endswith("down_proj.weight"):
            return torch.zeros(16, 8, dtype=torch.int8)
        if name.endswith(("down_proj.weight_scale", "down_proj.weight_offset")):
            return torch.zeros(16, 1)
        raise AssertionError(f"unexpected expert tensor: {name}")

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_compatible_w8a8_projection(
        self,
        prefix: str,
        proj: str,
        _shard_dims: dict | None = None,
        dynamic_activation: bool | None = None,
    ) -> bool:
        self.loaded.append(prefix + proj)
        assert dynamic_activation is self.dynamic_activation
        return self.dynamic_activation

    def w8a8_projection_uses_dynamic_activation(self, prefix: str, proj: str) -> bool:
        return self.dynamic_activation

    def load_fused_w8a8_projection(
        self,
        prefix: str,
        target_proj: str,
        source_projs: tuple[str, ...],
    ) -> None:
        self.fused_projections.append((prefix, target_proj, source_projs))

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, world, rank))


class _TensorStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@pytest.mark.parametrize("dynamic_activation", [False, True])
def test_compatible_attention_loader_allocates_only_selected_quantization(dynamic_activation: bool) -> None:
    projection = glm5_2._W8A8AttentionLinear(4, 6, torch.device("cpu"))
    model = torch.nn.Module()
    model.proj = projection
    tensors = {"proj.weight": torch.zeros(6, 4, dtype=torch.int8)}
    if dynamic_activation:
        tensors.update(
            {
                "proj.weight_scale": torch.ones(6, 1),
                "proj.weight_offset": torch.zeros(6, 1),
            }
        )
    else:
        tensors.update(
            {
                "proj.deq_scale": torch.ones(6),
                "proj.quant_bias": torch.zeros(6, dtype=torch.int32),
                "proj.input_scale": torch.ones(1, dtype=torch.bfloat16),
                "proj.input_offset": torch.zeros(1, dtype=torch.bfloat16),
            }
        )
    loader = W8A8WeightLoader(model, [_TensorStateDict(tensors)], tp_size=1, tp_rank=0)

    selected_dynamic = loader.w8a8_projection_uses_dynamic_activation("", "proj")
    projection._set_dynamic_activation(selected_dynamic)
    loader.load_compatible_w8a8_projection("", "proj", dynamic_activation=selected_dynamic)

    assert selected_dynamic is dynamic_activation
    assert torch.equal(projection.weight, tensors["proj.weight"])
    if dynamic_activation:
        assert projection.weight_scale.numel() == projection.out_features
        assert projection.weight_offset.numel() == projection.out_features
        assert projection.deq_scale.numel() == 0
        assert projection.quant_bias.numel() == 0
        assert projection.input_scale.numel() == 0
        assert projection.input_offset.numel() == 0
    else:
        assert projection.weight_scale.numel() == 0
        assert projection.weight_offset.numel() == 0
        assert projection.deq_scale.numel() == projection.out_features
        assert projection.quant_bias.numel() == projection.out_features
        assert projection.input_scale.numel() == 1
        assert projection.input_offset.numel() == 1


def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(".experts.4." in name or ".experts.5." in name for name in expert_names)
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    attention_prefix = "model.layers.0.self_attn."
    attention_projections = [name for name in loader.loaded if name.startswith(attention_prefix)]
    assert all(
        name in attention_projections
        for name in [
            attention_prefix + "q_a_proj",
            attention_prefix + "q_b_proj",
            attention_prefix + "kv_a_proj_with_mqa",
        ]
    )
    assert loader.fused_projections == []
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]


def test_glm_weight_loader_shards_proper_divisor_ep_with_moe_tp(monkeypatch) -> None:
    values = _config(ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    model = Glm52ForCausalLM(values)
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(any(f".experts.{expert}." in name for expert in range(4, 8)) for name in expert_names)
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 2, 1)]


@pytest.mark.parametrize("dynamic_activation", [False, True])
def test_glm_attention_selects_checkpoint_quantization(
    dynamic_activation: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    attention = model.model.layers[0].self_attn
    attention.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(_RecordingLoader, "dynamic_activation", dynamic_activation)
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    assert attention.q_a_proj._dynamic_activation is dynamic_activation
    assert attention.q_b_proj._dynamic_activation is dynamic_activation
    assert attention.kv_a_proj_with_mqa._dynamic_activation is dynamic_activation
    assert attention.o_proj._dynamic_activation is dynamic_activation
    assert attention.indexer is not None
    assert attention.indexer.wq_b._dynamic_activation is dynamic_activation
    projections = (
        attention.q_a_proj,
        attention.q_b_proj,
        attention.kv_a_proj_with_mqa,
        attention.o_proj,
        attention.indexer.wq_b,
    )
    for projection in projections:
        if dynamic_activation:
            assert projection.weight_scale.numel() == projection.out_features
            assert projection.weight_offset.numel() == projection.out_features
            assert projection.deq_scale.numel() == 0
            assert projection.quant_bias.numel() == 0
            assert projection.input_scale.numel() == 0
            assert projection.input_offset.numel() == 0
        else:
            assert projection.weight_scale.numel() == 0
            assert projection.weight_offset.numel() == 0
            assert projection.deq_scale.numel() == projection.out_features
            assert projection.quant_bias.numel() == projection.out_features
            assert projection.input_scale.numel() == 1
            assert projection.input_offset.numel() == 1

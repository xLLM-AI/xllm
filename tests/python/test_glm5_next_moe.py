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

import math
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.models import glm5_next


def test_native_test_oracle_uses_glm_clamp_not_the_operator_under_test() -> None:
    cases = runpy.run_path(str(Path(__file__).parents[1] / "core/kernels/npu/glm5_next_moe_cases.py"))
    gate_up = torch.tensor([[20.0, -20.0, 20.0, 20.0]])
    actual = cases["_activation_reference"](gate_up)
    expected = torch.tensor([[100.0 / (1.0 + math.exp(-10.0)), -200.0 / (1.0 + math.exp(20.0))]])
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0] < 100.0
    assert abs(actual[0, 1]) < 1e-5
    quantized, scale = cases["_quantize_reference"](torch.tensor([[0.0, 1.0, -1.0]]))
    torch.testing.assert_close(quantized, torch.tensor([[0, 127, -127]], dtype=torch.int8))
    torch.testing.assert_close(scale, torch.tensor([1.0 / 127.0]))


def test_native_npu_cases_do_not_replace_production_bindings() -> None:
    path = Path(__file__).parents[1] / "core/kernels/npu/glm5_next_moe_cases.py"
    source = path.read_text()
    assert "monkeypatch" not in source
    assert "torch_npu.npu_dequant_swiglu_quant" not in source
    assert "_dispatch_has_kernel_for_dispatch_key" in source
    assert not (Path(__file__).parent / "test_glm5_next_moe_npu.py").exists()


def _config(**overrides: object) -> glm5_next.Glm5NextConfig:
    values = dict(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        n_layers=1,
        first_k_dense_replace=0,
    )
    values.update(overrides)
    return glm5_next.Glm5NextConfig.from_dict(values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ep_size": 3}, "n_routed_experts"),
        ({"ep_size": 0}, "ep parallel"),
        ({"ep_size": 2, "ep_rank": 2}, "ep parallel"),
        ({"moe_tp_size": 3}, "moe_intermediate_size"),
        ({"dp_rank": -1}, "dp parallel"),
        ({"tp_size": 2, "moe_tp_size": 1}, "TP-only MoE"),
        ({"tp_size": 2, "tp_rank": 1, "moe_tp_size": 2, "moe_tp_rank": 0}, "TP-only MoE"),
        ({"tp_size": 4, "ep_size": 2, "moe_tp_size": 1}, "must equal"),
        ({"expert_parallel_degree": 2}, "ordinary EP level 1"),
        ({"enable_mega_moe": True}, "does not support enable_mega_moe"),
        ({"enable_fused_mc2": True}, "does not support enable_fused_mc2"),
    ],
)
def test_invalid_parallel_config(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_config_preserves_tp_defaults_and_nested_text_fields() -> None:
    config = glm5_next.Glm5NextConfig.from_dict(
        {"text_config": {"tp_size": 4, "tp_rank": 3}, "ep_size": 2, "moe_tp_size": 2, "moe_tp_rank": 1}
    )
    assert (config.tp_size, config.tp_rank) == (4, 3)
    assert (config.ep_size, config.moe_tp_size, config.moe_tp_rank) == (2, 2, 1)
    config = _config(tp_size=2, tp_rank=1)
    assert (config.moe_tp_size, config.moe_tp_rank) == (2, 1)


@pytest.mark.parametrize(
    "axes",
    [
        {"tp_size": 2, "tp_rank": 1},
        {"tp_size": 4, "tp_rank": 1, "ep_size": 2, "ep_rank": 0, "moe_tp_size": 2, "moe_tp_rank": 1},
        {
            "dp_size": 2,
            "dp_rank": 1,
            "tp_size": 2,
            "tp_rank": 1,
            "ep_size": 4,
            "ep_rank": 3,
            "moe_tp_size": 1,
            "moe_tp_rank": 0,
        },
    ],
)
def test_vl_nested_entry_merges_runtime_axes_before_moe_defaults(axes: dict[str, int]) -> None:
    from xllm.python.models.glm5_next_vl import Glm5NextVLModel

    config = {
        "device": "meta",
        "dtype": "bfloat16",
        "text_config": {"num_hidden_layers": 0},
        "vision_config": {"depth": 0},
        **axes,
    }
    model = Glm5NextVLModel(config)
    expected = glm5_next.Glm5NextConfig.from_dict(config)
    for name in ("tp_size", "tp_rank", "dp_size", "dp_rank", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank"):
        assert getattr(model.text_cfg, name) == getattr(expected, name)
    model.text_cfg._validate_moe_parallelism()


@pytest.mark.parametrize("degree", [0, 1])
def test_ep_level1_config_accepts_native_degree_settings(degree: int) -> None:
    config = _config(tp_size=2, ep_size=2, moe_tp_size=1, expert_parallel_degree=degree)
    assert config.expert_parallel_degree == degree


@pytest.mark.parametrize("graph_friendly", [False, True])
@pytest.mark.parametrize("moe_tp_size", [1, 2])
def test_bf16_expert_partition_matches_unsharded(graph_friendly: bool, moe_tp_size: int) -> None:
    torch.manual_seed(7)
    config = _config()
    full = glm5_next.Glm5NextExperts(config, torch.float32, torch.device("cpu"))
    nn.init.normal_(full.gate_up_proj, std=0.1)
    nn.init.normal_(full.down_proj, std=0.1)
    hidden = torch.randn(5, config.hidden_size)
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 2]])
    topk_weights = torch.rand(5, 2)
    expected = full._forward_eager(hidden, topk_ids, topk_weights)
    output = torch.zeros_like(expected)
    for ep_rank in range(2):
        for moe_tp_rank in range(moe_tp_size):
            shard_config = replace(config, ep_size=2, ep_rank=ep_rank, moe_tp_size=moe_tp_size, moe_tp_rank=moe_tp_rank)
            shard = glm5_next.Glm5NextExperts(shard_config, torch.float32, torch.device("cpu"))
            expert_slice = slice(ep_rank * 2, (ep_rank + 1) * 2)
            inter_slice = slice(moe_tp_rank * shard.intermediate_dim, (moe_tp_rank + 1) * shard.intermediate_dim)
            gate, up = full.gate_up_proj[expert_slice].chunk(2, dim=1)
            shard.gate_up_proj.data.copy_(torch.cat((gate[:, inter_slice], up[:, inter_slice]), dim=1))
            shard.down_proj.data.copy_(full.down_proj[expert_slice, :, inter_slice])
            forward = shard._forward_graph_friendly if graph_friendly else shard._forward_eager
            output += forward(hidden, topk_ids, topk_weights)
    torch.testing.assert_close(output, expected)


class _SharedExperts(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(hidden) * 3


@pytest.mark.parametrize(
    "dp_size,ep_size,moe_tp_size",
    [(1, 1, 1), (1, 1, 2), (1, 1, 8), (1, 2, 1), (1, 2, 2), (2, 1, 4), (2, 2, 2)],
)
def test_grouped_moe_collectives_and_local_range(
    monkeypatch: pytest.MonkeyPatch, dp_size: int, ep_size: int, moe_tp_size: int
) -> None:
    config = _config(
        tp_size=ep_size * moe_tp_size // dp_size,
        dp_size=dp_size,
        dp_rank=dp_size - 1,
        ep_size=ep_size,
        ep_rank=ep_size - 1,
        moe_tp_size=moe_tp_size,
    )
    layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    layer.use_w8a8 = True
    layer.shared_experts = _SharedExperts()
    for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
        setattr(layer, name, torch.empty(0))
    hidden = torch.randn(1, 2, config.hidden_size)
    calls = []

    def grouped_moe(inputs: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert inputs.shape[0] == (3 if dp_size == 2 else 2)
        assert kwargs["start_expert_id"] == layer.local_expert_start
        assert kwargs["num_experts_per_rank"] == layer.num_local_experts
        assert kwargs["num_total_experts"] == layer.num_experts
        assert kwargs["swiglu_limit"] == config.swiglu_limit
        calls.append("grouped")
        return torch.ones_like(inputs)

    def gather(inputs: torch.Tensor, counts: tuple[int, ...], rank: int) -> tuple[torch.Tensor, int]:
        assert counts == (1, 2)
        assert rank == 1
        calls.append("gather")
        return torch.cat((torch.zeros_like(inputs[:1]), inputs)), 1

    monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", grouped_moe, raising=False)
    monkeypatch.setattr(
        glm5_next.kernels,
        "grouped_moe",
        lambda *args, **kwargs: pytest.fail("W8A8 must use the clamped common path, including DP1/EP1"),
        raising=False,
    )
    monkeypatch.setattr(
        glm5_next.kernels,
        "moe_gate_routing",
        lambda *args, **kwargs: (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32)),
        raising=False,
    )
    monkeypatch.setattr(glm5_next.distributed, "gather_dp_execution_tokens", gather, raising=False)
    monkeypatch.setattr(
        glm5_next.distributed, "moe_tp_all_reduce", lambda output: calls.append("moe_tp"), raising=False
    )
    monkeypatch.setattr(
        glm5_next.distributed, "moe_ep_all_reduce", lambda output: calls.append("moe_ep"), raising=False
    )
    monkeypatch.setattr(glm5_next.distributed, "all_reduce_", lambda output: calls.append("tp"), raising=False)
    from xllm.python.layers import moe_dp

    monkeypatch.setattr(
        moe_dp,
        "get_forward_context",
        lambda: SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=(1, 2))),
    )
    output = layer(hidden)
    torch.testing.assert_close(output, torch.ones_like(hidden) * (config.routed_scaling_factor + 3))
    expected = ["gather"] if dp_size > 1 else []
    expected.append("grouped")
    if moe_tp_size > 1:
        expected.append("moe_tp")
    if ep_size > 1:
        expected.append("moe_ep")
    assert calls == expected


@pytest.mark.parametrize("tp_size", [1, 2])
def test_bf16_tp_only_preserves_one_combined_reduction(monkeypatch: pytest.MonkeyPatch, tp_size: int) -> None:
    config = _config(tp_size=tp_size)
    layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    assert layer.shared_experts.skip_tp_reduce
    layer.shared_experts = _SharedExperts()
    hidden = torch.randn(1, 3, config.hidden_size)
    calls = []

    def reject_w8a8(*args: object, **kwargs: object) -> None:
        pytest.fail("BF16 TP-only execution must not enter the W8A8 route")

    class Experts(nn.Module):
        def forward(self, inputs: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            calls.append("bf16")
            return torch.ones_like(inputs) * config.routed_scaling_factor

    def reduce(output: torch.Tensor) -> None:
        torch.testing.assert_close(output, torch.ones_like(hidden) * (config.routed_scaling_factor + 3))
        calls.append("tp")

    layer.experts = Experts()
    monkeypatch.setattr(layer, "_topk", lambda value: (None, torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.long)))
    monkeypatch.setattr(layer, "_forward_w8a8", reject_w8a8)
    monkeypatch.setattr(glm5_next, "dp_gather_tokens", reject_w8a8)
    monkeypatch.setattr(glm5_next.kernels, "grouped_moe", reject_w8a8, raising=False)
    monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", reject_w8a8, raising=False)
    monkeypatch.setattr(glm5_next.distributed, "all_reduce_", reduce, raising=False)
    monkeypatch.setattr(glm5_next.distributed, "moe_tp_all_reduce", reject_w8a8, raising=False)
    monkeypatch.setattr(glm5_next.distributed, "moe_ep_all_reduce", reject_w8a8, raising=False)
    output = layer(hidden)
    torch.testing.assert_close(output, torch.ones_like(hidden) * (config.routed_scaling_factor + 3))
    assert calls == ["bf16"] + (["tp"] if tp_size > 1 else [])


@pytest.mark.parametrize("dp_size,ep_size,moe_tp_size", [(1, 2, 1), (1, 2, 2), (2, 1, 4), (2, 2, 2)])
def test_shared_experts_keep_attention_tp_layout(dp_size: int, ep_size: int, moe_tp_size: int) -> None:
    config = _config(
        tp_size=ep_size * moe_tp_size // dp_size, dp_size=dp_size, ep_size=ep_size, moe_tp_size=moe_tp_size
    )
    layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    assert layer.shared_experts.skip_tp_reduce
    assert layer.shared_experts.gate_up_proj.weight.shape == (
        2 * config.moe_intermediate_size // config.tp_size,
        config.hidden_size,
    )
    assert layer.inter_local == config.moe_intermediate_size // config.moe_tp_size


@pytest.mark.parametrize(
    "dp_size,tp_size,ep_size", [(1, 1, 1), (1, 8, 1), (1, 8, 8), (2, 4, 8), (2, 4, 4), (4, 2, 2), (2, 4, 1)]
)
@pytest.mark.parametrize("empty_rank", [False, True])
def test_joint_reduction_injects_shared_once_in_its_dp_rows(
    monkeypatch: pytest.MonkeyPatch, dp_size: int, tp_size: int, ep_size: int, empty_rank: bool
) -> None:
    from xllm.python.layers import moe_dp

    world_size = dp_size * tp_size
    moe_tp_size = world_size // ep_size
    counts = tuple(1 if empty_rank and rank == 0 else rank + 2 for rank in range(dp_size))
    offsets = [sum(counts[:rank]) for rank in range(dp_size)]
    hidden = torch.arange(1, sum(counts) * 8 + 1, dtype=torch.float64).view(-1, 8)
    if empty_rank:
        hidden[:1].zero_()  # Native execution materializes a dummy row.
    combined_partials = []

    class SharedPartial(nn.Module):
        def __init__(self, rank: int, expected: torch.Tensor) -> None:
            super().__init__()
            self.rank = rank
            self.expected = expected

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            torch.testing.assert_close(inputs, self.expected, rtol=0, atol=0)
            return inputs * (self.rank + 1)

    monkeypatch.setattr(
        moe_dp,
        "get_forward_context",
        lambda: SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=counts)),
    )
    monkeypatch.setattr(
        glm5_next.distributed,
        "all_reduce_",
        lambda output: pytest.fail("shared must not issue a TP reduction"),
        raising=False,
    )
    for rank in range(world_size):
        dp_rank, tp_rank = divmod(rank, tp_size)
        config = _config(
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=rank // moe_tp_size,
            moe_tp_size=moe_tp_size,
            moe_tp_rank=rank % moe_tp_size,
            n_routed_experts=8,
            moe_intermediate_size=16,
        )
        layer = glm5_next.Glm5NextMoE(config, torch.float64, torch.device("cpu"))
        layer.use_w8a8 = True
        local = hidden.narrow(0, offsets[dp_rank], counts[dp_rank])
        layer.shared_experts = SharedPartial(tp_rank, local)
        for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
            setattr(layer, name, torch.empty(0))
        monkeypatch.setattr(
            glm5_next.distributed,
            "gather_dp_execution_tokens",
            lambda inputs, execution_counts, index: (hidden.clone(), offsets[index]),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_next.kernels,
            "moe_gate_routing",
            lambda *args, **kwargs: (torch.ones(sum(counts), 2), torch.zeros(sum(counts), 2, dtype=torch.int32)),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_next.kernels,
            "grouped_moe_with_selected_experts",
            lambda inputs, *args, rank=rank, **kwargs: inputs * (rank + 1),
            raising=False,
        )
        captures = []
        monkeypatch.setattr(
            glm5_next.distributed,
            "moe_tp_all_reduce",
            lambda output, captures=captures: captures.append(output.clone()),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_next.distributed,
            "moe_ep_all_reduce",
            lambda output, captures=captures: captures.append(output.clone()),
            raising=False,
        )
        output = layer(local)
        assert output.dtype == local.dtype
        assert len(captures) == int(moe_tp_size > 1) + int(ep_size > 1)
        expected = (hidden * ((rank + 1) * config.routed_scaling_factor)).float()
        expected.narrow(0, offsets[dp_rank], counts[dp_rank]).add_((local * (tp_rank + 1)).float())
        # TP1/EP1 has no collective to capture, but must account for the same
        # routed and shared contributions in its final output.
        partials = captures or [output.float()]
        for captured in partials:
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        combined_partials.append(partials[0])

    # An independent full-world sum checks contribution accounting, including
    # MoE-TP groups that cross attention-DP boundaries. This is a CPU contract
    # test, not evidence that real distributed collectives ran.
    expected_routed = hidden * (world_size * (world_size + 1) / 2 * config.routed_scaling_factor)
    expected_shared = hidden * (tp_size * (tp_size + 1) / 2)
    torch.testing.assert_close(
        torch.stack(combined_partials).double().sum(0), expected_routed + expected_shared, rtol=0, atol=0
    )


@pytest.mark.parametrize("dp_size,ep_size", [(1, 2), (2, 1), (2, 2)])
def test_bf16_ep_is_rejected_before_execution(monkeypatch: pytest.MonkeyPatch, dp_size: int, ep_size: int) -> None:
    config = _config(tp_size=2, dp_size=dp_size, ep_size=ep_size, moe_tp_size=2 * dp_size // ep_size)
    layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    monkeypatch.setattr(
        glm5_next, "dp_gather_tokens", lambda *args: pytest.fail("unsupported BF16 must fail before communication")
    )
    with pytest.raises(NotImplementedError, match="W8A8 expert weights only"):
        layer(torch.zeros(1, 8))


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.loaded: list[str] = []

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        return self.tensors[name]


@pytest.mark.parametrize("quantized", [False, True])
def test_expert_loader_only_reads_local_experts(quantized: bool) -> None:
    config = _config(tp_size=4, tp_rank=3, ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.cfg = config
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    moe = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    model.model.layers[0].mlp = moe
    tensors = {}
    for expert in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            weight = torch.arange(64).reshape(8, 8) + expert
            tensors[prefix + ".weight"] = weight.to(torch.int8 if quantized else torch.float32)
            tensors[prefix + ".weight_scale"] = torch.arange(8, dtype=torch.float32).reshape(8, 1) + expert
            tensors[prefix + ".weight_offset"] = torch.zeros(8, 1)
    state_dict = _StateDict(tensors)
    loader = W8A8WeightLoader(model, [state_dict], tp_size=4, tp_rank=3)
    load = model._load_experts_w8a8 if quantized else model._load_experts_bf16
    load(loader, "model.layers.0.mlp.", moe)
    assert not any("experts.0." in name or "experts.1." in name for name in state_dict.loaded)
    assert (loader.tp_size, loader.tp_rank) == (4, 3)
    gate_up = moe.experts_w13 if quantized else moe.experts.gate_up_proj
    down = moe.experts_w2 if quantized else moe.experts.down_proj
    assert gate_up.shape == (2, 8, 8)
    assert down.shape == (2, 8, 4)
    for local_index, expert in enumerate((2, 3)):
        expected = tensors[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"][4:]
        torch.testing.assert_close(gate_up[local_index], torch.cat((expected, expected)))
        expected_down = tensors[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"][:, 4:]
        torch.testing.assert_close(down[local_index], expected_down)
        if quantized:
            scale = tensors[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight_scale"][4:]
            torch.testing.assert_close(moe.experts_w13_scale[local_index], torch.cat((scale, scale)))
            torch.testing.assert_close(
                moe.experts_w2_scale[local_index],
                tensors[f"model.layers.0.mlp.experts.{expert}.down_proj.weight_scale"].to(torch.bfloat16),
            )
            assert torch.count_nonzero(moe.experts_w13_offset).item() == 0
            assert torch.count_nonzero(moe.experts_w2_offset).item() == 0


def test_bf16_packed_loader_slices_ep_then_moe_tp() -> None:
    config = _config(tp_size=4, tp_rank=3, ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.cfg = config
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
    gate_up = torch.arange(4 * 16 * 8, dtype=torch.float32).reshape(4, 16, 8)
    down = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    state = _StateDict(
        {
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
        }
    )
    loader = W8A8WeightLoader(model, [state], tp_size=4, tp_rank=3)
    model._load_experts_bf16(loader, "model.layers.0.mlp.", model.model.layers[0].mlp)
    experts = model.model.layers[0].mlp.experts
    torch.testing.assert_close(experts.gate_up_proj, torch.cat((gate_up[2:, 4:8], gate_up[2:, 12:16]), dim=1))
    torch.testing.assert_close(experts.down_proj, down[2:, :, 4:])
    assert (loader.tp_size, loader.tp_rank) == (4, 3)


@pytest.mark.parametrize("rows", [1, 7, 16])
def test_native_numerical_fixture_is_away_from_input_quantization_ties(rows: int) -> None:
    cases = runpy.run_path(str(Path(__file__).parents[1] / "core/kernels/npu/glm5_next_moe_cases.py"))
    case = cases["_make_case"](rows)
    _, scale = cases["_quantize_reference"](case["hidden"])
    normalized = case["hidden"].float() / scale[:, None]
    torch.testing.assert_close(normalized, normalized.round(), rtol=0, atol=0)
    # Graph tests retain the historical varying-input sample, not just this
    # numerically stable fixture.
    original = cases["_make_case"](rows, quantization_safe=False)
    assert not torch.equal(original["hidden"], case["hidden"])


@pytest.mark.parametrize("shape", [(1, 8), (4, 8), (1, 4, 8)])
def test_quantized_expert_scales_are_loaded_in_kernel_dtypes(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    config = glm5_next.Glm5NextConfig(
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        tp_size=1,
    )
    module = glm5_next.Glm5NextMoE(config, torch.bfloat16, torch.device("cpu"))
    monkeypatch.setattr(module.shared_experts, "forward", lambda hidden: hidden)
    module.use_w8a8 = True
    model = torch.nn.Module()
    model.cfg = config
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = module
    model.model.layers = torch.nn.ModuleList([layer])
    original = torch.linspace(0.001, 1.0, 32).view(4, 8, 1)
    gate_up_scale = torch.linspace(0.001, 1.0, 16).view(16, 1)
    tensors: dict[str, torch.Tensor] = {}
    for expert_index in range(config.n_routed_experts):
        expert_tensors = {
            "gate_proj.weight": torch.ones(16, 8, dtype=torch.int8),
            "gate_proj.weight_scale": gate_up_scale,
            "gate_proj.weight_offset": torch.zeros(16, 1),
            "up_proj.weight": torch.ones(16, 8, dtype=torch.int8),
            "up_proj.weight_scale": gate_up_scale,
            "up_proj.weight_offset": torch.zeros(16, 1),
            "down_proj.weight": torch.ones(8, 16, dtype=torch.int8),
            "down_proj.weight_scale": original[expert_index],
            "down_proj.weight_offset": torch.zeros(8, 1),
        }
        tensors.update(
            {f"model.layers.0.mlp.experts.{expert_index}.{name}": value for name, value in expert_tensors.items()}
        )
    loader = W8A8WeightLoader(
        model,
        [SimpleNamespace(has=tensors.__contains__, get_tensor=tensors.__getitem__)],
        tp_size=1,
        tp_rank=0,
    )
    monkeypatch.setattr(glm5_next.kernels, "format_cast_nz", lambda weight: weight, raising=False)

    glm5_next.Glm5NextForCausalLM._load_experts_w8a8(model, loader, "model.layers.0.mlp.", module)

    assert module.experts_w13_scale.dtype == torch.float32
    assert module.experts_w2_scale.dtype == torch.bfloat16
    torch.testing.assert_close(module.experts_w2_scale, original.bfloat16(), rtol=0, atol=0)
    module.process_weights_after_loading()

    expected_w13_scale = torch.cat((gate_up_scale, gate_up_scale)).view(1, 32).expand(4, 32)
    torch.testing.assert_close(module.experts_w13_scale, expected_w13_scale, rtol=0, atol=0)
    torch.testing.assert_close(module.experts_w2_scale, original.view(4, 8).bfloat16(), rtol=0, atol=0)
    assert module.get_buffer("experts_w2_scale") is module.experts_w2_scale
    assert not hasattr(module, "experts_w2_scale_compute")
    torch.testing.assert_close(module.state_dict()["experts_w2_scale"], original.view(4, 8).bfloat16(), rtol=0, atol=0)
    calls: list[object] = []

    def grouped_moe(hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert args[4] is module.experts_w13_scale
        calls.append(args[5])
        return torch.zeros_like(hidden)

    monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", grouped_moe, raising=False)
    monkeypatch.setattr(
        glm5_next.kernels,
        "moe_gate_routing",
        lambda logits, *args, **kwargs: (
            torch.ones(logits.shape[0], config.num_experts_per_tok),
            torch.zeros(logits.shape[0], config.num_experts_per_tok, dtype=torch.int32),
        ),
        raising=False,
    )
    hidden = torch.ones(shape, dtype=torch.bfloat16)
    for _ in range(2):
        torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)
    assert len(calls) == 2
    assert all(scale is module.experts_w2_scale for scale in calls)


def test_nonquantized_experts_do_not_allocate_quantization_scales() -> None:
    config = glm5_next.Glm5NextConfig(
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        tp_size=1,
    )
    module = glm5_next.Glm5NextMoE(config, torch.bfloat16, torch.device("cpu"))

    module.process_weights_after_loading()

    assert not hasattr(module, "experts_w13_scale")
    assert not hasattr(module, "experts_w2_scale")
    assert not hasattr(module, "experts_w2_scale_compute")

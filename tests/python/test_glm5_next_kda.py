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

import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.nn.functional as functional

from xllm.python.model_loader import QLinearWeightLoader
from xllm.python.models import glm5_next
from xllm.python.models.glm5_next import _KDA_IN_PROJ

_ATTENTION_PREFIX = "model.layers.0.self_attn."


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _make_model(
    tp_size: int, tp_rank: int
) -> tuple[glm5_next.Glm5NextForCausalLM, dict[str, torch.Tensor], glm5_next.Glm5NextKdaAttention]:
    config = glm5_next.Glm5NextConfig(hidden_size=32, kda_num_heads=8, kda_head_dim=8, tp_size=tp_size, tp_rank=tp_rank)
    attention = glm5_next.Glm5NextKdaAttention(config, 0, torch.float32, torch.device("cpu"))
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    torch.nn.Module.__init__(model)
    model.cfg = config
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = attention
    model.model.layers = torch.nn.ModuleList([layer])
    generator = torch.Generator().manual_seed(42)
    projection_size = config.kda_num_heads * config.kda_head_dim
    shapes = {
        "q_proj.weight": (projection_size, config.hidden_size),
        "k_proj.weight": (projection_size, config.hidden_size),
        "v_proj.weight": (projection_size, config.hidden_size),
        "b_proj.weight": (config.kda_num_heads, config.hidden_size),
        "forget_gate.f_a_proj.weight": (config.kda_head_dim, config.hidden_size),
        "g_a_proj.weight": (config.kda_head_dim, config.hidden_size),
        "forget_gate.f_b_proj.weight": (projection_size, config.kda_head_dim),
        "g_b_proj.weight": (projection_size, config.kda_head_dim),
        "forget_gate.A_log": (config.kda_num_heads,),
        "forget_gate.dt_bias": (projection_size,),
        "o_proj.weight": (config.hidden_size, projection_size),
        "o_norm.weight": (config.kda_head_dim,),
    }
    for channel in ("q", "k", "v"):
        shapes[f"{channel}_conv1d.weight"] = (projection_size, 1, config.short_conv_kernel_size)
    tensors = {
        _ATTENTION_PREFIX + name: torch.randn(shape, generator=generator) * 0.1 for name, shape in shapes.items()
    }
    loader = QLinearWeightLoader(model, [_StateDict(tensors)], tp_size, tp_rank)
    model._load_kda_attn(loader, _ATTENTION_PREFIX, 0)
    return model, tensors, attention


def _shard_rows(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    assert weight.size(0) % tp_size == 0, "test weights must be divisible by tp_size"
    shard_size = weight.size(0) // tp_size
    return weight.narrow(0, tp_rank * shard_size, shard_size).contiguous()


def _input_weights(tensors: dict[str, torch.Tensor], tp_size: int, tp_rank: int) -> list[torch.Tensor]:
    weights = []
    for projection, _, shard_dim in _KDA_IN_PROJ:
        weight = tensors[_ATTENTION_PREFIX + projection + ".weight"]
        if shard_dim is not None:
            weight = _shard_rows(weight, tp_size, tp_rank)
        weights.append(weight)
    return weights


def _reference_projections(
    hidden_states: torch.Tensor, tensors: dict[str, torch.Tensor], tp_size: int, tp_rank: int
) -> tuple[torch.Tensor, ...]:
    def _project(name: str, sharded: bool) -> torch.Tensor:
        weight = tensors[_ATTENTION_PREFIX + name + ".weight"]
        if sharded:
            weight = _shard_rows(weight, tp_size, tp_rank)
        return functional.linear(hidden_states, weight.to(device=hidden_states.device, dtype=hidden_states.dtype))

    return (
        _project("q_proj", True),
        _project("k_proj", True),
        _project("v_proj", True),
        _project("b_proj", True),
        _project("forget_gate.f_a_proj", False),
        _project("g_a_proj", False),
    )


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@torch.inference_mode()
def test_fg_batched_weights_preserve_head_shards_and_share_storage(tp_size: int) -> None:
    for tp_rank in range(tp_size):
        model, tensors = _make_model(tp_size, tp_rank)
        attention = model.model.layers[0].self_attn
        expected = torch.stack(
            [
                _shard_rows(tensors[_ATTENTION_PREFIX + name], tp_size, tp_rank)
                for name in ("forget_gate.f_b_proj.weight", "g_b_proj.weight")
            ]
        )
        torch.testing.assert_close(attention._fg_b_weight, expected, rtol=0, atol=0)
        for weight in (attention.forget_gate.f_b_proj.weight, attention.g_b_proj.weight):
            assert weight.untyped_storage().data_ptr() == attention._fg_b_weight.untyped_storage().data_ptr()
        attention.process_weights_after_loading()
        torch.testing.assert_close(attention._fg_b_weight, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_fg_batched_weights_reuse_storage_across_reloads() -> None:
    # A captured decode ACL graph records _fg_b_weight's storage address, so a
    # weight hot-reload must keep the packed buffer at a stable address instead
    # of reallocating; otherwise replay reads stale weights. Reproduces the
    # reload path: copy_in writes fresh weights into the existing .data storage,
    # then process_weights_after_loading re-packs.
    model, _ = _make_model(1, 0)
    attention = model.model.layers[0].self_attn
    captured = attention._fg_b_weight
    captured_ptr = captured.data_ptr()
    generator = torch.Generator().manual_seed(7)
    for _ in range(2):
        new_f = torch.randn(attention.forget_gate.f_b_proj.weight.shape, generator=generator)
        new_g = torch.randn(attention.g_b_proj.weight.shape, generator=generator)
        attention.forget_gate.f_b_proj.weight.data.copy_(new_f)
        attention.g_b_proj.weight.data.copy_(new_g)
        attention.process_weights_after_loading()
        assert attention._fg_b_weight.data_ptr() == captured_ptr
        torch.testing.assert_close(captured[0], new_f, rtol=0, atol=0)
        torch.testing.assert_close(captured[1], new_g, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 17), (3, 4)])
@torch.inference_mode()
def test_fg_batched_projection_matches_separate_inputs(
    batch_size: int, seq_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _make_model(2, 1)
    attention = model.model.layers[0].self_attn
    projected = attention.in_proj_qkvbfg_a(torch.randn(batch_size, seq_len, model.cfg.hidden_size))
    latents = projected[..., -2 * attention.head_dim :]
    forget_latent, output_latent = latents.chunk(2, dim=-1)
    hidden_shape = (batch_size, seq_len, attention.num_heads_local, attention.head_dim)
    expected = (
        attention.forget_gate.raw_projection(forget_latent),
        attention.g_b_proj(output_latent).view(hidden_shape),
    )
    bmm_calls = []
    original_bmm = torch.bmm

    def _bmm(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bmm_calls.append(inputs.shape)
        return original_bmm(inputs, weights)

    monkeypatch.setattr(torch, "bmm", _bmm)
    torch.testing.assert_close(attention._project_fg(latents), expected)
    assert bmm_calls == [torch.Size((2, batch_size * seq_len, attention.head_dim))]


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_merged_input_weights_shard_heads_and_replicate_latents(tp_size: int) -> None:
    for tp_rank in range(tp_size):
        _, tensors, attention = _make_model(tp_size, tp_rank)
        expected = torch.cat(_input_weights(tensors, tp_size, tp_rank))
        torch.testing.assert_close(attention.in_proj_qkvbfg_a.weight, expected, rtol=0, atol=0)
        assert attention.in_proj_qkvbfg_a.bias is None
        assert not any(name + ".weight" in dict(attention.named_parameters()) for name, _, _ in _KDA_IN_PROJ)


@pytest.mark.parametrize("projection", [name for name, _, _ in _KDA_IN_PROJ])
@pytest.mark.parametrize("quantization", ["int8", "deq_scale", "weight_scale"])
def test_merged_input_rejects_quantized_checkpoint(projection: str, quantization: str) -> None:
    model, tensors, attention = _make_model(2, 1)
    original_weight = attention.in_proj_qkvbfg_a.weight.detach().clone()
    prefix = _ATTENTION_PREFIX + projection
    if quantization == "int8":
        tensors[prefix + ".weight"] = tensors[prefix + ".weight"].to(torch.int8)
    else:
        tensors[prefix + "." + quantization] = torch.ones(tensors[prefix + ".weight"].size(0))
    loader = QLinearWeightLoader(model, [_StateDict(tensors)], 2, 1)
    with pytest.raises(ValueError, match="KDA input projection.*floating-point"):
        model._load_kda_attn(loader, _ATTENTION_PREFIX, 0)
    torch.testing.assert_close(attention.in_proj_qkvbfg_a.weight, original_weight, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (8, 7)])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 17), (3, 4)])
@torch.inference_mode()
def test_merged_input_forward_preserves_backend_arguments_and_output(
    tp_size: int, tp_rank: int, batch_size: int, seq_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(42)
    model, tensors, attention = _make_model(tp_size, tp_rank)
    hidden_states = torch.randn(batch_size, seq_len * 2, model.cfg.hidden_size)[:, ::2]
    query, key, value, beta_raw, forget_latent, output_latent = _reference_projections(
        hidden_states, tensors, tp_size, tp_rank
    )
    hidden_shape = (batch_size, seq_len, attention.num_heads_local, attention.head_dim)
    expected_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)
    expected_raw = functional.linear(forget_latent, attention.forget_gate.f_b_proj.weight).view(hidden_shape)
    core_output = value.view(hidden_shape)
    output_gate = functional.linear(output_latent, attention.g_b_proj.weight).view(hidden_shape)
    expected_output = attention.o_proj(attention.o_norm(core_output, output_gate).reshape(batch_size, seq_len, -1))
    calls = []
    reductions = []

    def _execute_linear(
        mixed_qkv: torch.Tensor, beta: torch.Tensor, layer: torch.nn.Module, *, raw_gate_proj: torch.Tensor
    ) -> torch.Tensor:
        assert layer is attention
        torch.testing.assert_close(mixed_qkv, expected_qkv)
        torch.testing.assert_close(beta, beta_raw.float().sigmoid())
        torch.testing.assert_close(raw_gate_proj, expected_raw)
        return core_output

    backend = SimpleNamespace(execute_linear=_execute_linear)
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: SimpleNamespace(attention_backend=backend))
    monkeypatch.setattr(glm5_next.distributed, "all_reduce_", lambda output: reductions.append(output), raising=False)
    handle = attention.in_proj_qkvbfg_a.register_forward_hook(lambda *_args: calls.append(True))
    output = attention(hidden_states, torch.arange(seq_len))
    handle.remove()
    torch.testing.assert_close(output, expected_output)
    assert len(calls) == 1
    assert len(reductions) == (1 if tp_size > 1 else 0)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (8, 7)])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("activation", ["none", "silu"])
@pytest.mark.parametrize(
    "conv_fn",
    [
        pytest.param(glm5_next._causal_conv1d_fn, id="prefill"),
        pytest.param(glm5_next._causal_conv1d_update, id="decode"),
        pytest.param(glm5_next._causal_conv1d_update_graph, id="graph_decode"),
    ],
)
@torch.inference_mode()
def test_merged_qkv_strides_preserve_causal_conv_and_state(
    tp_size: int, tp_rank: int, batch_size: int, activation: str, conv_fn: Callable[..., torch.Tensor]
) -> None:
    """Require exact conv/state equality; SiLU permits layout-dependent FP32 rounding."""
    torch.manual_seed(42)
    model, tensors, attention = _make_model(tp_size, tp_rank)
    for projection in ("q_proj", "k_proj", "v_proj"):
        weight = tensors[_ATTENTION_PREFIX + projection + ".weight"]
        weight.copy_(torch.randint(-4, 4, weight.shape).float() / 16)
    attention.in_proj_qkvbfg_a.weight.copy_(torch.cat(_input_weights(tensors, tp_size, tp_rank)))
    conv_weight = attention.conv1d.weight.squeeze(1)
    conv_state = torch.randn(batch_size, attention.conv_dim, attention.conv_kernel_size - 1)
    reference_state = conv_state.clone()
    for seq_len in (17, 1, 4, 1):
        hidden_states = torch.randint(-4, 4, (batch_size, seq_len, model.cfg.hidden_size)).float() / 8
        projected = attention.in_proj_qkvbfg_a(hidden_states)
        mixed_qkv = projected.split(attention.input_projection_sizes, dim=-1)[0].transpose(1, 2)
        query, key, value, *_ = _reference_projections(hidden_states, tensors, tp_size, tp_rank)
        reference_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        assert mixed_qkv.stride(-1) == sum(attention.input_projection_sizes)
        assert reference_qkv.stride(-1) == attention.conv_dim
        torch.testing.assert_close(mixed_qkv, reference_qkv, rtol=0, atol=0)
        if conv_fn is glm5_next._causal_conv1d_fn:
            actual = conv_fn(mixed_qkv, conv_weight, activation=activation)
            expected = conv_fn(reference_qkv, conv_weight, activation=activation)
        else:
            expected_state = torch.cat((reference_state, reference_qkv), dim=-1)[:, :, -conv_state.size(-1) :]
            actual = conv_fn(mixed_qkv, conv_state, conv_weight, activation=activation)
            expected = conv_fn(reference_qkv, reference_state, conv_weight, activation=activation)
            torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(conv_state, reference_state, rtol=0, atol=0)
        relative_tolerance = 2 * torch.finfo(actual.dtype).eps if activation == "silu" else 0
        torch.testing.assert_close(actual, expected, rtol=relative_tolerance, atol=0)


@pytest.fixture(scope="module")
def npu_runtime() -> ModuleType:
    return pytest.importorskip("torch_npu")


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 128])
@torch.inference_mode()
def test_merged_input_npu_projection_and_graph_replay(num_tokens: int, npu_runtime: ModuleType) -> None:
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    npu_runtime.npu.set_device(device)
    torch.manual_seed(42)
    model, tensors, attention = _make_model(8, 7)
    attention = attention.to(device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(1, num_tokens, model.cfg.hidden_size, device=device, dtype=torch.bfloat16)

    def _reference() -> torch.Tensor:
        return torch.cat(_reference_projections(hidden_states, tensors, 8, 7), dim=-1)

    for _ in range(3):
        attention.in_proj_qkvbfg_a(hidden_states)
    npu_runtime.npu.synchronize()
    graph = npu_runtime.npu.NPUGraph()
    with npu_runtime.npu.graph(graph):
        output = attention.in_proj_qkvbfg_a(hidden_states)
    for _ in range(3):
        hidden_states.copy_(torch.randn_like(hidden_states))
        graph.replay()
        npu_runtime.npu.synchronize()
        torch.testing.assert_close(output, attention.in_proj_qkvbfg_a(hidden_states), rtol=0, atol=0)
        torch.testing.assert_close(output, _reference())


@pytest.fixture(scope="module")
def real_kda_weights(npu_runtime: ModuleType) -> tuple[glm5_next.Glm5NextConfig, dict[str, torch.Tensor]]:
    model_path = os.getenv("XLLM_GLM_KDA_MODEL_PATH", "")
    if not model_path:
        pytest.skip("real KDA checkpoint not configured")
    from safetensors import safe_open

    checkpoint = Path(model_path)
    config = glm5_next.Glm5NextConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    names = {projection + ".weight" for projection, _, _ in _KDA_IN_PROJ}
    names.update(("forget_gate.f_b_proj.weight", "g_b_proj.weight", "forget_gate.A_log", "forget_gate.dt_bias"))
    aliases = {
        alias: name
        for name in names
        for alias in [_ATTENTION_PREFIX + name, *glm5_next._real_ckpt_aliases(_ATTENTION_PREFIX + name)]
    }
    tensors = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as source:
            for name in source.keys():
                if name in aliases:
                    tensors[_ATTENTION_PREFIX + aliases[name]] = source.get_tensor(name)
    assert set(tensors) == {_ATTENTION_PREFIX + name for name in names}
    return config, tensors


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("tp_rank", [0, 7])
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 128])
@torch.inference_mode()
def test_real_kda_merged_projection_graph_replay(
    real_kda_weights: tuple[glm5_next.Glm5NextConfig, dict[str, torch.Tensor]],
    npu_runtime: ModuleType,
    tp_rank: int,
    num_tokens: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Require exact graph/eager agreement and bounded BF16 fusion error.

    Peak-scaled and relative-L2 bounds account for cancellation near zero in
    the second projections without hiding large errors behind relative error.
    """
    config, tensors = real_kda_weights
    config.tp_size = 8
    config.tp_rank = tp_rank
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    npu_runtime.npu.set_device(device)
    torch.manual_seed(42)
    attention = glm5_next.Glm5NextKdaAttention(config, 0, torch.bfloat16, device).to(
        device=device, dtype=torch.bfloat16
    )
    weights = [weight.to(device=device, dtype=torch.bfloat16) for weight in _input_weights(tensors, 8, tp_rank)]
    attention.in_proj_qkvbfg_a.weight.copy_(torch.cat(weights))
    for projection in ("forget_gate.f_b_proj.weight", "g_b_proj.weight", "forget_gate.A_log", "forget_gate.dt_bias"):
        attention.get_parameter(projection).copy_(_shard_rows(tensors[_ATTENTION_PREFIX + projection], 8, tp_rank))
    attention.process_weights_after_loading()
    hidden_states = torch.randn(1, num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    hidden_shape = (1, num_tokens, attention.num_heads_local, attention.head_dim)

    def _project(merged: bool) -> tuple[torch.Tensor, ...]:
        if merged:
            projected = attention.in_proj_qkvbfg_a(hidden_states)
            mixed_qkv, beta_raw, fg_latents = projected.split(attention.input_projection_sizes, dim=-1)
            raw_gate, output_gate = attention._project_fg(fg_latents)
        else:
            query, key, value, beta_raw, forget_latent, output_latent = _reference_projections(
                hidden_states, tensors, 8, tp_rank
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
            raw_gate = attention.forget_gate.raw_projection(forget_latent)
            output_gate = attention.g_b_proj(output_latent).view(hidden_shape)
        return (
            mixed_qkv.transpose(1, 2),
            beta_raw.float().sigmoid(),
            raw_gate,
            attention.forget_gate.gate_from_raw(raw_gate),
            output_gate,
        )

    for _ in range(3):
        _project(True)
    npu_runtime.npu.synchronize()
    graph = npu_runtime.npu.NPUGraph()
    with npu_runtime.npu.graph(graph):
        outputs = _project(True)
    for replay_index in range(3):
        hidden_states.copy_(torch.randn_like(hidden_states))
        graph.replay()
        npu_runtime.npu.synchronize()
        torch.testing.assert_close(outputs, _project(True), rtol=0, atol=0)
        for name, actual, expected in zip(
            ("qkv", "beta", "raw_gate", "final_gate", "output_gate"), outputs, _project(False), strict=True
        ):
            difference = actual.float() - expected.float()
            max_error = difference.abs().max().item()
            reference_peak = expected.float().abs().max().item()
            relative_l2 = difference.norm().item() / max(expected.float().norm().item(), 1e-9)
            record_property(f"replay_{replay_index}_{name}_max_abs", max_error)
            record_property(f"replay_{replay_index}_{name}_relative_l2", relative_l2)
            assert max_error <= torch.finfo(torch.bfloat16).eps * reference_peak + 1e-5, name
            assert relative_l2 <= 1e-3, name

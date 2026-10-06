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
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from xllm.python.layers.qlinear import QLinear
from xllm.python.model_loader import QLinearWeightLoader
from xllm.python.models import glm5_next


def _make_attention(tp_size: int = 1, tp_rank: int = 0, bias: bool = False) -> glm5_next.Glm5NextMlaAttention:
    config = glm5_next.Glm5NextConfig(
        hidden_size=32,
        n_heads=8,
        n_kv_heads=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        indexer_types=["shared"],
        attention_bias=bias,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    attention = glm5_next.Glm5NextMlaAttention(config, 0, torch.float32, torch.device("cpu"))
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    for module in attention.modules():
        if isinstance(module, QLinear):
            module.resolve_quant(False)
    return attention


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 17), (3, 4)])
@torch.inference_mode()
def test_qkv_a_merged_forward_matches_separate_projections(
    bias: bool, batch_size: int, seq_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    attention = _make_attention(bias=bias)
    hidden_states = torch.randn(batch_size, seq_len, attention.hidden_size)
    topk = torch.zeros(batch_size * seq_len, 1, 4, dtype=torch.int32)
    mla_inputs = []

    def _execute_mla(
        query: torch.Tensor,
        query_rope: torch.Tensor | None,
        key: torch.Tensor,
        key_rope: torch.Tensor | None,
        layer: torch.nn.Module,
        *,
        topk: torch.Tensor,
    ) -> torch.Tensor:
        assert query_rope is None and key_rope is None
        assert layer is attention
        mla_inputs.append((query, key, topk))
        return query + key

    context = SimpleNamespace(attention_backend=SimpleNamespace(execute_mla=_execute_mla))
    monkeypatch.setattr(glm5_next, "get_forward_context", lambda: context)
    attention.process_weights_after_loading()
    attention._qkv_a_weight = None
    attention._qkv_a_bias = None
    expected = attention(hidden_states, torch.arange(seq_len), torch.ones(batch_size, seq_len), topk)
    attention.process_weights_after_loading()

    def _unexpected_projection(*_args: object) -> torch.Tensor:
        pytest.fail("packed Q/KV-A must not dispatch separate projections")

    monkeypatch.setattr(attention.q_a_proj, "forward", _unexpected_projection)
    monkeypatch.setattr(attention.kv_a_proj_with_mqa, "forward", _unexpected_projection)
    actual = attention(hidden_states, torch.arange(seq_len), torch.ones(batch_size, seq_len), topk)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mla_inputs[1], mla_inputs[0])
    for weight in (attention.q_a_proj.weight, attention.kv_a_proj_with_mqa.weight):
        assert weight.untyped_storage().data_ptr() == attention._qkv_a_weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("quantized", ["query", "kv", "both"])
@torch.inference_mode()
def test_qkv_a_quantized_projections_keep_independent_calls(quantized: str, monkeypatch: pytest.MonkeyPatch) -> None:
    attention = _make_attention()
    attention.process_weights_after_loading()
    inputs = torch.randn(3, attention.hidden_size)
    expected_query = attention.q_a_proj(inputs)
    expected_kv = attention.kv_a_proj_with_mqa(inputs)
    attention.q_a_proj.use_w8a8 = quantized in ("query", "both")
    attention.kv_a_proj_with_mqa.use_w8a8 = quantized in ("kv", "both")
    calls = []

    def _query(hidden: torch.Tensor) -> torch.Tensor:
        assert hidden is inputs
        calls.append("query")
        return expected_query

    def _kv(hidden: torch.Tensor) -> torch.Tensor:
        assert hidden is inputs
        calls.append("kv")
        return expected_kv

    monkeypatch.setattr(attention.q_a_proj, "forward", _query)
    monkeypatch.setattr(attention.kv_a_proj_with_mqa, "forward", _kv)
    attention.process_weights_after_loading()
    assert attention._qkv_a_weight is None
    torch.testing.assert_close(attention._project_qkv_a(inputs), (expected_query, expected_kv), rtol=0, atol=0)
    assert calls == ["query", "kv"]


@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_qkv_a_merged_weights_reuse_storage_across_reloads(bias: bool) -> None:
    # A captured decode ACL graph records _qkv_a_weight/_qkv_a_bias storage
    # addresses, so a weight hot-reload must keep the packed buffers stable
    # rather than reallocating via torch.cat; otherwise replay reads stale
    # weights. Mirrors the reload path (copy_in writes into the existing .data
    # storage, then process_weights_after_loading re-packs).
    attention = _make_attention(bias=bias)
    attention.process_weights_after_loading()
    captured_weight = attention._qkv_a_weight
    captured_weight_ptr = captured_weight.data_ptr()
    captured_bias = attention._qkv_a_bias
    generator = torch.Generator().manual_seed(11)
    for _ in range(2):
        new_q = torch.randn(attention.q_a_proj.weight.shape, generator=generator)
        new_kv = torch.randn(attention.kv_a_proj_with_mqa.weight.shape, generator=generator)
        attention.q_a_proj.weight.data.copy_(new_q)
        attention.kv_a_proj_with_mqa.weight.data.copy_(new_kv)
        if bias:
            attention.q_a_proj.bias.data.copy_(torch.randn(attention.q_a_proj.bias.shape, generator=generator))
            attention.kv_a_proj_with_mqa.bias.data.copy_(
                torch.randn(attention.kv_a_proj_with_mqa.bias.shape, generator=generator)
            )
        attention.process_weights_after_loading()
        assert attention._qkv_a_weight.data_ptr() == captured_weight_ptr
        torch.testing.assert_close(captured_weight[: attention.q_lora_rank], new_q, rtol=0, atol=0)
        torch.testing.assert_close(captured_weight[attention.q_lora_rank :], new_kv, rtol=0, atol=0)
        if bias:
            assert attention._qkv_a_bias.data_ptr() == captured_bias.data_ptr()


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (8, 7)])
@torch.inference_mode()
def test_qkv_a_loader_keeps_replicated_weights(tp_size: int, tp_rank: int) -> None:
    source = _make_attention()
    prefix = "model.layers.0.self_attn."
    tensors = {prefix + name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    state_dict = SimpleNamespace(has=lambda name: name in tensors, get_tensor=tensors.__getitem__)
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    torch.nn.Module.__init__(model)
    attention = _make_attention(tp_size, tp_rank)
    model.cfg = attention.cfg
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = attention
    model.model.layers = torch.nn.ModuleList([layer])
    loader = QLinearWeightLoader(model, [state_dict], tp_size, tp_rank)
    model._load_dsa_attn(loader, prefix, 0)
    expected = torch.cat((source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight))
    torch.testing.assert_close(attention._qkv_a_weight, expected, rtol=0, atol=0)
    tensors[prefix + "q_a_proj.weight"].add_(1)
    model._load_dsa_attn(loader, prefix, 0)
    torch.testing.assert_close(attention._qkv_a_weight[: attention.q_lora_rank], tensors[prefix + "q_a_proj.weight"])


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 8, 17, 128])
@torch.inference_mode()
def test_qkv_a_npu_graph_replay(num_tokens: int) -> None:
    runtime = pytest.importorskip("torch_npu")
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    attention = _make_attention().to(device=device, dtype=torch.bfloat16)
    attention.process_weights_after_loading()
    inputs = torch.randn(num_tokens, attention.hidden_size, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        attention._project_qkv_a(inputs)
    torch.npu.synchronize()
    graph = runtime.npu.NPUGraph()
    with runtime.npu.graph(graph):
        output = attention._project_qkv_a(inputs)
    for _ in range(3):
        inputs.copy_(torch.randn_like(inputs))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, attention._project_qkv_a(inputs), rtol=0, atol=0)
        expected = attention.q_a_proj(inputs), attention.kv_a_proj_with_mqa(inputs)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def real_dsa_weights() -> tuple[glm5_next.Glm5NextConfig, dict[str, torch.Tensor]]:
    model_path = os.getenv("XLLM_GLM_KDA_MODEL_PATH")
    if not model_path:
        pytest.skip("real GLM checkpoint not configured")
    from safetensors import safe_open

    checkpoint = Path(model_path)
    config = glm5_next.Glm5NextConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    layer_id = next(index for index in range(config.n_layers) if config.is_dsa(index))
    prefix = f"model.layers.{layer_id}.self_attn."
    names = ("q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_a_layernorm.weight", "kv_a_layernorm.weight")
    aliases = {alias: name for name in names for alias in (prefix + name, *glm5_next._real_ckpt_aliases(prefix + name))}
    tensors = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as source:
            for name in source.keys():
                if name in aliases:
                    tensors[aliases[name]] = source.get_tensor(name)
    assert set(tensors) == set(names)
    return config, tensors


@pytest.mark.parametrize("device_kind", ["cpu", "npu"])
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 128])
@torch.inference_mode()
def test_real_qkv_a_projection_and_graph_replay(
    real_dsa_weights: tuple[glm5_next.Glm5NextConfig, dict[str, torch.Tensor]],
    device_kind: str,
    num_tokens: int,
    record_property: Callable[[str, object], None],
) -> None:
    if device_kind == "npu":
        if not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"):
            pytest.skip("NPU device not configured")
        runtime = pytest.importorskip("torch_npu")
        device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
        torch.npu.set_device(device)
    else:
        device = torch.device("cpu")
    config, tensors = real_dsa_weights
    config.tp_size = 8
    layer_id = next(index for index in range(config.n_layers) if config.is_dsa(index))
    attention = glm5_next.Glm5NextMlaAttention(config, layer_id, torch.bfloat16, device).to(
        device=device, dtype=torch.bfloat16
    )
    for name, tensor in tensors.items():
        attention.get_parameter(name).copy_(tensor)
    attention.q_a_proj.resolve_quant(False)
    attention.kv_a_proj_with_mqa.resolve_quant(False)
    attention.kv_b_proj.weight.zero_()
    attention.process_weights_after_loading()
    torch.manual_seed(42)
    inputs = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)

    def _project(merged: bool) -> tuple[torch.Tensor, ...]:
        if merged:
            query, kv = attention._project_qkv_a(inputs)
        else:
            query, kv = attention.q_a_proj(inputs), attention.kv_a_proj_with_mqa(inputs)
        return query, kv, attention.q_a_layernorm(query), attention.kv_a_layernorm(kv)

    if device_kind == "npu":
        for _ in range(3):
            _project(True)
        torch.npu.synchronize()
        graph = runtime.npu.NPUGraph()
        with runtime.npu.graph(graph):
            outputs = _project(True)
    for replay_index in range(3):
        inputs.copy_(torch.randn_like(inputs))
        if device_kind == "npu":
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(outputs, _project(True), rtol=0, atol=0)
        else:
            outputs = _project(True)
        for name, actual, expected in zip(
            ("query", "kv", "query_norm", "kv_norm"), outputs, _project(False), strict=True
        ):
            difference = actual.float() - expected.float()
            max_error = difference.abs().max().item()
            reference_peak = expected.float().abs().max().item()
            relative_l2 = difference.norm().item() / max(expected.float().norm().item(), 1e-9)
            record_property(f"replay_{replay_index}_{name}_max_abs", max_error)
            record_property(f"replay_{replay_index}_{name}_relative_l2", relative_l2)
            assert max_error <= torch.finfo(torch.bfloat16).eps * reference_peak + 1e-5, name
            assert relative_l2 <= 1e-3, name

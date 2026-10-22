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

"""OLMo2 model semantics, strict checkpoint loading and registry contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Olmo2Config as HfOlmo2Config
from transformers import Olmo2ForCausalLM as HfOlmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm as HfOlmo2RMSNorm

from xllm import xllm_export
from xllm.pybind.utils import _infer_model_type_and_backend
from xllm.python import distributed, registry
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, LayerCache
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.olmo2 import (
    Olmo2Attention,
    Olmo2Config,
    Olmo2ForCausalLM,
    Olmo2RMSNorm,
    _Olmo2ColumnParallelLinear,
    _Olmo2RowParallelLinear,
)
from xllm.python.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="OLMo2 currently supports the CUDA runtime only")


def _config(**overrides: Any) -> dict[str, Any]:
    values = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "intermediate_size": 24,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "use_cache": False,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.reads: set[str] = set()

    def keys(self) -> list[str]:
        return list(self.tensors)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        self.reads.add(name)
        return self.tensors[name]


def _reference() -> HfOlmo2ForCausalLM:
    torch.manual_seed(20261019)
    cfg = HfOlmo2Config(**{key: value for key, value in _config().items() if key not in ("device", "dtype")})
    cfg._attn_implementation = "eager"
    model = HfOlmo2ForCausalLM(cfg).eval()
    # Nonuniform, signed gamma makes the full-projection norm axis observable.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2)
    return model


def test_official_configuration_and_cpp_field_aliases() -> None:
    fixture = Path(__file__).parents[1] / "models/fixtures/olmo2/config.json"
    cfg = Olmo2Config.from_dict(json.loads(fixture.read_text()))
    assert (cfg.hidden_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (2048, 16, 16, 16, 128)
    assert (cfg.intermediate_size, cfg.vocab_size, cfg.max_position_embeddings) == (8192, 100352, 4096)
    assert cfg.rms_norm_eps == 1e-6 and cfg.rope_theta == 500000.0
    config = _config()
    for hf_name, cpp_name in (
        ("num_hidden_layers", "n_layers"),
        ("num_attention_heads", "n_heads"),
        ("num_key_value_heads", "n_kv_heads"),
    ):
        config[cpp_name] = config.pop(hf_name)
    assert Olmo2Config.from_dict(config) == Olmo2Config.from_dict(_config())


def test_default_rope_parameters_preserve_theta() -> None:
    cfg = Olmo2Config.from_dict(_config(rope_parameters={"rope_type": "default", "rope_theta": 123456.0}))
    assert cfg.rope_theta == 123456.0


def test_legacy_default_rope_parameters_preserve_theta() -> None:
    cfg = Olmo2Config.from_dict(_config(rope_parameters={"type": "default", "rope_theta": 123456.0}))
    assert cfg.rope_theta == 123456.0


def test_legacy_longrope_parameters_are_rejected() -> None:
    parameters = {
        "type": "longrope",
        "rope_theta": 10000.0,
        "short_factor": [1.0, 1.0],
        "long_factor": [2.0, 2.0],
        "original_max_position_embeddings": 16,
    }
    with pytest.raises(ValueError, match="unscaled RoPE"):
        Olmo2Config.from_dict(_config(rope_parameters=parameters))


def test_canonical_rope_type_overrides_legacy_alias() -> None:
    parameters = {"type": "longrope", "rope_type": "default", "rope_theta": 123456.0}
    assert Olmo2Config.from_dict(_config(rope_parameters=parameters)).rope_theta == 123456.0
    parameters.update(type="default", rope_type="longrope")
    with pytest.raises(ValueError, match="unscaled RoPE"):
        Olmo2Config.from_dict(_config(rope_parameters=parameters))


def test_native_backend_discovery_for_formal_entrypoint() -> None:
    assert xllm_export.get_model_backend("olmo2") == "llm"
    fixture = Path(__file__).parents[1] / "models/fixtures/olmo2"
    assert _infer_model_type_and_backend(str(fixture)) == ("olmo2", "llm")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"hidden_size": 15}, "divisible"),
        ({"head_dim": 3}, "even head_dim"),
        ({"num_key_value_heads": 2}, "MHA"),
        ({"tp_size": 3}, "TP=1/2"),
        ({"tp_size": 2, "tp_rank": 2}, "rank"),
        ({"dp_size": 2}, "DP=CP=PP=1"),
        ({"cp_size": 2}, "DP=CP=PP=1"),
        ({"world_size": 2}, "DP=CP=PP=1"),
        ({"intermediate_size": 25, "tp_size": 2}, "divide evenly"),
        ({"rms_norm_eps": float("nan")}, "positive and finite"),
        ({"rope_theta": 0}, "positive and finite"),
        ({"hidden_act": "gelu"}, "SiLU"),
        ({"attention_bias": True}, "bias-free"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"rope_scaling": {"factor": 2}}, "unscaled"),
        ({"rope_parameters": {"rope_type": "linear", "factor": 2}}, "unscaled"),
        ({"rope_parameters": []}, "must be an object"),
        ({"quant_method": "awq"}, "unquantized"),
    ],
)
def test_unsupported_configuration_is_rejected(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Olmo2Config.from_dict(_config(**change))


def test_missing_required_dimension_is_rejected() -> None:
    config = _config()
    del config["num_hidden_layers"]
    with pytest.raises(ValueError, match="requires config field: n_layers"):
        Olmo2Config.from_dict(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_matches_reference_without_intermediate_low_precision_cast(dtype: torch.dtype) -> None:
    torch.manual_seed(13)
    value = (torch.randn(3, 16) * 3.0).to(dtype)
    weight = torch.randn(16).to(dtype)
    norm = Olmo2RMSNorm(16, 1e-6, dtype, torch.device("cpu"))
    reference = HfOlmo2RMSNorm(16, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        reference.weight.copy_(weight)
        torch.testing.assert_close(norm(value), reference(value), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_qk_norm_reduces_separate_global_statistics(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    cfg = Olmo2Config.from_dict(_config(tp_size=2, tp_rank=rank))
    attention = Olmo2Attention(cfg, 0, torch.float32, torch.device("cpu"))
    full_q = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    full_k = torch.arange(101, 133, dtype=torch.float32).reshape(2, 16)
    q, k = full_q.chunk(2, dim=-1)[rank], full_k.chunk(2, dim=-1)[rank]
    remote_q, remote_k = full_q.chunk(2, dim=-1)[1 - rank], full_k.chunk(2, dim=-1)[1 - rank]
    calls: list[torch.Tensor] = []

    def _sum(stats: torch.Tensor) -> None:
        calls.append(stats.clone())
        stats.add_(torch.stack((remote_q.square().sum(-1), remote_k.square().sum(-1)), dim=-1))

    monkeypatch.setattr(distributed, "tp_all_reduce", _sum)
    gamma = torch.linspace(-1.3, 2.1, 16)
    with torch.no_grad():
        attention.q_norm.weight.copy_(gamma.chunk(2)[rank])
        attention.k_norm.weight.copy_(gamma.flip(0).chunk(2)[rank])
        actual_q, actual_k = attention._normalize_qk(q, k)
        for full, local, weight in ((full_q, actual_q, gamma), (full_k, actual_k, gamma.flip(0))):
            reference = HfOlmo2RMSNorm(16, eps=cfg.rms_norm_eps)
            reference.weight.copy_(weight)
            torch.testing.assert_close(local, reference(full).chunk(2, dim=-1)[rank])
    assert len(calls) == 1 and calls[0].shape == (2, 2) and calls[0].dtype == torch.float32


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_checkpoint_loading_consumes_all_keys_and_preserves_logical_shards(tp_size: int, rank: int) -> None:
    state = _StateDict(dict(_reference().state_dict()))
    model = Olmo2ForCausalLM(_config(tp_size=tp_size, tp_rank=rank))
    model.load_weights([state], rank, tp_size)
    tensors = state.tensors
    assert set(model.loaded_checkpoint_keys) == set(tensors) == state.reads
    assert len(state.reads) == 25
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        expected_qkv = torch.cat(
            [tensors[prefix + f"self_attn.{name}_proj.weight"].chunk(tp_size, dim=0)[rank] for name in ("q", "k", "v")]
        )
        torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv, rtol=0, atol=0)
        for name in ("q", "k"):
            torch.testing.assert_close(
                getattr(layer.self_attn, f"{name}_norm").weight,
                tensors[prefix + f"self_attn.{name}_norm.weight"].chunk(tp_size)[rank],
                rtol=0,
                atol=0,
            )
        expected_mlp = torch.cat(
            [tensors[prefix + f"mlp.{name}_proj.weight"].chunk(tp_size, dim=0)[rank] for name in ("gate", "up")]
        )
        torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected_mlp, rtol=0, atol=0)
        torch.testing.assert_close(
            layer.self_attn.o_proj.weight, tensors[prefix + "self_attn.o_proj.weight"].chunk(tp_size, dim=1)[rank]
        )
        torch.testing.assert_close(
            layer.mlp.down_proj.weight, tensors[prefix + "mlp.down_proj.weight"].chunk(tp_size, dim=1)[rank]
        )
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            torch.testing.assert_close(getattr(layer, name).weight, tensors[prefix + name + ".weight"])
    torch.testing.assert_close(
        model.model.embed_tokens.weight, tensors["model.embed_tokens.weight"].chunk(tp_size, 1)[rank]
    )
    torch.testing.assert_close(model.lm_head.weight, tensors["lm_head.weight"].chunk(tp_size, 0)[rank])
    torch.testing.assert_close(model.model.norm.weight, tensors["model.norm.weight"])
    assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "wrong_shape"])
def test_invalid_checkpoint_fails_without_ignoring_tensors(fault: str) -> None:
    tensors = dict(_reference().state_dict())
    states = [_StateDict(tensors)]
    if fault == "missing":
        del tensors["lm_head.weight"]
    elif fault == "extra":
        tensors["model.layers.0.input_layernorm.weight"] = torch.ones(16)
    elif fault == "duplicate":
        states.append(_StateDict({"lm_head.weight": tensors["lm_head.weight"]}))
    else:
        tensors["model.layers.0.self_attn.q_norm.weight"] = torch.ones(4)
    model = Olmo2ForCausalLM(_config())
    with pytest.raises(ValueError, match="checkpoint"):
        model.load_weights(states, 0, 1)
    assert model.loaded_checkpoint_keys == ()


class _DenseAttentionBackend:
    """Unit reference for attention only; the model and CUDA MLP remain real."""

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        values = [value.view(1, -1, layer.num_heads, layer.head_dim).transpose(1, 2) for value in (q, k, v)]
        output = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True, scale=layer.scale)
        return output.transpose(1, 2).reshape(-1, layer.num_heads * layer.head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OLMo2 CUDA composition test requires a GPU")
def test_cuda_model_matches_tiny_transformers_at_nonzero_positions() -> None:
    device = torch.device("cuda:0")
    reference = _reference().to(device)
    model = Olmo2ForCausalLM(_config(device=str(device))).eval()
    model.load_weights([_StateDict(dict(reference.state_dict()))], 0, 1)
    tokens = torch.tensor([3, 5, 7], device=device)
    positions = torch.tensor([2, 5, 9], device=device)
    context = ForwardContext(
        attention_backend=cast(AttentionBackend, _DenseAttentionBackend()),
        device=device,
        metadata=cast(AttentionMetadata, SimpleNamespace()),
        layer_caches=[],
    )
    with torch.inference_mode(), forward_context(context):
        # Explicitly select one causal sequence: HF otherwise treats position gaps as packed sequence boundaries.
        expected = reference(
            tokens.unsqueeze(0),
            attention_mask=torch.ones_like(tokens.unsqueeze(0)),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).logits[0]
        actual = model.compute_logits(model.model(tokens, positions), None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert model.model.rotary.cos_sin_cache.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OLMo2 residual precision test requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_post_norm_updates_survive_repeated_residual_additions(dtype: torch.dtype) -> None:
    device = torch.device("cuda:0")
    model = Olmo2ForCausalLM(_config(device=str(device), dtype=dtype)).eval()
    layer = model.model.layers[0]
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        # A positive, uniform MLP output normalizes to gamma, independently of
        # its magnitude. Each layer should retain an approximately 0.001 update.
        layer.mlp.gate_up_proj.weight.fill_(0.125)
        layer.mlp.down_proj.weight.fill_(1 / 24)
        layer.post_feedforward_layernorm.weight.fill_(0.001)
    hidden = torch.ones(2, 16, dtype=dtype, device=device)
    cos = torch.ones(2, 2, device=device)
    sin = torch.zeros_like(cos)
    context = ForwardContext(
        attention_backend=cast(AttentionBackend, _DenseAttentionBackend()),
        device=device,
        metadata=cast(AttentionMetadata, SimpleNamespace()),
        layer_caches=[],
    )
    with torch.inference_mode(), forward_context(context):
        for _ in range(8):
            hidden = layer(hidden, cos, sin)
    torch.testing.assert_close(hidden.float(), torch.full_like(hidden.float(), 1.008), rtol=0, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 projection test requires CUDA")
@pytest.mark.parametrize("row_parallel", [False, True])
def test_bf16_projection_preserves_small_residual_input(row_parallel: bool) -> None:
    layer_type = _Olmo2RowParallelLinear if row_parallel else _Olmo2ColumnParallelLinear
    layer = layer_type(2, 1, 1, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0]], device="cuda:0"))
    value = torch.tensor([[1.001, 1.0], [1.0, 1.002]], device="cuda:0")
    with torch.inference_mode():
        actual = layer(value)
        expected = value @ layer.weight.float().t()
        rounded = torch.nn.functional.linear(value.bfloat16(), layer.weight).float()
    assert actual.dtype == torch.float32 and layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    assert torch.count_nonzero(rounded) == 0
    assert torch.count_nonzero(actual) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 projection test requires CUDA")
def test_bf16_row_projection_reduces_fp32_before_replicated_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = _Olmo2RowParallelLinear(2, 1, 2, bias=True, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0]], device="cuda:0"))
        layer.bias.fill_(0.25)
    layer.weight.data = layer.weight.data.t().contiguous()
    layer._weight_is_transposed = True
    calls: list[torch.Tensor] = []

    def _sum(value: torch.Tensor) -> None:
        calls.append(value.clone())
        value.add_(0.002)

    monkeypatch.setattr(distributed, "tp_all_reduce", _sum)
    with torch.inference_mode():
        actual = layer(torch.tensor([[1.001, 1.0]], device="cuda:0"))
    assert len(calls) == 1 and calls[0].dtype == torch.float32
    torch.testing.assert_close(calls[0], torch.full_like(calls[0], 0.001), rtol=0, atol=2e-6)
    torch.testing.assert_close(actual, torch.full_like(actual, 0.253), rtol=0, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 projection test requires CUDA")
def test_bf16_lm_head_gathers_fp32_vocabulary_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = _Olmo2ColumnParallelLinear(2, 1, 2, gather_output=True, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0]], device="cuda:0"))

    def _gather(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert value.dtype == torch.float32 and dim == -1 and world_size == 2
        return torch.cat((value, -value), dim=-1)

    monkeypatch.setattr(distributed, "tp_all_gather", _gather)
    with torch.inference_mode():
        actual = layer(torch.tensor([[1.001, 1.0]], device="cuda:0"))
    torch.testing.assert_close(actual, torch.tensor([[0.001, -0.001]], device="cuda:0"), rtol=0, atol=2e-6)


class _PagedModelRunner:
    """Drive the production FlashInfer backend with disjoint request pages."""

    def __init__(self, model: Olmo2ForCausalLM) -> None:
        from xllm.python.attention.flashinfer import FlashInferBackend

        self.model = model
        self.block_size = 128
        attention = model.model.layers[0].self_attn.attn
        self.backend = FlashInferBackend(
            attention.num_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.scale,
            0,
            model.device,
            model.dtype,
            use_tensor_core_decode=attention.use_tensor_core_decode,
        )
        shape = (8, self.block_size, attention.num_kv_heads, attention.head_dim)
        self.caches = [
            LayerCache(
                key=torch.full(shape, float("nan"), dtype=model.dtype, device=model.device),
                value=torch.full(shape, float("nan"), dtype=model.dtype, device=model.device),
            )
            for _ in model.model.layers
        ]
        self.backend.bind_kv_caches(self.caches)

    def run(
        self, queries: list[list[int]], past: list[int], pages: list[list[int]], *, chunked: bool = False
    ) -> torch.Tensor:
        totals = [n + len(query) for n, query in zip(past, queries, strict=True)]
        slots, positions, indices, indptr, qo = [], [], [], [0], [0]
        for query, n, total, request_pages in zip(queries, past, totals, pages, strict=True):
            slots.extend(
                request_pages[i // self.block_size] * self.block_size + i % self.block_size for i in range(n, total)
            )
            positions.extend(range(n, total))
            indices.extend(request_pages[: (total + self.block_size - 1) // self.block_size])
            indptr.append(len(indices))
            qo.append(qo[-1] + len(query))

        def _tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=self.model.device)

        metadata = cast(
            AttentionMetadata,
            SimpleNamespace(
                is_prefill=not any(past),
                is_chunked_prefill=chunked,
                slot_mapping=_tensor(slots),
                q_cu_seq_lens=_tensor(qo),
                kv_cu_seq_lens=_tensor(qo),
                qo_indptr=_tensor(qo),
                paged_kv_indptr=_tensor(indptr),
                paged_kv_indices=_tensor(indices),
                paged_kv_last_page_len=_tensor([(n - 1) % self.block_size + 1 for n in totals]),
                prepared_attention_state=None,
            ),
        )
        self.backend.prepare(metadata)
        tokens = torch.tensor([token for query in queries for token in query], device=self.model.device)
        position_ids = torch.tensor(positions, device=self.model.device)
        selected = torch.tensor([end - 1 for end in qo[1:]], device=self.model.device)
        context = ForwardContext(self.backend, self.model.device, metadata, self.caches)
        with torch.inference_mode(), forward_context(context):
            return self.model.compute_logits(self.model.model(tokens, position_ids), selected)


def _cached_cuda_models(dtype: torch.dtype) -> tuple[Olmo2ForCausalLM, HfOlmo2ForCausalLM]:
    torch.manual_seed(20261020)
    config = _config(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=512,
        max_position_embeddings=256,
        dtype=dtype,
        device="cuda:0",
    )
    hf_config = HfOlmo2Config(**{key: value for key, value in config.items() if key not in ("dtype", "device")})
    hf_config._attn_implementation = "eager"
    reference = HfOlmo2ForCausalLM(hf_config).to(device="cuda:0", dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "norm.weight" in name:
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel(), device=parameter.device))
    model = Olmo2ForCausalLM(config).eval()
    model.load_weights([_StateDict(dict(reference.state_dict()))], 0, 1)
    return model, reference


def _assert_cached_logits(
    actual: torch.Tensor,
    reference: HfOlmo2ForCausalLM,
    histories: list[list[int]],
    full_runner: _PagedModelRunner,
) -> None:
    with torch.inference_mode():
        for index, history in enumerate(histories):
            tokens = torch.tensor([history], device=actual.device)
            expected = reference(
                tokens,
                attention_mask=torch.ones_like(tokens),
                position_ids=torch.arange(len(history), device=actual.device)[None],
                use_cache=False,
            ).logits[:, -1]
            # Both compute dtypes exercise the real backend; use a common
            # comparison dtype because BF16 projections return FP32 logits.
            torch.testing.assert_close(actual[index : index + 1].float(), expected.float(), rtol=0.02, atol=0.02)
            if reference.dtype == torch.float16:
                # Retain the established FP16-vs-HF numerical requirement.
                error = (actual[index : index + 1].float() - expected.float()).square().mean().sqrt()
                assert error <= 0.02 * expected.float().square().mean().sqrt()
            # Cache invariance must compare the same model/precision policy.
            # Use separate cache storage so full prefill cannot repair a bad
            # incremental cache. Full-checkpoint BF16/HF budgets live in the
            # independent frozen numerical validation fixture.
            full = full_runner.run([history], [0], [[7, 2]])
            observed = actual[index : index + 1].float()
            torch.testing.assert_close(observed, full.float(), rtol=0.02, atol=0.02)
            error = (observed - full.float()).square().mean().sqrt()
            assert error <= 0.02 * full.float().square().mean().sqrt()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="paged OLMo2 validation requires a CUDA GPU")
@pytest.mark.parametrize("length", [127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_paged_cache_matches_full_history_across_block_boundary(length: int, dtype: torch.dtype) -> None:
    pytest.importorskip("flashinfer")
    model, reference = _cached_cuda_models(dtype)
    runner = _PagedModelRunner(model)
    full_runner = _PagedModelRunner(model)
    history = [(i * 7 + 3) % model.cfg.vocab_size for i in range(length + 3)]
    pages = [[7, 2]]
    initial = 60
    runner.run([history[:initial]], [0], pages)
    actual = runner.run([history[initial:length]], [initial], pages, chunked=True)
    _assert_cached_logits(actual, reference, [history[:length]], full_runner)
    for step in range(1, 4):
        actual = runner.run([[history[length + step - 1]]], [length + step - 1], pages)
        _assert_cached_logits(actual, reference, [history[: length + step]], full_runner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="batched OLMo2 validation requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixed_lengths_keep_request_cache_pages_isolated(dtype: torch.dtype) -> None:
    pytest.importorskip("flashinfer")
    model, reference = _cached_cuda_models(dtype)
    runner = _PagedModelRunner(model)
    full_runner = _PagedModelRunner(model)
    lengths = [3, 127, 129]
    histories = [
        [(i * 7 + request * 11) % model.cfg.vocab_size for i in range(n + 3)] for request, n in enumerate(lengths)
    ]
    pages = [[7], [4, 1], [6, 2]]
    actual = runner.run([history[:n] for history, n in zip(histories, lengths)], [0, 0, 0], pages)
    _assert_cached_logits(actual, reference, [history[:n] for history, n in zip(histories, lengths)], full_runner)
    for step in range(1, 4):
        actual = runner.run(
            [[history[n + step - 1]] for history, n in zip(histories, lengths)],
            [n + step - 1 for n in lengths],
            pages,
        )
        _assert_cached_logits(
            actual, reference, [history[: n + step] for history, n in zip(histories, lengths)], full_runner
        )


def test_registry_resolves_both_names_and_rejects_unverified_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    assert registry.get_model_class("olmo2") is Olmo2ForCausalLM
    assert registry.get_model_class("Olmo2ForCausalLM") is Olmo2ForCausalLM
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    with pytest.raises(NotImplementedError, match="olmo2.*npu"):
        registry.get_model_class("olmo2")

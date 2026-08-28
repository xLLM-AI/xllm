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

"""DeepSeek-V4 Python model: config parsing, registry, structure.

Pure-Python: does not load compiled operators or weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.models import deepseek_v4, deepseek_v32
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4HyperConnection,
    DeepseekV4Model,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
)
from xllm.python.models.deepseek_v32 import DeepseekV3MLP, W8A8DynamicLinear, _swiglu_with_clamp
from xllm.python.registry import get_model_class

_DSV4_CONFIG = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 4,
    "num_attention_heads": 64,
    "head_dim": 512,
    "vocab_size": 129280,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "original_max_position_embeddings": 65536,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "compress_ratios": [0, 4, 128, 4],
    "window_size": 128,
    "sliding_window": 128,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "n_activated_experts": 6,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "scoring_func": "sqrtsoftplus",
    "scale_fmt": "ue8m0",
    "n_routed_experts": 256,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 0,
    "tie_word_embeddings": False,
}


def test_config_from_dict_reads_dsv4_fields() -> None:
    cfg = DeepseekV4Config.from_dict(_DSV4_CONFIG)
    assert cfg.model_type == "deepseek_v4"
    assert cfg.n_layers == 4
    assert cfg.compress_ratios == [1, 4, 128, 4]
    assert cfg.window_size == 128
    assert cfg.o_lora_rank == 1024
    assert cfg.o_groups == 8
    assert cfg.hc_mult == 4
    assert cfg.index_topk == 512
    assert cfg.rope_scaling_factor == 16.0


def test_config_prefers_dsv4_model_args_over_zero_legacy_rope_fields() -> None:
    cfg = DeepseekV4Config.from_dict(
        {
            **_DSV4_CONFIG,
            "rope_scaling": None,
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "rope_scaling_attn_factor": 1.0,
            "rope_scaling_factor": 0.0,
            "rope_scaling_beta_fast": 0.0,
            "rope_scaling_beta_slow": 0.0,
        }
    )

    assert cfg.rope_scaling_factor == 16.0
    assert cfg.rope_beta_fast == 32
    assert cfg.rope_beta_slow == 1
    assert cfg.rope_mscale == 1.0


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"num_hash_layers": 2}, 2),
        ({"n_hash_layers": 0}, 0),
        ({"n_hash_layers": 1, "num_hash_layers": 2}, 1),
    ],
)
def test_config_accepts_native_and_legacy_hash_layer_fields(fields: dict, expected: int) -> None:
    cfg = DeepseekV4Config.from_dict({**_DSV4_CONFIG, **fields})
    assert cfg.n_hash_layers == expected


def test_rotary_cache_matches_cpp_cpu_float32_construction() -> None:
    rotary = DeepseekV4RotaryEmbedding(
        rotary_dim=64,
        max_position_embeddings=87,
        scaling_factor=16.0,
        theta=10000.0,
        beta_fast=32,
        beta_slow=1,
        old_context_len=1048576,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    # Position 86, frequency 1 is the first observed CPU/NPU rounding split.
    # Lock the Python cache to the value produced by the C++ CPU path.
    assert rotary.cos_sin_cache[86, 1].item() == -0.087890625


def test_rotary_cache_shares_identical_descriptors() -> None:
    args = dict(
        rotary_dim=64,
        max_position_embeddings=87,
        scaling_factor=16.0,
        theta=160000.0,
        beta_fast=32,
        beta_slow=1,
        old_context_len=1048576,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    c4 = DeepseekV4RotaryEmbedding(**args)
    c128 = DeepseekV4RotaryEmbedding(**args)
    default = DeepseekV4RotaryEmbedding(**{**args, "theta": 10000.0})

    assert c4.cos_sin_cache.data_ptr() == c128.cos_sin_cache.data_ptr()
    assert c4.cos_sin_cache.data_ptr() != default.cos_sin_cache.data_ptr()


def test_registry_resolves_deepseek_v4() -> None:
    cls = get_model_class("deepseek_v4")
    assert cls.__name__ == "DeepseekV4ForCausalLM"


def test_hyper_connection_shapes() -> None:
    cfg = DeepseekV4Config.from_dict(_DSV4_CONFIG)
    hc = DeepseekV4HyperConnection(cfg, torch.float32, torch.device("cpu"))
    assert hc.hc_mult_local == 4  # hc_mult is NOT TP-sharded
    # hc_*_fn: [mix_hc, hc_dim] = [(2+mult)*mult, mult*hidden] = [24, 16384].
    assert hc.hc_attn_fn.shape == ((2 + 4) * 4, 4 * 4096)
    assert hc.hc_attn_base.shape == ((2 + 4) * 4,)
    assert hc.hc_attn_scale.shape == (3,)
    # hc_pre calls the compiled NPU kernel, which is not available in the
    # pure-Python unit-test context; only shape construction is verified here.


def test_decoder_layer_builds() -> None:
    """A C4 decoder layer builds attention + HC + MoE without error."""
    cfg = DeepseekV4Config.from_dict(_DSV4_CONFIG)
    layer = DeepseekV4DecoderLayer(cfg, layer_id=1, dtype=torch.float32, device=torch.device("cpu"))
    assert layer.self_attn.layer_id == 1
    assert layer.self_attn.indexer is not None
    assert layer.hc.hc_mult_local == 4


def test_attention_builds_compression_modules_only_for_matching_ratios() -> None:
    cfg = DeepseekV4Config.from_dict(_DSV4_CONFIG)
    c1 = DeepseekV4DecoderLayer(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu")).self_attn
    c4 = DeepseekV4DecoderLayer(cfg, layer_id=1, dtype=torch.float32, device=torch.device("cpu")).self_attn
    c128 = DeepseekV4DecoderLayer(cfg, layer_id=2, dtype=torch.float32, device=torch.device("cpu")).self_attn

    assert c1.indexer is None
    assert not hasattr(c1, "cmp_wkv")
    assert c4.indexer is not None
    assert hasattr(c4, "cmp_wkv")
    assert c128.indexer is None
    assert hasattr(c128, "cmp_wkv")
    assert c1.attn_sink_loaded is False
    assert c4.attn_sink_loaded is False
    assert c128.attn_sink_loaded is False


def test_compressor_weights_are_cached_as_non_persistent_bf16(monkeypatch) -> None:
    cfg = DeepseekV4Config.from_dict(_DSV4_CONFIG)
    attention = DeepseekV4DecoderLayer(cfg, layer_id=1, dtype=torch.float32, device=torch.device("cpu")).self_attn
    for projection in (attention.q_a_proj, attention.kv_proj, attention.q_b_proj, attention.indexer.wq_b):
        projection.weight_offset.zero_()
    monkeypatch.setattr(deepseek_v32.kernels, "prepare_quant_weight", lambda weight: weight, raising=False)

    attention.process_weights_after_loading()

    assert attention._cmp_wkv_bf16.dtype == torch.bfloat16
    assert attention._cmp_wgate_bf16.dtype == torch.bfloat16
    assert attention._cmp_norm_bf16.dtype == torch.bfloat16
    assert attention.indexer._compressor_wkv_bf16.dtype == torch.bfloat16
    assert attention.indexer._compressor_wgate_bf16.dtype == torch.bfloat16
    assert attention.indexer._compressor_norm_bf16.dtype == torch.bfloat16
    assert not any(name.endswith("_bf16") for name in attention.state_dict())


def test_moe_gate_state_matches_cpp_parameter_ownership() -> None:
    cfg_dict = dict(_DSV4_CONFIG)
    cfg_dict["n_hash_layers"] = 3
    cfg = DeepseekV4Config.from_dict(cfg_dict)

    hash_moe = DeepseekV4MoE(cfg, layer_id=2, dtype=torch.float32, device=torch.device("cpu"))
    non_hash_moe = DeepseekV4MoE(cfg, layer_id=3, dtype=torch.float32, device=torch.device("cpu"))

    hash_params = dict(hash_moe.named_parameters())
    non_hash_params = dict(non_hash_moe.named_parameters())
    assert "tid2eid" in hash_params
    assert not hash_params["tid2eid"].requires_grad
    assert "e_score_correction_bias" in non_hash_params
    assert not non_hash_params["e_score_correction_bias"].requires_grad


def test_clamped_swiglu_matches_cpp_activation_formula() -> None:
    x = torch.tensor([[-20.0, 5.0, 20.0, 12.0, -15.0, 3.0]], dtype=torch.bfloat16)
    gate, up = x.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float().clamp_max(10.0)) * up.float().clamp(min=-10.0, max=10.0)).to(
        x.dtype
    )

    torch.testing.assert_close(_swiglu_with_clamp(x, 10.0), expected)


def test_dynamic_linear_preserves_v3_and_v4_weight_layout_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_quant_matmul(x, weight, transpose2, scale, offset, pertoken, bias, output_dtype):
        calls.append(transpose2)
        return torch.empty((x.size(0), scale.numel()), dtype=output_dtype)

    monkeypatch.setattr(deepseek_v32.kernels, "quant_matmul", fake_quant_matmul, raising=False)
    monkeypatch.setattr(
        deepseek_v32.kernels,
        "prepare_quant_weight",
        lambda weight: weight.transpose(0, 1).contiguous(),
        raising=False,
    )
    x = torch.ones((2, 3), dtype=torch.int8)
    pertoken = torch.ones((2,), dtype=torch.float32)

    v3 = W8A8DynamicLinear(3, 4, torch.device("cpu"))
    v3.weight_offset.zero_()
    v3.process_weights_after_loading()
    assert v3.weight.shape == (3, 4)
    v3.forward_quantized(x, pertoken)

    v4 = W8A8DynamicLinear(3, 4, torch.device("cpu"), transpose_weight_after_loading=False)
    v4.weight_offset.zero_()
    v4.process_weights_after_loading()
    assert v4.weight.shape == (4, 3)
    v4.forward_quantized(x, pertoken)

    assert calls == [False, True]


def test_dense_mlp_uses_native_aware_tp_reduce(monkeypatch) -> None:
    cfg = DeepseekV4Config.from_dict({**_DSV4_CONFIG, "tp_size": 2})
    mlp = DeepseekV3MLP(cfg, cfg.moe_intermediate_size, torch.float32, torch.device("cpu"))
    mlp.gate_up_proj.forward = MagicMock(return_value=torch.ones(1, 2 * mlp.gate_up_proj.out_features))
    mlp.down_proj.forward = MagicMock(return_value=torch.ones(1, cfg.hidden_size))
    monkeypatch.setattr(deepseek_v32.kernels, "silu_and_mul", lambda tensor: tensor[..., : tensor.shape[-1] // 2])
    tp_reduce = MagicMock()
    monkeypatch.setattr(deepseek_v32.distributed, "tp_all_reduce", tp_reduce, raising=False)

    mlp(torch.ones(1, cfg.hidden_size))

    tp_reduce.assert_called_once()


def test_model_rejects_cp_until_cp_context_is_available() -> None:
    cfg = DeepseekV4Config.from_dict({**_DSV4_CONFIG, "cp_size": 2})
    with pytest.raises(NotImplementedError, match="CP context PR"):
        DeepseekV4Model(cfg, torch.float32, torch.device("cpu"))


def test_causal_lm_rejects_data_parallelism_before_building_model() -> None:
    with pytest.raises(NotImplementedError, match="dp_size > 1"):
        DeepseekV4ForCausalLM({**_DSV4_CONFIG, "dp_size": 2})


def test_dense_mlp_loader_maps_dsv4_weight_names() -> None:
    tensors = {
        "layers.0.ffn.w1.weight": torch.arange(12, dtype=torch.int8).reshape(4, 3),
        "layers.0.ffn.w3.weight": torch.arange(12, 24, dtype=torch.int8).reshape(4, 3),
        "layers.0.ffn.w2.weight": torch.arange(12, dtype=torch.int8).reshape(3, 4),
        "layers.0.ffn.w1.weight_scale": torch.arange(4, dtype=torch.float32).reshape(4, 1),
        "layers.0.ffn.w3.weight_scale": torch.arange(4, 8, dtype=torch.float32).reshape(4, 1),
        "layers.0.ffn.w2.weight_scale": torch.arange(3, dtype=torch.float32).reshape(3, 1),
        "layers.0.ffn.w1.weight_offset": torch.zeros(4, 1),
        "layers.0.ffn.w3.weight_offset": torch.zeros(4, 1),
        "layers.0.ffn.w2.weight_offset": torch.zeros(3, 1),
    }

    class FakeLoader:
        def __init__(self) -> None:
            self.loaded: dict[str, torch.Tensor] = {}

        def load_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def shard(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return tensor.chunk(2, dim=dim)[1].contiguous()

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    class FakeMlp:
        processed = False

        def process_weights_after_loading(self) -> None:
            self.processed = True

    loader = FakeLoader()
    mlp = FakeMlp()
    DeepseekV4ForCausalLM._load_dsv4_dense_mlp(loader, "layers.0.", "model.layers.0.", mlp)

    torch.testing.assert_close(
        loader.loaded["model.layers.0.mlp.gate_up_proj.weight"],
        torch.cat([tensors["layers.0.ffn.w1.weight"][2:], tensors["layers.0.ffn.w3.weight"][2:]]),
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.mlp.down_proj.weight"],
        tensors["layers.0.ffn.w2.weight"][:, 2:],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.mlp.down_proj.weight_scale"],
        tensors["layers.0.ffn.w2.weight_scale"],
    )
    assert mlp.processed


def test_moe_uses_dedicated_group_sizes() -> None:
    cfg_dict = dict(_DSV4_CONFIG)
    cfg_dict.update(
        tp_size=4,
        tp_rank=1,
        ep_size=8,
        ep_rank=3,
        moe_tp_size=1,
        moe_tp_rank=0,
        cp_size=1,
        cp_rank=0,
    )
    cfg = DeepseekV4Config.from_dict(cfg_dict)
    moe = DeepseekV4MoE(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu"))

    assert moe.moe_tp_size == 1
    assert moe.moe_tp_rank == 0
    assert moe.start_expert_id == 3 * (cfg.n_routed_experts // cfg.ep_size)


def test_moe_tp_ep_reduction_order_matches_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeDistributed:
        @staticmethod
        def moe_tp_all_reduce(tensor: torch.Tensor) -> None:
            calls.append("moe_tp")
            tensor.add_(10)

        @staticmethod
        def moe_ep_all_reduce(tensor: torch.Tensor) -> None:
            calls.append("moe_ep")
            tensor.add_(100)

    monkeypatch.setattr(deepseek_v4, "distributed", FakeDistributed)
    owner = SimpleNamespace(
        cfg=SimpleNamespace(ep_size=2, tp_size=1),
        moe_tp_size=2,
    )
    routed = torch.zeros(2)
    shared = torch.ones(2)

    output = DeepseekV4MoE._reduce_moe_outputs(owner, routed, shared)

    assert calls == ["moe_tp", "moe_ep", "moe_tp"]
    assert torch.equal(output, torch.full((2,), 121.0))


def test_moe_ep_only_reduces_routed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeDistributed:
        @staticmethod
        def moe_ep_all_reduce(tensor: torch.Tensor) -> None:
            calls.append("moe_ep")
            tensor.add_(100)

    monkeypatch.setattr(deepseek_v4, "distributed", FakeDistributed)
    owner = SimpleNamespace(
        cfg=SimpleNamespace(ep_size=2, tp_size=1),
        moe_tp_size=1,
    )

    output = DeepseekV4MoE._reduce_moe_outputs(owner, torch.zeros(1), torch.ones(1))

    assert calls == ["moe_ep"]
    assert torch.equal(output, torch.full((1,), 101.0))


def test_moe_tp_only_combines_before_one_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeDistributed:
        @staticmethod
        def moe_tp_all_reduce(tensor: torch.Tensor) -> None:
            calls.append("moe_tp")
            tensor.mul_(2)

    monkeypatch.setattr(deepseek_v4, "distributed", FakeDistributed)
    owner = SimpleNamespace(
        cfg=SimpleNamespace(ep_size=1, tp_size=2),
        moe_tp_size=2,
    )

    output = DeepseekV4MoE._reduce_moe_outputs(owner, torch.full((1,), 2.0), torch.full((1,), 3.0))

    assert calls == ["moe_tp"]
    assert torch.equal(output, torch.full((1,), 10.0))

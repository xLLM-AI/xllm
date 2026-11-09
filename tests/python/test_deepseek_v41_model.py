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

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

import pytest
import torch

from tests.python._engram_reference import _tiny_engram_hash_state
from xllm.python import distributed
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _indexer_score,
    _indexer_topk,
    _select_candidate_blocks,
    _window_visible_idxs,
    apply_interleaved_rope,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    DsaMetadata,
    build_cache_specs_v41,
)
from xllm.python.layers import moe_parallel
from xllm.python.layers.attention import Attention
from xllm.python.model_executor import executor as executor_module
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.models import deepseek_v41 as dq
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_FULL,
    CSA2_MODE_REINDEX,
    CSA2_MODE_REUSE,
    CSA2_MODE_SWA,
    V41_QUANT_ASCEND,
    V41_QUANT_FP8,
    V41_QUANT_NONE,
    CSA2LayerPlan,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DeepseekV41MoE,
    fp4_act_qdq_e8m0,
)
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramLayout,
    build_compressed_token_map_from_texts,
    load_quarot_gate_unrotate,
    normalize_token_text,
)
from xllm.python.platform import current_platform


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


npu_only = pytest.mark.skipif(
    not _npu_available(),
    reason="DeepSeek-V4.1 requires an NPU attention backend",
)


def _active_device() -> torch.device:
    return torch.device("npu:0") if _npu_available() else torch.device("cpu")


_DSV41_CONFIG = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "head_dim": 512,
    "vocab_size": 129280,
    "rms_norm_eps": 1e-20,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "type": "yarn"},
    "q_lora_rank": 1280,
    "qk_rope_head_dim": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "compress_rope_theta": 160000.0,
    "window_size": 128,
    "sliding_window": 128,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 512,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_compressed_vocab_size": 99092,
    "engram_pad_token_id": 2,
    "n_activated_experts": 6,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "n_shared_experts": 1,
    "scoring_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "swiglu_limit": 10.0,
    "image_token_id": 129264,
    "first_k_dense_replace": 0,
    "tie_word_embeddings": False,
}


_ENGRAM_TINY_OVERRIDE = {
    "engram_layer_ids": [1],
    "engram_num_embeddings": [290],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 64,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_compressed_vocab_size": 16,
    "engram_pad_token_id": 0,
}

_DSV41_TINY_CONFIG = {
    **_DSV41_CONFIG,
    **_ENGRAM_TINY_OVERRIDE,
    "num_hidden_layers": 6,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "head_dim": 32,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "o_groups": 2,
    "qk_rope_head_dim": 8,
    "vocab_size": 64,
    "n_routed_experts": 4,
    "n_activated_experts": 2,
    "moe_intermediate_size": 32,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 4,
    "compress_ratios": [0, 0, 2, 2, 1, 1],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
}

_DSV41_FORWARD_CONFIG = {
    **_DSV41_TINY_CONFIG,
    "num_attention_heads": 2,
    "head_dim": 64,
    "qk_rope_head_dim": 16,
    "index_head_dim": 32,
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
    "index_source_layer_ids": [2, 4, 5],
    "candidate_source_layer_id": 4,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 4,
}


@pytest.fixture(scope="module", autouse=True)
def _engram_sidecar(tmp_path_factory: pytest.TempPathFactory) -> None:
    model_path = tmp_path_factory.mktemp("engram")
    state = _tiny_engram_hash_state()
    torch.save(
        {"token_map": state.token_map, "multipliers": state.multipliers},
        model_path / "engram_hash_state.bin",
    )
    for config in (_ENGRAM_TINY_OVERRIDE, _DSV41_TINY_CONFIG, _DSV41_FORWARD_CONFIG):
        config["model_path"] = str(model_path)


def test_csa2_plan_matches_reference_layer_assignment() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    plan = CSA2LayerPlan.build(cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids)
    full_layers = {2, 8, 14, 20}
    reindex_layers = {24, 28, 32, 36}
    for layer_id in range(cfg.n_layers):
        mode = plan.mode(layer_id)
        if layer_id in (0, 1):
            assert mode.mode == CSA2_MODE_SWA
            assert mode.compress_ratio == 0
        elif layer_id in full_layers:
            assert mode.mode == CSA2_MODE_FULL
        elif layer_id in reindex_layers:
            assert mode.mode == CSA2_MODE_REINDEX
        else:
            assert mode.mode == CSA2_MODE_REUSE
        if layer_id >= 2:
            assert mode.compress_ratio == (2 if layer_id < 20 else 1)


@pytest.mark.parametrize(
    ("n_layers", "ratios", "kv_sources", "index_sources"),
    [
        (40, [1] * 40, [45], [45, 2]),
        (40, [1] * 40, [2], [8]),
        (40, [1] * 40, [8], [2, 8]),
        (40, [0] * 40, [2], [2, 8]),
    ],
)
def test_csa2_plan_rejects_invalid_source_layouts(
    n_layers: int,
    ratios: list[int],
    kv_sources: list[int],
    index_sources: list[int],
) -> None:
    with pytest.raises(ValueError):
        CSA2LayerPlan.build(n_layers, ratios, kv_sources, index_sources)


def _tiny_model() -> DeepseekV41Model:
    return DeepseekV41Model(
        DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG),
        torch.float32,
        torch.device("cpu"),
        config=_DSV41_TINY_CONFIG,
    )


def test_moe_bias_vl_shifts_only_visual_tokens() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    moe = model.layers[2].mlp

    moe.gate_bias_vl_loaded = True
    with torch.no_grad():
        moe.gate_bias_vl.fill_(1.5)
    gate_input = torch.zeros(2, cfg.hidden_size)
    input_ids = torch.tensor([1, cfg.image_token_id])
    logits, scores = moe._selection_logits(gate_input, input_ids)
    assert torch.allclose(logits[1], logits[0] + 1.5)
    assert torch.allclose(scores[1], scores[0])
    text_logits, _ = moe._selection_logits(gate_input, None)
    assert torch.allclose(text_logits, scores.expand(2, -1))


@pytest.mark.parametrize("tokens", [0, 1, 6])
@pytest.mark.parametrize("topk,normalize,limit", [(1, True, 2.0), (2, True, 2.0), (2, False, 0.0)])
@pytest.mark.parametrize("ep_size,ep_rank,selected", [(1, 0, 193), (2, 1, 193), (2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_delegates_to_one_grouped_kernel_and_matches_all_expert_reference(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    topk: int,
    normalize: bool,
    limit: float,
    ep_size: int,
    ep_rank: int,
    selected: int,
    dtype: torch.dtype,
) -> None:
    from xllm.python import distributed, kernels

    torch.manual_seed(41)
    cfg = DeepseekV41Config.from_dict(
        {
            **_DSV41_TINY_CONFIG,
            "hidden_size": 8,
            "moe_intermediate_size": 6,
            "n_routed_experts": 384,
            "n_activated_experts": topk,
            "norm_topk_prob": normalize,
            "swiglu_limit": limit,
        }
    )
    cfg.ep_size, cfg.ep_rank = ep_size, ep_rank
    moe = DeepseekV41MoE(cfg, dtype, torch.device("cpu"))
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(std=0.7)
        moe.e_score_correction_bias.zero_()
        moe.e_score_correction_bias[selected : selected + 2] = 100.0
        moe.gate_bias_vl.zero_()
        moe.gate_bias_vl[selected + 2 : selected + 4] = 100.0
    moe.gate_bias_vl_loaded = True
    hidden = torch.randn(1, tokens, cfg.hidden_size, dtype=dtype)
    input_ids = torch.tensor([cfg.image_token_id if i % 2 else 1 for i in range(tokens)])
    x = hidden.reshape(-1, cfg.hidden_size)
    logits, scores = moe._selection_logits(x, input_ids)
    indices = logits.topk(topk, dim=-1).indices
    weights = scores.gather(1, indices)
    if normalize and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * moe.routed_scaling
    expected = torch.zeros_like(x, dtype=torch.float32)
    for local_slot in range(moe.num_experts_per_rank):
        rows, slots = torch.where(indices == moe.start_expert_id + local_slot)
        if rows.numel() == 0:
            continue
        gate = torch.nn.functional.linear(x[rows], moe.experts_w13[local_slot, : cfg.moe_intermediate_size]).float()
        up = torch.nn.functional.linear(x[rows], moe.experts_w13[local_slot, cfg.moe_intermediate_size :]).float()
        if 0.0 < limit < 1_000_000.0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        intermediate = weights[rows, slots, None] * (torch.nn.functional.silu(gate) * up)
        output = torch.nn.functional.linear(intermediate.to(dtype), moe.experts_w2[local_slot])
        expected.index_add_(0, rows, output.float())

    reduced: list[torch.Tensor] = []

    def fake_all_reduce(partial: torch.Tensor) -> None:
        reduced.append(partial.clone())
        partial.mul_(2)

    monkeypatch.setattr(distributed, "moe_ep_all_reduce", fake_all_reduce)
    dispatched_kernel = kernels.grouped_moe_bf16
    kernel_calls: list[tuple[tuple, dict]] = []

    def recording_grouped_moe(*args: object, **kwargs: object) -> torch.Tensor:
        kernel_calls.append((args, kwargs))
        return dispatched_kernel(*args, **kwargs)

    monkeypatch.setattr(kernels, "grouped_moe_bf16", recording_grouped_moe)
    with monkeypatch.context() as sync_guard:
        sync_guard.setattr(
            torch.Tensor,
            "cpu",
            MagicMock(side_effect=AssertionError("routed MoE must not copy routing metadata to the host")),
        )
        actual = moe(hidden, input_ids)
    assert len(kernel_calls) == 1
    kernel_args, kernel_kwargs = kernel_calls[0]
    assert kernel_kwargs == {"swiglu_limit": limit, "router_weight_before_down": True}
    (
        kernel_hidden,
        kernel_weights,
        kernel_ids,
        kernel_w13,
        kernel_w2,
        kernel_total,
        kernel_start,
        kernel_local,
    ) = kernel_args
    torch.testing.assert_close(kernel_hidden, x)
    torch.testing.assert_close(kernel_weights, weights)
    assert torch.equal(kernel_ids, indices)
    assert kernel_ids.dtype is torch.int32
    assert kernel_w13.data_ptr() == moe.experts_w13.data_ptr()
    assert kernel_w2.data_ptr() == moe.experts_w2.data_ptr()
    assert (kernel_total, kernel_start, kernel_local) == (
        cfg.n_routed_experts,
        moe.start_expert_id,
        moe.num_experts_per_rank,
    )
    if ep_size > 1:
        assert len(reduced) == 1
        torch.testing.assert_close(reduced[0], expected)
        expected *= 2
    else:
        assert not reduced
    expected += dq._swiglu_clamped(x, moe.shared_w1(x), moe.shared_w3(x), moe.shared_w2, moe.swiglu_limit).float()
    torch.testing.assert_close(actual, expected.to(dtype).view_as(hidden))


def test_attention_backend_rejects_v41_off_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor import executor as executor_module

    monkeypatch.setattr(executor_module.current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(executor_module.current_platform, "is_cuda", lambda: True)
    with pytest.raises(NotImplementedError, match="runs on NPU only"):
        executor_module._create_attention_backend(
            first_attention=object(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            config={"model_type": "deepseek_v41"},
        )


class _FakeStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _build_tiny_checkpoint(cfg: DeepseekV41Config) -> dict[str, torch.Tensor]:
    hidden = cfg.hidden_size
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    n_heads = cfg.n_heads
    inter = cfg.moe_intermediate_size
    n_experts = cfg.n_routed_experts
    o_lora = cfg.o_lora_rank

    def w(out: int, inp: int) -> torch.Tensor:
        return torch.randn(out, inp)

    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg.vocab_size, hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg.vocab_size, hidden),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        tensors[p + "attn_norm.weight"] = torch.ones(hidden)
        tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
        for part in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", ((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * hidden)),
                ("scale", (3,)),
                ("base", ((2 + cfg.hc_mult) * cfg.hc_mult,)),
            ):
                tensors[p + f"hc_{part}_{suffix}"] = torch.randn(*shape)
        tensors[p + "attn.wq_a.weight"] = w(q_lora, hidden)
        tensors[p + "attn.wq_b.weight"] = w(n_heads * head_dim, q_lora)
        tensors[p + "attn.wkv.weight"] = w(head_dim, hidden)
        tensors[p + "attn.wo_a.weight"] = w(cfg.o_groups * o_lora, (n_heads * head_dim) // cfg.o_groups)
        tensors[p + "attn.wo_b.weight"] = w(hidden, cfg.o_groups * o_lora)
        tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
        tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
        tensors[p + "attn.attn_sink"] = torch.zeros(n_heads)
        if i in (2, 4):
            tensors[p + "attn.compressor.wkv.weight"] = w(head_dim, hidden)
            if cfg.compress_ratios[i] > 1:
                tensors[p + "attn.compressor.wgate.weight"] = w(head_dim, hidden)
            tensors[p + "attn.compressor.norm.weight"] = torch.ones(head_dim)
            tensors[p + "attn.indexer.wq_b.weight"] = w(cfg.index_n_heads * cfg.index_head_dim, q_lora)
            tensors[p + "attn.indexer.wk.weight"] = w(cfg.index_head_dim, head_dim)
            tensors[p + "attn.indexer.k_norm.weight"] = torch.ones(cfg.index_head_dim)
            tensors[p + "attn.indexer.weights_proj.weight"] = w(cfg.index_n_heads, hidden)
        tensors[p + "ffn.gate.weight"] = torch.randn(n_experts, hidden)
        tensors[p + "ffn.gate.bias"] = torch.randn(n_experts)
        tensors[p + "ffn.gate.bias_vl"] = torch.randn(n_experts)
        for expert in range(n_experts):
            for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
                tensors[p + f"ffn.experts.{expert}.{name}.weight"] = w(out, inp)
        for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
            tensors[p + f"ffn.shared_experts.{name}.weight"] = w(out, inp)
        if i in cfg.engram_layer_ids:
            n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            head_dim = cfg.engram_head_dim
            rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(i)]
            tensors[p + "engram.embed.weight"] = torch.randint(0, 256, (rows, head_dim), dtype=torch.uint8)
            tensors[p + "engram.embed.scale"] = torch.randint(126, 130, (rows, head_dim // 32), dtype=torch.uint8)
            tensors[p + "engram.wkv.weight"] = w(hidden * (cfg.hc_mult + 1), n_hash_cols * head_dim)
            tensors[p + "engram.q_weight"] = w(cfg.hc_mult, hidden)
            tensors[p + "engram.k_weight"] = w(cfg.hc_mult, hidden)
    return tensors


def test_load_weights_maps_v41_checkpoint_layout() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    tensors = _build_tiny_checkpoint(cfg)
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)

    assert torch.equal(lm.model.layers[0].self_attn.q_a_proj.weight, tensors["layers.0.attn.wq_a.weight"])
    assert lm.model.layers[3].self_attn.attn_sink_loaded
    assert torch.equal(lm.model.layers[2].self_attn.cmp_wkv.weight, tensors["layers.2.attn.compressor.wkv.weight"])
    assert torch.equal(lm.model.layers[2].self_attn.indexer.wk.weight, tensors["layers.2.attn.indexer.wk.weight"])
    assert torch.equal(lm.model.layers[5].mlp.e_score_correction_bias, tensors["layers.5.ffn.gate.bias"])
    assert lm.model.layers[5].mlp.gate_bias_vl_loaded
    assert torch.equal(
        lm.model.layers[1].mlp.experts_w13[3, lm.model.layers[1].mlp.cfg.moe_intermediate_size :],
        tensors["layers.1.ffn.experts.3.w3.weight"],
    )
    assert torch.equal(lm.model.layers[0].mlp.shared_w1.weight, tensors["layers.0.ffn.shared_experts.w1.weight"])
    eng = lm.model.layers[1].engram
    assert eng is not None
    assert torch.equal(eng.embed_tokens.weight, tensors["layers.1.engram.embed.weight"])
    assert torch.equal(eng.embed_tokens.weight_scale, tensors["layers.1.engram.embed.scale"])
    assert torch.equal(eng.wkv.weight, tensors["layers.1.engram.wkv.weight"])
    assert torch.equal(eng.q_weight, tensors["layers.1.engram.q_weight"])
    assert torch.equal(eng.k_weight, tensors["layers.1.engram.k_weight"])
    for layer_id in (0, 2, 3, 4, 5):
        assert lm.model.layers[layer_id].engram is None
    assert torch.equal(lm.model.norm.weight, tensors["norm.weight"])
    assert torch.equal(lm.lm_head.weight, tensors["head.weight"])


def test_load_weights_requires_output_head() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    tensors = _build_tiny_checkpoint(cfg)
    del tensors["head.weight"]
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    with pytest.raises(KeyError, match="checkpoint output-head weight"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


_DSV41_ASCEND_CONFIG = {
    **_DSV41_CONFIG,
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "head_dim": 16,
    "q_lora_rank": 8,
    "o_lora_rank": 4,
    "o_groups": 2,
    "qk_rope_head_dim": 4,
    "vocab_size": 32,
    "moe_intermediate_size": 16,
    "first_k_dense_replace": 1,
    "compress_ratios": [0],
    "kv_source_layer_ids": [],
    "index_source_layer_ids": [],
    "candidate_source_layer_id": -1,
    "candidate_topk_blocks": 0,
    "candidate_block_size": 0,
    "engram_layer_ids": [],
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
}

_DSV41_ASCEND_FULL_CONFIG = {
    **_DSV41_ASCEND_CONFIG,
    "compress_ratios": [2],
    "kv_source_layer_ids": [0],
    "index_source_layer_ids": [0],
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 4,
}


def _make_ascend_lm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict = _DSV41_ASCEND_CONFIG,
) -> DeepseekV41ForCausalLM:
    from xllm.python import kernels

    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    (tmp_path / "quant_model_description.json").write_text("{}")
    lm = DeepseekV41ForCausalLM({**config, "device": "cpu", "dtype": "float32", "model_path": str(tmp_path)})
    assert lm.quant_mode == "ascend"
    return lm


def _ascend_tensors(omit_attn_offsets: bool = False, *, config: dict | None = None) -> dict[str, torch.Tensor]:
    cfg = _DSV41_ASCEND_CONFIG if config is None else config
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    o_groups = cfg["o_groups"]
    o_lora = cfg["o_lora_rank"]
    inter = DeepseekV41Config.from_dict(cfg).dense_intermediate_size
    p = "layers.0."
    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg["vocab_size"], hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg["vocab_size"], hidden),
    }
    tensors[p + "attn_norm.weight"] = torch.ones(hidden)
    tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
    hc_mult = cfg["hc_mult"]
    for part in ("attn", "ffn"):
        tensors[p + f"hc_{part}_fn"] = torch.randn((2 + hc_mult) * hc_mult, hc_mult * hidden)
        tensors[p + f"hc_{part}_scale"] = torch.randn(3)
        tensors[p + f"hc_{part}_base"] = torch.randn((2 + hc_mult) * hc_mult)

    def w8a8(prefix: str, out: int, inp: int) -> None:
        tensors[prefix + ".weight"] = torch.zeros(out, inp, dtype=torch.int8)
        tensors[prefix + ".weight_scale"] = torch.ones(out, 1)
        if not omit_attn_offsets:
            tensors[prefix + ".weight_offset"] = torch.zeros(out, 1)

    w8a8(p + "attn.wq_a", q_lora, hidden)
    w8a8(p + "attn.wq_b", n_heads * head_dim, q_lora)
    w8a8(p + "attn.wkv", head_dim, hidden)
    tensors[p + "attn.wo_a.weight"] = torch.randn(o_groups * o_lora, (n_heads * head_dim) // o_groups)
    tensors[p + "attn.wo_b.weight"] = torch.randn(hidden, o_groups * o_lora)
    tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
    tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
    for name, rows, cols in (
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ):
        tensors[p + f"ffn.{name}.weight"] = torch.zeros(rows, cols, dtype=torch.int8)
        tensors[p + f"ffn.{name}.weight_scale"] = torch.ones(rows, 1)
        tensors[p + f"ffn.{name}.weight_offset"] = torch.zeros(rows, 1)
    return tensors


def _ascend_full_tensors() -> dict[str, torch.Tensor]:
    cfg = _DSV41_ASCEND_FULL_CONFIG
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    head_dim = cfg["head_dim"]
    index_dim = cfg["index_head_dim"]
    index_heads = cfg["index_n_heads"]
    p = "layers.0."
    tensors = _ascend_tensors()
    tensors[p + "attn.compressor.wkv.weight"] = torch.randn(head_dim, hidden)
    tensors[p + "attn.compressor.wgate.weight"] = torch.randn(head_dim, hidden)
    tensors[p + "attn.compressor.norm.weight"] = torch.ones(head_dim)
    tensors[p + "attn.indexer.wq_b.weight"] = torch.zeros(index_heads * index_dim, q_lora, dtype=torch.int8)
    tensors[p + "attn.indexer.wq_b.weight_scale"] = torch.ones(index_heads * index_dim, 1)
    tensors[p + "attn.indexer.weights_proj.weight"] = torch.randn(index_heads, hidden)
    tensors[p + "attn.indexer.wk.weight"] = torch.randn(index_dim, head_dim)
    tensors[p + "attn.indexer.k_norm.weight"] = torch.ones(index_dim)
    return tensors


def _poison_w8a8_offsets(lm: DeepseekV41ForCausalLM) -> None:
    for module in lm.modules():
        if hasattr(module, "weight_offset"):
            module.weight_offset.data.fill_(float("nan"))


def _ascend_attn_offsets(lm: DeepseekV41ForCausalLM) -> list[torch.Tensor]:
    attn = lm.model.layers[0].self_attn
    return [getattr(attn, name).weight_offset for name in ("q_a_proj", "q_b_proj", "kv_proj")]


def test_ascend_load_zero_fills_missing_weight_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    lm.load_weights([_FakeStateDict(_ascend_tensors(omit_attn_offsets=True))], 0, 1)
    offsets = _ascend_attn_offsets(lm)
    assert all(torch.all(offset == 0) for offset in offsets)
    assert all(offset.dtype == torch.float32 for offset in offsets)
    assert [offset.numel() for offset in offsets] == [8, 32, 16]


def test_ascend_load_rejects_nonzero_weight_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    tensors = _ascend_tensors()
    tensors["layers.0.attn.wq_a.weight_offset"] = torch.ones(8, 1)
    with pytest.raises(ValueError, match="zero weight_offset"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_load_maps_compressor_and_indexer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _make_ascend_lm(tmp_path, monkeypatch, _DSV41_ASCEND_FULL_CONFIG)
    tensors = _ascend_full_tensors()
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)
    attn = lm.model.layers[0].self_attn
    for name, key in (
        ("cmp_wkv", "attn.compressor.wkv.weight"),
        ("cmp_wgate", "attn.compressor.wgate.weight"),
        ("cmp_norm", "attn.compressor.norm.weight"),
    ):
        assert torch.equal(getattr(attn, name).weight, tensors["layers.0." + key])
    assert torch.equal(attn.indexer.weights_proj.weight, tensors["layers.0.attn.indexer.weights_proj.weight"])
    assert torch.equal(attn.indexer.wk.weight, tensors["layers.0.attn.indexer.wk.weight"])
    assert torch.equal(attn.indexer.k_norm.weight, tensors["layers.0.attn.indexer.k_norm.weight"])
    assert torch.all(attn.indexer.wq_b.weight_offset == 0)


_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1_slow(value: float) -> float:
    magnitude = abs(value)
    below = max((g for g in _E2M1_GRID if g <= magnitude), default=0.0)
    above = min((g for g in _E2M1_GRID if g >= magnitude), default=6.0)
    if magnitude - below < above - magnitude:
        nearest = below
    elif magnitude - below > above - magnitude:
        nearest = above
    else:
        nearest = below if _E2M1_GRID.index(below) % 2 == 0 else above
    return -nearest if value < 0 else nearest


def _expected_fp8_act_qdq(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    t = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return (normalized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e8m0(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    t = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e4m3(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-9)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _make_block_quantized(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_out = -(-out_dim // 32)
    blocks_in = -(-in_dim // 32)
    exponents = torch.randint(-8, 9, (blocks_out, blocks_in))
    scale_bytes = (exponents + 127).to(torch.uint8)
    scales = (
        torch.exp2(exponents.double())
        .repeat_interleave(32, dim=0)
        .repeat_interleave(32, dim=1)[:out_dim, :in_dim]
        .float()
    )
    base = (torch.randn(out_dim, in_dim) * 100).clamp(-448, 448)
    base = base.to(torch.float8_e4m3fn).float()
    weight = (base * scales).to(torch.bfloat16)
    quantized = (weight.float() / scales).to(torch.float8_e4m3fn)
    return quantized, scale_bytes, weight


def test_ue8m0_to_scale() -> None:
    scale_bytes = torch.tensor([0x01, 0x7F, 0x80, 0x00, 0xFE], dtype=torch.uint8)
    expected = torch.tensor([2.0**-126, 1.0, 2.0, 2.0**-127, 2.0**127])
    assert torch.equal(dq.ue8m0_to_scale(scale_bytes), expected)
    raw = torch.tensor([0x80, 0x7F], dtype=torch.uint8).view(torch.int8)
    assert torch.equal(dq.ue8m0_to_scale(raw), torch.tensor([2.0, 1.0]))
    with pytest.raises(ValueError):
        dq.ue8m0_to_scale(torch.tensor([0x01, 0xFF, 0x02], dtype=torch.uint8))


def test_dequant_fp8_block_roundtrip_exact() -> None:
    torch.manual_seed(0)
    quantized, scale_bytes, weight = _make_block_quantized(96, 160)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, weight)
    torch.manual_seed(1)
    quantized, scale_bytes, weight = _make_block_quantized(70, 100)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert torch.equal(out, weight)
    with pytest.raises(ValueError):
        dq.dequant_fp8_block(quantized, scale_bytes[:-1])


def test_unpack_fp4_preserves_nibble_order() -> None:
    packed = torch.tensor([[0x12, 0x21, 0x98, 0x9A, 0x00, 0x77, 0xF0]], dtype=torch.uint8)
    out = dq.unpack_fp4(packed)
    expected_row = [1.0, 0.5, 0.5, 1.0, -0.0, -0.5, -1.0, -0.5, 0.0, 0.0, 6.0, 6.0, 0.0, -6.0]
    assert torch.equal(out.float(), torch.tensor([expected_row]))


def test_dequant_fp4_group_known_scales() -> None:
    torch.manual_seed(3)
    n, k = 8, 96
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.int8)
    scale_bytes = torch.tensor([[0x7F, 0x80, 0x7D]] * n, dtype=torch.uint8)
    out = dq.dequant_fp4_group(packed, scale_bytes)
    unpacked = dq.unpack_fp4(packed, dtype=torch.float32)
    scales = torch.tensor([1.0, 2.0, 0.25]).repeat_interleave(32)
    assert torch.equal(out, (unpacked * scales).to(torch.bfloat16))
    zero_out = dq.dequant_fp4_group(packed[:, :16], torch.zeros(n, 1, dtype=torch.uint8))
    assert torch.equal(zero_out, (unpacked[:, :32] * 2.0**-127).to(torch.bfloat16))


def test_fp8_act_qdq_known_and_reference() -> None:
    x = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 100.0, 0.5, 300.0]])
    expected = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 96.0, 0.5, 288.0]])
    assert torch.equal(dq.fp8_act_qdq(x, group=8), expected)
    assert torch.all(dq.fp8_act_qdq(torch.full((1, 32), 449.0), group=32) == 448.0)
    torch.manual_seed(6)
    xr = (torch.randn(4, 128) * 7).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp8_act_qdq(xr, group=32)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp8_act_qdq(xr, 32))
    assert torch.all(out[:, :5] == 0)


def test_fp4_act_qdq_e8m0_known_and_reference() -> None:
    x = torch.tensor(
        [[6.0, 3.0, 1.5, 0.75, 0.25, 0.0, -6.0, -0.75, 12.0, 5.0, 2.5, 3.5, 0.5, 1.0, 4.0, 0.26]],
        dtype=torch.bfloat16,
    )
    expected = torch.tensor([[6, 3, 2, 1, 0, 0, -6, -1, 12, 4, 2, 4, 0, 1, 4, 0]], dtype=torch.bfloat16)
    assert torch.equal(dq.fp4_act_qdq_e8m0(x, group=16), expected)
    torch.manual_seed(8)
    xr = (torch.randn(4, 128) * 3).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp4_act_qdq_e8m0(xr, group=32)
    assert torch.equal(out.float(), _expected_fp4_e8m0(xr, 32))
    assert torch.all(dq.fp4_act_qdq_e8m0(torch.zeros(2, 32, dtype=torch.bfloat16), group=32) == 0)


def test_fp4_act_qdq_e4m3_scale_known_and_reference() -> None:
    cases = [
        (6.0, 6.0),  # s = e4m3(1) = 1
        (1.0, 1.03125),  # s = e4m3(1/6) = 0.171875
        (0.0234375, 0.0234375),  # 6 * 2**-8: s = 2**-8 exactly
        (5.0, 4.875),  # s = e4m3(5/6) = 0.8125; 5/0.8125 rounds to 6
        (0.0, 0.0),  # all-zero group: amax floor 6 * 2**-9
    ]
    for value, expected in cases:
        out = dq.fp4_act_qdq_e4m3_scale(torch.tensor([value]), group=1)
        assert out.item() == expected, (value, out.item(), expected)
    torch.manual_seed(9)
    xr = (torch.randn(4, 128) * 2).to(torch.bfloat16)
    xr[:, :3] = 0
    out = dq.fp4_act_qdq_e4m3_scale(xr, group=16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp4_e4m3(xr, 16))


def test_build_cache_specs_v41() -> None:
    caches_info, group_infos = build_cache_specs_v41(
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4],
        window_size=128,
        n_layers=6,
    )
    expected_groups = [
        (DSA_CACHE_SLIDING_WINDOW, 1, 128),
        (DSA_CACHE_TOKEN, 1, 128),
        (DSA_CACHE_TOKEN, 2, 128),
    ]
    assert [(g.cache_type, g.ratio, g.block_size) for g in group_infos] == expected_groups
    _, unsorted = build_cache_specs_v41(
        compress_ratios=[1, 1, 2, 2, 1, 1],
        kv_source_layer_ids=[4, 2],
        window_size=128,
        n_layers=6,
    )
    assert [(g.cache_type, g.ratio, g.block_size) for g in unsorted] == expected_groups

    def entries(layer_id: int) -> list[tuple[int, int, int]]:
        return [(c.cache_type, c.ratio, c.group_id) for c in caches_info[layer_id]]

    swa = (DSA_CACHE_SLIDING_WINDOW, 1, 0)
    assert entries(0) == [swa]
    assert entries(3) == [swa]
    assert entries(2) == [(DSA_CACHE_TOKEN, 2, 2), (DSA_CACHE_TOKEN, 2, 2), swa, swa, swa]
    assert entries(4) == [(DSA_CACHE_TOKEN, 1, 1), (DSA_CACHE_TOKEN, 1, 1), swa]

    backend = _make_csa2_backend(DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG))
    full_r2 = backend._resolve_cache_mapping(2, 2)
    assert (
        full_r2.cmp_cache_idx,
        full_r2.index_cache_idx,
        full_r2.ori_cache_idx,
        full_r2.kv_state_cache_idx,
        full_r2.score_state_cache_idx,
    ) == (0, 1, 2, 3, 4)
    full_r1 = backend._resolve_cache_mapping(4, 1)
    assert (full_r1.cmp_cache_idx, full_r1.index_cache_idx, full_r1.ori_cache_idx) == (0, 1, 2)
    assert (full_r1.kv_state_cache_idx, full_r1.score_state_cache_idx) == (-1, -1)
    reindex = backend._resolve_cache_mapping(5, 1)
    assert (reindex.cmp_cache_idx, reindex.ori_cache_idx) == (-1, 0)


_TOKEN_BLOCK_SIZE = 128
_SWA_RING_BLOCKS = 2
_HOST_DEVICE = torch.device("cpu")


def _make_csa2_backend(
    cfg: DeepseekV41Config,
    device: torch.device | None = None,
    *,
    quant_mode: str = V41_QUANT_NONE,
    acl_graph_enabled: bool = False,
    max_model_len: int = 0,
) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=list(cfg.compress_ratios),
        window_size=cfg.window_size,
        n_layers=cfg.n_layers,
        num_heads=cfg.n_heads,
        attn_head_dim=cfg.head_dim,
        index_topk=cfg.index_topk,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        device=device or torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=list(cfg.kv_source_layer_ids),
        index_source_layer_ids=list(cfg.index_source_layer_ids),
        candidate_source_layer_id=cfg.candidate_source_layer_id,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        quant_mode=quant_mode,
        acl_graph_enabled=acl_graph_enabled,
        max_model_len=max_model_len,
    )


def test_csa2_acl_graph_capacity_is_checked_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", raising=False)
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)

    derived = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=32768,
    )
    assert derived.graph_token_capacity_granularity > 0

    with pytest.raises(RuntimeError, match="static committed-row capacity"):
        _make_csa2_backend(
            cfg,
            torch.device("cpu"),
            quant_mode=V41_QUANT_ASCEND,
            acl_graph_enabled=True,
            max_model_len=0,
        )

    _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_NONE,
        acl_graph_enabled=True,
        max_model_len=0,
    )

    monkeypatch.setenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", "8192")
    explicit = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=0,
    )
    assert explicit.graph_token_capacity_granularity == 8192


def _random_init_v41(lm: DeepseekV41ForCausalLM) -> None:
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if param.dtype == torch.uint8:
                fill = 127 if name.endswith("weight_scale") else 0x38
                param.copy_(torch.full_like(param, fill))
                continue
            if name == "model.norm.weight" or name.endswith(("layernorm.weight", "cmp_norm.weight", "k_norm.weight")):
                param.copy_(torch.ones_like(param))
            elif name.endswith(("attn_sink", "e_score_correction_bias", "gate_bias_vl")):
                param.copy_(torch.zeros_like(param))
            else:
                param.normal_(mean=0.0, std=0.05)
            param.requires_grad_(False)


def _allocate_v41_layer_caches(
    cfg: DeepseekV41Config,
    n_seqs: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
    *,
    reserve_token_sink: bool = False,
) -> list[LayerCache]:
    device = device or torch.device("cpu")

    def swa(dtype_: torch.dtype = dtype) -> torch.Tensor:
        return torch.zeros(_SWA_RING_BLOCKS * n_seqs, cfg.window_size, cfg.head_dim, dtype=dtype_, device=device)

    token_blocks = n_seqs + 1 if reserve_token_sink else n_seqs
    caches: list[LayerCache] = []
    for layer_id in range(cfg.n_layers):
        if layer_id not in cfg.kv_source_layer_ids:
            caches.append(LayerCache(key=None, value=None, swa=swa()))
            continue
        ratio = cfg.compress_ratios[layer_id]
        states = ratio > 1
        caches.append(
            LayerCache(
                key=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.head_dim, dtype=dtype, device=device),
                value=None,
                index=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.index_head_dim, dtype=dtype, device=device),
                swa=swa(),
                compress_kv_state=swa(torch.float32) if states else None,
                compress_score_state=swa(torch.float32) if states else None,
            )
        )
    return caches


class _V41ForwardHarness:
    def __init__(
        self,
        lm: DeepseekV41ForCausalLM,
        config: dict,
        n_seqs: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        self.lm = lm
        self.cfg = lm.cfg
        self.n_seqs = n_seqs
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.swa_bt = torch.arange(_SWA_RING_BLOCKS * n_seqs, dtype=torch.int32, device=self.device).view(n_seqs, -1)
        self.tok2_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.tok1_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.layer_caches = _allocate_v41_layer_caches(
            self.cfg, n_seqs, dtype, device=self.device, reserve_token_sink=True
        )
        self.executor = ModelExecutor(lm, dict(config), max_seqs_per_batch=n_seqs)
        self.executor.bind_kv_caches(list(self.layer_caches))
        self.backend = self.executor.attention_backend

    def step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: list[int],
        q_lens: list[int],
        *,
        is_prefill: bool,
        is_chunked_prefill: bool = False,
    ) -> torch.Tensor:
        metadata = SimpleNamespace(
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            is_mixed=False,
            is_spec_verify=False,
            multi_block_tables=[self.swa_bt, self.tok1_bt, self.tok2_bt],
            kv_seq_lens_host=torch.tensor(kv_lens, dtype=torch.int32, device=_HOST_DEVICE),
            kv_seq_lens_host_values=list(kv_lens),
            q_seq_lens_host=torch.tensor(q_lens, dtype=torch.int32, device=_HOST_DEVICE),
        )
        return self.executor.execute(input_ids, positions, metadata)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.lm.compute_logits(hidden, None).float()


def _make_forward_lm(
    dtype: torch.dtype,
    seed: int,
    device: torch.device | None = None,
) -> DeepseekV41ForCausalLM:
    torch.manual_seed(seed)
    lm = DeepseekV41ForCausalLM({**_DSV41_FORWARD_CONFIG, "device": str(device or "cpu"), "dtype": dtype})
    _random_init_v41(lm)
    return lm


def test_compressor_math() -> None:
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    def build_dsa(kv_len: int, q_len: int, start: int) -> DsaMetadata:
        return backend._builder.build(
            multi_block_tables=[
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
            ],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(start, start + q_len, dtype=torch.int64),
            is_prefill=False,
            is_chunked_prefill=True,
        )

    def expected_key_row(latent: torch.Tensor, position: int) -> torch.Tensor:
        rotated = apply_interleaved_rope(latent.unsqueeze(0), torch.tensor([position]), cos_sin)
        return _expected_fp4_e4m3(rotated, 16)[0]

    def expected_index_rows(latent: torch.Tensor, starts: list[int]) -> torch.Tensor:
        index_k = layer.indexer_key(latent)
        return fp4_act_qdq_e8m0(apply_interleaved_rope(index_k, torch.tensor(starts), cos_sin), 32)

    x1 = torch.randn(3, cfg.hidden_size)
    ctx1 = Csa2LayerContext(
        x=x1,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3),
        cos_sin_cache=cos_sin,
    )
    shared = CSA2SharedRuntime()
    backend._run_compressor(
        layer, ctx1, build_dsa(3, 3, 0), layer_caches[2], mapping, [3], torch.tensor([3], dtype=torch.int64), shared
    )

    kv1, score1 = layer.compress_kv_score(x1)
    gate0 = score1[:2].softmax(dim=0)
    pooled0 = (kv1[:2] * gate0).sum(dim=0)
    latent0 = layer.compress_rmsnorm(pooled0.unsqueeze(0))[0]
    key_cache = layer_caches[2].key
    assert torch.allclose(key_cache[0, 0], expected_key_row(latent0, 0), atol=1e-6)
    assert torch.allclose(layer_caches[2].index[0, 0], expected_index_rows(latent0.unsqueeze(0), [0])[0], atol=1e-6)
    assert (key_cache[0, : 3 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 3 // 2 :] == 0).all()
    state_kv = layer_caches[2].compress_kv_state.view(-1, cfg.head_dim)
    state_score = layer_caches[2].compress_score_state.view(-1, cfg.head_dim)
    assert torch.allclose(state_kv[2], kv1[2], atol=1e-6)
    assert torch.allclose(state_score[2], score1[2], atol=1e-6)

    x2 = torch.randn(3, cfg.hidden_size)
    ctx2 = Csa2LayerContext(
        x=x2,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3, 6),
        cos_sin_cache=cos_sin,
    )
    backend._run_compressor(
        layer, ctx2, build_dsa(6, 3, 3), layer_caches[2], mapping, [3], torch.tensor([6], dtype=torch.int64), shared
    )

    kv2, score2 = layer.compress_kv_score(x2)
    members = torch.stack((state_kv[2].clone(), kv2[0]))
    gates = torch.stack((state_score[2].clone(), score2[0])).softmax(dim=0)
    pooled1 = (members * gates).sum(dim=0)
    latent1 = layer.compress_rmsnorm(pooled1.unsqueeze(0))[0]
    gate2 = score2[1:3].softmax(dim=0)
    pooled2 = (kv2[1:3] * gate2).sum(dim=0)
    latent2 = layer.compress_rmsnorm(pooled2.unsqueeze(0))[0]
    assert torch.allclose(key_cache[0, 1], expected_key_row(latent1, 2), atol=1e-6)
    assert torch.allclose(key_cache[0, 2], expected_key_row(latent2, 4), atol=1e-6)
    expected_index = expected_index_rows(torch.stack((latent1, latent2)), [2, 4])
    assert torch.allclose(layer_caches[2].index[0, 1:3], expected_index, atol=1e-6)
    assert (key_cache[0, : 6 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 6 // 2 :] == 0).all()

    layer4 = model.layers[4].self_attn
    mapping4 = backend._resolve_cache_mapping(4, layer4.mode.compress_ratio)
    x = torch.randn(5, cfg.hidden_size)
    ctx = Csa2LayerContext(
        x=x,
        qr=torch.randn(5, cfg.q_lora_rank),
        positions=torch.arange(5),
        cos_sin_cache=cos_sin,
    )
    shared4 = CSA2SharedRuntime()
    backend._run_compressor(
        layer4, ctx, build_dsa(5, 5, 0), layer_caches[4], mapping4, [5], torch.tensor([5], dtype=torch.int64), shared4
    )
    kv, _ = layer4.compress_kv_score(x)
    latent = layer4.compress_rmsnorm(kv)
    expected = _expected_fp4_e4m3(apply_interleaved_rope(latent, torch.arange(5), cos_sin), 16)
    assert torch.allclose(layer_caches[4].key[0, :5], expected, atol=1e-6)
    assert (layer_caches[4].key[0, 5:] == 0).all()
    assert layer_caches[4].compress_kv_state is None
    assert layer_caches[4].compress_score_state is None


def test_compressor_padded_row_uses_nonnegative_rope_position() -> None:
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    assert layer.mode.compress_ratio == 2
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    dsa = backend._builder.build(
        multi_block_tables=[
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        ],
        kv_seq_lens=[0],
        q_seq_lens=[1],
        positions=torch.arange(0, 1, dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=True,
    )
    ctx = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.arange(0, 1),
        cos_sin_cache=cos_sin,
    )
    key_before = layer_caches[2].key.clone()
    index_before = layer_caches[2].index.clone()

    backend._run_compressor(
        layer, ctx, dsa, layer_caches[2], mapping, [1], torch.tensor([0], dtype=torch.int64), CSA2SharedRuntime()
    )

    assert torch.equal(layer_caches[2].key, key_before)
    assert torch.equal(layer_caches[2].index, index_before)


def test_indexer_scoring() -> None:
    torch.manual_seed(21)
    q = torch.randn(4, 3, 32)
    index_k = torch.randn(6, 32)
    weights = torch.randn(4, 3)
    score = _indexer_score(q, index_k, weights)
    manual = torch.einsum("shd,td->sht", q, index_k).relu_() * weights.unsqueeze(-1)
    assert torch.allclose(score, manual.sum(dim=1), atol=1e-5)

    lm = _make_forward_lm(torch.float32, seed=22)
    layer = lm.model.layers[2].self_attn
    x = torch.randn(4, lm.cfg.hidden_size)
    scale = layer.indexer.head_dim**-0.5 * layer.indexer.n_heads**-0.5
    assert torch.allclose(layer.indexer_weights(x), layer.indexer.weights_proj(x) * scale)

    score_row = torch.tensor([[3.0, 1.0, 2.0, float("-inf"), float("-inf")]])
    idxs = _indexer_topk(score_row, torch.tensor([3]), index_topk=4, offset=7)
    assert idxs.tolist() == [[7, 8, 9, -1]]

    score_pool = torch.tensor([[3.0, float("-inf"), 1.0, 2.0, float("-inf")]])
    idxs_pool = _indexer_topk(score_pool, torch.tensor([5]), index_topk=4, offset=7)
    assert idxs_pool.tolist() == [[7, -1, 9, 10]]

    score_pool = torch.tensor(
        [[1.0, 2.0, float("-inf"), float("-inf"), 3.0, 4.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")]]
    )
    pinned = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=1, block_size=4)
    assert pinned.tolist() == [[False] * 4 + [True] * 4 + [False] * 2]
    both = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=2, block_size=4)
    assert both.tolist() == [[True] * 8 + [False] * 2]


def test_indexer_causal_visibility_and_candidate_pool() -> None:
    lm = _make_forward_lm(torch.float32, seed=23)
    cfg = lm.cfg
    backend = _make_csa2_backend(cfg)
    cos_sin = lm.model.compress_rotary.cos_sin_cache

    layer2 = lm.model.layers[2].self_attn
    index_cache = torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5
    shared = CSA2SharedRuntime(
        index_cache=index_cache,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=2,
    )
    q_len, ctx_len = 8, 8
    ctx = Csa2LayerContext(
        x=torch.randn(q_len, cfg.hidden_size),
        qr=torch.randn(q_len, cfg.q_lora_rank),
        positions=torch.arange(q_len),
        cos_sin_cache=cos_sin,
    )
    topk = backend._run_indexer(layer2, ctx, None, [q_len], torch.tensor([ctx_len], dtype=torch.int64), shared)
    offset = cfg.window_size + q_len
    for query in range(q_len):
        visible = (query + 1) // 2
        picks = topk[0, query]
        valid = picks[picks >= 0] - offset
        assert ((valid >= 0) & (valid < visible)).all()
        assert valid.numel() == min(cfg.index_topk, visible)
    assert shared.candidates is None

    layer4 = lm.model.layers[4].self_attn
    shared4 = CSA2SharedRuntime(
        index_cache=torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=1,
    )
    backend._run_indexer(layer4, ctx, None, [q_len], torch.tensor([ctx_len], dtype=torch.int64), shared4)
    assert len(shared4.candidates) == 1

    layer5 = lm.model.layers[5].self_attn
    pool = torch.zeros(1, ctx_len, dtype=torch.bool)
    pool[0, [1, 3, 4, 6]] = True
    shared5 = CSA2SharedRuntime(
        index_cache=shared4.index_cache,
        index_block_table=shared4.index_block_table,
        compress_ratio=1,
        candidates=[pool],
    )
    ctx_late = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.tensor([ctx_len - 1]),
        cos_sin_cache=cos_sin,
    )
    topk5 = backend._run_indexer(layer5, ctx_late, None, [1], torch.tensor([ctx_len], dtype=torch.int64), shared5)
    picks = topk5[0, 0]
    assert (picks[picks >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]

    backend.index_topk = 512
    topk_wide = backend._run_indexer(layer5, ctx_late, None, [1], torch.tensor([ctx_len], dtype=torch.int64), shared5)
    picks_wide = topk_wide[0, 0]
    assert picks_wide.numel() == ctx_len
    assert (picks_wide[picks_wide >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]
    assert (picks_wide == -1).sum().item() == ctx_len - 4


def test_swa_window_lower_bound_is_per_query() -> None:
    cases = [
        (8, 0, 9),
        (2, 5, 3),
        (128, 200, 2),
        (8, 0, 8),
    ]
    for win, prev_ctx, q_len in cases:
        ring_pos = (prev_ctx - win) + torch.arange(win, dtype=torch.int64)
        positions_seq = prev_ctx + torch.arange(q_len, dtype=torch.int64)
        col_pos = torch.cat((ring_pos, positions_seq), dim=0)
        cols = torch.arange(col_pos.numel(), dtype=torch.int64)
        idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)
        for i in range(q_len):
            p = prev_ctx + i
            picked = idxs[i][idxs[i] >= 0]
            got = col_pos[picked].tolist()
            expected = list(range(max(p - win + 1, 0), p + 1))
            assert got == expected, f"win={win} prev_ctx={prev_ctx} q_len={q_len} query={i}: {got} != {expected}"


@npu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [0, 1, 7])
def test_hc_post_preserves_v41_matrix_orientation(dtype: torch.dtype, num_tokens: int) -> None:
    torch.manual_seed(4109)
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    hc = dq.DeepseekV41HyperConnection(cfg, _active_device())
    x = torch.randn(num_tokens, cfg.hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, cfg.hc_mult, cfg.hidden_size, dtype=dtype)
    post = torch.randn(num_tokens, cfg.hc_mult, dtype=torch.float32)
    comb = torch.arange(num_tokens * cfg.hc_mult**2, dtype=torch.float32).reshape(
        num_tokens, cfg.hc_mult, cfg.hc_mult
    ) / (cfg.hc_mult**2)
    expected = post.unsqueeze(-1) * x.float().unsqueeze(-2) + torch.bmm(comb, residual.float())
    actual = hc.hc_post(
        x.to(_active_device()),
        residual.to(_active_device()),
        post.to(_active_device()),
        comb.to(_active_device()),
    )
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=tolerance, atol=tolerance)


@npu_only
def test_mhc_final_merge_is_learned_weighted_sum() -> None:
    device = _active_device()
    lm = _make_forward_lm(torch.float32, seed=31, device=device)
    with torch.no_grad():
        for layer in lm.model.layers:
            layer.hc.hc_attn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_ffn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_attn_scale.fill_(1.0)
            layer.hc.hc_ffn_scale.fill_(1.0)
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs=1, dtype=torch.float32, device=device)

    captured: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in (4, 5):
        decoder_layer = lm.model.layers[layer_id]
        original = decoder_layer.forward

        def make_wrapper(
            orig: Callable[..., tuple[torch.Tensor, torch.Tensor]], store: dict[str, torch.Tensor]
        ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
            def wrapper(
                hidden: torch.Tensor,
                positions: torch.Tensor,
                cos_sin_cache: torch.Tensor,
                input_ids: torch.Tensor | None = None,
                pre_mix: torch.Tensor | None = None,
                engram_hashes: torch.Tensor | None = None,
                engram_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                out, ffn_pre = orig(hidden, positions, cos_sin_cache, input_ids, pre_mix, engram_hashes, engram_mask)
                store["hidden"] = out.clone()
                store["ffn_pre"] = ffn_pre.clone()
                return out, ffn_pre

            return wrapper

        decoder_layer.forward = make_wrapper(original, captured.setdefault(layer_id, {}))

    input_ids = torch.randint(0, lm.cfg.vocab_size, (5,)).to(device)
    with torch.inference_mode():
        hidden = harness.step(input_ids, torch.arange(5, device=device), [5], [5], is_prefill=True)

    last = captured[5]
    earlier = captured[4]
    assert (last["ffn_pre"].std(dim=-1) > 1e-3).all()

    with torch.inference_mode():
        hc = lm.model.layers[5].hc
        expected = lm.model.norm(hc.hc_pre(last["hidden"], last["ffn_pre"]))
        mean_merge = lm.model.norm(last["hidden"].float().mean(dim=-2))
        earlier_merge = lm.model.norm(hc.hc_pre(last["hidden"], earlier["ffn_pre"]))
    assert torch.allclose(hidden, expected, atol=1e-5)
    assert not torch.allclose(hidden, mean_merge, atol=1e-3)
    assert not torch.allclose(hidden, earlier_merge, atol=1e-3)


@npu_only
def test_forward_smoke_and_prefill_decode_consistency() -> None:
    device = _active_device()
    lm = _make_forward_lm(torch.bfloat16, seed=41, device=device)
    cfg = lm.cfg
    n_seqs, prompt_len, decode_steps = 2, cfg.window_size + 1, 4
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)

    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device).repeat(n_seqs),
            [prompt_len] * n_seqs,
            [prompt_len] * n_seqs,
            is_prefill=True,
        )
    assert torch.isfinite(hidden.float()).all()
    prefill_logits = harness.logits(hidden)
    assert torch.isfinite(prefill_logits).all()

    swa = harness.layer_caches[0].swa
    assert (swa[0].abs().sum(dim=-1) > 0).all()
    assert (swa[2].abs().sum(dim=-1) > 0).all()
    assert swa[1, 0].abs().sum(dim=-1) > 0
    assert (swa[1, 1:] == 0).all()
    assert swa[3, 0].abs().sum(dim=-1) > 0
    assert (swa[3, 1:] == 0).all()

    key2 = harness.layer_caches[2].key
    key4 = harness.layer_caches[4].key
    assert (key2[1, : prompt_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, prompt_len // 2 :] == 0).all()
    assert (key4[1, :prompt_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, prompt_len:] == 0).all()

    shared = harness.backend._v41_shared
    topk5 = shared.topk_idxs
    compress_offset = cfg.window_size + prompt_len
    for seq in range(n_seqs):
        pool = shared.candidates[seq]
        for query in range(prompt_len):
            valid = topk5[seq, query] - compress_offset
            for value in valid[valid >= 0].tolist():
                assert pool[query, value]

    decode_ids = torch.randint(0, cfg.vocab_size, (n_seqs, decode_steps)).to(device)
    for step in range(decode_steps):
        position = prompt_len + step
        with torch.inference_mode():
            step_hidden = harness.step(
                decode_ids[:, step],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        assert torch.isfinite(step_hidden.float()).all()
    second_block_rows = prompt_len - cfg.window_size + decode_steps
    assert (swa[1, :second_block_rows].abs().sum(dim=-1) > 0).all()
    assert (swa[1, second_block_rows:] == 0).all()
    assert (swa[3, :second_block_rows].abs().sum(dim=-1) > 0).all()
    ctx_len = prompt_len + decode_steps
    assert (key2[1, : ctx_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, ctx_len // 2 :] == 0).all()
    assert (key4[1, :ctx_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, ctx_len:] == 0).all()

    replay = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    replay_logits = []
    for position in range(prompt_len):
        with torch.inference_mode():
            step_hidden = replay.step(
                input_ids[:, position],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        replay_logits.append(replay.logits(step_hidden))
    for position in range(prompt_len):
        for seq in range(n_seqs):
            expected = prefill_logits[seq * prompt_len + position]
            diff = (replay_logits[position][seq] - expected).abs().max().item()
            assert diff < 2e-2, f"decode replay diverged at position {position} of sequence {seq}: {diff}"


def test_engram_layout_prime_buckets() -> None:
    layout = EngramLayout(
        layer_ids=[1, 14],
        num_embeddings=[384006168, 384016682],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    assert layout.primes[0][0] == (67, 71)
    assert layout.primes[0][1] == (73, 79)
    assert layout.primes[1][0] == (83, 89)
    assert layout.primes[1][1] == (97, 101)
    assert layout.n_hash_cols == 4
    assert torch.equal(layout.offsets, torch.tensor([[0, 67, 138, 211], [0, 83, 172, 269]]))


def test_engram_token_map_normalization_chain() -> None:
    lookup, size = build_compressed_token_map_from_texts(["The", "the", "THE", " thé ", "x", "y"])
    assert size == 3
    assert lookup[0] == lookup[1] == lookup[2] == lookup[3]
    assert lookup[4] != lookup[5]
    assert normalize_token_text("  A\tB\nC ") == "a b c"
    assert normalize_token_text(" ") == " "


def test_engram_embedding_tp_sharding_nondivisible() -> None:
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(1234)
    payload = torch.randint(0, 256, (num_rows, dim), dtype=torch.uint8, generator=g)
    payload = torch.where((payload == 0x7F) | (payload == 0xFF), payload - 1, payload)
    scales = torch.randint(126, 130, (num_rows, dim // 32), dtype=torch.uint8, generator=g)

    def make(tp_size: int, tp_rank: int) -> EngramEmbedding:
        emb = EngramEmbedding(num_rows, dim, head_sizes, tp_size, tp_rank)
        with torch.no_grad():
            emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
            emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
        return emb

    ref = make(1, 0)
    t = 6
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])

    for tp_size in (2, 3, 5, 8):
        part = (len(head_sizes) + tp_size - 1) // tp_size
        blocks = [make(tp_size, rank)._local_block(ids) for rank in range(tp_size)]
        for rank in range(tp_size):
            if rank * part >= len(head_sizes):
                assert not blocks[rank].any()
        gathered = torch.cat(blocks, dim=1)[:, : ref.n_hash_cols]
        assert torch.equal(gathered, ref.lookup(ids))


def _quarot_gate_reference(
    eng: Engram,
    hidden: torch.Tensor,
    staged_rows: torch.Tensor,
    unrotate: torch.Tensor | None,
) -> torch.Tensor:
    rows = staged_rows.to(eng.wkv.weight.dtype)
    kv = eng.wkv(rows.flatten(-2))
    num_tokens, hc_mult, dim = hidden.shape
    keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
    value = kv[:, hc_mult * dim :].float()
    h = hidden.float()
    gate_hidden = h @ unrotate.to(torch.float32) if unrotate is not None else h
    hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + eng.eps)
    key_rms = torch.rsqrt(keys.square().mean(-1) + eng.eps)
    dot = (gate_hidden * eng.q_weight.float() * eng.k_weight.float() * keys).sum(-1)
    dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
    gate_input = dot.abs().clamp_min(1e-6).sqrt()
    gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(hidden.dtype)


def test_engram_gate_quarot_unrotation() -> None:
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    dim, hc_mult = 48, 2
    g = torch.Generator().manual_seed(99)
    q_orth = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]

    def make(gate_unrotate: torch.Tensor | None) -> Engram:
        eng = Engram(
            layout,
            0,
            dim,
            hc_mult,
            rms_norm_eps=1e-6,
            tp_size=1,
            tp_rank=0,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            storage=ENGRAM_STORAGE_INT8,
            gate_unrotate=gate_unrotate,
        )
        with torch.no_grad():
            eng.wkv.weight.copy_(torch.randn(eng.wkv.weight.shape, generator=g).bfloat16())
            eng.q_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
            eng.k_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
        return eng

    t, n_hash_cols = 7, layout.n_hash_cols
    staged = torch.randn(t, n_hash_cols, layout.head_dim, generator=g).bfloat16()
    hidden = torch.randn(t, hc_mult, dim, generator=g).bfloat16()

    plain = make(None)
    plain._staged_rows = staged
    out_plain = plain(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_plain, _quarot_gate_reference(plain, hidden, staged, None))

    rotated = make(q_orth.t().contiguous())
    rotated.wkv.load_state_dict(plain.wkv.state_dict())
    with torch.no_grad():
        rotated.q_weight.copy_(plain.q_weight)
        rotated.k_weight.copy_(plain.k_weight)
    rotated._staged_rows = staged
    out_rot = rotated(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_rot, _quarot_gate_reference(rotated, hidden, staged, q_orth.t()))
    assert not torch.equal(out_plain, out_rot)


def test_load_quarot_gate_unrotate(tmp_path: Path) -> None:
    import json as _json

    import safetensors.torch as _st
    from safetensors import SafetensorError

    model_path = tmp_path / "eco"
    (model_path / "optional").mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    q = torch.linalg.qr(torch.randn(64, 64, generator=g))[0]

    def write_config(rotated: bool) -> None:
        (model_path / "config.json").write_text(
            _json.dumps({"engram_rotation_config": {"value_projection_rotated": rotated}})
        )

    write_config(False)
    assert load_quarot_gate_unrotate(str(model_path), torch.device("cpu")) is None

    write_config(True)
    _st.save_file({"global_rotation": q.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded is not None
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, q.t().contiguous().to(torch.float32))

    q_bf = q.to(torch.bfloat16)
    _st.save_file({"global_rotation": q_bf.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded_bf = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded_bf is not None
    assert loaded_bf.dtype == torch.float32
    assert torch.equal(loaded_bf, q_bf.t().contiguous().to(torch.float32))

    (model_path / "optional" / "quarot.safetensors").write_bytes(b"garbage-bytes")
    with pytest.raises(SafetensorError):
        load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))


def test_engram_gate_math_and_mask_passthrough() -> None:
    lm = _make_forward_lm(torch.float32, seed=43)
    cfg = lm.cfg
    eng = lm.model.layers[1].engram
    assert eng is not None
    with torch.no_grad():
        eng.q_weight.normal_(0.0, 0.05)
        eng.k_weight.normal_(0.0, 0.05)
        eng.wkv.weight.normal_(0.0, 0.05)
    eng.embed_tokens.weight.zero_()
    payload = torch.tensor([0x38, 0xC0], dtype=torch.uint8)
    eng.embed_tokens.weight[:, 0] = payload[0]
    eng.embed_tokens.weight[:, 1] = payload[1]
    eng.embed_tokens.weight_scale.fill_(127)

    t, hc, hidden = 2, cfg.hc_mult, cfg.hidden_size
    hash_ids = torch.zeros(t, eng.embed_tokens.n_hash_cols, dtype=torch.int64)
    hidden_states = torch.randn(t, hc, hidden)

    out = eng(hidden_states, hash_ids)
    expected = _quarot_gate_reference(eng, hidden_states, eng.embed_tokens.lookup(hash_ids), None)
    assert torch.allclose(out, expected, atol=1e-4)

    mask = torch.tensor([True, False])
    out_masked = eng(hidden_states, hash_ids, mask)
    assert torch.allclose(out_masked[0], out[0], atol=1e-6)
    assert torch.allclose(out_masked[1], hidden_states[1], atol=1e-6)

    assert cfg.image_token_id == ENGRAM_IMAGE_SENTINEL_BASE_ID


@npu_only
def test_engram_injection_shifts_residual_stream() -> None:
    device = _active_device()
    lm = _make_forward_lm(torch.float32, seed=45, device=device)
    cfg = lm.cfg
    n_seqs, prompt_len = 1, 6
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.float32, device=device)
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)
    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device),
            [prompt_len],
            [prompt_len],
            is_prefill=True,
        )
    assert torch.isfinite(hidden).all()
    assert lm.model.engram_hash is not None
    assert lm.model.engram_hash._cache is not None
    assert lm.model.engram_hash._cache.numel() == harness.layer_caches[0].swa.numel() // cfg.head_dim
    state = lm.model.engram_hash
    comp = state.token_map[input_ids.reshape(-1).to(torch.int64)]
    flat_cache = state._cache
    for pos, value in enumerate(comp.tolist()):
        slot = int(harness.swa_bt[0, pos // cfg.window_size]) * cfg.window_size + pos % cfg.window_size
        assert int(flat_cache[slot]) == value
    eng = lm.model.layers[1].engram
    assert eng is not None and eng._staged_rows is not None
    assert torch.isfinite(eng._staged_rows.float()).all()
    hashes = state(
        input_ids.reshape(-1),
        torch.arange(prompt_len, device=device),
        torch.tensor([0, prompt_len], device=device),
        torch.zeros(prompt_len, dtype=torch.bool, device=device),
        None,
        None,
        0,
    )
    assert (hashes >= 0).all() and (hashes < cfg.engram_num_embeddings[0]).all()


@pytest.mark.parametrize("ring_blocks", [1, 2])
@pytest.mark.parametrize("full_table", [False, True])
def test_engram_cached_long_chunk_preserves_history(ring_blocks: int, full_table: bool) -> None:
    state = _tiny_engram_hash_state()
    reference = _tiny_engram_hash_state()
    block_size, prefix_len, chunk_len = 8, 19, 21
    capacity = ring_blocks * block_size
    state._cache = torch.zeros(capacity, dtype=torch.int32)
    ids = torch.randint(1, 64, (prefix_len + chunk_len,), generator=torch.Generator().manual_seed(311))
    positions = torch.arange(ids.numel())
    table_columns = (ids.numel() + block_size - 1) // block_size if full_table else ring_blocks
    blocks = torch.arange(table_columns) % ring_blocks
    table = blocks.to(torch.int32).unsqueeze(0)
    dead = torch.zeros(ids.numel(), dtype=torch.bool)
    state(
        ids[:prefix_len],
        positions[:prefix_len],
        torch.tensor([0, prefix_len]),
        dead[:prefix_len],
        (positions[:prefix_len] % capacity).to(torch.int32),
        table,
        block_size,
    )
    actual = state(
        ids[prefix_len:],
        positions[prefix_len:],
        torch.tensor([0, chunk_len]),
        dead[prefix_len:],
        (positions[prefix_len:] % capacity).to(torch.int32),
        table,
        block_size,
    )
    expected = reference(ids, positions, torch.tensor([0, ids.numel()]), dead, None, None, 0)
    torch.testing.assert_close(actual, expected[prefix_len:])


@pytest.mark.parametrize("ring_blocks", [1, 2])
def test_swa_cached_long_chunk_preserves_history(monkeypatch: pytest.MonkeyPatch, ring_blocks: int) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, 1, torch.float32)
    backend.bind_kv_caches(caches)
    prefix_len, chunk_len = 19, 21
    values = torch.arange(1, prefix_len + chunk_len + 1, dtype=torch.float32)
    assert caches[0].swa is not None
    flat = caches[0].swa.view(-1, cfg.head_dim)
    capacity = ring_blocks * cfg.window_size
    for pos in range(prefix_len):
        flat[pos % capacity].fill_(values[pos])
    positions = torch.arange(prefix_len, prefix_len + chunk_len)
    dsa = backend._builder.build(
        multi_block_tables=[
            torch.arange(ring_blocks, dtype=torch.int32).unsqueeze(0),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        ],
        kv_seq_lens=[prefix_len + chunk_len],
        q_seq_lens=[chunk_len],
        positions=positions,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    monkeypatch.setattr(backend, "_current_forward_metadata", lambda: SimpleNamespace(dsa_metadata=dsa))
    layer = SimpleNamespace(
        layer_id=0,
        mode=CSA2LayerPlan.build(
            cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids
        ).mode(0),
        attn_sink=torch.zeros(cfg.n_heads),
        attn_sink_loaded=True,
        _csa2_ctx=SimpleNamespace(x=torch.zeros(chunk_len, cfg.hidden_size)),
    )
    query = torch.zeros(chunk_len, cfg.n_heads, cfg.head_dim)
    current_kv = values[prefix_len:, None].expand(-1, cfg.head_dim).contiguous()
    actual = backend.execute_csa2_layer(query, current_kv, layer)
    expected = torch.stack(
        [
            values[pos - cfg.window_size + 1 : pos + 1].sum() / (cfg.window_size + 1)
            for pos in range(prefix_len, prefix_len + chunk_len)
        ]
    )
    torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))


def test_csa2_sink_normalizer_follows_loaded_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, 1, torch.float32)
    backend.bind_kv_caches(caches)
    prefix_len, chunk_len = 9, 1
    values = torch.arange(1, prefix_len + chunk_len + 1, dtype=torch.float32)
    assert caches[0].swa is not None
    flat = caches[0].swa.view(-1, cfg.head_dim)
    for pos in range(prefix_len):
        flat[pos % cfg.window_size].fill_(values[pos])
    dsa = backend._builder.build(
        multi_block_tables=[
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        ],
        kv_seq_lens=[prefix_len + chunk_len],
        q_seq_lens=[chunk_len],
        positions=torch.arange(prefix_len, prefix_len + chunk_len),
        is_prefill=False,
        is_chunked_prefill=True,
    )
    monkeypatch.setattr(backend, "_current_forward_metadata", lambda: SimpleNamespace(dsa_metadata=dsa))
    sink = torch.linspace(-1.0, 1.0, cfg.n_heads)
    layer = SimpleNamespace(
        layer_id=0,
        mode=CSA2LayerPlan.build(
            cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids
        ).mode(0),
        attn_sink=sink,
        attn_sink_loaded=False,
        _csa2_ctx=SimpleNamespace(x=torch.zeros(chunk_len, cfg.hidden_size)),
    )
    query = torch.zeros(chunk_len, cfg.n_heads, cfg.head_dim)
    current_kv = values[prefix_len:, None].expand(-1, cfg.head_dim).contiguous()
    visible = values[prefix_len - cfg.window_size + 1 : prefix_len + 1].sum()

    absent = backend.execute_csa2_layer(query, current_kv, layer)
    torch.testing.assert_close(absent, (visible / cfg.window_size).expand_as(absent))

    layer.attn_sink_loaded = True
    loaded = backend.execute_csa2_layer(query, current_kv, layer)
    expected_loaded = visible / (cfg.window_size + torch.exp(sink))
    torch.testing.assert_close(loaded, expected_loaded[None, :, None].expand_as(loaded))


def test_compressor_cached_long_chunk_preserves_partial_group() -> None:
    lm = _make_forward_lm(torch.float32, seed=113)
    cfg, model = lm.cfg, lm.model
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, 1, torch.float32)
    backend.bind_kv_caches(caches)
    layer = model.layers[2].self_attn
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache
    shared = CSA2SharedRuntime()

    def run(x: torch.Tensor, start: int) -> None:
        end = start + x.shape[0]
        positions = torch.arange(start, end)
        dsa = backend._builder.build(
            multi_block_tables=[torch.tensor([[0, 1]], dtype=torch.int32)]
            + [torch.tensor([[0]], dtype=torch.int32)] * 2,
            kv_seq_lens=[end],
            q_seq_lens=[x.shape[0]],
            positions=positions,
            is_prefill=False,
            is_chunked_prefill=True,
        )
        ctx = Csa2LayerContext(
            x=x, qr=torch.zeros(x.shape[0], cfg.q_lora_rank), positions=positions, cos_sin_cache=cos_sin
        )
        backend._run_compressor(
            layer, ctx, dsa, caches[2], mapping, [x.shape[0]], torch.tensor([end], dtype=torch.int64), shared
        )

    prefix = torch.randn(3, cfg.hidden_size)
    chunk = torch.randn(2 * cfg.window_size + 3, cfg.hidden_size)
    old_kv, old_score = layer.compress_kv_score(prefix)
    new_kv, new_score = layer.compress_kv_score(chunk)
    members = torch.stack((old_kv[2], new_kv[0]))
    gates = torch.stack((old_score[2], new_score[0])).softmax(dim=0)
    latent = layer.compress_rmsnorm((members * gates).sum(dim=0, keepdim=True))
    expected = _expected_fp4_e4m3(apply_interleaved_rope(latent, torch.tensor([2]), cos_sin), 16)[0]
    run(prefix, 0)
    run(chunk, 3)
    assert caches[2].key is not None
    torch.testing.assert_close(caches[2].key[0, 1], expected)


@pytest.mark.parametrize("counts", [(1, 3), (3, 1), (1, 1)])
@torch.inference_mode()
def test_moe_dp_aligns_expert_rows_and_restores_local_output(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, int]
) -> None:
    torch.manual_seed(129)
    cfg = DeepseekV41Config.from_dict(
        {
            **_DSV41_TINY_CONFIG,
            "hidden_size": 8,
            "moe_intermediate_size": 6,
            "n_routed_experts": 2,
            "n_activated_experts": 2,
        }
    )
    baseline = DeepseekV41MoE(cfg, torch.float32, torch.device("cpu"))
    for parameter in baseline.parameters():
        parameter.normal_(std=0.3)
    baseline.gate_bias_vl_loaded = True
    hidden = torch.randn(sum(counts), cfg.hidden_size)
    ids = torch.tensor([cfg.image_token_id if i % 2 else 1 for i in range(sum(counts))])
    full_output = baseline(hidden, ids)
    shared = dq._swiglu_clamped(
        hidden, baseline.shared_w1(hidden), baseline.shared_w3(hidden), baseline.shared_w2, baseline.swiglu_limit
    )
    routed = full_output - shared
    snapshot = baseline.state_dict()
    monkeypatch.setattr(
        moe_parallel,
        "get_forward_context",
        lambda: SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=counts)),
    )

    def run_rank(rank: int, count: int, offset: int) -> None:
        local_ref = DeepseekV41MoE(cfg, torch.float32, torch.device("cpu"))
        local_ref.load_state_dict(snapshot)
        local_ref.gate_bias_vl_loaded = True
        local_ref.experts_w2[1 - rank].zero_()
        expected_partial = local_ref(hidden, ids) - shared
        replica_cfg = replace(cfg, dp_size=2, dp_rank=rank, ep_size=2, ep_rank=rank)
        replica = DeepseekV41MoE(replica_cfg, torch.float32, torch.device("cpu"))
        replica.load_state_dict(
            {
                name: value[rank : rank + 1] if name.startswith("experts_w") else value
                for name, value in snapshot.items()
            }
        )
        replica.gate_bias_vl_loaded = True
        logits, scores = baseline._selection_logits(hidden, ids)
        gathered_values = (hidden, torch.cat((logits, scores), dim=-1))
        gathers: list[torch.Tensor] = []
        reductions: list[torch.Tensor] = []

        def gather(local: torch.Tensor, execution_counts: tuple[int, ...], dp_rank: int) -> tuple[torch.Tensor, int]:
            assert tuple(execution_counts) == counts
            assert dp_rank == rank
            expected_gathered = gathered_values[len(gathers)]
            torch.testing.assert_close(local, expected_gathered[offset : offset + count])
            gathers.append(local)
            return expected_gathered.clone(), offset

        def reduce(partial: torch.Tensor) -> None:
            torch.testing.assert_close(partial, expected_partial)
            reductions.append(partial.clone())
            partial.copy_(routed)

        monkeypatch.setattr(distributed, "gather_dp_execution_tokens", gather)
        monkeypatch.setattr(distributed, "moe_ep_all_reduce", reduce)
        actual = replica(hidden[offset : offset + count], ids[offset : offset + count])
        torch.testing.assert_close(actual, full_output[offset : offset + count])
        assert len(gathers) == 2
        assert len(reductions) == 1

    offset = 0
    for rank, count in enumerate(counts):
        run_rank(rank, count, offset)
        offset += count


@pytest.mark.parametrize("quant_mode", [V41_QUANT_NONE, V41_QUANT_FP8, V41_QUANT_ASCEND])
def test_attention_backend_reuses_model_quant_mode(monkeypatch: pytest.MonkeyPatch, quant_mode: str) -> None:
    config = _DSV41_FORWARD_CONFIG.copy()
    config.pop("candidate_topk_blocks")
    config.pop("candidate_block_size")
    cfg = DeepseekV41Config.from_dict(config)
    attention = Attention(
        num_heads=cfg.n_heads,
        num_kv_heads=1,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim**-0.5,
        sliding_window=cfg.window_size,
        layer_id=0,
    )
    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    monkeypatch.setattr(dq, "_detect_v41_quant_mode", lambda _: pytest.fail("checkpoint format was reread"))
    backend = executor_module._create_attention_backend(
        attention,
        torch.device("cpu"),
        torch.float32,
        config,
        quant_mode=quant_mode,
    )
    assert isinstance(backend, Csa2AttentionBackend)
    assert backend.moe_graph_capturable == (quant_mode == V41_QUANT_ASCEND)
    assert backend.candidate_topk_blocks == cfg.candidate_topk_blocks
    assert backend.candidate_block_size == cfg.candidate_block_size


@pytest.mark.parametrize("ctx_lens", [[0, 1], [1, 0], [0, 2], [2, 0], [0, 0]])
def test_compressor_mixed_padding_preserves_live_rows(ctx_lens: list[int]) -> None:
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, n_seqs=2, dtype=torch.float32)
    backend.bind_kv_caches(caches)
    layer = lm.model.layers[2].self_attn
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    positions = torch.tensor([max(length - 1, 0) for length in ctx_lens])
    dsa = backend._builder.build(
        multi_block_tables=[
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            torch.tensor([[0], [1]], dtype=torch.int32),
            torch.tensor([[0], [1]], dtype=torch.int32),
        ],
        kv_seq_lens=ctx_lens,
        q_seq_lens=[1, 1],
        positions=positions,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    ctx = Csa2LayerContext(
        x=torch.randn(2, cfg.hidden_size),
        qr=torch.randn(2, cfg.q_lora_rank),
        positions=positions,
        cos_sin_cache=lm.model.compress_rotary.cos_sin_cache,
    )
    before = {
        name: getattr(caches[2], name).clone() for name in ("key", "index", "compress_kv_state", "compress_score_state")
    }
    backend._run_compressor(layer, ctx, dsa, caches[2], mapping, [1, 1], dsa.seq_lens, CSA2SharedRuntime())
    kv, scores = layer.compress_kv_score(ctx.x)
    for seq, length in enumerate(ctx_lens):
        if length == 0:
            for name in ("key", "index"):
                assert torch.equal(getattr(caches[2], name)[seq], before[name][seq])
            for name in ("compress_kv_state", "compress_score_state"):
                assert torch.equal(getattr(caches[2], name)[2 * seq : 2 * seq + 2], before[name][2 * seq : 2 * seq + 2])
        elif length % layer.mode.compress_ratio:
            slot = 2 * seq * cfg.window_size + length - 1
            assert torch.allclose(caches[2].compress_kv_state.view(-1, cfg.head_dim)[slot], kv[seq], atol=1e-6)
            assert torch.allclose(caches[2].compress_score_state.view(-1, cfg.head_dim)[slot], scores[seq], atol=1e-6)
        else:
            for name in ("compress_kv_state", "compress_score_state"):
                assert torch.equal(getattr(caches[2], name), before[name])
            assert torch.count_nonzero(caches[2].key[seq, 0]) > 0
            assert torch.count_nonzero(caches[2].index[seq, 0]) > 0


def test_load_weights_rejects_invalid_engram_scale_before_lookup() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    tensors = _build_tiny_checkpoint(cfg)
    tensors["layers.1.engram.embed.scale"][0, 0] = 255
    with pytest.raises(ValueError, match="ue8m0 scale byte 0xFF"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_engram_lookup_does_not_read_device_values_in_python(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    tensors = _build_tiny_checkpoint(cfg)
    tensors["layers.1.engram.embed.weight"].fill_(0x38)
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)
    table = lm.model.layers[1].engram.embed_tokens
    ids = torch.zeros((1, table.n_hash_cols), dtype=torch.int64)
    expected = table.lookup(ids)
    assert torch.isfinite(expected).all()

    def reject_bool(_: torch.Tensor) -> None:
        raise AssertionError("lookup must not synchronize a device scalar")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", reject_bool)
        actual = table.lookup(ids)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("max_model_len", [512, 513, 4096, 4097])
def test_graph_tables_preserve_logical_swa_and_token_capacity(max_model_len: int) -> None:
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    window = cfg.window_size
    backend = _make_csa2_backend(cfg)
    columns = (max_model_len + window - 1) // window + 1
    tables = backend.create_graph_block_tables(2, max_model_len, columns)
    assert tables[0].shape == (2, columns)
    storage = tables[0].data_ptr()
    backend.bind_kv_caches(_allocate_v41_layer_caches(cfg, n_seqs=2, dtype=torch.float32))
    static = SimpleNamespace(multi_block_tables=tables)
    runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
    swa_flat = torch.arange(5 * window).reshape(-1, 1)
    for length in (window - 1, window, window + 1, 2 * window - 1, 2 * window, 2 * window + 1, max_model_len):
        live_begin = max(0, length - window) // window
        page_count = (length + window - 1) // window
        source = torch.full((2, page_count), -1, dtype=torch.int32)
        for seq in range(2):
            for column in range(live_begin, page_count):
                source[seq, column] = 1 + 2 * seq + column % 2
        runner._fill_dsa_block_tables(
            static, SimpleNamespace(multi_block_tables=(source, *([None] * (len(tables) - 1)))), 2
        )
        assert tables[0].data_ptr() == storage
        dsa = backend._builder.build(
            multi_block_tables=tables,
            kv_seq_lens=[length, length],
            q_seq_lens=[1, 1],
            positions=torch.tensor([length - 1, length - 1]),
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=columns,
        )
        positions = torch.arange(max(0, length - window), length)
        for seq in range(2):
            slots = backend._gather_swa_positions(swa_flat, dsa.block_tables[0][0], seq, positions, window)[:, 0]
            expected_blocks = source[seq, positions // window].long()
            assert torch.equal(slots, expected_blocks * window + positions % window)
    for table, group in zip(tables[1:], backend.group_infos[1:], strict=True):
        tokens_per_block = group.ratio * group.block_size
        assert table.shape == (2, (max_model_len + tokens_per_block - 1) // tokens_per_block)


@pytest.mark.parametrize("quant_mode", ["none", "ascend"])
@pytest.mark.parametrize(("explicit", "moe_width", "expected"), [(64, 32, 64), (-1, 32, 32), (0, 32, 32), (0, 0, 128)])
def test_dense_layers_honor_explicit_intermediate_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quant_mode: str, explicit: int, moe_width: int, expected: int
) -> None:
    config = {**_DSV41_ASCEND_CONFIG, "intermediate_size": explicit, "moe_intermediate_size": moe_width}
    if quant_mode == "ascend":
        lm = _make_ascend_lm(tmp_path, monkeypatch, config)
        lm.load_weights([_FakeStateDict(_ascend_tensors(config=config))], 0, 1)
        assert lm.model.layers[0].mlp.down_proj.weight.shape == (config["hidden_size"], expected)
    else:
        lm = DeepseekV41ForCausalLM({**config, "device": "cpu", "dtype": "float32"})
        assert lm.model.layers[0].mlp.w1.weight.shape == (expected, config["hidden_size"])
        assert lm.model.layers[0].mlp.w2.weight.shape == (config["hidden_size"], expected)


@pytest.mark.parametrize("quant_mode", ["none", "ascend"])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("sink_layout", ["global", "local", "absent"])
def test_attention_extras_preserve_sink_tp_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quant_mode: str, tp_rank: int, sink_layout: str
) -> None:
    from xllm.python.model_loader import W8A8WeightLoader

    config = {**_DSV41_ASCEND_CONFIG, "tp_size": 2, "tp_rank": tp_rank}
    if quant_mode == "ascend":
        lm = _make_ascend_lm(tmp_path, monkeypatch, config)
    else:
        lm = DeepseekV41ForCausalLM({**config, "device": "cpu", "dtype": "float32"})
    tensors = _ascend_tensors()
    global_sink = torch.tensor([0.25, 0.75])
    if sink_layout != "absent":
        tensors["layers.0.attn.attn_sink"] = (
            global_sink if sink_layout == "global" else global_sink.narrow(0, tp_rank, 1)
        )
    loader = W8A8WeightLoader(lm, [_FakeStateDict(tensors)], 2, tp_rank)
    attn = lm.model.layers[0].self_attn

    def no_indexer_projection(source: str, target: str) -> None:
        pytest.fail(f"SWA-only layer unexpectedly loaded indexer projection {source} into {target}")

    lm._load_attention_extras(loader, "layers.0.", "model.layers.0.", attn, no_indexer_projection)
    assert attn.attn_sink_loaded == (sink_layout != "absent")
    torch.testing.assert_close(
        attn.attn_sink, global_sink[tp_rank : tp_rank + 1] if attn.attn_sink_loaded else torch.zeros(1)
    )


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_swa_first_prefill_ignores_unwritten_cache_rows(monkeypatch: pytest.MonkeyPatch, poison: float) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, 1, torch.float32)
    caches[0].swa.fill_(poison)
    backend.bind_kv_caches(caches)
    length = 3
    dsa = backend._builder.build(
        multi_block_tables=[torch.tensor([[0, 1]], dtype=torch.int32)] * 3,
        kv_seq_lens=[length],
        q_seq_lens=[length],
        positions=torch.arange(length),
        is_prefill=True,
        is_chunked_prefill=False,
    )
    monkeypatch.setattr(backend, "_current_forward_metadata", lambda: SimpleNamespace(dsa_metadata=dsa))
    layer = SimpleNamespace(
        layer_id=0,
        mode=CSA2LayerPlan.build(
            cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids
        ).mode(0),
        attn_sink=torch.zeros(cfg.n_heads),
        attn_sink_loaded=False,
        _csa2_ctx=SimpleNamespace(x=torch.zeros(length, cfg.hidden_size)),
    )
    query = torch.zeros(length, cfg.n_heads, cfg.head_dim)
    current_kv = torch.ones(length, cfg.head_dim)
    actual = backend.execute_csa2_layer(query, current_kv, layer)
    torch.testing.assert_close(actual, torch.ones_like(actual))


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_compressed_attention_ignores_uncommitted_capacity(poison: float) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, 1, torch.float32)
    backend.bind_kv_caches(caches)
    length = 3
    layer_id = 2
    caches[layer_id].swa.fill_(poison)
    cache = caches[layer_id].key
    cache.fill_(poison)
    cache[0, 0].fill_(3)
    table = torch.tensor([[0]], dtype=torch.int32)
    dsa = backend._builder.build(
        multi_block_tables=[torch.tensor([[0, 1]], dtype=torch.int32), table, table],
        kv_seq_lens=[length],
        q_seq_lens=[length],
        positions=torch.arange(length),
        is_prefill=True,
        is_chunked_prefill=False,
    )
    dsa.is_acl_graph = True
    dsa.graph_token_capacity = 8
    layer = SimpleNamespace(
        layer_id=layer_id,
        mode=CSA2LayerPlan.build(
            cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids
        ).mode(layer_id),
        attn_sink=torch.zeros(cfg.n_heads),
        attn_sink_loaded=False,
    )
    shared = CSA2SharedRuntime(key_cache=cache, key_block_table=table)
    topk = torch.tensor([[[-1], [cfg.window_size + length], [cfg.window_size + length]]], dtype=torch.int32)
    query = torch.zeros(length, cfg.n_heads, cfg.head_dim)
    actual = backend._csa2_attention(
        query,
        torch.ones(length, cfg.head_dim),
        layer,
        dsa,
        [length],
        torch.tensor([length], dtype=torch.int32),
        topk,
        shared,
    )
    expected = torch.tensor([1.0, 5.0 / 3.0, 1.5])[:, None, None].expand_as(actual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_empty_dp_attention_has_no_history_or_cache_writes(
    monkeypatch: pytest.MonkeyPatch, graph_mode: bool, rows: int, poison: float
) -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)
    backend = _make_csa2_backend(cfg)
    caches = _allocate_v41_layer_caches(cfg, rows, torch.float32)
    caches[0].swa.fill_(poison)
    before = caches[0].swa.clone()
    backend.bind_kv_caches(caches)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.ones(rows, dtype=torch.int32),
        q_seq_lens_host=torch.ones(rows, dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_metadata=None,
        dsa_positions=torch.zeros(rows, dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=graph_mode,
        dsa_graph_block_table_cols=8,
        dsa_graph_token_capacity=8,
    )
    dsa = backend._build_dsa_metadata_for_forward(metadata)
    assert dsa.seq_lens.tolist() == [0] * rows
    assert dsa.seq_lens_q.tolist() == [1] * rows
    assert dsa.start_pos.tolist() == [-1] * rows
    for layer_slots in dsa.slot_mappings:
        for slots in layer_slots:
            assert torch.all(slots == -1)
    monkeypatch.setattr(backend, "_current_forward_metadata", lambda: SimpleNamespace(dsa_metadata=dsa))
    layer = SimpleNamespace(
        layer_id=0,
        mode=CSA2LayerPlan.build(
            cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids
        ).mode(0),
        attn_sink=torch.zeros(cfg.n_heads),
        attn_sink_loaded=False,
        _csa2_ctx=SimpleNamespace(x=torch.zeros(rows, cfg.hidden_size)),
    )
    query = torch.full((rows, cfg.n_heads, cfg.head_dim), poison)
    actual = backend.execute_csa2_layer(query, torch.full((rows, cfg.head_dim), poison), layer)
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    torch.testing.assert_close(caches[0].swa, before, equal_nan=True)


def test_engram_cache_reuses_resolved_token_ids() -> None:
    state = _tiny_engram_hash_state()
    reference = _tiny_engram_hash_state()
    state._cache = torch.zeros(8, dtype=torch.int32)
    ids = torch.tensor([-1, 7, state.token_map.numel() + 1])
    positions = torch.arange(ids.numel())
    dead = torch.tensor([True, False, True])
    boundaries = torch.tensor([0, ids.numel()])
    actual = state(
        ids,
        positions,
        boundaries,
        dead,
        positions.to(torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        8,
    )
    expected = reference(ids, positions, boundaries, dead, None, None, 0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state._cache[:3], torch.tensor([-1, int(state.token_map[7]), -1], dtype=torch.int32))

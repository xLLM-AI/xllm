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

import torch

from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4HyperConnection,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    _build_deepseek_v4_cp_context,
)
from xllm.python.models.deepseek_v32 import _swiglu_with_clamp
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
    assert cfg.compress_ratios == [0, 4, 128, 4]
    assert cfg.window_size == 128
    assert cfg.o_lora_rank == 1024
    assert cfg.o_groups == 8
    assert cfg.hc_mult == 4
    assert cfg.index_topk == 512
    assert cfg.rope_scaling_factor == 16.0


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
    layer = DeepseekV4DecoderLayer(
        cfg, layer_id=1, dtype=torch.float32, device=torch.device("cpu")
    )
    assert layer.self_attn.layer_id == 1
    assert layer.self_attn.indexer is not None
    assert layer.hc.hc_mult_local == 4


def test_moe_gate_state_matches_cpp_parameter_ownership() -> None:
    cfg_dict = dict(_DSV4_CONFIG)
    cfg_dict["n_hash_layers"] = 3
    cfg = DeepseekV4Config.from_dict(cfg_dict)

    hash_moe = DeepseekV4MoE(
        cfg, layer_id=2, dtype=torch.float32, device=torch.device("cpu")
    )
    non_hash_moe = DeepseekV4MoE(
        cfg, layer_id=3, dtype=torch.float32, device=torch.device("cpu")
    )

    hash_params = dict(hash_moe.named_parameters())
    non_hash_params = dict(non_hash_moe.named_parameters())
    assert "tid2eid" in hash_params
    assert not hash_params["tid2eid"].requires_grad
    assert "e_score_correction_bias" in non_hash_params
    assert not non_hash_params["e_score_correction_bias"].requires_grad


def test_clamped_swiglu_matches_cpp_activation_formula() -> None:
    x = torch.tensor(
        [[-20.0, 5.0, 20.0, 12.0, -15.0, 3.0]], dtype=torch.bfloat16
    )
    gate, up = x.chunk(2, dim=-1)
    expected = (
        torch.nn.functional.silu(gate.float().clamp_max(10.0))
        * up.float().clamp(min=-10.0, max=10.0)
    ).to(x.dtype)

    torch.testing.assert_close(_swiglu_with_clamp(x, 10.0), expected)


def test_v4_cp_context_matches_cpp_contiguous_split() -> None:
    positions = torch.arange(8, dtype=torch.int64)
    rank0 = _build_deepseek_v4_cp_context(
        2, 0, [5, 3], [9, 3], positions
    )
    rank1 = _build_deepseek_v4_cp_context(
        2, 1, [5, 3], [9, 3], positions
    )

    assert rank0.local_row_indices.tolist() == [0, 1, 2, 5, 6]
    assert rank1.local_row_indices.tolist() == [3, 4, 7]
    assert rank0.tokens_per_rank == [5, 3]
    assert rank0.restore_indices.tolist() == [0, 1, 2, 5, 6, 3, 4, 7]
    assert rank0.local_q_seq_lens == [3, 2]
    assert rank1.local_q_seq_lens == [2, 1]
    # cached prefix (9 - 5) plus one past the local segment end
    assert rank0.local_kv_seq_lens == [7, 2]
    assert rank1.local_kv_seq_lens == [9, 3]
    assert rank0.local_kv_cu_seq_lens.tolist() == [0, 7, 9]
    assert rank0.global_q_cu_seq_lens.tolist() == [0, 5, 8]


def test_v4_cp_context_keeps_empty_high_rank_segments() -> None:
    context = _build_deepseek_v4_cp_context(
        4,
        3,
        [2],
        [6],
        torch.arange(2, dtype=torch.int64),
    )

    assert context.local_row_indices.numel() == 0
    assert context.tokens_per_rank == [1, 1, 0, 0]
    assert context.local_q_seq_lens == [0]
    assert context.local_kv_seq_lens == [6]


def test_moe_uses_dedicated_group_sizes_under_cp() -> None:
    cfg_dict = dict(_DSV4_CONFIG)
    cfg_dict.update(
        tp_size=4,
        tp_rank=1,
        ep_size=8,
        ep_rank=3,
        moe_tp_size=1,
        moe_tp_rank=0,
        cp_size=2,
        cp_rank=0,
    )
    cfg = DeepseekV4Config.from_dict(cfg_dict)
    moe = DeepseekV4MoE(
        cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu")
    )

    assert moe.moe_tp_size == 1
    assert moe.moe_tp_rank == 0
    assert moe.start_expert_id == 3 * (cfg.n_routed_experts // cfg.ep_size)

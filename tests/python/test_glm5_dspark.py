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

"""MLA DSpark checkpoint, rotary and compressed-attention contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.glm5_dspark import Glm5DSparkConfig, Glm5DSparkForCausalLM, _apply_rope


def _config(**overrides: object) -> dict:
    values = dict(
        # The executor classifies glm5_dspark via the model_type predicate in
        # model_platform_support, so this matches what production sees.
        model_type="glm5_dspark",
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        vocab_size=16,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        max_position_embeddings=32,
        markov_rank=2,
        dtype="float32",
        device="cpu",
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return values


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Glm5DSparkForCausalLM:
    def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight

    monkeypatch.setattr(kernels, "rms_norm", rms_norm)
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", lambda weight: (weight, False))
    result = Glm5DSparkForCausalLM(_config(enable_confidence_head=True, confidence_head_with_markov=True))
    torch.manual_seed(17)
    with torch.no_grad():
        result.model.context_proj.load_weight(torch.randn(8, 16), 0)
        for parameter in result.parameters():
            parameter.normal_(0, 0.2)
    return result


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_speculators_config_uses_mla_dimensions_and_nested_rope() -> None:
    config = Glm5DSparkConfig.from_dict(
        {
            "speculators_model_type": "dspark",
            "markov_rank": 256,
            "transformer_layer_config": dict(
                _config(
                    hidden_size=6144,
                    num_attention_heads=64,
                    num_hidden_layers=5,
                    q_lora_rank=2048,
                    kv_lora_rank=512,
                    qk_nope_head_dim=192,
                    qk_rope_head_dim=64,
                    v_head_dim=256,
                    head_dim=64,
                    rope_parameters={"rope_theta": 8000000},
                    sliding_window=None,
                ),
            ),
        }
    )
    config.validate()
    assert config.head_dim == 256
    assert config.n_kv_heads == 1
    assert config.rope_theta == 8000000
    assert config.n_layers == 5
    assert config.kv_lora_rank + config.qk_rope_head_dim == 576


@pytest.mark.parametrize("override", [{"q_lora_rank": 0}, {"qk_rope_head_dim": 3}, {"attention_bias": True}])
def test_invalid_projection_geometry_is_rejected(override: dict) -> None:
    with pytest.raises(ValueError):
        Glm5DSparkForCausalLM(_config(**override))


def test_interleaved_rope_rotates_adjacent_pairs() -> None:
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    output = _apply_rope(value, torch.zeros(1, 2), torch.ones(1, 2), interleaved=True)
    torch.testing.assert_close(output, torch.tensor([[[-2.0, 1.0, -4.0, 3.0]]]))


def test_absorbed_mla_matches_expanded_dense_attention(model: Glm5DSparkForCausalLM) -> None:
    attention = model.model.layers[0].self_attn
    hidden = torch.randn(3, 8)
    cos, sin = torch.ones(3, 1), torch.zeros(3, 1)
    q = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(hidden))).view(3, 2, 4)
    latent, rope = attention.project_context(hidden, cos, sin)
    expanded = attention.kv_b_proj(latent.squeeze(1)).view(3, 2, 6)
    k_nope, values = expanded.split((2, 4), -1)
    keys = torch.cat((k_nope, rope.expand(-1, 2, -1)), -1)
    scores = torch.einsum("thd,shd->hts", q, keys) * attention.scale
    reference = attention.o_proj(torch.einsum("hts,shv->thv", scores.softmax(-1), values).reshape(3, 8))

    class _Backend:
        def execute_mla(
            self,
            query: torch.Tensor,
            query_rope: torch.Tensor,
            key: torch.Tensor,
            key_rope: torch.Tensor,
            layer: object,
        ) -> torch.Tensor:
            assert query.is_contiguous()
            scores = (
                torch.einsum("thr,shr->hts", query, key.expand(-1, 2, -1))
                + torch.einsum("thd,shd->hts", query_rope, key_rope.expand(-1, 2, -1))
            ) * attention.scale
            return torch.einsum("hts,shr->thr", scores.softmax(-1), key.expand(-1, 2, -1))

    with forward_context(ForwardContext(_Backend(), torch.device("cpu"), SimpleNamespace(), [])):
        actual = attention(hidden, cos, sin)
    torch.testing.assert_close(actual, reference)
    for weight in attention._kv_weights():
        assert weight.untyped_storage().data_ptr() == attention.kv_b_proj.weight.untyped_storage().data_ptr()


def test_context_writes_compressed_latent_and_rotary_cache_only(
    model: Glm5DSparkForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def scatter(
        slots: torch.Tensor,
        latent: torch.Tensor,
        rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> None:
        valid = slots >= 0
        latent_cache.flatten(0, 1)[slots[valid]] = latent[valid]
        rope_cache.flatten(0, 1)[slots[valid]] = rope[valid]
        calls.append((latent.shape, rope.shape))

    monkeypatch.setattr(kernels, "reshape_paged_cache", scatter)
    caches = [(torch.full((1, 8, 1, 4), 42.0), torch.full((1, 8, 1, 2), 43.0)) for _ in range(2)]
    hidden = torch.randn(3, 16)
    model.write_context_kv(hidden, torch.arange(3), torch.tensor([1, -1, 3]), caches, None)
    assert calls == [(torch.Size([3, 1, 4]), torch.Size([3, 1, 2]))] * 2
    for latent_cache, rope_cache in caches:
        assert torch.all(latent_cache[0, 2] == 42)
        assert torch.all(rope_cache[0, 2] == 43)


def test_attention_tp_loads_head_shards_and_replicated_latents(model: Glm5DSparkForCausalLM) -> None:
    tensors = {key: value.detach().clone() for key, value in model.model.layers[0].self_attn.state_dict().items()}
    cfg = Glm5DSparkConfig.from_dict(_config(tp_size=2, tp_rank=1))
    cfg.validate()
    from xllm.python.models.glm5_dspark import Glm5DSparkAttention

    attention = Glm5DSparkAttention(cfg, 0, torch.float32, torch.device("cpu"))
    attention.load_weights(ScopedWeightLoader([_StateDict(tensors)]), ParallelLoadContext(1, 2))
    torch.testing.assert_close(attention.q_a_proj.weight, tensors["q_a_proj.weight"])
    torch.testing.assert_close(attention.kv_a_proj_with_mqa.weight, tensors["kv_a_proj_with_mqa.weight"])
    torch.testing.assert_close(attention.q_b_proj.weight, tensors["q_b_proj.weight"][4:])
    torch.testing.assert_close(attention.kv_b_proj.weight, tensors["kv_b_proj.weight"][6:])
    torch.testing.assert_close(attention.o_proj.weight, tensors["o_proj.weight"][:, 4:])


def test_checkpoint_loads_original_mla_and_context_weight_names(model: Glm5DSparkForCausalLM) -> None:
    tensors = {name: value.detach().clone() for name, value in model.state_dict().items()}
    tensors = {name.removeprefix("model."): value for name, value in tensors.items()}
    for index in range(model.cfg.n_layers):
        fused = tensors.pop(f"layers.{index}.mlp.gate_up_proj.weight")
        tensors[f"layers.{index}.mlp.gate_proj.weight"], tensors[f"layers.{index}.mlp.up_proj.weight"] = fused.chunk(2)
    tensors["embed_tokens.weight"] = torch.randn(16, 8)
    tensors["lm_head.weight"] = torch.randn(16, 8)
    model.load_weights([_StateDict(tensors)], 0, 1)
    torch.testing.assert_close(model.model.context_projection_weight, tensors["context_proj.weight"])
    torch.testing.assert_close(model.model.context_norm.weight, tensors["context_norm.weight"])
    torch.testing.assert_close(model.model.embed_tokens.weight, tensors["embed_tokens.weight"])
    torch.testing.assert_close(model.lm_head.weight, tensors["lm_head.weight"])
    torch.testing.assert_close(model.markov_head.markov_w1.weight, tensors["markov_head.markov_w1.weight"])


@torch.inference_mode()
def test_real_mla_dspark_graph_and_context_replay_match_eager() -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.attention.backend import LayerCache
    from xllm.python.layers import HiddenParallelEmbedding
    from xllm.python.model_executor.executor import ModelExecutor
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata

    config = _config(
        model_type="glm5_dspark",
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        dtype="bfloat16",
        device="npu:0",
        enable_mla=True,
        num_speculative_tokens=8,
        python_graph_backend="aclgraph",
        enable_task_pipeline=True,
    )
    torch.manual_seed(89)
    model = Glm5DSparkForCausalLM(config)
    model.model.embed_tokens = HiddenParallelEmbedding(16, 64, 1, dtype=model.dtype, device=model.device)
    model.model.context_proj.load_weight(torch.randn(64, 128), tp_rank=0)
    for parameter in model.parameters():
        parameter.normal_(std=0.1)
    caches = [
        (
            torch.randn(1, 128, 1, 512, dtype=model.dtype, device=model.device),
            torch.randn(1, 128, 1, 64, dtype=model.dtype, device=model.device),
        )
        for _ in range(model.cfg.n_layers)
    ]
    executor = ModelExecutor(model, config, max_seqs_per_batch=1)
    executor.bind_kv_caches([LayerCache(*pair) for pair in caches])
    tokens = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=model.device)
    positions = torch.arange(8, dtype=torch.int32, device=model.device)
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=positions.clone(),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=model.device),
        q_seq_lens=torch.tensor([8], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([8], dtype=torch.int32, device=model.device),
        q_cu_seq_lens_host_values=[8],
        kv_seq_lens=torch.tensor([12], dtype=torch.int32, device=model.device),
        kv_seq_lens_host_values=[32],
        is_chunked_prefill=True,
    )
    stream = torch.npu.Stream(device=model.device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        executor.prepare_metadata(metadata)
        executor.warmup_prepared_graph(tokens, positions, metadata)
        for length in (12, 24, 8, 32):
            positions.copy_(torch.arange(length - 8, length, dtype=torch.int32, device=model.device))
            metadata.slot_mapping.copy_(positions)
            metadata.kv_seq_lens.fill_(length)
            executor.prepare_metadata(metadata)
            reference = executor.execute(tokens, positions, metadata).clone()
            actual = executor.execute(tokens, positions, metadata, enable_graph=True)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, reference)
        assert executor.prepared_graph_runner.prepared_replays == 4

        context = torch.randn(8, 128, dtype=model.dtype, device=model.device)
        context_positions = torch.arange(8, dtype=torch.int32, device=model.device)
        slots = torch.arange(40, 48, dtype=torch.int32, device=model.device)
        runner = executor.context_kv_graph_runner
        runner.execute_borrowed(context, context_positions, slots, warmup=True)
        slots[3:].fill_(-1)
        context.mul_(0.5)
        expected = model.write_context_kv(context, context_positions, slots, caches, None).clone()
        expected_cache = [(latent.clone(), rope.clone()) for latent, rope in caches]
        actual = runner.execute_borrowed(context, context_positions, slots, warmup=False)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)
        for actual_pair, expected_pair in zip(caches, expected_cache):
            for actual_cache, reference_cache in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual_cache, reference_cache)

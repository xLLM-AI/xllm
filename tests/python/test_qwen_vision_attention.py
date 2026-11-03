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

"""Checkpoint head partitioning and real native-width vision attention."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from tests.python.qwen3_5_test_utils import StateDict
from xllm.python import kernels
from xllm.python.model_loader import ScopedWeightLoader
from xllm.python.models.qwen3_vl import Qwen3VLVisionAttention, Qwen3VLVisionConfig, Qwen3VLVisionTransformer
from xllm.python.platform import current_platform


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_packed_checkpoint_preserves_each_ranks_qkv_heads(tp_size: int) -> None:
    cfg = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=4,
        intermediate_size=32,
        out_hidden_size=8,
        patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    full = Qwen3VLVisionTransformer(cfg, dtype=torch.float32, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(17)
    checkpoint = {
        name.replace("patch_embed.", "patch_embed.proj."): torch.randn(param.shape, generator=generator)
        for name, param in full.named_parameters()
    }
    full.load_weights(ScopedWeightLoader([StateDict(checkpoint)]))
    hidden = torch.randn(3, cfg.hidden_size, generator=generator)
    projected = full.blocks[0].attn.qkv(hidden).view(3, 3, cfg.num_heads, -1)
    expanded = full.blocks[0].mlp.linear_fc1(hidden)
    local_heads = cfg.num_heads // tp_size
    local_intermediate = cfg.intermediate_size // tp_size
    for rank in range(tp_size):
        sharded = Qwen3VLVisionTransformer(cfg, dtype=torch.float32, device=torch.device("cpu"), tp_size=tp_size)
        sharded.load_weights(ScopedWeightLoader([StateDict(checkpoint)]), rank)
        block = sharded.blocks[0]
        actual = block.attn.qkv(hidden).view(3, 3, local_heads, -1)
        torch.testing.assert_close(actual, projected[:, :, rank * local_heads : (rank + 1) * local_heads])
        torch.testing.assert_close(
            block.mlp.linear_fc1(hidden), expanded[:, rank * local_intermediate : (rank + 1) * local_intermediate]
        )
        torch.testing.assert_close(block.attn.proj.bias, full.blocks[0].attn.proj.bias)
        torch.testing.assert_close(block.mlp.linear_fc2.bias, full.blocks[0].mlp.linear_fc2.bias)
        torch.testing.assert_close(sharded.merger.linear_fc1.weight, full.merger.linear_fc1.weight)


def test_vision_heads_must_partition_evenly() -> None:
    with pytest.raises(ValueError, match="num_heads must be divisible"):
        Qwen3VLVisionAttention(72 * 6, 6, tp_size=4)


@pytest.mark.parametrize("layout", ["BSH", "TND"])
def test_other_attention_layouts_retain_torch_npu_dispatch(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    torch_npu = pytest.importorskip("torch_npu")
    query = torch.zeros(2, 3, 96) if layout == "BSH" else torch.zeros(3, 2, 128)
    q_lengths, kv_lengths = ([], []) if layout == "BSH" else ([1, 3], [2, 3])
    called = []

    def fusion_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor]:
        called.append(kwargs)
        assert q is query and k is query and v is query
        return (query,)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", fusion_attention)
    actual = kernels.vision_fusion_attention(
        query,
        query,
        query,
        actual_seq_qlen=q_lengths,
        actual_seq_kvlen=kv_lengths,
        num_heads=2,
        scale=0.125,
        input_layout=layout,
    )
    torch.testing.assert_close(actual, query)
    assert len(called) == 1
    assert called[0]["input_layout"] == layout
    assert called[0]["actual_seq_qlen"] == q_lengths
    assert called[0]["actual_seq_kvlen"] == kv_lengths


@pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1" or not current_platform.is_npu(),
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)
@pytest.mark.parametrize("head_dim", [64, 72, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[16], [16, 7, 11]])
@torch.inference_mode()
def test_native_width_attention_matches_per_frame_sdpa(head_dim: int, dtype: torch.dtype, lengths: list[int]) -> None:
    generator = torch.Generator().manual_seed(23)
    tensors = [torch.randn(sum(lengths), 4, head_dim, generator=generator).to(dtype) for _ in range(3)]
    expected = []
    start = 0
    for length in lengths:
        q, k, v = (x[start : start + length].float().transpose(0, 1) for x in tensors)
        expected.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1))
        start += length
    boundaries = list(torch.tensor(lengths).cumsum(0).tolist())
    query, key, value = (x.to("npu:0") for x in tensors)
    actual = kernels.vision_fusion_attention(
        query,
        key,
        value,
        actual_seq_qlen=boundaries,
        actual_seq_kvlen=boundaries,
        num_heads=4,
        scale=head_dim**-0.5,
    )
    assert actual.shape == query.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu().float(), torch.cat(expected), rtol=0.02, atol=0.015)

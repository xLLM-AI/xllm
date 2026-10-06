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

import os

import pytest
import torch

from xllm.python.models import glm5_next


def _indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
    )
    return glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device=device, dtype=torch.bfloat16)


@torch.inference_mode()
def test_pool_selection_length_input_matches_dense_fallback() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    indexer = _indexer(device)
    lengths = torch.tensor([3, 8, 17], dtype=torch.int32)
    key_valid = torch.arange(32)[None] < lengths[:, None]
    cache = torch.randn(8, 4, 1, 128, dtype=torch.bfloat16)
    tables = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    hidden = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, dtype=torch.bool)
    expected = indexer.select_topk(
        hidden, hidden, mask, 32, 32, key_valid=key_valid, pool_cache=cache, pool_block_table=tables
    )
    actual = indexer.select_topk(
        hidden, hidden, mask, 32, 32, kv_seq_lens=lengths, pool_cache=cache, pool_block_table=tables
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("always_select_tail", [False, True])
@torch.inference_mode()
def test_fused_pool_selection_does_not_materialize_history_mask(
    monkeypatch: pytest.MonkeyPatch, always_select_tail: bool
) -> None:
    pytest.importorskip("torch_npu")
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    indexer = _indexer(device)
    indexer.index_kpool_always_select_tail = always_select_tail
    lengths = torch.tensor([3, 8, 32768], device=device, dtype=torch.int32)
    hidden = torch.randn(3, 1, 16, device=device, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, device=device, dtype=torch.bool)
    cache = torch.zeros(8, 4, 1, 128, device=device, dtype=torch.bfloat16)
    tables = torch.ones(3, 2048, device=device, dtype=torch.int32)
    output = torch.zeros(3, 1, 11, device=device, dtype=torch.int32)
    calls = []

    def _pool_indexer(
        query: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        tail: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append((tail, kwargs["actual_seq_k"]))
        return output, None

    def _unexpected_arange(*_args: object, **_kwargs: object) -> None:
        pytest.fail("fused paged selection must not construct a length-sized mask")

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _pool_indexer, raising=False)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    monkeypatch.setattr(torch, "arange", _unexpected_arange)
    actual = indexer.select_topk(
        hidden, hidden, mask, 32768, 32768, kv_seq_lens=lengths, pool_cache=cache, pool_block_table=tables
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0].cpu(), torch.tensor([3, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(calls[0][1].cpu(), torch.tensor([0, 2, 8192], dtype=torch.int32))
    output_width = 11 if always_select_tail else 8
    torch.testing.assert_close(actual, output[..., :output_width].long())

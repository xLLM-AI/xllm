# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the multimodal embedding scatter in Qwen3-VL.

``PyExecutorImpl`` calls ``get_input_embeddings(input_ids, image_embeds,
video_embeds)`` (each embedding optional). The method masks each modality's
placeholder-token positions (``image_token_id`` / ``video_token_id``) and
scatters the main + multiscale (deepstack) parts. These tests cover the pure
helper ``_scatter_multimodal`` and the method with synthetic tensors — no model
weights or compiled operators are needed (conftest stubs the platform binding).
"""

from __future__ import annotations

import types

import pytest
import torch

from xllm.python.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    _scatter_multimodal,
)

# Distinct placeholder token ids used by the stub.
IMAGE_TID = 100
VIDEO_TID = 101


def _make_embeds(n_rows: int, visual_dim: int, num_level: int, base: float) -> torch.Tensor:
    """Distinct, recognizable embeddings: (n_rows, visual_dim*(1+num_level))."""
    return (
        torch.arange(
            n_rows * visual_dim * (1 + num_level), dtype=torch.float32
        ).reshape(n_rows, -1)
        + base
    )


def test_scatter_image_and_video_disjoint() -> None:
    """Image rows land at image_mask, video rows at video_mask; deepstack accumulates both."""
    seq_len, visual_dim, num_level = 20, 8, 3
    inputs_embeds = torch.zeros(seq_len, visual_dim, dtype=torch.float32)
    deepstack_acc = torch.zeros(seq_len, num_level * visual_dim, dtype=torch.float32)

    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    image_mask[2:5] = True  # 3 image tokens
    video_mask = torch.zeros(seq_len, dtype=torch.bool)
    video_mask[10:12] = True  # 2 video tokens

    img_embeds = _make_embeds(3, visual_dim, num_level, base=100.0)
    vid_embeds = _make_embeds(2, visual_dim, num_level, base=200.0)

    _scatter_multimodal(
        inputs_embeds, deepstack_acc, img_embeds, image_mask, visual_dim, num_level
    )
    _scatter_multimodal(
        inputs_embeds, deepstack_acc, vid_embeds, video_mask, visual_dim, num_level
    )

    assert torch.equal(inputs_embeds[2:5], img_embeds[:, :visual_dim])
    assert torch.equal(inputs_embeds[10:12], vid_embeds[:, :visual_dim])
    assert torch.count_nonzero(inputs_embeds[0:2]) == 0
    assert torch.count_nonzero(inputs_embeds[5:10]) == 0
    assert torch.count_nonzero(inputs_embeds[12:]) == 0

    ds = deepstack_acc.view(seq_len, num_level, visual_dim)
    assert torch.equal(ds[2:5].reshape(3, -1), img_embeds[:, visual_dim:])
    assert torch.equal(ds[10:12].reshape(2, -1), vid_embeds[:, visual_dim:])
    assert torch.count_nonzero(ds[0:2]) == 0


def test_scatter_subset() -> None:
    """A subset mask scatters only the matching rows (used per modality / per batch)."""
    seq_len, visual_dim, num_level = 10, 4, 3
    inputs_embeds = torch.zeros(seq_len, visual_dim, dtype=torch.float32)
    deepstack_acc = torch.zeros(seq_len, num_level * visual_dim, dtype=torch.float32)

    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[3:6] = True
    embeds = _make_embeds(3, visual_dim, num_level, base=1.0)

    _scatter_multimodal(
        inputs_embeds, deepstack_acc, embeds, mask, visual_dim, num_level
    )

    assert torch.equal(inputs_embeds[3:6], embeds[:, :visual_dim])
    assert torch.count_nonzero(inputs_embeds[0:3]) == 0
    assert torch.count_nonzero(inputs_embeds[6:]) == 0
    ds = deepstack_acc.view(seq_len, num_level, visual_dim)
    assert torch.equal(ds[3:6].reshape(3, -1), embeds[:, visual_dim:])


def _make_stub(seq_len: int, visual_dim: int, num_level: int) -> types.SimpleNamespace:
    """A lightweight stand-in for Qwen3VLForConditionalGeneration (no weights)."""
    model = types.SimpleNamespace()

    def embed_tokens(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.shape[0], visual_dim, dtype=torch.float32)

    model.embed_tokens = embed_tokens
    return types.SimpleNamespace(
        visual_dim=visual_dim,
        deepstack_num_level=num_level,
        multiscale_dim=visual_dim * num_level,
        image_token_id=IMAGE_TID,
        video_token_id=VIDEO_TID,
        model=model,
    )


def test_get_input_embeddings_both_modalities() -> None:
    seq_len, visual_dim, num_level = 12, 8, 3
    stub = _make_stub(seq_len, visual_dim, num_level)
    fn = types.MethodType(Qwen3VLForConditionalGeneration.get_input_embeddings, stub)

    input_ids = torch.zeros(seq_len, dtype=torch.long)
    input_ids[1:4] = IMAGE_TID  # 3 image tokens
    input_ids[7:9] = VIDEO_TID  # 2 video tokens
    img_embeds = _make_embeds(3, visual_dim, num_level, base=100.0)
    vid_embeds = _make_embeds(2, visual_dim, num_level, base=200.0)

    out = fn(input_ids, image_embeds=img_embeds, video_embeds=vid_embeds)

    assert out.shape == (seq_len, visual_dim)
    assert torch.equal(out[1:4], img_embeds[:, :visual_dim])
    assert torch.equal(out[7:9], vid_embeds[:, :visual_dim])
    assert torch.count_nonzero(out[4:7]) == 0

    ds = stub.model.deepstack_input_embeds
    assert isinstance(ds, list)
    assert len(ds) == num_level
    for t in ds:
        assert t.shape == (seq_len, visual_dim)
    ds_cat = torch.stack(ds, dim=1)  # (seq_len, num_level, visual_dim)
    assert torch.equal(ds_cat[1:4].reshape(3, -1), img_embeds[:, visual_dim:])
    assert torch.equal(ds_cat[7:9].reshape(2, -1), vid_embeds[:, visual_dim:])

    assert stub.model._inputs_embeds is out


def test_get_input_embeddings_no_modality_clears_attrs() -> None:
    """Decode/text steps (no modality) must clear the attributes for the runner."""
    stub = _make_stub(seq_len=5, visual_dim=8, num_level=3)
    fn = types.MethodType(Qwen3VLForConditionalGeneration.get_input_embeddings, stub)

    out = fn(torch.zeros(5, dtype=torch.long))

    assert stub.model._inputs_embeds is None
    assert stub.model.deepstack_input_embeds is None
    assert out.shape == (5, 8)


def test_get_input_embeddings_image_only() -> None:
    """Image-only path under the (image, video) signature."""
    seq_len, visual_dim, num_level = 8, 6, 3
    stub = _make_stub(seq_len, visual_dim, num_level)
    fn = types.MethodType(Qwen3VLForConditionalGeneration.get_input_embeddings, stub)

    input_ids = torch.zeros(seq_len, dtype=torch.long)
    input_ids[0:4] = IMAGE_TID
    img_embeds = _make_embeds(4, visual_dim, num_level, base=10.0)

    out = fn(input_ids, image_embeds=img_embeds)

    assert torch.equal(out[0:4], img_embeds[:, :visual_dim])
    assert len(stub.model.deepstack_input_embeds) == num_level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

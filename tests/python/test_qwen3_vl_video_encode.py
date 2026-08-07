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

"""Unit tests for the video token layout in the Qwen3-VL vision tower.

Video support reuses the same ``encode(pixel_values, grid_thw)`` path as still
images: a video is just an item with ``grid_thw[i][0] == T > 1``. Two invariants
must hold for the ViT to process video in one forward and for the C++ per-item
split (``grid_thw.prod(-1) / merge**2``) to slice embeddings correctly:

1. ``_compute_cu_seqlens`` emits one varlen attention segment per frame
   (``h*w`` tokens), repeated ``T`` times — so a mixed image+video batch keeps
   per-frame boundaries.
2. The post-merge token count per item is ``T * H * W / merge**2`` (identical
   to images, just with ``T`` frames).

Pure-tensor checks — no weights or compiled operators (conftest stubs the
platform binding).
"""

from __future__ import annotations

import pytest
import torch

from xllm.python.models.qwen3_vl import Qwen3VLVisionTransformer

# spatial_merge_size for Qwen3-VL (vision_config.spatial_merge_size).
MERGE = 2


def test_video_cu_seqlens_one_per_frame() -> None:
    """grid_t=4 → 4 frames of h*w tokens each → cu_seqlens with 4 equal segments."""
    grid_thw = torch.tensor([[4, 8, 8]])  # 4 frames, 8x8 patches each
    cu = Qwen3VLVisionTransformer._compute_cu_seqlens(grid_thw, torch.device("cpu"))

    frame_tokens = 8 * 8  # h * w
    expected = [0]
    for _ in range(4):
        expected.append(expected[-1] + frame_tokens)
    assert cu.tolist() == expected
    assert cu.dtype == torch.int32


def test_video_token_count_matches_split_formula() -> None:
    """Per-item post-merge token count == grid_thw.prod(-1) / merge**2 (C++ split sizes)."""
    grid_thw = torch.tensor([[4, 8, 8]])  # one video, 4 frames
    tokens = (grid_thw.prod(-1) / MERGE / MERGE).to(torch.int64)
    # 4 * 8 * 8 / 4 == 64
    assert tokens.tolist() == [64]
    # equals sum of per-frame merged tokens: T * (h/merge * w/merge)
    assert tokens[0].item() == 4 * (8 // MERGE) * (8 // MERGE)


def test_mixed_image_and_video_token_counts() -> None:
    """A batch of [image(grid_t=1), video(grid_t=2)] splits into [4, 8] embeddings."""
    grid_thw = torch.tensor([[1, 4, 4], [2, 4, 4]])
    tokens = (grid_thw.prod(-1) / MERGE / MERGE).to(torch.int64)
    # image: 1*4*4/4 == 4 ; video: 2*4*4/4 == 8
    assert tokens.tolist() == [4, 8]
    # cu_seqlens: image → 1 segment of 16; video → 2 segments of 16 each.
    cu = Qwen3VLVisionTransformer._compute_cu_seqlens(grid_thw, torch.device("cpu"))
    assert cu.tolist() == [0, 16, 32, 48]


def test_multiple_videos_token_counts() -> None:
    """Two videos with different frame counts split independently."""
    grid_thw = torch.tensor([[4, 4, 4], [2, 6, 6]])
    tokens = (grid_thw.prod(-1) / MERGE / MERGE).to(torch.int64)
    # video0: 4*4*4/4 == 16 ; video1: 2*6*6/4 == 18
    assert tokens.tolist() == [16, 18]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

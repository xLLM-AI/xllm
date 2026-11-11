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

"""Row alignment and request isolation contracts for Qwen3-VL metadata work."""

import pytest
import torch
from torch import nn

from xllm.python.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionTransformer,
    _VisionHostMetadata,
)


@pytest.fixture(params=["cpu", "npu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("NPU is unavailable")
    return torch.device(request.param)


def _model(levels: int, device: torch.device) -> Qwen3VLForConditionalGeneration:
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.image_token_id = 10
    model.video_token_id = 11
    model.visual_dim = 4
    model.deepstack_num_level = levels
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(64, dtype=torch.float32, device=device).reshape(16, 4)
    )
    decoder._inputs_embeds = None
    decoder.deepstack_input_embeds = None
    model.model = decoder
    return model


@pytest.mark.parametrize("levels", [0, 3])
def test_image_video_main_and_deepstack_rows(levels: int, device: torch.device) -> None:
    model = _model(levels, device)
    tokens = torch.tensor([1, 10, 10, 2, 11, 3], device=device)
    width = 4 * (levels + 1)
    image = torch.arange(2 * width, dtype=torch.float32, device=device).reshape(2, width) + 100
    video = torch.arange(width, dtype=torch.float32, device=device).reshape(1, width) + 500
    text = model.model.embed_tokens(tokens)
    expected = text.clone()
    expected[1:3] = image[:, :4]
    expected[4] = video[0, :4]

    actual = model.get_input_embeddings(tokens, image, video)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.model._inputs_embeds, expected)
    if levels == 0:
        assert model.model.deepstack_input_embeds is None
    else:
        for level, actual_level in enumerate(model.model.deepstack_input_embeds):
            expected_level = torch.zeros_like(text)
            expected_level[1:3] = image[:, 4 * (level + 1) : 4 * (level + 2)]
            expected_level[4] = video[0, 4 * (level + 1) : 4 * (level + 2)]
            torch.testing.assert_close(actual_level, expected_level)


def test_new_visual_rows_replace_previous_request_and_text_clears_state(device: torch.device) -> None:
    model = _model(3, device)
    for rows, value in [(2, 100.0), (2, 200.0), (1, 300.0)]:
        tokens = torch.tensor([1] + [10] * rows, device=device)
        image = torch.full((rows, 16), value, device=device)
        actual = model.get_input_embeddings(tokens, image)
        torch.testing.assert_close(actual[1:], image[:, :4])
        for level in model.model.deepstack_input_embeds:
            torch.testing.assert_close(level[0], torch.zeros(4, device=device))
            torch.testing.assert_close(level[1:], torch.full((rows, 4), value, device=device))

    tokens = torch.tensor([4, 5], device=device)
    torch.testing.assert_close(model.get_input_embeddings(tokens), model.model.embed_tokens(tokens))
    assert model.model._inputs_embeds is None
    assert model.model.deepstack_input_embeds is None


def test_empty_modality_rows_preserve_text_embeddings(device: torch.device) -> None:
    model = _model(3, device)
    tokens = torch.tensor([1, 2], device=device)
    actual = model.get_input_embeddings(tokens, torch.empty(0, 16, device=device), torch.empty(0, 16, device=device))
    torch.testing.assert_close(actual, model.model.embed_tokens(tokens))
    for level in model.model.deepstack_input_embeds or []:
        torch.testing.assert_close(level, torch.zeros(2, 4, device=device))


def test_frame_boundaries_preserve_image_video_order() -> None:
    grid = torch.tensor([[1, 4, 4], [2, 4, 4], [1, 4, 6]], dtype=torch.int32)
    actual = Qwen3VLVisionTransformer._compute_cu_seqlens(grid, torch.device("cpu"))
    torch.testing.assert_close(actual, torch.tensor([0, 16, 32, 48, 72], dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [[[1, 4, 4]], [[3, 4, 6]], [[1, 4, 4], [2, 4, 6]]])
def test_host_frame_boundaries_match_tensor_path(rows: list[list[int]], dtype: torch.dtype) -> None:
    # A strided grid also occurs when metadata is selected from a larger batch.
    storage = torch.zeros(len(rows), 6, dtype=dtype)
    grid = storage[:, ::2]
    grid.copy_(torch.tensor(rows, dtype=dtype))
    num_patches = int(grid.prod(dim=1).sum())
    metadata = _VisionHostMetadata.from_grid(grid, num_patches, spatial_merge_size=2)
    expected = Qwen3VLVisionTransformer._compute_cu_seqlens(grid, torch.device("cpu"))
    assert metadata.actual_seq == expected[1:].tolist()
    grid.fill_(1)
    assert metadata.grid_thw == tuple(tuple(row) for row in rows)


@pytest.mark.parametrize(
    "grid,num_patches",
    [
        (torch.ones(3, dtype=torch.int32), 1),
        (torch.empty(0, 3, dtype=torch.int32), 0),
        (torch.tensor([[1.0, 4.0, 4.0]]), 16),
        (torch.tensor([[0, 4, 4]]), 0),
        (torch.tensor([[1, 3, 4]]), 12),
        (torch.tensor([[1, 4, 4]]), 15),
        (torch.empty(1, 3, dtype=torch.int32, device="meta"), 16),
    ],
)
def test_host_metadata_rejects_inconsistent_inputs(grid: torch.Tensor, num_patches: int) -> None:
    with pytest.raises(ValueError):
        _VisionHostMetadata.from_grid(grid, num_patches, spatial_merge_size=2)

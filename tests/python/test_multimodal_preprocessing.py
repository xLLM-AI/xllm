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

"""Verify decoded image preprocessing against the PIL input contract."""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessorFast

from xllm.pybind import multimodal


@pytest.mark.parametrize("transpose", [False, True])
def test_decoded_tensors_match_pil_preprocessing(monkeypatch: pytest.MonkeyPatch, transpose: bool) -> None:
    processor = Qwen2VLImageProcessorFast(
        size={"shortest_edge": 1024, "longest_edge": 65536},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )
    monkeypatch.setattr(multimodal, "__cache_image_processor", lambda *args, **kwargs: processor)
    images = [(torch.arange(3 * 64 * width) % 256).to(torch.uint8).reshape(3, 64, width) for width in (96, 128)]
    if transpose:
        images = [image.transpose(1, 2) for image in images]
    originals = [image.clone() for image in images]
    pil_images = [Image.fromarray(np.asarray(image.permute(1, 2, 0).contiguous()), mode="RGB") for image in images]
    expected = processor.preprocess(pil_images, return_tensors="pt").data

    actual = multimodal.preprocess_tensors(images, "unused-local-model")

    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for image, original in zip(images, originals):
        torch.testing.assert_close(image, original, rtol=0, atol=0)

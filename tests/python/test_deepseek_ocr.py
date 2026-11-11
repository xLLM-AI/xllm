# Copyright 2025-2026 The xLLM Authors.
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
# ==============================================================================

"""DeepSeek-OCR preprocessing and token-grid contracts (CPU-only).

The C++ prompt processor (xllm/processors/deepseek_ocr_prompt_processor.cpp)
and the Python vision model (xllm/python/models/deepseek_ocr.py) share an
implicit contract: the number of ``<image>`` tokens a prompt expands to must
equal the number of embedding rows ``DeepseekVisionModel.encode`` produces
per image. These tests pin that contract (and the grid quirks it depends on)
without needing a checkpoint or an NPU.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from xllm.python.models.deepseek_ocr import (
    DeepseekOCRConfig,
    DeepseekVisionModel,
    _find_closest_aspect_ratio,
    _queries,
    preprocess_images,
)

# HF infer() constants for the Gundam recipe.
BASE_SIZE = 1024
CROP_SIZE = 640
PATCH = 16
DOWNSAMPLE = 4


def prompt_grid_tokens(base_size: int, crop_size: int, wc: int, hc: int) -> int:
    """Python mirror of DeepseekOCRPromptProcessor's per-image token count.

    NOTE: this mirrors the C++ formula in
    xllm/processors/deepseek_ocr_prompt_processor.cpp (grid_length());
    it is NOT linked to the C++ code, so changing the formula only there
    would leave these tests green. The end-to-end golden-set parity runs
    against the HF reference are the real cross-language guard; any formula
    change must update both sides plus the golden expectations.
    """
    q_base = math.ceil((base_size // PATCH) / DOWNSAMPLE)
    q_crop = math.ceil((crop_size // PATCH) / DOWNSAMPLE)
    length = q_base * (q_base + 1) + 1
    if wc > 1 or hc > 1:
        length += (q_crop * wc + 1) * (q_crop * hc)
    return length


@pytest.fixture(scope="module")
def vision_model() -> DeepseekVisionModel:
    """One shared fp32 model for the module: SAM-B + CLIP-L is ~400M params,
    so per-test construction would dominate the suite's time and memory.

    Eval-only. Deliberately NOT requires_grad_(False)/inference_mode: under
    torch_npu 2.9 both flip the SDPA dispatch onto a native path that
    segfaults the NPU ops library for these shapes (verified by bisect on
    the NPU host); forwards therefore keep autograd enabled and pay the
    activation memory, which stays acceptable for a single-image test.
    """
    model = DeepseekVisionModel(1280, dtype=torch.float32, device=torch.device("cpu"))
    model.eval()
    return model


class TestTokenGridContract:
    def test_grid_formula_matches_encoder_rows_no_crops(self, vision_model: DeepseekVisionModel) -> None:
        pixel_values = torch.randn(2, 3, BASE_SIZE, BASE_SIZE)
        spatial = torch.tensor([[1, 1], [1, 1]])
        embeds = vision_model.encode(pixel_values, None, spatial, BASE_SIZE, CROP_SIZE)
        per_image = prompt_grid_tokens(BASE_SIZE, CROP_SIZE, 1, 1)
        assert embeds.shape == (2 * per_image, 1280)

    def test_grid_formula_matches_encoder_rows_with_crops(self, vision_model: DeepseekVisionModel) -> None:
        # Two images: one 2x2-tiled, one untiled; crops concatenated in order.
        pixel_values = torch.randn(2, 3, BASE_SIZE, BASE_SIZE)
        crops = torch.randn(4, 3, CROP_SIZE, CROP_SIZE)
        spatial = torch.tensor([[2, 2], [1, 1]])
        embeds = vision_model.encode(pixel_values, crops, spatial, BASE_SIZE, CROP_SIZE)
        expected = prompt_grid_tokens(BASE_SIZE, CROP_SIZE, 2, 2) + prompt_grid_tokens(BASE_SIZE, CROP_SIZE, 1, 1)
        assert embeds.shape == (expected, 1280)

    def test_row_order_is_crops_then_global_then_separator(self, vision_model: DeepseekVisionModel) -> None:
        """The HF trained-with quirk: crop rows precede the global grid.

        Works on a throwaway copy (the shared module fixture must stay
        untouched). Zero every weight so refined rows carry only the
        projector bias (7.0), tag image_newline with -1.0 and
        view_seperator with -2.0: the crops block, the global block and the
        separator must appear in exactly that order.
        """
        model = copy.deepcopy(vision_model)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear | torch.nn.Conv2d):
                    torch.nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(model.projector.layers.bias, 7.0)
            torch.nn.init.constant_(model.image_newline, -1.0)
            torch.nn.init.constant_(model.view_seperator, -2.0)

            pixel_values = torch.zeros(1, 3, BASE_SIZE, BASE_SIZE)
            crops = torch.zeros(2 * 2, 3, CROP_SIZE, CROP_SIZE)
            spatial = torch.tensor([[2, 2]])
            embeds = model.encode(pixel_values, crops, spatial, BASE_SIZE, CROP_SIZE)

        n_crops_rows = (10 * 2 + 1) * (10 * 2)  # q_crop=10, 2x2 crop grid: 20 rows of 21
        n_global_rows = 16 * 17  # q_base=16: 16 rows of (16 tokens + 1 newline)
        assert embeds.shape[0] == n_crops_rows + n_global_rows + 1

        first = embeds[:, 0]
        # The two grids differ in row width, so the newline (-1.0) positions
        # pin the ORDER, not just the counts: a crops block has rows of
        # q_crop*wc+1 = 21 (newlines at 20, 41, ...), a global block has rows
        # of q_base+1 = 17 (newlines at n_crops_rows+16, +33, ...). A
        # [global, crops] swap would put 17-wide rows first and fail here.
        crops_newlines = torch.tensor([r * 21 + 20 for r in range(20)])
        global_newlines = torch.tensor([n_crops_rows + r * 17 + 16 for r in range(16)])
        assert first[crops_newlines].eq(-1.0).all()
        assert first[global_newlines].eq(-1.0).all()
        # No newline anywhere else, and the separator is strictly last.
        all_newlines = first.eq(-1.0)
        assert all_newlines.sum().item() == 36
        assert first[-1].item() == -2.0
        # Everything else is a refined (7.0) row.
        assert first.eq(7.0).sum().item() == 693 - 36 - 1


class TestCropGridInterleave:
    def test_crop_rows_interleave_across_tiles(self, vision_model: DeepseekVisionModel) -> None:
        """``_crop_grid_tokens`` must interleave crop rows across tiles.

        Tiles are emitted row-major (tile index i = crop row i//wc, crop
        column i%wc). The output stitches every tile's crop-row r side by
        side: output row r concatenates the r-th crop row of tiles 0..wc-1
        (crop columns), then row r+q continues with the next crop row. Whole
        crops back-to-back (row r entirely from one tile) is the bug this
        guards against. Each output row carries q*wc tokens plus a trailing
        image_newline embedding, so rows sit q*wc+1 apart.
        """
        model = vision_model
        wc, hc, q, dim = 2, 2, 10, 1280
        row_width = q * wc + 1  # +1: trailing image_newline position
        # Unique tag per (tile, crop row): (tile+1)*1000 + crop_row.
        tiles = torch.zeros(wc * hc, q * q, dim)
        for tile in range(wc * hc):
            for crop_row in range(q):
                tiles[tile, crop_row * q : (crop_row + 1) * q, 0] = (tile + 1) * 1000 + crop_row
        grid = model._crop_grid_tokens(tiles, wc, hc, q, dim)
        # Output row 0: crop row 0 of tile 0 (crop col 0) then tile 1 (col 1).
        assert grid[0, 0].item() == 1000
        assert grid[q, 0].item() == 2000  # tile 1, crop row 0
        # Output row q (after the newline): crop row 1 of tiles 0 and 1.
        assert grid[row_width, 0].item() == 1001
        assert grid[row_width + q, 0].item() == 2001
        # The newline embedding ends every output row: it never carries a tag.
        assert grid[row_width - 1, 0].item() not in (1000, 2000, 1001, 2001)


class TestDynamicPreprocess:
    def test_aspect_ratio_selection(self) -> None:
        assert _find_closest_aspect_ratio(1.0, [(1, 1), (1, 2), (2, 1)], 100, 100, 640) == (1, 1)
        assert _find_closest_aspect_ratio(2.0, [(1, 1), (1, 2), (2, 1)], 640, 320, 640) == (2, 1)

    def test_small_image_not_tiled(self) -> None:
        img = torch.zeros(3, 400, 400, dtype=torch.uint8)
        item = preprocess_images([img], BASE_SIZE, CROP_SIZE)[0]
        assert item["pixel_values"].shape == (1, 3, BASE_SIZE, BASE_SIZE)
        assert "pixel_values_crops" not in item
        assert item["image_spatial_crop"].tolist() == [[1, 1]]

    def test_large_image_tiled(self) -> None:
        img = torch.zeros(3, 1300, 900, dtype=torch.uint8)
        item = preprocess_images([img], BASE_SIZE, CROP_SIZE)[0]
        wc, hc = item["image_spatial_crop"][0].tolist()
        assert wc > 1 or hc > 1
        assert item["pixel_values_crops"].shape == (wc * hc, 3, CROP_SIZE, CROP_SIZE)

    def test_global_view_aspect_fill_uses_mean_color(self) -> None:
        """``ImageOps.pad`` scales to FIT INSIDE the square (contain) and
        pads the remainder with the mean color: a 200x100 red image becomes
        1024x512, so the top and bottom bands show the mean fill (127/255)
        that HF's BasicImageTransform uses."""
        img = torch.zeros(3, 100, 200, dtype=torch.uint8)
        img[0] = 255  # red in RGB channel order
        item = preprocess_images([img], BASE_SIZE, CROP_SIZE)[0]
        view = item["pixel_values"][0]
        assert view.shape == (3, BASE_SIZE, BASE_SIZE)
        # Center: the red image body ((255/255 - .5)/.5 == 1 in ch0, -1 else).
        assert view[0, BASE_SIZE // 2, BASE_SIZE // 2].item() == pytest.approx(1.0, abs=1e-3)
        assert view[1, BASE_SIZE // 2, BASE_SIZE // 2].item() == pytest.approx(-1.0, abs=1e-3)
        # Top band: pure mean fill ((127/255 - .5)/.5 ≈ -0.0039).
        assert view[0, 100, BASE_SIZE // 2].item() == pytest.approx(-0.0039, abs=1e-3)
        assert view[1, 100, BASE_SIZE // 2].item() == pytest.approx(-0.0039, abs=1e-3)


class TestConfig:
    def test_kv_heads_defaults_to_mha(self) -> None:
        cfg = DeepseekOCRConfig.from_dict({"hidden_size": 1280, "n_heads": 10})
        assert cfg.n_kv_heads == 10
        assert cfg.head_dim == 128

    def test_shared_expert_size_derived(self) -> None:
        cfg = DeepseekOCRConfig.from_dict({"moe_intermediate_size": 896, "n_shared_experts": 2})
        assert cfg.shared_expert_intermediate_size == 1792

    def test_moe_rejects_unsupported_routing(self) -> None:
        from xllm.python.models.deepseek_ocr import DeepseekMoE

        cfg = DeepseekOCRConfig.from_dict(
            {
                "routed_scaling_factor": 2.0,
                "moe_intermediate_size": 64,
                "n_shared_experts": 2,
                "shared_expert_intermediate_size": 128,
            }
        )
        with pytest.raises(NotImplementedError):
            DeepseekMoE(cfg, torch.float32, torch.device("cpu"))


class TestQueriesHelper:
    @pytest.mark.parametrize(
        "view,expected",
        [(1024, 16), (640, 10), (1280, 20), (512, 8)],
    )
    def test_queries_per_edge(self, view: int, expected: int) -> None:
        assert _queries(view, PATCH, DOWNSAMPLE) == expected

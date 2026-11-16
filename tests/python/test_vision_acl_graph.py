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

"""Vision encoder graph admission, layout keying and output ownership."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.model_executor.runners.vision_acl_graph import VisionAclGraphRunner
from xllm.python.models.qwen3_vl import Qwen3VLVisionTransformer, _host_grid_rows

_PIXEL_COLUMNS = 8


def _pixels(patches: int, fill: float = 1.0) -> torch.Tensor:
    return torch.full((patches, _PIXEL_COLUMNS), fill)


@pytest.fixture
def vision_model() -> SimpleNamespace:
    """A stand-in encoder: one output row per spatial-merge group of patches."""

    def forward(pixel_values: torch.Tensor, grid_rows: list[list[int]]) -> torch.Tensor:
        rows = sum(t * h * w for t, h, w in grid_rows) // 4
        return pixel_values[:rows, :2] * 2.0

    return SimpleNamespace(side_effect=Mock(side_effect=forward))


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, vision_model: SimpleNamespace) -> VisionAclGraphRunner:
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=Mock(return_value=Mock()),
            Stream=Mock(return_value=Mock()),
            stream=Mock(side_effect=lambda *args: nullcontext()),
            NPUGraph=Mock(side_effect=Mock),
            graph=Mock(side_effect=lambda *args, **kwargs: nullcontext()),
            graph_pool_handle=Mock(return_value=(0, 0)),
            mem_get_info=Mock(return_value=(1 << 40, 1 << 40)),
        ),
        raising=False,
    )
    # The runner only calls the encoder, so a plain callable stands in for it.
    model = vision_model.side_effect
    return VisionAclGraphRunner(model, torch.device("cpu"), max_graphs=2)


def test_layout_is_captured_once_and_replayed(runner: VisionAclGraphRunner) -> None:
    grid = [[1, 2, 2]]
    first = runner.execute(_pixels(4), grid)
    captures = torch.npu.graph.call_count
    second = runner.execute(_pixels(4, fill=3.0), grid)

    assert runner.captured_graphs == 1
    assert torch.npu.graph.call_count == captures, "a second call of a known layout must not capture again"
    assert first.shape == second.shape


def test_distinct_layouts_get_distinct_graphs(runner: VisionAclGraphRunner) -> None:
    runner.execute(_pixels(4), [[1, 2, 2]])
    runner.execute(_pixels(16), [[1, 4, 4]])
    assert runner.captured_graphs == 2

    # Two images of a captured geometry are a layout of their own: the grouped
    # encode is what fills the device, so it is what gets captured, and the
    # budget decides whether this one can be.
    captures = torch.npu.graph.call_count
    runner.execute(_pixels(8), [[1, 2, 2], [1, 2, 2]])
    assert runner.captured_graphs == 2, "the budget is two graphs"
    assert torch.npu.graph.call_count == captures
    assert runner.eager_encodes == 1


def test_budget_exhaustion_keeps_a_layout_eager(runner: VisionAclGraphRunner, vision_model: SimpleNamespace) -> None:
    runner.execute(_pixels(4), [[1, 2, 2]])
    runner.execute(_pixels(16), [[1, 4, 4]])
    captures = torch.npu.graph.call_count
    calls = vision_model.side_effect.call_count

    rejected = runner.execute(_pixels(36), [[1, 6, 6]])
    assert torch.npu.graph.call_count == captures
    assert vision_model.side_effect.call_count == calls + 1, "the over-budget layout must run the encoder eagerly"
    assert rejected.shape[0] == 9
    assert runner.eager_encodes == 1

    # An over-budget layout is not retried, and nothing is remembered per
    # layout: the budget never frees up, so such state would grow with the
    # variety of the traffic rather than with the deployment's geometry count.
    runner.execute(_pixels(36), [[1, 6, 6]])
    runner.execute(_pixels(64), [[1, 8, 8]])
    assert torch.npu.graph.call_count == captures
    assert runner.eager_encodes == 3
    assert runner.captured_graphs == 2
    assert not [name for name in vars(runner) if "reject" in name]


def test_replay_output_is_not_the_captured_buffer(runner: VisionAclGraphRunner) -> None:
    grid = [[1, 2, 2]]
    runner.execute(_pixels(4), grid)
    kept = runner.execute(_pixels(4, fill=2.0), grid)
    captured = next(iter(runner._graphs.values())).static_output
    # The framework keeps the encoder output across prefill chunks and in its
    # encoder cache, while the next replay overwrites the captured buffer, so
    # the two must not be the same storage.
    assert kept.data_ptr() != captured.data_ptr()
    assert torch.equal(kept, captured)


def test_pixel_rows_must_match_the_grid(runner: VisionAclGraphRunner) -> None:
    # A grid that does not describe these pixels would slice the wrong rows.
    with pytest.raises(ValueError, match="pixel rows"):
        runner.execute(_pixels(5), [[1, 2, 2]])


def test_frame_boundaries_match_the_device_formula() -> None:
    for grid_rows in ([[1, 2, 2]], [[1, 4, 6], [1, 2, 2]], [[3, 2, 4]], [[2, 4, 4], [1, 6, 6], [4, 2, 2]]):
        grid = torch.tensor(grid_rows, dtype=torch.int32)
        expected = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
        assert Qwen3VLVisionTransformer._compute_frame_boundaries(grid_rows) == expected.tolist()


def test_host_grid_rows_accepts_both_grid_forms() -> None:
    # The captured path resolves the grid to Host rows; the Qwen3.5 composer
    # still passes a device tensor, and both have to reach the same rows.
    rows = [[1, 4, 6], [2, 2, 2]]
    assert _host_grid_rows(rows) is rows
    converted = _host_grid_rows(torch.tensor(rows, dtype=torch.int32))
    assert converted == rows
    assert all(isinstance(value, int) for row in converted for value in row)

# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Test pose conversion and the real DROID checkpoint processor."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from xllm.vla.gr00t_processing import Gr00tDroidProcessor, _rotation_matrix


def test_rotation_orthogonalizes_nonunit_axes() -> None:
    matrix = _rotation_matrix(np.asarray([[2.0, 0.0, 0.0, 1.0, 3.0, 0.0]]))
    np.testing.assert_allclose(matrix, np.eye(3)[None], atol=1e-12)


@pytest.mark.parametrize("axes", [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
def test_rejects_degenerate_rotation(axes: list[float]) -> None:
    with pytest.raises(ValueError, match="axis|axes"):
        _rotation_matrix(np.asarray(axes))


@pytest.fixture
def processor() -> Gr00tDroidProcessor:
    checkpoint = os.environ.get("GR00T_CHECKPOINT")
    path = os.environ.get("GR00T_PROCESSOR_PATH")
    if not checkpoint or not path:
        pytest.skip("Set GR00T_CHECKPOINT and GR00T_PROCESSOR_PATH for checkpoint tests")
    return Gr00tDroidProcessor(Path(checkpoint), Path(path))


def test_zero_relative_motion_preserves_robot_pose(processor: Gr00tDroidProcessor) -> None:
    state = {
        "eef_9d": np.asarray([[[0.4, -0.2, 0.3, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        "gripper_position": np.asarray([[[0.5]]], dtype=np.float32),
        "joint_position": np.arange(7, dtype=np.float32).reshape(1, 1, 7) / 10,
    }
    physical = np.zeros((1, 40, 17))
    physical[..., 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    normalized = np.zeros((1, 40, 132), dtype=np.float32)
    offset = 0
    for key in processor.modalities["action"]["modality_keys"]:
        relative = key != "gripper_position"
        stats = processor.statistics["relative_action" if relative else "action"][key]
        low_key, high_key = ("q01", "q99") if processor.config["use_percentiles"] and not relative else ("min", "max")
        low, high = np.asarray(stats[low_key]), np.asarray(stats[high_key])
        width = low.shape[-1]
        normalized[..., offset : offset + width] = (physical[..., offset : offset + width] - low) / (high - low) * 2 - 1
        offset += width
    actions = processor.decode(normalized, state)
    np.testing.assert_allclose(actions["eef_9d"], np.repeat(state["eef_9d"], 40, axis=1), atol=2e-6)
    np.testing.assert_allclose(actions["joint_position"], np.repeat(state["joint_position"], 40, axis=1), atol=2e-6)

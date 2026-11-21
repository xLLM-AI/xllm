# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
# Inference transformations adapted from vLLM-Omni (Apache-2.0), commit
# 74b022ec52668b7914bd3da742649f7d44975b99, diffusion/models/gr00t/.
"""DROID observation processing and action decoding for GR00T N1.7."""

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from transformers import Qwen3VLProcessor


def _resize_short_edge(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / float(min(height, width))
    dimensions = (round(width * scale), round(height * scale))
    if dimensions == (width, height):
        return image
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def _rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    shape = rotation.shape[:-1]
    rows = np.asarray(rotation, dtype=np.float64).reshape(-1, 2, 3)
    first = rows[:, 0]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm == 0):
        raise ValueError("Invalid zero axis in DROID 6D rotation")
    first = first / first_norm
    second = rows[:, 1] - np.sum(first * rows[:, 1], axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm == 0):
        raise ValueError("Invalid parallel axes in DROID 6D rotation")
    second = second / second_norm
    matrices = np.stack((first, second, np.cross(first, second)), axis=-2)
    return Rotation.from_matrix(matrices).as_matrix().reshape(*shape, 3, 3)


class Gr00tDroidProcessor:
    """Process the checkpoint's DROID embodiment, using its saved statistics.

    Cameras are batched RGB uint8 arrays [1,T,H,W,3]. States are float32
    [1,1,D], and language is observation['language'][language_key] = [[text]].
    The caller supplies every historical frame requested by delta_indices.
    """

    embodiment_tag = "oxe_droid_relative_eef_relative_joint"

    def __init__(self, model_path: str | Path, processor_path: str | Path) -> None:
        path = Path(model_path)
        self.config = json.loads((path / "processor_config.json").read_text())["processor_kwargs"]
        self.modalities = self.config["modality_configs"][self.embodiment_tag]
        self.statistics = json.loads((path / "statistics.json").read_text())[self.embodiment_tag]
        self.embodiment_id = json.loads((path / "embodiment_id.json").read_text())[self.embodiment_tag]
        if not self.config.get("use_albumentations", False) or self.config.get("letter_box_transform", False):
            raise ValueError(
                "DROID processor requires the checkpoint's aspect-preserving albumentations eval transform"
            )
        if self.config.get("apply_sincos_state_encoding") or self.config.get("exclude_state"):
            raise ValueError("This DROID processor requires ordinary normalized state inputs")
        if self.modalities["state"]["modality_keys"] != ["eef_9d", "gripper_position", "joint_position"]:
            raise ValueError("Unsupported DROID state schema")
        expected_actions = [
            {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "eef_9d"},
            {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "gripper_position"},
            {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "joint_position"},
        ]
        if (
            self.modalities["action"]["modality_keys"] != self.modalities["state"]["modality_keys"]
            or self.modalities["action"]["action_configs"] != expected_actions
        ):
            raise ValueError("Unsupported DROID action schema")
        if not self.config.get("use_relative_action"):
            raise ValueError("DROID policy requires relative action statistics")
        if self.modalities["state"]["delta_indices"] != [0]:
            raise ValueError("DROID policy requires a single current robot state")
        self.language_key = self.modalities["language"]["modality_keys"][0]
        self.processor = Qwen3VLProcessor.from_pretrained(processor_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"

    def _image(self, frame: np.ndarray) -> Image.Image:
        size = self.config["shortest_image_edge"]
        image = _resize_short_edge(frame, size)
        height, width = image.shape[:2]
        fraction = self.config["crop_fraction"]
        crop_height, crop_width = max(1, int(height * fraction)), max(1, int(width * fraction))
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        image = _resize_short_edge(image[top : top + crop_height, left : left + crop_width], size)
        return Image.fromarray(np.ascontiguousarray(image))

    def prepare(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        states = []
        for key in self.modalities["state"]["modality_keys"]:
            stats = self.statistics["state"][key]
            values = observation["state"][key]
            dimension = len(stats["min"])
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float32
                or values.shape != (1, 1, dimension)
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"DROID state {key} must be finite float32 with shape [1,1,{dimension}]")
            low_key, high_key = ("q01", "q99") if self.config["use_percentiles"] else ("min", "max")
            low, high = np.asarray(stats[low_key]), np.asarray(stats[high_key])
            normalized = np.zeros_like(values)
            varying = ~np.isclose(high, low)
            normalized[..., varying] = (values[..., varying] - low[varying]) / (high[varying] - low[varying])
            normalized[..., varying] = 2 * normalized[..., varying] - 1
            states.append(np.clip(normalized, -1, 1))
        state = np.concatenate(states, axis=-1)
        if state.shape[-1] > self.config["max_state_dim"]:
            raise ValueError("DROID state exceeds checkpoint padding dimension")
        state = np.pad(state, ((0, 0), (0, 0), (0, self.config["max_state_dim"] - state.shape[-1])))
        cameras = self.modalities["video"]["modality_keys"]
        history = len(self.modalities["video"]["delta_indices"])
        for key in cameras:
            video = observation["video"][key]
            if (
                not isinstance(video, np.ndarray)
                or video.dtype != np.uint8
                or video.ndim != 5
                or video.shape[:2] != (1, history)
                or video.shape[-1] != 3
            ):
                raise ValueError(f"DROID camera {key} must be uint8 RGB with shape [1,{history},H,W,3]")
        images = [self._image(observation["video"][key][0, t]) for t in range(history) for key in cameras]
        language = observation["language"][self.language_key]
        if (
            not isinstance(language, list)
            or len(language) != 1
            or not isinstance(language[0], list)
            or len(language[0]) != 1
            or not isinstance(language[0][0], str)
        ):
            raise ValueError("DROID language must have the form [[instruction]]")
        text = language[0][0]
        if self.config["formalize_language"]:
            text = re.sub(r"[^\w\s]", "", text.lower())
        conversation = [
            {
                "role": "user",
                "content": [*[{"type": "image", "image": image} for image in images], {"type": "text", "text": text}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = dict(self.processor(text=[prompt], images=images, return_tensors="pt", padding=True))
        inputs["state"] = torch.from_numpy(state)
        inputs["embodiment_id"] = torch.tensor([self.embodiment_id], dtype=torch.int64)
        return inputs

    def decode(self, actions: np.ndarray, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        horizon = len(self.modalities["action"]["delta_indices"])
        if actions.shape != (1, horizon, self.config["max_action_dim"]) or not np.isfinite(actions).all():
            raise ValueError("Normalized DROID action tensor has an invalid shape or nonfinite values")
        output = {}
        offset = 0
        for key in self.modalities["action"]["modality_keys"]:
            # Upstream replaces the entire stats record for relative groups;
            # these use min/max even when use_percentiles is true globally.
            relative = key in ("eef_9d", "joint_position")
            stats = self.statistics["relative_action" if relative else "action"][key]
            low_key, high_key = ("q01", "q99") if self.config["use_percentiles"] and not relative else ("min", "max")
            low, high = np.asarray(stats[low_key]), np.asarray(stats[high_key])
            dimension = low.shape[-1]
            values = (np.clip(actions[..., offset : offset + dimension], -1, 1) + 1) / 2 * (high - low) + low
            offset += dimension
            if key == "joint_position":
                values = values + state[key][:, -1:, :]
            if key == "eef_9d":
                reference = state[key][:, -1:, :].astype(np.float64)
                reference_rotation = _rotation_matrix(reference[..., 3:])
                relative_rotation = _rotation_matrix(values[..., 3:])
                absolute_rotation = reference_rotation @ relative_rotation
                absolute_rotation = (
                    Rotation.from_matrix(absolute_rotation.reshape(-1, 3, 3)).as_matrix().reshape(1, horizon, 3, 3)
                )
                translation = (reference_rotation @ values[..., :3, None])[..., 0] + reference[..., :3]
                values = np.concatenate((translation, absolute_rotation[..., :2, :].reshape(1, horizon, 6)), axis=-1)
            output[key] = values.astype(np.float32)
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "embodiment_tag": self.embodiment_tag,
            "action_horizon": len(self.modalities["action"]["delta_indices"]),
            "action_keys": self.modalities["action"]["modality_keys"],
            "video": self.modalities["video"],
            "state": self.modalities["state"],
            "language_key": self.language_key,
        }

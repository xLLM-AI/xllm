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

"""Qwen3.5-family vision tower composed with the Python hybrid decoder."""

from __future__ import annotations

import torch

from xllm.python.model_loader import ScopedWeightLoader
from xllm.python.models.qwen3_5 import Qwen3_5ForCausalLM
from xllm.python.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionTransformer


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    def __init__(self, config: dict) -> None:
        vision_config = dict(config)
        vision_config.setdefault("mm_deepstack_visual_indexes", [])
        self.vision_cfg = Qwen3VLVisionConfig.from_dict(vision_config)
        if self.vision_cfg.deepstack_visual_indexes:
            raise ValueError("Qwen3.5 VLM does not support DeepStack visual indexes")
        if self.vision_cfg.out_hidden_size != config["hidden_size"]:
            raise ValueError("vision output dimension must match the language model hidden size")
        super().__init__(config)
        self.supports_mtp_kv_capacity_graph = self.cfg.num_experts == 0
        self.image_token_id = int(config.get("image_token_id", 248056))
        self.video_token_id = int(config.get("video_token_id", 248057))
        self.vision_model = Qwen3VLVisionTransformer(
            self.vision_cfg, dtype=self.dtype, device=self.device, tp_size=self.cfg.tp_size
        )

    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode image patches or video frames with the shared vision tower."""
        return self.vision_model(
            pixel_values.to(device=self.device, dtype=self.dtype),
            grid_thw.to(device=self.device, dtype=torch.int32),
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        video_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge the current chunk's visual rows selected by the C++ MM gather."""
        modalities = [(image_embeds, self.image_token_id), (video_embeds, self.video_token_id)]
        masks = []
        for embeds, token_id in modalities:
            if embeds is None:
                continue
            mask = input_ids == token_id
            expected_shape = (int(mask.sum().item()), self.cfg.hidden_size)
            if tuple(embeds.shape) != expected_shape:
                raise ValueError(f"visual embedding shape {tuple(embeds.shape)} does not match {expected_shape}")
            if expected_shape[0] == 0:
                continue
            masks.append((embeds, mask))

        inputs_embeds = self.model.embed_tokens(input_ids)
        for embeds, mask in masks:
            inputs_embeds.masked_scatter_(
                mask.unsqueeze(-1), embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            )
        return inputs_embeds

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        self.vision_model.load_weights(ScopedWeightLoader(state_dicts, src_prefixes=("model.visual.",)), tp_rank)
        super().load_weights(state_dicts, tp_rank, tp_size)

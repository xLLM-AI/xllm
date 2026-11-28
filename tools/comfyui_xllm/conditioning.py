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

"""Translate MiniMax-H3 text encoder output at the ComfyUI boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class H3Conditioning:
    prompt: str
    embedding: torch.Tensor
    token_tags: torch.Tensor

    @classmethod
    def from_tensors(cls, prompt: str, embedding: torch.Tensor, token_tags: torch.Tensor) -> H3Conditioning:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("MiniMax-H3 conditioning requires its original prompt")
        if not isinstance(embedding, torch.Tensor) or not embedding.is_floating_point():
            raise ValueError("MiniMax-H3 prompt_embed must be a floating point tensor")
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 2 or embedding.shape[0] == 0 or embedding.shape[1] != 5120:
            raise ValueError("MiniMax-H3 prompt_embed must have shape [L, 5120] or [1, L, 5120]")
        if not isinstance(token_tags, torch.Tensor) or token_tags.dtype != torch.int64:
            raise ValueError("MiniMax-H3 prompt_token_tags must be an int64 tensor")
        if token_tags.ndim != 1 or token_tags.shape[0] != embedding.shape[0]:
            raise ValueError("MiniMax-H3 requires one prompt_token_tag per embedding row")
        embedding = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        token_tags = token_tags.detach().cpu().contiguous()
        if not torch.isfinite(embedding).all():
            raise ValueError("MiniMax-H3 prompt_embed must contain finite values")
        if not torch.all(token_tags == 1):
            raise ValueError("xLLM Python MiniMax-H3 currently supports text-only conditioning (all token tags are 1)")
        return cls(prompt, embedding, token_tags)

    @classmethod
    def from_comfy(cls, prompt: str, conditioning: list[Any]) -> H3Conditioning:
        if len(conditioning) != 1:
            raise ValueError("xLLM MiniMax-H3 requires a single, unscheduled text conditioning")
        embedding, metadata = conditioning[0]
        if "start_percent" in metadata or "end_percent" in metadata:
            raise ValueError("Scheduled conditioning is not supported by xLLM MiniMax-H3")
        tags = metadata.get("minimax_token_tags")
        if tags is None:
            raise ValueError("Use ComfyUI's MiniMax-H3 CLIP encoder; minimax_token_tags is missing")
        return cls.from_tensors(prompt, embedding, tags)

    @classmethod
    def from_file(cls, path: Path) -> H3Conditioning:
        value = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_tensors(value["prompt"], value["prompt_embed"], value["prompt_token_tags"])

    def to_input(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_embed": {
                "name": "prompt_embed",
                "datatype": "FP32",
                "shape": list(self.embedding.shape),
                "contents": {"fp32_contents": self.embedding.flatten().tolist()},
                "parameters": {"prompt_token_tags": {"int64_tensor_param": self.token_tags.tolist()}},
            },
        }

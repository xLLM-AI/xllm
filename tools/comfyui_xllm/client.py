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

"""xLLM video generation protocol, independent of ComfyUI and native xLLM."""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .conditioning import H3Conditioning


@dataclass(frozen=True)
class XLLMServer:
    base_url: str
    model: str
    timeout_seconds: float = 3600

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
            raise ValueError("Server URL must be an http(s) base URL, without a query or fragment")
        if not self.model.strip():
            raise ValueError("The served model ID is required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive and finite")

    async def generate_h3(
        self,
        conditioning: H3Conditioning,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        steps: int,
        seed: int,
        output_directory: Path,
    ) -> tuple[Path, dict[str, Any]]:
        if width <= 0 or height <= 0 or steps <= 0 or num_frames <= 0:
            raise ValueError("Width, height, frames and steps must be positive")
        if not math.isfinite(fps) or fps <= 0 or num_frames / fps < 5 or (num_frames + 16) / fps > 15:
            raise ValueError("MiniMax-H3 requires 5–15 seconds, including its frame alignment margin")
        if not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative signed 64-bit integer")
        payload = {
            "model": self.model,
            "input": conditioning.to_input(),
            "parameters": {
                "size": f"{width}*{height}",
                "num_frames": num_frames,
                "fps": fps,
                "num_inference_steps": steps,
                "seed": seed,
                "sampling_rate": 32000,
                "num_videos_per_prompt": 1,
                "output_type": "base64",
            },
        }
        # A cancelled or timed-out request is never resubmitted: the server may
        # still be running the original generation.
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(self.base_url.rstrip("/") + "/v1/video/generation", json=payload) as response,
        ):
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"xLLM HTTP {response.status}: {text[:4096]}")
        result = json.loads(text)
        if "error" in result:
            raise RuntimeError(f"xLLM generation failed: {result['error']}")
        outputs = result["output"]["results"]
        if len(outputs) != 1:
            raise ValueError("Expected exactly one MiniMax-H3 video from xLLM")
        output = outputs[0]
        video = base64.b64decode(output["video"]["base64"], validate=True)
        if not video:
            raise ValueError("xLLM returned an empty video")
        metadata = {key: value for key, value in output.items() if key != "video"}
        metadata.update(request_id=result.get("id", ""), model=result.get("model", self.model))
        # The file must outlive this node: VIDEO and downstream cached nodes
        # read it lazily. ComfyUI owns its temporary directory's lifecycle.
        output_directory.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix="xllm_h3_", suffix=".mp4", dir=output_directory)
        path = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(video)
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path, metadata

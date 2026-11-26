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

"""ComfyUI Server Mode nodes for the Python MiniMax-H3 T2VA pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, io

from .client import XLLMServer
from .conditioning import H3Conditioning

if TYPE_CHECKING:
    from comfy.sd import CLIP

_SERVER = io.Custom("XLLM_SERVER")
_CONDITIONING = io.Custom("XLLM_H3_CONDITIONING")


class XLLMServerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XLLMServer",
            display_name="xLLM Server",
            category="xLLM",
            inputs=[
                io.String.Input("base_url", default="http://127.0.0.1:8000"),
                io.String.Input("model", default="minimax-h3"),
                io.Float.Input("timeout_seconds", default=3600, min=1),
            ],
            outputs=[_SERVER.Output()],
        )

    @classmethod
    def execute(cls, base_url: str, model: str, timeout_seconds: float) -> io.NodeOutput:
        return io.NodeOutput(XLLMServer(base_url, model, timeout_seconds))


class XLLMH3TextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XLLMH3TextEncode",
            display_name="xLLM MiniMax-H3 Text Encode",
            category="xLLM/conditioning",
            description="Use CLIPLoader with type minimax and matching H3 text-encoder weights.",
            inputs=[io.Clip.Input("clip"), io.String.Input("prompt", multiline=True)],
            outputs=[_CONDITIONING.Output()],
        )

    @classmethod
    def execute(cls, clip: CLIP, prompt: str) -> io.NodeOutput:
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(H3Conditioning.from_comfy(prompt, conditioning))


class XLLMH3ConditioningLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = sorted(path.name for path in Path(folder_paths.get_input_directory()).glob("*.pt"))
        return io.Schema(
            node_id="XLLMH3ConditioningLoader",
            display_name="xLLM MiniMax-H3 Conditioning Loader",
            category="xLLM/conditioning",
            description="Load a trusted H3 conditioning.pt containing prompt, prompt_embed and prompt_token_tags.",
            inputs=[io.Combo.Input("filename", options=files)],
            outputs=[_CONDITIONING.Output()],
        )

    @classmethod
    def execute(cls, filename: str) -> io.NodeOutput:
        return io.NodeOutput(H3Conditioning.from_file(Path(folder_paths.get_input_directory()) / filename))

    @classmethod
    def fingerprint_inputs(cls, filename: str) -> str:
        path = Path(folder_paths.get_input_directory()) / filename
        return hashlib.sha256(path.read_bytes()).hexdigest()


class XLLMH3Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XLLMH3Generate",
            display_name="xLLM Generate MiniMax-H3",
            category="xLLM",
            description="Generate one video with audio on a running xLLM Python H3 server. Cancel stops waiting; "
            "the server may finish the running generation. Results use ComfyUI caching; change seed to regenerate.",
            inputs=[
                _SERVER.Input("server"),
                _CONDITIONING.Input("conditioning"),
                io.Int.Input("width", default=1344, min=1),
                io.Int.Input("height", default=768, min=1),
                io.Int.Input("num_frames", default=124, min=1),
                io.Float.Input("fps", default=24.0, min=0.01),
                io.Int.Input("steps", default=50, min=1),
                io.Int.Input("seed", default=42, min=0, max=2**63 - 1, control_after_generate=True),
            ],
            outputs=[io.Video.Output(), io.String.Output(display_name="metadata")],
        )

    @classmethod
    async def execute(
        cls,
        server: XLLMServer,
        conditioning: H3Conditioning,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        steps: int,
        seed: int,
    ) -> io.NodeOutput:
        request = asyncio.create_task(
            server.generate_h3(
                conditioning,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                seed=seed,
                output_directory=Path(folder_paths.get_temp_directory()),
            )
        )
        try:
            while not request.done():
                comfy.model_management.throw_exception_if_processing_interrupted()
                await asyncio.wait({request}, timeout=0.5)
            path, metadata = await request
        finally:
            if not request.done():
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), json.dumps(metadata, ensure_ascii=False))


class XLLMExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [XLLMServerNode, XLLMH3TextEncode, XLLMH3ConditioningLoader, XLLMH3Generate]

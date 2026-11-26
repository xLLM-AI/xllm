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

"""Capture H3 denoising steps; schedulers and media decoders stay eager."""

from dataclasses import dataclass, replace

import torch

from scripts.logger import logger
from xllm.python.dit.minimax_h3_model import MiniMaxH3Denoiser, PackedConditioning


@dataclass
class _DenoiseGraph:
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, torch.Tensor]
    graph: object


class MiniMaxH3AclGraphRunner:
    def __init__(self, denoiser: MiniMaxH3Denoiser, device: torch.device) -> None:
        self._denoiser = denoiser
        self._device = device
        self._stream = torch.npu.Stream(device=device)
        self._pool = torch.npu.graph_pool_handle()
        self._layout_key: tuple[object, ...] | None = None
        self._graphs: dict[int, _DenoiseGraph] = {}
        self.captured_steps = 0
        self.replayed_steps = 0

    @torch.no_grad()
    def __call__(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        conditioning: PackedConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = conditioning.layout
        # These are all tensor values read by MiniMaxH3Denoiser.forward.
        # Positions were already converted to rotary tables before this call.
        inputs = (
            video,
            audio,
            timestep,
            timestep_indices,
            conditioning.text,
            *conditioning.rotary,
            layout.tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
        )
        layout_key = (
            tuple((tensor.shape, tensor.dtype) for index, tensor in enumerate(inputs) if index != 2),
            conditioning.local_start,
            conditioning.local_rows,
            conditioning.padded_rows,
        )
        if layout_key != self._layout_key:
            # Keep only one request layout and its one-/two-timestep graphs.
            # Retire outstanding replays before releasing their memory pool.
            torch.npu.current_stream(self._device).synchronize()
            self._graphs.clear()
            self._pool = torch.npu.graph_pool_handle()
            self._layout_key = layout_key
        entry = self._graphs.get(timestep.numel())
        if entry is None:
            entry = self._capture(inputs, conditioning)
            self._graphs[timestep.numel()] = entry
        for target, source in zip(entry.inputs, inputs, strict=True):
            target.copy_(source)
        # Capture records operations without producing the first prediction.
        entry.graph.replay()
        self.replayed_steps += 1
        # The scheduler consumes both predictions before the next replay.
        return entry.outputs

    def _capture(self, inputs: tuple[torch.Tensor, ...], conditioning: PackedConditioning) -> _DenoiseGraph:
        static = tuple(tensor.clone() for tensor in inputs)
        static_conditioning = replace(
            conditioning,
            text=static[4],
            rotary=(static[5], static[6]),
            layout=replace(
                conditioning.layout,
                tags=static[7],
                video_indices=static[8],
                audio_indices=static[9],
                text_indices=static[10],
            ),
        )
        caller_stream = torch.npu.current_stream(self._device)
        self._stream.wait_stream(caller_stream)
        with torch.npu.stream(self._stream):
            # Initialize operator workspaces and HCCL communicators outside capture.
            for _ in range(2):
                self._denoiser(*static[:4], static_conditioning)
            self._stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=self._pool, stream=self._stream, capture_error_mode="thread_local"):
                outputs = self._denoiser(*static[:4], static_conditioning)
        caller_stream.wait_stream(self._stream)
        self.captured_steps += 1
        logger.info(
            "Python MiniMax-H3 ACL graph captured: device=%s video_rows=%d audio_rows=%d text_rows=%d timesteps=%d",
            self._device,
            static[0].shape[1],
            static[1].shape[1],
            static[4].shape[1],
            static[2].numel(),
        )
        return _DenoiseGraph(static, outputs, graph)

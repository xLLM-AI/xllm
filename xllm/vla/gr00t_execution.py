# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Shared Flash eager execution and bounded action-head HIP graph replay."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.logger import logger
from xllm.vla.gr00t_action import Gr00tActionHead


@dataclass
class _FastGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class _Gr00tActionRunner:
    def __init__(self, head: Gr00tActionHead, execution: str) -> None:
        if head._flash_keys is None or execution not in ("eager", "graph"):
            raise ValueError("GR00T action runner requires Flash attention and eager or graph execution")
        self._head, self._execution = head, execution
        self._graphs: OrderedDict[tuple[int, int, int], _FastGraph] = OrderedDict()
        self.requests = 0
        self.captures = 0
        self.replays = 0

    def _forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        features, state, ids, image_mask, attention_mask, noise, text_indices, image_indices = inputs
        self._head._flash_keys.indices = (text_indices, image_indices)
        return self._head._forward(features, state, ids, image_mask, attention_mask, noise, None)

    @torch.inference_mode()
    @sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    def run(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        ids: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self._head._validate_inputs(features, state, ids, image_mask, attention_mask)
        shape = (1, self._head.config.action_horizon, self._head.config.max_action_dim)
        noise = (
            torch.randn(shape, device=features.device, dtype=features.dtype, generator=generator)
            if initial_noise is None
            else initial_noise
        )
        if noise.shape != shape or noise.dtype != features.dtype or noise.device != features.device:
            raise ValueError("Frozen initial noise differs in shape, dtype or device")
        text_indices, image_indices = self._head._prepare_flash_indices(image_mask, attention_mask)
        inputs = (features, state, ids, image_mask, attention_mask, noise, text_indices, image_indices)
        self.requests += 1
        if self._execution == "eager":
            output = self._forward(inputs)
            return output
        cache_key = (features.shape[1], text_indices.numel(), image_indices.numel())
        if cache_key not in self._graphs:
            static = tuple(value.clone() for value in inputs)
            stream = torch.cuda.Stream(device=features.device)
            stream.wait_stream(torch.cuda.current_stream(features.device))
            with torch.cuda.stream(stream):
                self._forward(static)
                for _ in range(2):
                    self._forward(static)
            torch.cuda.current_stream(features.device).wait_stream(stream)
            torch.cuda.synchronize(features.device)
            if len(self._graphs) == 2:
                self._graphs.popitem(last=False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._forward(static)
            self._graphs[cache_key] = _FastGraph(graph, static, output)
            self.captures += 1
            logger.info("Captured Flash action graph for length/text/image=%s", cache_key)
        entry = self._graphs[cache_key]
        self._graphs.move_to_end(cache_key)
        for target, source in zip(entry.inputs, inputs, strict=True):
            target.copy_(source)
        entry.graph.replay()
        self.replays += 1
        return entry.output.clone()

    def summary(self) -> dict[str, object]:
        return {
            "execution": self._execution,
            "requests": self.requests,
            "captures": self.captures,
            "replays": self.replays,
            "cache_keys": [list(key) for key in self._graphs],
            "mask_indices": "Recomputed from each actual observation and copied into static graph inputs",
        }

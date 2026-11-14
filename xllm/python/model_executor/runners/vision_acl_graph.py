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

"""ACL graphs for the Qwen3-VL vision encoder.

``enable_graph`` captures decode steps only, so the vision encoder runs eager
on every request that carries an image: one Host dispatch per operator, per
ViT layer, before the language model can start.

Every shape and every Host scalar inside the encoder is a function of the image
grid alone -- patch count, frame boundaries, rotary tables and interpolation
weights all follow from ``(t, h, w)``. Keying a graph on the grid therefore
pins all of them, so replay only has to supply new pixels, and the Host-side
attention lengths captured inside the graph stay correct without task updates.

The key is the whole encode layout: every image's ``(t, h, w)`` in order. The
encoder is separable per image, which would key the graph by geometry alone and
bound the graph count by the resolutions a deployment serves, but it is not
*efficient* per image: one image of 784 patches leaves the device unsaturated,
so replaying four of them in sequence takes 74% longer than the grouped encode
they replace. The grouped call is the unit that fills the device, so it is the
unit that gets captured, and replay then returns the same values the eager
encoder returns, bit for bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.model_executor.runners.acl_graph import _get_graph_pool
from xllm.python.model_executor.runners.decode_cuda_graph import _CAPTURE_WARMUP_STEPS
from xllm.python.platform import current_platform

# The layout of one encode: every image's (t, h, w) in order, plus the
# patch-embedding input width. Together they fix every shape and every Host
# scalar inside the encoder.
_VisionGraphKey = tuple[tuple[tuple[int, int, int], ...], int]

# How often an encode that found the budget full reports coverage. Traffic can
# hold more layouts than the budget, so this is sampled rather than per layout.
_EAGER_LOG_INTERVAL = 1024


def _device_capture_lock(device: torch.device) -> object:
    """Hold the worker's per-device capture lock for the duration of a capture.

    The worker takes this lock around the H2D copies that prepare the next
    batch, because they break an ACL graph capture even when the streams
    differ and the capture mode is thread-local. These captures happen on live
    traffic -- a layout is only known when its first image arrives -- so they
    have to take the same lock. A build without the native bridge, such as a
    standalone test, has nothing to serialise against.
    """
    try:
        import xllm_runtime
    except ImportError:
        return nullcontext()
    lock = getattr(xllm_runtime, "DeviceCaptureLock", None)
    if lock is None:
        return nullcontext()
    return lock(device_index=device.index or 0)


@dataclass(slots=True)
class _VisionGraph:
    static_pixels: torch.Tensor
    static_output: torch.Tensor
    graph: object


class VisionAclGraphRunner:
    """Capture one encoder graph per encode layout and replay it per step."""

    def __init__(
        self,
        vision_model: nn.Module,
        device: torch.device,
        max_graphs: int,
    ) -> None:
        self.vision_model = vision_model
        self.device = device
        self._max_graphs = max(0, int(max_graphs))
        self._graphs: dict[_VisionGraphKey, _VisionGraph] = {}
        self._capture_stream: torch.npu.Stream | None = None
        self.replayed_encodes = 0
        self.eager_encodes = 0

    @property
    def captured_graphs(self) -> int:
        return len(self._graphs)

    @staticmethod
    def _patches(row: tuple[int, int, int]) -> int:
        return row[0] * row[1] * row[2]

    def _has_capture_budget(self) -> bool:
        """Capture only while the operator's graph budget allows it.

        The budget is a count, not free memory: the encoder's parallel ranks
        must reach the same admission decision, and they see the same sequence
        of geometries but not the same free memory. A rank that captured while
        a peer went eager would diverge inside the encoder's tensor-parallel
        collectives.
        """
        return len(self._graphs) < self._max_graphs

    def execute(self, pixel_values: torch.Tensor, grid_thw_list: list[list[int]]) -> torch.Tensor:
        """Return the encoder output for the images in ``pixel_values``."""
        rows = tuple((int(t), int(h), int(w)) for t, h, w in grid_thw_list)
        key = (rows, int(pixel_values.shape[1]))
        entry = self._graphs.get(key)
        if entry is None:
            if not rows or not self._has_capture_budget():
                # Declared eager path, selected before the encoder runs. No
                # per-layout state is kept: the budget never frees up, so a
                # layout that misses it now misses it again, and remembering
                # the keys would grow with the traffic's variety. The sampled
                # line below is what an operator sizing the budget reads, so
                # its volume cannot grow that way either.
                self.eager_encodes += 1
                if rows and self.eager_encodes % _EAGER_LOG_INTERVAL == 1:
                    logger.info(
                        "Python vision ACL graph budget reached: captured=%d layout=%s "
                        "replayed_encodes=%d eager_encodes=%d",
                        len(self._graphs),
                        rows,
                        self.replayed_encodes,
                        self.eager_encodes,
                    )
                return self.vision_model(pixel_values, list(grid_thw_list))
            # The captured graph owns a pixel buffer of this exact size, so the
            # grid it was keyed on has to describe these pixels.
            if pixel_values.shape[0] != sum(self._patches(row) for row in rows):
                raise ValueError("vision graph pixel rows must match the grid patch count")
            entry = self._capture(key, pixel_values)
        # Capture records the encoder without running it, so the first values
        # come from this replay, exactly like every later call's.
        self._replay(entry, pixel_values)
        self.replayed_encodes += 1
        # Cloned outside inference mode, where capture and replay have to run,
        # so the framework receives an ordinary tensor as it does from the
        # eager encoder -- it keeps this output across prefill chunks and in
        # its encoder cache, while the next replay overwrites the captured
        # buffer.
        return entry.static_output.clone()

    @torch.inference_mode()
    def _capture(self, key: _VisionGraphKey, pixels: torch.Tensor) -> _VisionGraph:
        static_pixels = torch.empty_like(pixels)
        static_pixels.copy_(pixels)
        grid_rows = [list(row) for row in key[0]]
        caller_stream = torch.npu.current_stream(self.device)
        if self._capture_stream is None:
            self._capture_stream = torch.npu.Stream(device=self.device)
        stream = self._capture_stream
        # Capture reads the pixels copied above and the encoder weights, both
        # written on the caller's stream.
        stream.wait_stream(caller_stream)
        capture_lock = _device_capture_lock(self.device)
        with torch.npu.stream(stream):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self.vision_model(static_pixels, grid_rows)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            # The lock keeps the worker's next-batch H2D out of the capture;
            # thread_local keeps another thread's capture out of it.
            with (
                capture_lock,
                torch.npu.graph(
                    graph,
                    pool=_get_graph_pool(),
                    stream=stream,
                    capture_error_mode="thread_local",
                ),
            ):
                static_output = self.vision_model(static_pixels, grid_rows)
        caller_stream.wait_stream(stream)
        entry = _VisionGraph(
            static_pixels=static_pixels,
            static_output=static_output,
            graph=graph,
        )
        self._graphs[key] = entry
        free_bytes, _ = torch.npu.mem_get_info(self.device)
        logger.info(
            "Python vision ACL graph captured: layout=%s images=%d patches=%d tokens=%d graphs=%d "
            "free_mb=%d input_prep_locked=%s",
            key[0],
            len(key[0]),
            static_pixels.shape[0],
            static_output.shape[0],
            len(self._graphs),
            free_bytes // (1024 * 1024),
            not isinstance(capture_lock, nullcontext),
        )
        return entry

    @torch.inference_mode()
    def _replay(self, entry: _VisionGraph, pixels: torch.Tensor) -> None:
        # Inference mode: the pixel buffer and the graph's own bookkeeping were
        # allocated inside it during capture, and updating them in place from
        # outside is rejected. The layout is part of the key and the dtype comes
        # from the caller's cast, so the copy is checked by construction.
        entry.static_pixels.copy_(pixels)
        entry.graph.replay()


def build_vision_graph_runner(
    vision_model: nn.Module,
    config: Mapping[str, object],
) -> VisionAclGraphRunner | None:
    """Return a runner when the worker captures this encoder, else ``None``."""
    if not (config.get("enable_graph", False) and current_platform.is_npu()):
        return None
    if not config.get("enable_graph_mm_encoder", False):
        return None
    max_captures = int(config.get("mm_encoder_graph_capture_shapes", 0) or 0)
    logger.info("Python vision ACL graph config: max_captures=%d", max_captures)
    if max_captures <= 0:
        return None
    return VisionAclGraphRunner(vision_model, vision_model.device, max_captures)

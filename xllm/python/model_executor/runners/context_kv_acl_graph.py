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

"""DSpark context projection and KV scatter graphs with warmed input buffers."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.acl_graph import _get_graph_pool
from xllm.python.model_executor.runners.decode_cuda_graph import _CAPTURE_WARMUP_STEPS


@dataclass(slots=True)
class _ContextKvGraph:
    hidden: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    graph: object
    output: torch.Tensor
    valid_rows: int = 0


class ContextKvAclGraphRunner:
    """Graph the context update with owned input buffers.

    Accepted rows are compacted by the worker before this call. Unused bucket
    rows use negative slots, the same no-write contract as block-draft padding.
    The caller orders producers, updates and cache readers on its compute
    stream and consumes the output before another update.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._graphs: dict[int, _ContextKvGraph] = {}
        self._buckets: list[int] = []
        self._capture_stream: torch.npu.Stream | None = None
        self.replays = 0

    def bind_kv_caches(self, caches: list[LayerCache]) -> None:
        bindings = []
        for cache in caches:
            if cache.key is None or cache.value is None:
                raise ValueError("context KV graphs require key and value caches for every layer")
            bindings.append((cache.key, cache.value))
        if self._kv_caches is not None:
            if len(bindings) != len(self._kv_caches) or any(
                old.data_ptr() != new.data_ptr()
                or old.shape != new.shape
                or old.stride() != new.stride()
                or old.dtype != new.dtype
                or old.device != new.device
                for old_pair, new_pair in zip(self._kv_caches, bindings)
                for old, new in zip(old_pair, new_pair)
            ):
                raise ValueError("context KV cache bindings changed after initialization")
            return
        self._kv_caches = bindings

    @torch.inference_mode()
    def warmup(self, max_rows: int) -> None:
        if self._kv_caches is None or max_rows <= 0:
            raise ValueError("context KV graph warmup requires bound caches and positive capacity")
        weight = self.model.model.context_projection_weight
        if weight.shape[1] == 0:
            raise RuntimeError("context projection weights must be loaded before graph warmup")
        bucket = 1
        while bucket < max_rows:
            bucket *= 2
        missing_rows = [
            1 << power for power in reversed(range(bucket.bit_length())) if (1 << power) not in self._graphs
        ]
        if not missing_rows:
            return
        caller_stream = torch.npu.current_stream(self.device)
        if self._capture_stream is None:
            self._capture_stream = torch.npu.Stream(device=self.device)
        stream = self._capture_stream
        # Startup warmup can run on the default stream, which cannot capture.
        # Order weight/cache initialization before our private capture stream,
        # then hand the completed captures back to the caller for replay.
        stream.wait_stream(caller_stream)
        # Larger captures precede smaller ones, matching the shared ACL pool's
        # allocation order. Warmup never scatters into live or padding KV.
        with torch.npu.stream(stream):
            for rows in missing_rows:
                hidden = torch.zeros((rows, weight.shape[1]), dtype=weight.dtype, device=self.device)
                positions = torch.zeros(rows, dtype=torch.int32, device=self.device)
                slots = torch.full((rows,), -1, dtype=torch.int32, device=self.device)
                for _ in range(_CAPTURE_WARMUP_STEPS):
                    self.model.write_context_kv(hidden, positions, slots, self._kv_caches, None)
                stream.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=_get_graph_pool(), stream=stream):
                    output = self.model.write_context_kv(hidden, positions, slots, self._kv_caches, None)
                if output is None:
                    raise RuntimeError("context KV graph capture produced no output")
                self._graphs[rows] = _ContextKvGraph(hidden, positions, slots, graph, output)
                logger.info("Python context KV ACL graph captured: model=%s rows=%d", type(self.model).__name__, rows)
        caller_stream.wait_stream(stream)
        self._buckets = sorted(self._graphs)

    @torch.inference_mode()
    def execute(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.dim() != 2 or positions.dim() != 1 or slots.dim() != 1:
            raise ValueError("context KV graph inputs require hidden rows and one-dimensional positions/slots")
        rows = hidden.shape[0]
        if rows <= 0 or positions.numel() != rows or slots.numel() != rows:
            raise ValueError("context KV graph input row counts must agree and be positive")
        if not self._buckets:
            raise RuntimeError("context KV ACL graphs require startup warmup")
        index = bisect_left(self._buckets, rows)
        if index == len(self._buckets):
            # Declared out-of-capacity eager path, selected before any update.
            # Graph capture errors and missing startup warmup never fall back.
            return self.model.write_context_kv(hidden, positions, slots, self._kv_caches, None)
        entry = self._graphs[self._buckets[index]]
        if hidden.shape[1] != entry.hidden.shape[1] or hidden.dtype != entry.hidden.dtype:
            raise ValueError("context KV graph hidden geometry/dtype changed after warmup")
        if any(tensor.device != self.device for tensor in (hidden, positions, slots)):
            raise ValueError("context KV graph inputs must be on the configured device")
        if positions.dtype not in (torch.int32, torch.int64) or slots.dtype != torch.int32:
            raise ValueError("context KV graph positions/slots must use the native integer layout")
        entry.hidden[:rows].copy_(hidden)
        entry.positions[:rows].copy_(positions)
        entry.slots[:rows].copy_(slots)
        if rows < entry.valid_rows:
            # Projection, norm and RoPE are row-local. Retain finite padded
            # inputs; only stale write slots need invalidation on shrink.
            entry.slots[rows : entry.valid_rows].fill_(-1)
        entry.valid_rows = rows
        entry.graph.replay()
        self.replays += 1
        return entry.output[:rows]

"""Breakable CUDA graph: piecewise capture with eager break points.

Simplified implementation inspired by SGLang's breakable_cuda_graph. Captures
the model forward as a sequence of CUDA graph segments, with attention ops
running eagerly between segments. All segments share one graph memory pool so
intermediate tensor addresses are stable across replays.

Usage (in the graph runner):
    graph = BreakableCUDAGraph()
    with BreakableCUDAGraphCapture(graph, pool=pool, stream=stream):
        output = model(input_ids, positions)
    # replay:
    graph.replay()

Break points are inserted via the `eager_break` decorator on attention:
    @eager_break
    def unified_attention(q, k, v, output, ...):
        ...
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

import torch

_current_capture: contextvars.ContextVar["BreakableCUDAGraphCapture | None"] = (
    contextvars.ContextVar("_current_capture", default=None)
)


def is_capturing() -> bool:
    return _current_capture.get() is not None


class BreakableCUDAGraph:
    """Holds captured segments + interleaved eager break functions."""

    __slots__ = ("_segments", "_break_fns", "_output")

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph] = []
        self._break_fns: list[Callable] = []
        self._output: torch.Tensor | None = None

    def replay(self) -> None:
        for i, seg in enumerate(self._segments):
            seg.replay()
            if i < len(self._break_fns):
                self._break_fns[i]()


class BreakableCUDAGraphCapture:
    """Context manager that captures a forward pass as piecewise CUDA graphs.

    Between segments, decorated break-point functions (attention) run eagerly.
    All segments share ``pool`` for address stability.
    """

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool: tuple[int, int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._graph = cuda_graph
        self._pool = pool
        self._stream = stream
        self._current_seg: torch.cuda.CUDAGraph | None = None
        self._token: contextvars.Token | None = None
        self._stream_ctx: Any = None

    def __enter__(self) -> "BreakableCUDAGraphCapture":
        if self._stream is not None:
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        self._token = _current_capture.set(self)
        self._begin_segment()
        return self

    def __exit__(self, *args) -> None:
        try:
            self._end_segment()
        finally:
            _current_capture.reset(self._token)
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)

    def _begin_segment(self) -> None:
        g = torch.cuda.CUDAGraph()
        g.capture_begin(pool=self._pool)
        self._current_seg = g

    def _end_segment(self) -> None:
        if self._current_seg is not None:
            self._current_seg.capture_end()
            self._graph._segments.append(self._current_seg)
            self._current_seg = None

    def break_here(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """Called by the `eager_break` decorator during capture."""
        # End preceding segment.
        self._end_segment()
        # Run the break function eagerly (attention).
        result = fn(*args, **kwargs)
        # Store replay closure (re-runs the same function with same addresses).
        self._graph._break_fns.append(lambda: fn(*args, **kwargs))
        # Begin next segment.
        self._begin_segment()
        return result


def eager_break(fn: Callable) -> Callable:
    """Decorator: marks a function as a graph-break point during capture.

    Outside capture, the function runs normally. During capture, it ends the
    current graph segment, runs eagerly, stores a replay closure, and begins
    the next segment.
    """

    def wrapper(*args, **kwargs):
        capture = _current_capture.get()
        if capture is None:
            return fn(*args, **kwargs)
        return capture.break_here(fn, args, kwargs)

    wrapper.__wrapped__ = fn
    return wrapper

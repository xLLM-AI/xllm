# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit NPU timing; graph failures propagate and are never timed as eager."""

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from xllm_kernel.numerics.distributed import agree
from xllm_kernel.numerics.suite import PreparedCase


def measure_graph(prepared: PreparedCase, row: dict[str, Any], *, iterations: int, repeats: int) -> None:
    """Validate graph replay before timing and retain captured outputs throughout."""
    capture_start = time.perf_counter()
    graph = torch.npu.NPUGraph()
    divisor = 64 if prepared.reset is None else 1
    if prepared.reset is not None:
        prepared.reset()
    with torch.npu.graph(graph):
        outputs = [prepared.function() for _ in range(divisor)]
    if prepared.reset is not None:
        prepared.reset()
    graph.replay()
    torch.npu.synchronize()
    row["graph_accuracy"] = prepared.validate(lambda output=outputs[-1]: output)
    row["status"] = "passed" if agree(row["graph_accuracy"]["status"] == "passed") else "failed"
    row["capture_s"] = time.perf_counter() - capture_start
    if row["status"] == "passed":
        row["timing"] = measure(
            graph.replay, iterations=iterations, repeats=repeats, divisor=divisor, reset=prepared.reset
        )


def measure(
    function: Callable[[], Any],
    *,
    iterations: int,
    repeats: int,
    divisor: int = 1,
    reset: Callable[[], None] | None = None,
) -> dict:
    if reset is not None:
        return _measure_mutating(function, reset, iterations=iterations, repeats=repeats)
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        host_start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        host_end = time.perf_counter_ns()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        count = iterations * divisor
        samples.append(
            {
                "host_enqueue_us": (host_end - host_start) / 1000 / count,
                "device_interval_us": start.elapsed_time(end) * 1000 / count,
                "wall_us": (wall_end - wall_start) / 1000 / count,
            }
        )
    report = {"samples": samples, "iterations": iterations, "repeats": repeats, "operations_per_call": divisor}
    for metric in samples[0]:
        values = sorted(row[metric] for row in samples)
        median = statistics.median(values)
        report[metric] = {
            "median": median,
            "p90_nearest_rank": values[(9 * len(values) + 9) // 10 - 1],
            "median_absolute_deviation": statistics.median(abs(v - median) for v in values),
        }
    return report


def _measure_mutating(function: Callable[[], Any], reset: Callable[[], None], *, iterations: int, repeats: int) -> dict:
    """Time individual calls after restoring inputs; resets are outside event pairs."""
    samples = []
    for _ in range(repeats):
        events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iterations)]
        torch.npu.synchronize()
        wall_start = time.perf_counter_ns()
        enqueue_ns = 0
        for start, end in events:
            reset()
            start.record()
            before = time.perf_counter_ns()
            function()
            enqueue_ns += time.perf_counter_ns() - before
            end.record()
        events[-1][1].synchronize()
        samples.append(
            {
                "device_interval_us": sum(start.elapsed_time(end) for start, end in events) * 1000 / iterations,
                "host_enqueue_us": enqueue_ns / 1000 / iterations,
                "wall_us": (time.perf_counter_ns() - wall_start) / 1000 / iterations,
            }
        )
    report = {
        "samples": samples,
        "iterations": iterations,
        "repeats": repeats,
        "operations_per_call": 1,
        "reset_policy": "before_each_call_outside_device_and_host_intervals; wall_includes_reset",
    }
    for metric in samples[0]:
        values = sorted(row[metric] for row in samples)
        median = statistics.median(values)
        report[metric] = {
            "median": median,
            "p90_nearest_rank": values[(9 * len(values) + 9) // 10 - 1],
            "median_absolute_deviation": statistics.median(abs(value - median) for value in values),
        }
    return report

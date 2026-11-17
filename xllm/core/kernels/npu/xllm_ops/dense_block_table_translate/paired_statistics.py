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

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np


def summarize(
    baseline: Sequence[float], candidate: Sequence[float], acceptance: dict[str, Any], case_id: str
) -> dict[str, Any]:
    rounds = acceptance["measurement_protocol"]["rounds"]
    if len(baseline) != rounds or len(candidate) != rounds:
        raise ValueError("missing paired rounds")
    if any(not math.isfinite(x) or x <= 0 for x in [*baseline, *candidate]):
        raise ValueError("latencies must be positive and finite")
    differences = np.asarray(baseline, dtype=np.float64) - np.asarray(candidate)
    rng = np.random.default_rng(acceptance["seed"])
    indices = rng.integers(0, rounds, size=(acceptance["resamples"], rounds))
    resampled = differences[indices].mean(axis=1)
    ci = np.percentile(resampled, [2.5, 97.5], method=acceptance["percentile"]).tolist()
    mean_b, mean_c = (statistics.mean(baseline), statistics.mean(candidate))
    median_b, median_c = (statistics.median(baseline), statistics.median(candidate))
    target = acceptance["max_mean_us"].get(case_id)
    checks = {
        "positive_mean_saving": mean_b - mean_c > 0,
        "positive_median_saving": median_b - median_c > 0,
        "positive_ci95_lower": ci[0] > 0,
        "target_mean": target is None or mean_c <= target,
    }
    return {
        "id": case_id,
        "required": target is not None,
        "status": ("PASS" if all(checks.values()) else "FAIL") if target else "MEASURED",
        "baseline_rounds_us": baseline,
        "candidate_rounds_us": candidate,
        "baseline_mean_us": mean_b,
        "candidate_mean_us": mean_c,
        "baseline_median_us": median_b,
        "candidate_median_us": median_c,
        "saved_mean_us": mean_b - mean_c,
        "saved_median_us": median_b - median_c,
        "saved_mean_ci95_us": ci,
        "max_mean_us": target,
        "checks": checks,
        "resamples": acceptance["resamples"],
        "seed": acceptance["seed"],
        "rng": "numpy.random.default_rng/PCG64",
        "percentile": acceptance["percentile"],
    }

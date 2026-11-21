# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Official paired-round acceptance statistics, independent of device libraries."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_statistics(baseline: list[float], candidate: list[float], limit: float | None = None) -> dict[str, Any]:
    b, c = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if b.shape != (6,) or c.shape != (6,) or not np.isfinite([b, c]).all():
        raise ValueError("exactly six finite paired rounds are required")
    if (b <= 0).any() or (c <= 0).any():
        raise ValueError("latencies must be positive")
    ids = np.random.default_rng(20261031).integers(0, 6, size=(10000, 6))
    lo, hi = np.percentile((b[ids] - c[ids]).mean(axis=1), [2.5, 97.5], method="linear")
    mean_diff = float(b.mean() - c.mean())
    median_diff = float(np.median(b) - np.median(c))
    checks = {
        "mean_improved": mean_diff > 0,
        "median_improved": median_diff > 0,
        "paired_ci_lower_positive": bool(lo > 0),
    }
    if limit is not None:
        checks["within_target"] = bool(c.mean() <= limit)
    return {
        "baseline_round_us": b.tolist(),
        "candidate_round_us": c.tolist(),
        "baseline_mean_us": float(b.mean()),
        "candidate_mean_us": float(c.mean()),
        "mean_difference_us": mean_diff,
        "median_difference_us": median_diff,
        "paired_mean_difference_ci95_us": [float(lo), float(hi)],
        "bootstrap": {"resamples": 10000, "seed": 20261031, "percentile": "linear"},
        "limit_us": limit,
        "checks": checks,
        "status": ("PASS" if all(checks.values()) else "FAIL") if limit is not None else "MEASURED",
    }

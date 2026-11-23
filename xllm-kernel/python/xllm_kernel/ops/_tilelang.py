# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Ascend vector-core count for the TileLang launch adapters."""

import torch
from triton.runtime import driver

VECTOR_CORES = driver.active.utils.get_device_properties(device=torch.npu.current_device())["num_vectorcore"]
if VECTOR_CORES <= 0 or VECTOR_CORES % 2:
    raise RuntimeError(f"Invalid NPU vector-core count {VECTOR_CORES}")

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Load the package's private NPU binding for startup adapters."""

from pathlib import Path

import torch
import torch_npu  # noqa: F401

torch.ops.load_library(str(Path(__file__).with_name("_atb.so")))

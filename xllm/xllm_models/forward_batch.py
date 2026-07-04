# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight, half-explicit forward batch (SGLang-style).

The C++ ``PyCausalLM::forward`` builds this and passes it as the 3rd forward
argument. Paged-KV / flashinfer metadata is intentionally NOT carried here: the
CUDA ``xllm_ops.attention`` op reads that from the C++ thread-local forward
context. This object only carries what the Python graph itself needs
(currently just token positions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ForwardBatch:
    # Token positions, [num_tokens], int (same tensor passed to forward()).
    positions: Optional[torch.Tensor] = None

    # Retained only because the C++ bridge constructs ForwardBatch with a
    # ``native_attention=False`` keyword (py_causal_lm.cpp). The Python graph no
    # longer branches on it — every op runs the fused xllm_ops kernel — so this
    # is an inert compatibility field; dropping it would require changing the
    # C++ py::arg and rebuilding the binary.
    native_attention: bool = False

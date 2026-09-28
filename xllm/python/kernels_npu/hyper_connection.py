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

"""NPU hyper-connection kernels for multi-stream residual (mHC) layers.

``hc_pre`` fuses rsqrt RMS + linear projection + sigmoid gating + Sinkhorn
normalisation + weighted-sum reduction into a single call. ``hc_post``
applies the post-attention/FFN recombination (post * x + comb^T @ residual).
"""

from __future__ import annotations

import torch

try:
    hc_pre = torch.ops.xllm_ops.hc_pre
except (AttributeError, RuntimeError):
    hc_pre = None

try:
    hc_post = torch.ops.xllm_ops.hc_post
except (AttributeError, RuntimeError):
    hc_post = None

__all__ = [
    "hc_pre",
    "hc_post",
]

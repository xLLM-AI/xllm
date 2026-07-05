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

"""Op dispatch layer for the Python model graph.

Each op forwards to the fused kernel exposed under the single
``torch.ops.xllm_ops.*`` namespace (the *same* vendor kernel the C++ decoder
path uses), which torch dispatches to the per-device implementation by tensor
device — so the Python graph reuses one symbol per op across hardware backends
with no ``#ifdef`` (CUDA today; NPU/others register into the same namespace).

The FakeTensor (meta) impls needed for ``torch.compile`` / graph capture live in
:mod:`python.ops.fake_impls` and are registered lazily (only once a compile
backend is enabled) so the default eager parity path stays clean. The op
dispatch layer depends only on the kernel backends (:mod:`python.kernels`).
"""

from .dispatch import (
    all_gather,
    all_reduce,
    attention,
    fused_add_rms_norm,
    fused_qk_norm_rope,
    rms_norm,
    silu_and_mul,
)

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "fused_qk_norm_rope",
    "attention",
    "all_reduce",
    "all_gather",
]

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

"""Public interfaces for Python model weight loading."""

from .module_loaders import (
    load_causal_lm_weights,
    load_gqa_fused_attention,
)
from .parallel_load_context import ParallelLoadContext
from .scoped_weight_loader import (
    ScopedWeightLoader,
    StateDictLike,
)
from .sharding import (
    gqa_head_split,
    gqa_qkv_shards,
)

__all__ = [
    "ParallelLoadContext",
    "ScopedWeightLoader",
    "StateDictLike",
    "gqa_head_split",
    "gqa_qkv_shards",
    "load_causal_lm_weights",
    "load_gqa_fused_attention",
]

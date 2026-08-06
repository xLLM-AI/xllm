# Copyright 2025-2026 The xLLM Authors.
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

"""Reusable model layers (RMSNorm, rotary embedding, tensor-parallel linear /
embedding).

Layers call :mod:`python.kernels`, which resolves to the package of the active
hardware platform; they carry no hardware branch of their own. The dependency
direction is ``models -> layers -> kernels``.
"""

from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.layernorm import GemmaRMSNorm, RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.rotary_embedding import RotaryEmbedding

__all__ = [
    "Attention",
    "FusedMoE",
    "GatedMLP",
    "RMSNorm",
    "GemmaRMSNorm",
    "RotaryEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "HiddenParallelEmbedding",
]

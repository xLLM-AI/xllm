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

"""Attention layer that delegates runtime execution to the current backend.

The layer owns layer-local attention semantics. Wrapper state, plan, workspace
and KV cache reside in the FlashInferBackend, accessed via ForwardContext.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import get_forward_context

if TYPE_CHECKING:
    from xllm.python.attention.mla_index_context import MlaIndexContextBuilder


class Attention(nn.Module):
    """Thin attention layer that dispatches to the current backend."""

    # Specialized graph backends must opt in explicitly; a tensor shape
    # shared by unrelated models is not a model capability.
    supports_cp_nope_prefill: bool = False
    # A specialized MLA layer may opt into a typed context. The backend only
    # invokes the builder and never imports that layer's type.
    mla_index_context_builder: type[MlaIndexContextBuilder] | None = None

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        layer_id: int,
        causal: bool = True,
        fia_sparse_mode: int | None = None,
        fia_pre_tokens: int = 2147483647,
        fia_next_tokens: int = 2147483647,
        fia_use_attention_mask: bool = False,
        use_tensor_core_decode: bool | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.layer_id = layer_id
        self.causal = causal
        self.fia_sparse_mode = fia_sparse_mode
        self.fia_pre_tokens = fia_pre_tokens
        self.fia_next_tokens = fia_next_tokens
        self.fia_use_attention_mask = fia_use_attention_mask
        self.use_tensor_core_decode = use_tensor_core_decode
        # External transfers update framework-owned caches only. Derived
        # model-local caches must be reconstructed from those tensors.
        self.cache_transfer_enabled = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        backend = get_forward_context().attention_backend
        return backend.execute(q, k, v, self)

# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Flash attention for GR00T's batch-one, text/image key-mask contract."""

import torch
from diffusers.models.attention import Attention
from torch.nn import functional as F


class _FlashKeyIndices:
    def __init__(self) -> None:
        self.indices: tuple[torch.Tensor, torch.Tensor] | None = None


class _Gr00tFlashProcessor:
    """Use request-owned key indices without changing global Torch operators.

    GR00T Attention modules use three-dimensional inputs, no Q/K normalization,
    and no processor-level residual or spatial normalization. Selecting only
    visible K/V rows preserves its boolean key-mask semantics.
    """

    def __init__(self, keys: _FlashKeyIndices, kind: int | None = None) -> None:
        self._keys = keys
        self._kind = kind

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.to_q(hidden_states)
        key, value = attn.to_k(context), attn.to_v(context)
        if attention_mask is not None:
            if self._kind is None or self._keys.indices is None:
                raise ValueError("GR00T Flash key indices must be bound before masked attention")
            indices = self._keys.indices[self._kind]
            key, value = key.index_select(1, indices), value.index_select(1, indices)
        batch, sequence, width = query.shape
        head_dim = width // attn.heads
        query = query.view(batch, sequence, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).reshape(batch, sequence, width).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](output))

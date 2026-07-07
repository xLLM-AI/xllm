"""Forward context: carries per-step attention metadata and kv_caches outside
the torch.compile graph inputs.

Set once per forward step (by PyModelBase.forward, BEFORE the compiled runner),
accessed inside attention layers (which run in eager due to graph-break). This
keeps the compiled graph's signature stable at (input_ids, positions) and
enables Dynamo loop-caching across decoder layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .attn_metadata import AttentionMetadata


class ForwardContext:
    __slots__ = ("attn_metadata", "kv_caches")

    def __init__(
        self,
        attn_metadata: "AttentionMetadata",
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.attn_metadata = attn_metadata
        self.kv_caches = kv_caches


_current_context: Optional[ForwardContext] = None


def set_forward_context(
    attn_metadata: "AttentionMetadata",
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    global _current_context
    _current_context = ForwardContext(attn_metadata, kv_caches)


def get_forward_context() -> ForwardContext:
    assert _current_context is not None, "forward context not set"
    return _current_context

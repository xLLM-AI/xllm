# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reshape_paged_cache implementations supplied by torch; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    def run(
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        key_cache.view(-1, *keys.shape[1:]).index_copy_(0, slots.long(), keys)
        value_cache.view(-1, *values.shape[1:]).index_copy_(0, slots.long(), values)
        return key_cache

    return run

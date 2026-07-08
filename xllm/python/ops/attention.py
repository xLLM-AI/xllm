from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# KV cache write (capturable — pure kernel, no external state dependency)
# ---------------------------------------------------------------------------
reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache


@torch.library.register_fake("xllm_ops::reshape_paged_cache")
def _(slot_mapping, k, v, k_cache, v_cache):
    return k_cache

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# KV cache write (capturable — pure kernel, no external state dependency)
# ---------------------------------------------------------------------------
reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata


@torch.library.register_fake("xllm_ops::reshape_paged_cache")
def _(slot_mapping, k, v, k_cache, v_cache):
    return k_cache


@torch.library.register_fake("xllm_ops::update_decode_graph_metadata")
def _(
    tokens,
    positions,
    slot_mapping,
    kv_seq_lens,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    dst_tokens,
    dst_positions,
    dst_slot_mapping,
    dst_kv_seq_lens,
    dst_kv_seq_lens_delta,
    dst_paged_kv_indptr,
    dst_paged_kv_indices,
    dst_paged_kv_last_page_len,
    padded_num_tokens,
):
    return dst_tokens

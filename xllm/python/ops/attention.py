from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# KV cache write (capturable — pure kernel, no external state dependency)
# ---------------------------------------------------------------------------
reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache


@torch.library.register_fake("xllm_ops::reshape_paged_cache")
def _(slot_mapping, k, v, k_cache, v_cache):
    return k_cache


# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------
batch_prefill = torch.ops.xllm_ops.batch_prefill
batch_decode = torch.ops.xllm_ops.batch_decode
batch_chunked_prefill = torch.ops.xllm_ops.batch_chunked_prefill

# ---------------------------------------------------------------------------
# Attention plan (scheduling)
# ---------------------------------------------------------------------------
update_prefill_plan = torch.ops.xllm_ops.update_prefill_plan
update_decode_plan = torch.ops.xllm_ops.update_decode_plan
update_chunked_prefill_plan = torch.ops.xllm_ops.update_chunked_prefill_plan

# ---------------------------------------------------------------------------
# Attention kernels run in eager (depend on volatile plan/workspace state).
# ---------------------------------------------------------------------------
for _op in (
    batch_prefill,
    batch_decode,
    batch_chunked_prefill,
):
    torch._dynamo.disallow_in_graph(getattr(_op, "default", _op))

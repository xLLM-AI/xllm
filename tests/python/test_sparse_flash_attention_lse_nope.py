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

"""NPU tests for sparse_flash_attention_lse true-NoPE (query_rope/key_rope is None)."""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_bool, c_char_p, c_double, c_int64, c_uint64, c_void_p
from itertools import accumulate
from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu", reason="NoPE SFA tests require torch_npu")

_ACL_BF16 = 27
_ACL_FLOAT = 0
_ACL_INT32 = 3
_ACL_FORMAT_ND = 2
_ACL_MEM_MALLOC_HUGE_FIRST = 0
_OPAPI = os.environ.get("SFA_LSE_OPAPI", "")
_OPP = os.path.dirname(os.path.dirname(os.path.dirname(_OPAPI))) if _OPAPI else ""


def _ensure_custom_opp() -> None:
    if not _OPP:
        return
    cur = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if _OPP not in cur:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{_OPP}:{cur}" if cur else _OPP


# Must run at import time when a custom OPP is provided: the tiling library
# is located via ASCEND_CUSTOM_OPP_PATH when the first NPU op triggers aclInit.
if _OPAPI:
    _ensure_custom_opp()


def _load_aclnn() -> tuple[ctypes.CDLL, ctypes.CDLL, Any, Any]:
    nnop = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    opapi = ctypes.CDLL(_OPAPI, mode=ctypes.RTLD_GLOBAL)
    nnop.aclCreateTensor.restype = c_void_p
    nnop.aclCreateTensor.argtypes = [
        POINTER(c_int64),
        c_uint64,
        ctypes.c_int,
        POINTER(c_int64),
        c_int64,
        ctypes.c_int,
        POINTER(c_int64),
        c_uint64,
        c_void_p,
    ]
    nnop.aclDestroyTensor.argtypes = [c_void_p]
    acl.aclrtMalloc.argtypes = [POINTER(c_void_p), ctypes.c_size_t, ctypes.c_uint]
    acl.aclrtMalloc.restype = ctypes.c_int
    acl.aclrtFree.argtypes = [c_void_p]
    get_ws = opapi.aclnnSparseFlashAttentionLseGetWorkspaceSize
    get_ws.restype = ctypes.c_int
    get_ws.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_double,
        c_int64,
        c_char_p,
        c_char_p,
        c_int64,
        c_int64,
        c_int64,
        c_int64,
        c_bool,
        c_void_p,
        c_void_p,
        c_void_p,
        POINTER(c_uint64),
        POINTER(c_void_p),
    ]
    run = opapi.aclnnSparseFlashAttentionLse
    run.restype = ctypes.c_int
    run.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    return nnop, acl, get_ws, run


def _acl_dtype(tensor: torch.Tensor) -> int:
    if tensor.dtype == torch.bfloat16:
        return _ACL_BF16
    if tensor.dtype == torch.float32:
        return _ACL_FLOAT
    if tensor.dtype == torch.int32:
        return _ACL_INT32
    raise TypeError(f"unsupported dtype {tensor.dtype}")


def _make_acl_tensor(nnop: Any, tensor: torch.Tensor | None) -> tuple[Any, tuple[Any, ...]]:
    if tensor is None:
        return None, ()
    tensor = tensor.contiguous()
    dims = (c_int64 * tensor.dim())(*[int(s) for s in tensor.shape])
    strides = (c_int64 * tensor.dim())(*[int(s) for s in tensor.stride()])
    handle = nnop.aclCreateTensor(
        dims,
        tensor.dim(),
        _acl_dtype(tensor),
        strides,
        0,
        _ACL_FORMAT_ND,
        dims,
        tensor.dim(),
        c_void_p(tensor.data_ptr()),
    )
    if not handle:
        raise RuntimeError("aclCreateTensor failed")
    return handle, (tensor, dims, strides)


def _run_aclnn_sfa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    block_table: torch.Tensor | None,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    layout_kv: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_custom_opp()
    nnop, acl, get_ws, run = _load_aclnn()
    group = query.size(1) // KV_HEADS
    attn_out = torch.empty_like(query)
    softmax_max = torch.empty((KV_HEADS, query.size(0), group), device=query.device, dtype=torch.float32)
    softmax_sum = torch.empty_like(softmax_max)
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "sparse": sparse_indices,
        "bt": block_table,
        "seq_q": actual_seq_q,
        "seq_kv": actual_seq_kv,
        "q_rope": query_rope,
        "k_rope": key_rope,
        "out": attn_out,
        "smax": softmax_max,
        "ssum": softmax_sum,
    }
    handles = {}
    keep = []
    try:
        for name, ten in tensors.items():
            handle, refs = _make_acl_tensor(nnop, ten)
            handles[name] = handle
            keep.append(refs)
        layout_q = ctypes.create_string_buffer(b"TND")
        layout_k = ctypes.create_string_buffer(layout_kv.encode())
        ws_size = c_uint64(0)
        executor = c_void_p()
        status = get_ws(
            c_void_p(handles["query"]),
            c_void_p(handles["key"]),
            c_void_p(handles["value"]),
            c_void_p(handles["sparse"]),
            c_void_p(handles["bt"] or 0),
            c_void_p(handles["seq_q"]),
            c_void_p(handles["seq_kv"]),
            c_void_p(handles["q_rope"] or 0),
            c_void_p(handles["k_rope"] or 0),
            c_double(SCALE),
            c_int64(1),
            layout_q,
            layout_k,
            c_int64(0),
            c_int64(9223372036854775807),
            c_int64(9223372036854775807),
            c_int64(2),
            c_bool(True),
            c_void_p(handles["out"]),
            c_void_p(handles["smax"]),
            c_void_p(handles["ssum"]),
            ctypes.byref(ws_size),
            ctypes.byref(executor),
        )
        if status != 0:
            raise RuntimeError(f"aclnnSparseFlashAttentionLseGetWorkspaceSize failed: {status}")
        workspace = c_void_p()
        if ws_size.value:
            malloc_ret = acl.aclrtMalloc(ctypes.byref(workspace), ws_size.value, _ACL_MEM_MALLOC_HUGE_FIRST)
            if malloc_ret != 0:
                raise RuntimeError(f"aclrtMalloc failed: {malloc_ret}")
        stream = c_void_p(int(torch.npu.current_stream().npu_stream))
        status = run(workspace, ws_size, executor, stream)
        if status != 0:
            raise RuntimeError(f"aclnnSparseFlashAttentionLse failed: {status}")
        torch.npu.synchronize()
        return attn_out, softmax_max, softmax_sum
    finally:
        if "workspace" in locals() and workspace:
            acl.aclrtFree(workspace)
        for handle in handles.values():
            if handle:
                nnop.aclDestroyTensor(c_void_p(handle))


HEAD_DIM = 512
ROPE_DIM = 64
Q_HEADS = 8
KV_HEADS = 1
SCALE = HEAD_DIM**-0.5
BF16_RTOL = 2e-2
BF16_ATOL = 2e-2


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _sfa_lse_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    if ops is not None and hasattr(ops, "sparse_flash_attention_lse"):
        return True
    return os.path.isfile(_OPAPI)


pytestmark = [
    pytest.mark.skipif(not _npu_available(), reason="NPU is not available"),
    pytest.mark.skipif(not _OPAPI, reason="SFA_LSE_OPAPI is not set"),
    pytest.mark.skipif(
        not _sfa_lse_available(),
        reason="torch.ops.xllm_ops.sparse_flash_attention_lse is not registered",
    ),
]


def _prefix_sums(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(accumulate(seq_lens)), dtype=torch.int32, device=device)


def _pytorch_sfa_tnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense PyTorch reference: 512-d nope dot + sparse gather + softmax.

    Layouts: query/key/value TND [T, N, D], sparse_indices [T, N2, K],
    actual seq lengths are prefix sums. softmax_max/sum layout is NTG [N2, T, G].
    """
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    tokens = query_f.size(0)
    q_heads = query_f.size(1)
    group = q_heads // KV_HEADS
    attn_out = torch.zeros_like(query_f)
    softmax_max = torch.full((KV_HEADS, tokens, group), -float("inf"), dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros((KV_HEADS, tokens, group), dtype=torch.float32, device=query.device)

    q_starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=query.device), actual_seq_q.long()[:-1]])
    kv_starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=query.device), actual_seq_kv.long()[:-1]])
    batch = actual_seq_q.numel()
    for batch_idx in range(batch):
        q_beg = int(q_starts[batch_idx].item())
        q_end = int(actual_seq_q[batch_idx].item())
        kv_beg = int(kv_starts[batch_idx].item())
        kv_end = int(actual_seq_kv[batch_idx].item())
        kv_len = kv_end - kv_beg
        for q_pos in range(q_beg, q_end):
            idx = sparse_indices[q_pos, 0].long()
            valid = (idx >= 0) & (idx < kv_len)
            local = idx.clamp(min=0)
            gathered_k = key_f[kv_beg + local]
            gathered_v = value_f[kv_beg + local]
            q_tok = query_f[q_pos]
            scores = torch.einsum("hd,kd->hk", q_tok, gathered_k.squeeze(1)) * scale
            scores = scores.masked_fill(~valid.unsqueeze(0), -float("inf"))
            max_s = torch.amax(scores, dim=-1)
            exp_s = torch.exp(scores - max_s.unsqueeze(-1))
            exp_s = exp_s.masked_fill(~valid.unsqueeze(0), 0.0)
            sum_s = torch.sum(exp_s, dim=-1).clamp_min(1e-12)
            attn = exp_s / sum_s.unsqueeze(-1)
            attn_out[q_pos] = torch.einsum("hk,kvd->hd", attn, gathered_v)
            softmax_max[0, q_pos] = max_s.view(group)
            softmax_sum[0, q_pos] = sum_s.view(group)
    return attn_out.to(query.dtype), softmax_max, softmax_sum


def _run_sfa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    block_table: torch.Tensor | None,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    layout_kv: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ops = getattr(torch.ops, "xllm_ops", None)
    if ops is not None and hasattr(ops, "sparse_flash_attention_lse"):
        return torch.ops.xllm_ops.sparse_flash_attention_lse(
            query,
            key,
            value,
            sparse_indices,
            block_table,
            actual_seq_q,
            actual_seq_kv,
            query_rope,
            key_rope,
            SCALE,
            1,
            "TND",
            layout_kv,
            0,
            9223372036854775807,
            9223372036854775807,
            2,
            True,
        )
    return _run_aclnn_sfa(
        query,
        key,
        value,
        sparse_indices,
        block_table=block_table,
        actual_seq_q=actual_seq_q,
        actual_seq_kv=actual_seq_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_kv=layout_kv,
    )


def _make_tnd_case(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260909)
    seq_q = [2, 1]
    seq_kv = [8, 4]
    tokens_q = sum(seq_q)
    tokens_kv = sum(seq_kv)
    query = torch.randn(tokens_q, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(tokens_kv, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = key.clone()
    actual_seq_q = _prefix_sums(seq_q, device)
    actual_seq_kv = _prefix_sums(seq_kv, device)
    topk = 4
    sparse = torch.full((tokens_q, KV_HEADS, topk), -1, dtype=torch.int32, device=device)
    q_starts = [0, seq_q[0]]
    kv_lens = seq_kv
    for batch_idx, q_len in enumerate(seq_q):
        for local_q in range(q_len):
            tok = q_starts[batch_idx] + local_q
            valid = min(topk, kv_lens[batch_idx])
            sparse[tok, 0, :valid] = torch.arange(valid, device=device, dtype=torch.int32)
    zeros_q_rope = torch.zeros(tokens_q, Q_HEADS, ROPE_DIM, device=device, dtype=torch.bfloat16)
    zeros_k_rope = torch.zeros(tokens_kv, KV_HEADS, ROPE_DIM, device=device, dtype=torch.bfloat16)
    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse,
        "actual_seq_q": actual_seq_q,
        "actual_seq_kv": actual_seq_kv,
        "query_rope": zeros_q_rope,
        "key_rope": zeros_k_rope,
    }


def _pack_pa(key_tnd: torch.Tensor, seq_kv: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(seq_kv)
    max_blocks = (max(seq_kv) + block_size - 1) // block_size
    pages: list[torch.Tensor] = []
    block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=key_tnd.device)
    offset = 0
    page_id = 0
    for batch_idx, kv_len in enumerate(seq_kv):
        seq = key_tnd[offset : offset + kv_len]
        pad = (-kv_len) % block_size
        if pad:
            seq = torch.cat(
                [seq, torch.zeros(pad, KV_HEADS, HEAD_DIM, device=key_tnd.device, dtype=key_tnd.dtype)],
                dim=0,
            )
        n_blocks = seq.size(0) // block_size
        for blk in range(n_blocks):
            pages.append(seq[blk * block_size : (blk + 1) * block_size])
            block_table[batch_idx, blk] = page_id
            page_id += 1
        offset += kv_len
    key_pa = torch.stack(pages, dim=0)
    return key_pa, block_table


def test_nope_tnd_tnd_matches_zero_rope_and_pytorch() -> None:
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    device = torch.device("npu:0")
    tensors = _make_tnd_case(device)
    nope = _run_sfa(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["sparse_indices"],
        block_table=None,
        actual_seq_q=tensors["actual_seq_q"],
        actual_seq_kv=tensors["actual_seq_kv"],
        query_rope=None,
        key_rope=None,
        layout_kv="TND",
    )
    rope_zero = _run_sfa(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["sparse_indices"],
        block_table=None,
        actual_seq_q=tensors["actual_seq_q"],
        actual_seq_kv=tensors["actual_seq_kv"],
        query_rope=tensors["query_rope"],
        key_rope=tensors["key_rope"],
        layout_kv="TND",
    )
    ref = _pytorch_sfa_tnd(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["sparse_indices"],
        tensors["actual_seq_q"],
        tensors["actual_seq_kv"],
        SCALE,
    )
    torch.testing.assert_close(nope, rope_zero, rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(nope[0].float(), ref[0].float(), rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(nope[1], ref[1], rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(nope[2], ref[2], rtol=BF16_RTOL, atol=BF16_ATOL)


def test_nope_tnd_pa_bsnd_matches_zero_rope() -> None:
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    device = torch.device("npu:0")
    tensors = _make_tnd_case(device)
    seq_kv = [8, 4]
    block_size = 16
    key_pa, block_table = _pack_pa(tensors["key"], seq_kv, block_size)
    key_rope_pa = key_pa.new_zeros((*key_pa.shape[:-1], ROPE_DIM))

    nope = _run_sfa(
        tensors["query"],
        key_pa,
        key_pa,
        tensors["sparse_indices"],
        block_table=block_table,
        actual_seq_q=tensors["actual_seq_q"],
        actual_seq_kv=tensors["actual_seq_kv"],
        query_rope=None,
        key_rope=None,
        layout_kv="PA_BSND",
    )
    rope_zero = _run_sfa(
        tensors["query"],
        key_pa,
        key_pa,
        tensors["sparse_indices"],
        block_table=block_table,
        actual_seq_q=tensors["actual_seq_q"],
        actual_seq_kv=tensors["actual_seq_kv"],
        query_rope=tensors["query_rope"],
        key_rope=key_rope_pa,
        layout_kv="PA_BSND",
    )
    torch.testing.assert_close(nope, rope_zero, rtol=0.0, atol=0.0, equal_nan=True)

    ref = _pytorch_sfa_tnd(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["sparse_indices"],
        tensors["actual_seq_q"],
        tensors["actual_seq_kv"],
        SCALE,
    )
    torch.testing.assert_close(nope[0].float(), ref[0].float(), rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(nope[1], ref[1], rtol=BF16_RTOL, atol=BF16_ATOL)
    torch.testing.assert_close(nope[2], ref[2], rtol=BF16_RTOL, atol=BF16_ATOL)


def test_zero_dim_rope_tensor_is_rejected() -> None:
    """A non-null rope tensor whose last dim is 0 must be rejected by host checks.

    Regression test: ropeHeadDim_ must stay bound to rope tensor presence.
    Previously ropeHeadDim=0 passed the shape check while GenTilingKey still
    selected the HAS_ROPE=1 kernel, which would read 64 dims out of bounds.
    """
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    device = torch.device("npu:0")
    tensors = _make_tnd_case(device)
    zero_dim_q_rope = torch.zeros(tensors["query"].size(0), Q_HEADS, 0, device=device, dtype=torch.bfloat16)
    zero_dim_k_rope = torch.zeros(tensors["key"].size(0), KV_HEADS, 0, device=device, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="GetWorkspaceSize failed"):
        _run_aclnn_sfa(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            tensors["sparse_indices"],
            block_table=None,
            actual_seq_q=tensors["actual_seq_q"],
            actual_seq_kv=tensors["actual_seq_kv"],
            query_rope=zero_dim_q_rope,
            key_rope=zero_dim_k_rope,
            layout_kv="TND",
        )


def test_nope_npugraph_capture_replay() -> None:
    ops = getattr(torch.ops, "xllm_ops", None)
    if ops is None or not hasattr(ops, "sparse_flash_attention_lse"):
        pytest.skip("NPUGraph smoke requires torch.ops.xllm_ops.sparse_flash_attention_lse")
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    device = torch.device("npu:0")
    tensors = _make_tnd_case(device)
    s = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(s):
        graph.capture_begin()
        out0, max0, sum0 = _run_sfa(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            tensors["sparse_indices"],
            block_table=None,
            actual_seq_q=tensors["actual_seq_q"],
            actual_seq_kv=tensors["actual_seq_kv"],
            query_rope=None,
            key_rope=None,
            layout_kv="TND",
        )
        graph.capture_end()
    eager = _run_sfa(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["sparse_indices"],
        block_table=None,
        actual_seq_q=tensors["actual_seq_q"],
        actual_seq_kv=tensors["actual_seq_kv"],
        query_rope=None,
        key_rope=None,
        layout_kv="TND",
    )
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out0, eager[0], rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(max0, eager[1], rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(sum0, eager[2], rtol=0.0, atol=0.0, equal_nan=True)

# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resident-page ownership on CPU; writer/query numerics and replay on SM90."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from xllm.python.attention.xqa_decode import _XQA_PAGE_SIZES

from .test_native_kv_prefill_dispatch_cpu import _load_method


def test_fp8_aliases_preserve_owner_and_whole_page_copy() -> None:
    bind = _load_method(
        "bind_kv_caches", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch, "logger": Mock(), "json": json}
    )
    owner_k = torch.zeros((3, 128, 4, 256), dtype=torch.uint8)
    owner_v = torch.zeros_like(owner_k)
    linear = SimpleNamespace(key=None, value=None)
    caches = [linear, SimpleNamespace(key=owner_k, value=owner_v)]
    backend = SimpleNamespace(
        head_dim=256,
        num_kv_heads=4,
        _decode_backend="xqa",
        _decode_workspace=torch.empty(1),
        native_kv_runtime_info=lambda: {},
    )
    bind(backend, caches)
    key, value = backend._pages[1]
    assert key.dtype == value.dtype == torch.float8_e4m3fn
    assert key.data_ptr() == owner_k.data_ptr()
    assert value.data_ptr() == owner_v.data_ptr()
    assert backend._kv_caches is caches
    assert 0 not in backend._pages
    owner_k[0].fill_(17)
    owner_v[0].fill_(29)
    owner_k[2].copy_(owner_k[0])
    owner_v[2].copy_(owner_v[0])
    assert torch.equal(key[2].view(torch.uint8), owner_k[0])
    assert torch.equal(value[2].view(torch.uint8), owner_v[0])
    assert not owner_k[1].any()


def test_runtime_byte_count_uses_full_attention_storage() -> None:
    report = _load_method("native_kv_runtime_info")
    owners = (torch.empty((2, 128, 4, 256), dtype=torch.uint8), torch.empty((2, 128, 4, 256), dtype=torch.uint8))
    pages = tuple(tensor.view(torch.float8_e4m3fn) for tensor in owners)
    backend = SimpleNamespace(
        _cache_format="fp8_e4m3",
        _fixed_scales=(0.125, 0.5),
        _pages={1: pages},
        _prefill_backend="fa3",
        _decode_backend="xqa",
        _prefill_query_dtype=torch.float8_e4m3fn,
        dtype=torch.bfloat16,
        _max_context_tokens=32768,
    )
    info = report(backend)
    actual_storage = sum(tensor.untyped_storage().nbytes() for tensor in owners)
    assert info["resident_KV_bytes"] == actual_storage
    assert info["KV_tensor_address_bytes"] == actual_storage
    assert info["resident_global_scale_bytes"] == 0


@pytest.mark.parametrize("kind", ["dtype", "shape", "strides", "missing_value"])
def test_fp8_binding_rejects_wrong_physical_storage(kind: str) -> None:
    bind = _load_method("bind_kv_caches", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch})
    key = torch.zeros((2, 128, 4, 256), dtype=torch.uint8)
    if kind == "dtype":
        key = key.to(torch.bfloat16)
    elif kind == "shape":
        key = key[..., :128].contiguous()
    elif kind == "strides":
        key = key.transpose(0, 1)
    value = None if kind == "missing_value" else key.clone()
    backend = SimpleNamespace(head_dim=256, num_kv_heads=4, _decode_backend="xqa", _decode_workspace=torch.empty(1))
    with pytest.raises(ValueError, match="contiguous resident byte pages"):
        bind(backend, [SimpleNamespace(key=key, value=value)])


@pytest.mark.parametrize("shape", [(2, 64, 4, 256), (3, 128, 4, 256)])
def test_fp8_layers_must_share_page_geometry(shape: tuple[int, ...]) -> None:
    bind = _load_method(
        "bind_kv_caches", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch, "logger": Mock(), "json": json}
    )
    first = torch.zeros((2, 128, 4, 256), dtype=torch.uint8)
    other = torch.zeros(shape, dtype=torch.uint8)
    backend = SimpleNamespace(
        head_dim=256,
        num_kv_heads=4,
        _decode_backend="xqa",
        _decode_workspace=torch.empty(1),
        native_kv_runtime_info=lambda: {},
    )
    with pytest.raises(ValueError, match="same page geometry"):
        bind(
            backend, [SimpleNamespace(key=first, value=first.clone()), SimpleNamespace(key=other, value=other.clone())]
        )


@pytest.mark.parametrize("shape", [(0, 128, 4, 256), (2, 0, 4, 256)])
def test_fp8_requires_nonempty_resident_pages(shape: tuple[int, ...]) -> None:
    bind = _load_method(
        "bind_kv_caches", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch, "logger": Mock(), "json": json}
    )
    key = torch.empty(shape, dtype=torch.uint8)
    backend = SimpleNamespace(
        head_dim=256,
        num_kv_heads=4,
        _decode_backend="xqa",
        _decode_workspace=torch.empty(1),
        native_kv_runtime_info=lambda: {},
    )
    with pytest.raises(ValueError, match="nonempty resident pages"):
        bind(backend, [SimpleNamespace(key=key, value=key.clone())])


def test_fp8_xqa_rejects_unsupported_page_size_when_binding() -> None:
    bind = _load_method(
        "bind_kv_caches", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch, "logger": Mock(), "json": json}
    )
    key = torch.empty((2, 256, 4, 256), dtype=torch.uint8)
    backend = SimpleNamespace(
        head_dim=256,
        num_kv_heads=4,
        _decode_backend="xqa",
        _decode_workspace=torch.empty(1),
        native_kv_runtime_info=lambda: {},
    )
    with pytest.raises(ValueError, match="XQA page size"):
        bind(backend, [SimpleNamespace(key=key, value=key.clone())])


@pytest.mark.parametrize("cache_dtype", ["nvfp4", "mxfp4", "fp8_e5m2"])
def test_pr_backend_rejects_other_formats(cache_dtype: str) -> None:
    constructor = _load_method("__init__")
    with pytest.raises(ValueError, match="fp8_e4m3"):
        constructor(object(), 24, 4, 256, 0.0625, 0, torch.device("cuda"), torch.bfloat16, cache_dtype)


def test_fp8_rejects_unqualified_virtual_pages() -> None:
    import math

    constructor = _load_method("__init__", globals_={"_XQA_PAGE_SIZES": _XQA_PAGE_SIZES, "torch": torch, "math": math})
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        pytest.raises(ValueError, match="VMM is not qualified"),
    ):
        constructor(
            object(), 24, 4, 256, 0.0625, 0, torch.device("cuda"), torch.bfloat16, "fp8_e4m3", virtual_memory=True
        )


_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("graph", [False, True])
def test_fp8_writer_matches_independent_cast_and_replays(graph: bool) -> None:
    pytest.importorskip("triton")
    from xllm.python.attention.native_kv_writer import write_fp8_kv

    key = torch.randn((5, 4, 512), device="cuda", dtype=torch.bfloat16)[..., ::2]
    value = torch.randn_like(key) * 100
    slots = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32, device="cuda")
    owners = tuple(torch.zeros((3, 128, 4, 256), dtype=torch.uint8, device="cuda") for _ in range(2))
    pages = tuple(tensor.view(torch.float8_e4m3fn) for tensor in owners)
    scales = (0.125, 0.5)
    write_fp8_kv(key, value, slots, pages, scales)
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            write_fp8_kv(key, value, slots, pages, scales)
        key.mul_(2)
        value.mul_(0.5)
        captured.replay()
    for source, owner, scale in zip((key, value), owners, scales):
        expected = torch.zeros_like(owner).view(-1, 4, 256)
        quantized = (source.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        expected[slots[:-1].long()] = quantized[:-1]
        assert torch.equal(owner, expected.view_as(owner))


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
def test_query_conversion_preserves_strided_values() -> None:
    pytest.importorskip("triton")
    from xllm.python.attention.query_quantization import quantize_query_fp8

    query = torch.randn((17, 24, 512), dtype=torch.bfloat16, device="cuda")[..., ::2]
    actual = quantize_query_fp8(query, 1.0)
    expected = query.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("phase", ["chunked_mixed", "decode", "decode_graph"])
def test_native_attention_matches_dense_reference(phase: str) -> None:
    pytest.importorskip("triton")
    pytest.importorskip("flashinfer")
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend

    # The first sequence crosses a page boundary. The second stays in one page.
    # Mixed prefill appends two queries and one decode token in a single call.
    device = torch.device("cuda")
    page, dim, heads = 16, 128, 8
    k_scale, v_scale = 0.125, 0.5
    backend = FlashInferNativeKVBackend(
        heads,
        1,
        dim,
        dim**-0.5,
        0,
        device,
        torch.bfloat16,
        "fp8_e4m3",
        k_scale,
        v_scale,
        max_context_tokens=32,
    )
    rng = torch.Generator(device=device).manual_seed(73)

    def samples(count: int, nheads: int) -> torch.Tensor:
        # Exactly representable inputs make paging, scale and causal errors visible.
        return torch.randint(-4, 5, (count, nheads, dim), generator=rng, device=device).bfloat16() / 2

    mixed = phase == "chunked_mixed"
    prefix = 15 if mixed else 16
    q = samples(3 if mixed else 2, heads)
    keys, values = samples(20, 1), samples(20, 1)
    slots = [*range(32, 48), 16, 48, 49, 50]
    new_indices = [*range(prefix, 17), 19]
    cache_slots = torch.tensor(slots, device=device)
    caches = []
    for source, scale in ((keys, k_scale), (values, v_scale)):
        owner = torch.zeros((4, page, 1, dim), device=device, dtype=torch.uint8)
        owner.view(-1, 1, dim)[cache_slots] = (source / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        owner.view(-1, 1, dim)[cache_slots[new_indices]] = 0
        caches.append(owner)
    backend.bind_kv_caches([LayerCache(key=caches[0], value=caches[1])])
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    tails = torch.tensor([1, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=mixed,
        qo_indptr=torch.tensor([0, q.size(0) - 1, q.size(0)], device=device, dtype=torch.int32),
        paged_kv_indptr_host=indptr,
        paged_kv_last_page_len_host=tails,
        paged_kv_indptr=indptr.cuda(),
        paged_kv_last_page_len=tails.cuda(),
        paged_kv_indices=torch.tensor([2, 1, 3], device=device, dtype=torch.int32),
        slot_mapping=cache_slots[new_indices].int(),
    )
    graph_mode = phase == "decode_graph"
    backend.prepare(metadata, graph_mode=graph_mode)
    k, v = keys[new_indices], values[new_indices]
    layer = SimpleNamespace(layer_id=0)
    output = backend.execute(q, k, v, layer)
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = backend.execute(q, k, v, layer)
        q.neg_()
        graph.replay()

    contexts = [(0, end) for end in range(prefix + 1, 18)] + [(17, 20)]
    reference, rounded_reference = [], []
    fp8 = torch.finfo(torch.float8_e4m3fn)
    for query, (start, stop) in zip(q, contexts, strict=True):
        logits = query.float() @ keys[start:stop, 0].float().T * dim**-0.5
        dense_values = values[start:stop, 0].float()
        reference.append(logits.softmax(dim=-1) @ dense_values)
        # These short contexts fit in one attention tile. FA3 and SM90 XQA
        # round the softmax numerator to E4M3 before the product with V.
        weights = (logits - logits.amax(dim=-1, keepdim=True)).exp()
        rounded = (weights * fp8.max).to(torch.float8_e4m3fn).float() / fp8.max
        rounded_reference.append((rounded @ dense_values) / weights.sum(dim=-1, keepdim=True))

    actual = output.float().view_as(q)
    expected = torch.stack(reference)
    relative_l2 = (actual - expected).norm(dim=-1) / expected.norm(dim=-1)
    assert relative_l2.max().item() < fp8.eps / 2
    torch.testing.assert_close(actual, torch.stack(rounded_reference), atol=0.005, rtol=0.005)

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

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.model_loader.quantized_weight_loader import load_kv_cache_scales

if TYPE_CHECKING:
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend


_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _SM90, reason="requires SM90")
def test_default_cuda_worker_does_not_load_native_fp8_ops() -> None:
    # conftest has already bound the FP8 test worker. Exercise default startup
    # in a fresh process with the same native extension as the parent.
    script = """
import sys
import pytest
from xllm import xllm_export
from xllm.python import initialize_runtime

initialize_runtime()
from xllm_kernel.ops.kv_cache import WRITE_FP8_KV_OPERATION
assert WRITE_FP8_KV_OPERATION.prepared is None
assert "xllm_kernel.ops.kv_cache.native_kv_writer" not in sys.modules
assert "xllm_kernel.ops.quantization.query_quantization" not in sys.modules
with pytest.raises(RuntimeError, match="restart the worker"):
    initialize_runtime(kv_cache_dtype="fp8_e4m3")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.fixture
def backend_type() -> type[FlashInferNativeKVBackend]:
    pytest.importorskip("triton")
    pytest.importorskip("flashinfer")
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend

    return FlashInferNativeKVBackend


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (1.0, 1.0)),
        ({"quantization_config": {"quant_method": "fp8"}}, (1.0, 1.0)),
        ({"quantization_config": {"kv_cache_scales": {"k": 0.125}}}, (0.125, 1.0)),
        ({"quantization_config": {"kv_cache_scales": {"v": 0.5}}}, (1.0, 0.5)),
    ],
)
def test_checkpoint_scale_defaults(tmp_path: Path, config: dict, expected: tuple[float, float]) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert load_kv_cache_scales(tmp_path) == expected


@pytest.mark.parametrize(
    "scales",
    [
        None,
        {"k": False},
        {"k": "0.125"},
        {"v": None},
        {"k": 0},
        {"v": -1},
        {"v": 1e-40},
        {"k": 1e39},
        {"v": float("inf")},
        {"k": float("nan")},
    ],
)
def test_invalid_checkpoint_scales(tmp_path: Path, scales: object) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"kv_cache_scales": scales}}))
    with pytest.raises(ValueError, match="kv_cache_scales"):
        load_kv_cache_scales(tmp_path)


@pytest.fixture
def storage_backend(backend_type: type[FlashInferNativeKVBackend]) -> FlashInferNativeKVBackend:
    # Binding needs the workspace device, but does not allocate or launch CUDA kernels.
    backend = backend_type.__new__(backend_type)
    backend.head_dim = 256
    backend.num_kv_heads = 4
    backend._decode_workspace = torch.empty(0)
    backend._fixed_scales = (0.125, 0.5)
    backend._max_context_tokens = 32768
    backend.dtype = torch.bfloat16
    return backend


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _SM90, reason="requires SM90"))]
)
def test_fp8_binding_preserves_storage_and_whole_page_copy(
    storage_backend: FlashInferNativeKVBackend, device: str
) -> None:
    storage_backend._decode_workspace = torch.empty(0, device=device)
    owner_k = torch.zeros((3, 128, 4, 256), dtype=torch.float8_e4m3fn, device=device)
    owner_v = torch.zeros_like(owner_k)
    caches = [LayerCache(None, None), LayerCache(owner_k, owner_v)]
    storage_backend.bind_kv_caches(caches)
    key, value = storage_backend._pages[1]
    assert key.dtype == value.dtype == torch.float8_e4m3fn
    assert key.data_ptr() == owner_k.data_ptr()
    assert value.data_ptr() == owner_v.data_ptr()
    assert storage_backend._kv_caches is caches
    assert 0 not in storage_backend._pages
    info = storage_backend.native_kv_runtime_info()
    actual_storage = owner_k.untyped_storage().nbytes() + owner_v.untyped_storage().nbytes()
    assert info["resident_KV_bytes"] == info["KV_tensor_address_bytes"] == actual_storage
    owner_k[0].view(torch.uint8).fill_(17)
    owner_v[0].view(torch.uint8).fill_(29)
    owner_k[2].copy_(owner_k[0])
    owner_v[2].copy_(owner_v[0])
    assert torch.equal(key[2].view(torch.uint8), owner_k[0].view(torch.uint8))
    assert torch.equal(value[2].view(torch.uint8), owner_v[0].view(torch.uint8))
    assert not owner_k[1].view(torch.uint8).any()


@pytest.mark.parametrize("kind", ["bf16", "byte", "shape", "strides", "missing_value"])
def test_fp8_binding_rejects_wrong_physical_storage(storage_backend: FlashInferNativeKVBackend, kind: str) -> None:
    key = torch.zeros((2, 128, 4, 256), dtype=torch.float8_e4m3fn)
    if kind == "byte":
        key = key.view(torch.uint8)
    elif kind == "bf16":
        key = key.to(torch.bfloat16)
    elif kind == "shape":
        key = key[..., :128].contiguous()
    elif kind == "strides":
        key = key.transpose(0, 1)
    value = None if kind == "missing_value" else key.clone()
    with pytest.raises(ValueError, match="contiguous resident FP8 E4M3 pages"):
        storage_backend.bind_kv_caches([LayerCache(key, value)])


@pytest.mark.parametrize("shape", [(2, 64, 4, 256), (3, 128, 4, 256)])
def test_fp8_layers_must_share_page_geometry(
    storage_backend: FlashInferNativeKVBackend, shape: tuple[int, ...]
) -> None:
    first = torch.zeros((2, 128, 4, 256), dtype=torch.float8_e4m3fn)
    other = torch.zeros(shape, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="same page geometry"):
        storage_backend.bind_kv_caches([LayerCache(first, first.clone()), LayerCache(other, other.clone())])


@pytest.mark.parametrize(
    ("blocks", "page_size", "error"),
    [(0, 128, "nonempty resident pages"), (2, 0, "nonempty resident pages"), (2, 256, "XQA page size")],
)
def test_fp8_rejects_invalid_pages(
    storage_backend: FlashInferNativeKVBackend, blocks: int, page_size: int, error: str
) -> None:
    key = torch.empty((blocks, page_size, 4, 256), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=error):
        storage_backend.bind_kv_caches([LayerCache(key, key.clone())])


@pytest.mark.parametrize("cache_dtype", ["nvfp4", "mxfp4", "fp8_e5m2"])
def test_backend_rejects_other_formats(backend_type: type[FlashInferNativeKVBackend], cache_dtype: str) -> None:
    with pytest.raises(ValueError, match="fp8_e4m3"):
        backend_type(24, 4, 256, 0.0625, 0, torch.device("cuda"), torch.bfloat16, cache_dtype)


@pytest.mark.parametrize(
    ("capability", "heads", "kv_heads", "dim", "window", "vmm", "error"),
    [
        ((8, 6), 24, 4, 256, 0, False, "SM90"),
        ((9, 0), 24, 4, 64, 0, False, "head_dim"),
        ((9, 0), 8, 4, 256, 0, False, "GQA"),
        ((9, 0), 17, 4, 256, 0, False, "GQA"),
        ((9, 0), 24, 4, 256, 0, True, "VMM"),
        ((9, 0), 24, 4, 256, 128, False, "full attention"),
    ],
)
def test_rejects_unsupported_attention_before_allocation(
    backend_type: type[FlashInferNativeKVBackend],
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    heads: int,
    kv_heads: int,
    dim: int,
    window: int,
    vmm: bool,
    error: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(ValueError, match=error):
        backend_type(
            heads,
            kv_heads,
            dim,
            dim**-0.5,
            window,
            torch.device("cuda"),
            torch.bfloat16,
            "fp8_e4m3",
            virtual_memory=vmm,
        )


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("graph", [False, True])
def test_fp8_writer_matches_independent_cast_and_replays(graph: bool) -> None:
    pytest.importorskip("triton")
    from xllm_kernel.ops.kv_cache import write_fp8_kv

    key = torch.randn((5, 4, 512), device="cuda", dtype=torch.bfloat16)[..., ::2]
    value = torch.randn_like(key) * 100
    slots = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32, device="cuda")
    pages = tuple(torch.zeros((3, 128, 4, 256), dtype=torch.float8_e4m3fn, device="cuda") for _ in range(2))
    scales = (0.125, 0.5)
    write_fp8_kv(key, value, slots, *pages, *scales)
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            write_fp8_kv(key, value, slots, *pages, *scales)
        key.mul_(2)
        value.mul_(0.5)
        captured.replay()
    for source, page, scale in zip((key, value), pages, scales):
        expected = torch.zeros_like(page, dtype=torch.uint8).view(-1, 4, 256)
        quantized = (source.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        expected[slots[:-1].long()] = quantized[:-1]
        assert torch.equal(page.view(torch.uint8), expected.view_as(page))


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("graph", [False, True])
def test_query_conversion_preserves_strided_values(graph: bool) -> None:
    pytest.importorskip("triton")
    from xllm_kernel.ops.quantization import quantize_query_fp8

    query = torch.randn((17, 24, 512), dtype=torch.bfloat16, device="cuda")[..., ::2]
    actual = quantize_query_fp8(query, 1.0)
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            actual = quantize_query_fp8(query, 1.0)
        query.mul_(2)
        captured.replay()
    expected = query.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    assert actual.is_contiguous() and actual.data_ptr() != query.data_ptr()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not _SM90, reason="Native FP8 integration targets SM90")
@pytest.mark.parametrize("phase", ["prefill", "chunked_mixed", "decode", "decode_graph"])
@pytest.mark.parametrize(("dtype", "v_scale"), [(torch.bfloat16, 0.5), (torch.float16, 0.0625)])
def test_native_attention_matches_dense_reference(
    phase: str, dtype: torch.dtype, v_scale: float, tmp_path: Path
) -> None:
    pytest.importorskip("triton")
    pytest.importorskip("flashinfer")
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend

    # The first sequence crosses a page boundary. The second stays in one page.
    # Mixed prefill appends two queries and one decode token in a single call.
    device = torch.device("cuda")
    page, dim, heads = 16, 128, 8
    k_scale = 0.125
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"kv_cache_scales": {"k": k_scale, "v": v_scale}}})
    )
    backend = FlashInferNativeKVBackend(
        heads,
        1,
        dim,
        dim**-0.5,
        0,
        device,
        dtype,
        "fp8_e4m3",
        fixed_scales=load_kv_cache_scales(tmp_path),
        max_context_tokens=32,
    )
    rng = torch.Generator(device=device).manual_seed(73)

    def samples(count: int, nheads: int) -> torch.Tensor:
        # Exactly representable inputs make paging, scale and causal errors visible.
        return torch.randint(-4, 5, (count, nheads, dim), generator=rng, device=device).to(dtype) / 2

    prefill = phase == "prefill"
    mixed = phase == "chunked_mixed"
    prefix = 0 if prefill else (15 if mixed else 16)
    second_prefix = 0 if prefill else 2
    new_indices = [*range(prefix, 17), *range(17 + second_prefix, 20)]
    q = samples(len(new_indices), heads)
    keys, values = samples(20, 1), samples(20, 1)
    slots = [*range(32, 48), 16, 48, 49, 50]
    cache_slots = torch.tensor(slots, device=device)
    caches = []
    for source, scale in ((keys, k_scale), (values, v_scale)):
        owner = torch.zeros((4, page, 1, dim), device=device, dtype=torch.float8_e4m3fn)
        page_bytes = owner.view(torch.uint8).view(-1, 1, dim)
        page_bytes[cache_slots] = (source / scale).to(torch.float8_e4m3fn).view(torch.uint8)
        page_bytes[cache_slots[new_indices]] = 0
        caches.append(owner)
    backend.bind_kv_caches([LayerCache(key=caches[0], value=caches[1])])
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    tails = torch.tensor([1, 3], dtype=torch.int32)
    qo_indptr = torch.tensor([0, 17 - prefix, q.size(0)], device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=prefill,
        is_chunked_prefill=mixed,
        qo_indptr=None if prefill else qo_indptr,
        q_cu_seq_lens=qo_indptr,
        paged_kv_indptr_host=indptr,
        paged_kv_last_page_len_host=tails,
        paged_kv_indptr=indptr.cuda(),
        paged_kv_last_page_len=tails.cuda(),
        paged_kv_indices=torch.tensor([2, 1, 3], device=device, dtype=torch.int32),
        slot_mapping=cache_slots[new_indices].int(),
    )
    graph_mode = phase == "decode_graph"
    k, v = keys[new_indices], values[new_indices]
    layer = SimpleNamespace(layer_id=0)
    if phase == "decode":
        # A failed plan must leave the original pages intact even if execute is called.
        metadata.paged_kv_indptr_host = None
        with pytest.raises(ValueError, match="CPU planner metadata"):
            backend.prepare(metadata)
        with pytest.raises(RuntimeError, match="before writing KV"):
            backend.execute(q, k, v, layer)
        for owner in caches:
            assert not owner.view(torch.uint8).view(-1, 1, dim)[cache_slots[new_indices]].any()
        metadata.paged_kv_indptr_host = indptr
    backend.prepare(metadata, graph_mode=graph_mode)
    if phase in ("decode", "decode_graph"):
        plan = backend._active_xqa_plan
        backend.prepare(metadata, graph_mode=graph_mode)
        assert backend._active_xqa_plan is plan
    output = backend.execute(q, k, v, layer)
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = backend.execute(q, k, v, layer)
        q.neg_()
        graph.replay()

    contexts = [(0, end) for end in range(prefix + 1, 18)] + [(17, end) for end in range(18 + second_prefix, 21)]
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

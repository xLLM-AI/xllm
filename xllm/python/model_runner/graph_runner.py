"""Graph runner: owns plan + forward_context + CUDA-graph orchestration.

Selected by the C++ ``--python_graph_backend`` flag (passed via config dict):

- ``off`` (default): eager pass-through, parity baseline.
- ``cudagraphs``: decode = manual full CUDA-graph capture bucketed by batch
  size; prefill / chunked-prefill = breakable piecewise CUDA-graph (manual
  capture with eager break points at attention ops, shared graph pool for
  address stability). This mirrors the C++ ``CudaGraphExecutorImpl`` design but
  keeps all capture/replay orchestration on the Python side.
- any other value: ``torch.compile(model, backend=<value>)``.

Decode full graph:
  Every decode step has ``num_tokens == batch_size`` (block_size=1, q_len=1 per
  seq). We capture one full graph per padded batch bucket. The captured graph
  records ALL kernels including attention. Volatile inputs and attention metadata
  live in fixed-address static buffers refreshed before each replay; the
  FlashInfer workspace and the per-layer ``plan_info`` are already fixed-address.
  The flashinfer plan runs OUTSIDE the capture region with
  ``enable_cuda_graph=True`` (fixed-layout schedule) once before capture and
  again before every replay.

Prefill piecewise graph:
  Captures the model forward as a sequence of CUDA graph segments. At each
  attention layer, the ``@eager_break`` decorator on ``_attention_dispatch``
  ends the current segment, runs attention eagerly (KV write + kernel), and
  starts the next segment. All segments share one ``graph_pool_handle()`` so
  intermediate tensor addresses (including the pre-allocated attention output)
  are stable across replays. Bucketed by total num_tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..attn_metadata import AttentionMetadata
from ..forward_context import set_forward_context

# Warmup eager runs before capture (pays one-time kernel setup / workspace init).
_CAPTURE_WARMUP_STEPS = 2


def _decode_bucket(batch_size: int) -> int:
    """Padded bucket for a decode batch, mirroring C++ get_bucket_num_tokens:
    {1, 2, 4, 8} then round up to a multiple of 16."""
    if batch_size <= 1:
        return 1
    if batch_size <= 2:
        return 2
    if batch_size <= 4:
        return 4
    if batch_size <= 8:
        return 8
    return ((batch_size + 15) // 16) * 16


class _DecodeGraphEntry:
    """Per-bucket captured graph plus its fixed-address static buffers."""

    __slots__ = (
        "batch_pad",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_slot_mapping",
        "static_paged_kv_indptr",
        "static_paged_kv_indices",
        "static_paged_kv_last_page_len",
        "static_meta",
    )


class DecodeFullGraphRunner:
    """Manual full CUDA-graph capture/replay for decode, bucketed by batch size.

    One graph per padded batch bucket, captured lazily on first encounter.
    """

    def __init__(self, model: nn.Module, max_batch: int = 256) -> None:
        self._model = model
        self._graphs: dict[int, _DecodeGraphEntry] = {}
        self._max_batch = max_batch
        # Dedicated stream for the whole decode-graph lifecycle (plan + capture +
        # replay), mirroring the C++ CudaGraphExecutorImpl which runs plan/update
        # AND capture/replay on one capture stream so the plan's int-workspace
        # write is ordered before the (captured) attention kernels.
        self._stream: torch.cuda.Stream | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttentionMetadata,
        kv_caches: list,
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        batch_pad = _decode_bucket(batch)

        # Batch too large to capture → eager fallback (plan with the real meta).
        if batch_pad > self._max_batch:
            self._model.plan_attention(meta, kv_caches)
            set_forward_context(meta, kv_caches)
            return self._model(input_ids, positions)

        entry = self._graphs.get(batch_pad)
        first_capture = entry is None
        if first_capture:
            entry = self._alloc_entry(batch_pad, input_ids, positions, meta, kv_caches)
            self._graphs[batch_pad] = entry

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=input_ids.device)

        # Run plan + capture + replay on one dedicated stream so the plan's
        # int-workspace write precedes the attention kernels. Wait on the current
        # stream first (prefill KV writes / the sampled decode token land there).
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            # Refresh fixed-address static buffers with this step's real data.
            self._fill_buffers(entry, input_ids, positions, meta, batch)
            # Attention reads kv_caches + metadata from the forward context.
            set_forward_context(entry.static_meta, kv_caches)
            # Plan OUTSIDE the capture region: fixed-layout schedule
            # (enable_cuda_graph=True), writes per-step tiling into the
            # fixed-address int workspace and refreshes plan_info in place.
            self._model.plan_attention(
                entry.static_meta, kv_caches, enable_cuda_graph=True
            )

            if first_capture:
                self._capture(entry)
            # CUDA graph capture RECORDS without executing: replay to run
            # this step's forward (produce output + write current KV).
            entry.graph.replay()
            # Clone the padded output slice: static_output is overwritten
            # next step (safe under schedule-overlap).
            result = entry.static_output[:batch].clone()

        # Make the caller's stream wait for the graph output before it is used.
        torch.cuda.current_stream().wait_stream(self._stream)
        return result

    def _alloc_entry(
        self,
        batch_pad: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttentionMetadata,
        kv_caches: list,
    ) -> _DecodeGraphEntry:
        dev = input_ids.device
        # paged_kv_indices upper bound = total blocks in the KV cache: block ids
        # across a batch can never exceed the cache's block count.
        num_kv_blocks = kv_caches[0][0].size(0)

        e = _DecodeGraphEntry()
        e.batch_pad = batch_pad
        e.graph = None
        e.static_output = None
        e.static_input_ids = torch.zeros(
            batch_pad, dtype=input_ids.dtype, device=dev
        )
        e.static_positions = torch.zeros(
            batch_pad, dtype=positions.dtype, device=dev
        )
        e.static_slot_mapping = torch.zeros(
            batch_pad, dtype=meta.slot_mapping.dtype, device=dev
        )
        e.static_paged_kv_indptr = torch.zeros(
            batch_pad + 1, dtype=meta.paged_kv_indptr.dtype, device=dev
        )
        e.static_paged_kv_indices = torch.zeros(
            num_kv_blocks, dtype=meta.paged_kv_indices.dtype, device=dev
        )
        e.static_paged_kv_last_page_len = torch.zeros(
            batch_pad, dtype=meta.paged_kv_last_page_len.dtype, device=dev
        )
        # A single AttentionMetadata bound to the static buffers; reused every
        # step (the buffers' contents change, their addresses do not).
        e.static_meta = AttentionMetadata(
            {
                "slot_mapping": e.static_slot_mapping,
                "paged_kv_indptr": e.static_paged_kv_indptr,
                "paged_kv_indices": e.static_paged_kv_indices,
                "paged_kv_last_page_len": e.static_paged_kv_last_page_len,
                "is_prefill": False,
                "is_chunked_prefill": False,
                "enable_cuda_graph": False,
                "use_tensor_core": False,
            }
        )
        return e

    def _fill_buffers(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttentionMetadata,
        batch: int,
    ) -> None:
        batch_pad = entry.batch_pad
        entry.static_input_ids[:batch].copy_(input_ids)
        entry.static_positions[:batch].copy_(positions)
        entry.static_slot_mapping[:batch].copy_(meta.slot_mapping)
        entry.static_paged_kv_indptr[: batch + 1].copy_(meta.paged_kv_indptr)
        n_idx = meta.paged_kv_indices.numel()
        entry.static_paged_kv_indices[:n_idx].copy_(meta.paged_kv_indices)
        entry.static_paged_kv_last_page_len[:batch].copy_(
            meta.paged_kv_last_page_len
        )

        if batch_pad > batch:
            # Pad each dummy seq with one reserved block (id 0). Block 0 is the
            # block manager's reserved padding block, so real seqs are unaffected.
            # n_idx == real total blocks == indptr[batch].
            dev = entry.static_input_ids.device
            entry.static_input_ids[batch:batch_pad].zero_()
            entry.static_positions[batch:batch_pad].zero_()
            entry.static_slot_mapping[batch:batch_pad].zero_()
            pad_indptr = torch.arange(
                n_idx + 1,
                n_idx + 1 + (batch_pad - batch),
                dtype=entry.static_paged_kv_indptr.dtype,
                device=dev,
            )
            entry.static_paged_kv_indptr[batch + 1 : batch_pad + 1].copy_(
                pad_indptr
            )
            entry.static_paged_kv_indices[n_idx : n_idx + (batch_pad - batch)].zero_()
            entry.static_paged_kv_last_page_len[batch:batch_pad].fill_(1)

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        # Called while the current stream is self._stream (set by __call__), and
        # after the pre-capture plan ran on that same stream. Warmup eager runs
        # pay one-time kernel/cuBLAS setup; then capture on the SAME stream so the
        # plan's int-workspace write and the captured kernels share ordering.
        model = self._model
        ii, pp = entry.static_input_ids, entry.static_positions
        for _ in range(_CAPTURE_WARMUP_STEPS):
            model(ii, pp)
        entry.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(entry.graph, stream=self._stream):
            entry.static_output = model(ii, pp)


_PREFILL_BUCKETS: list[int] = (
    list(range(128, 513, 128))          # 128,256,384,512        step 128
    + list(range(768, 2049, 256))       # 768,1024,...,2048      step 256
    + list(range(2560, 4097, 512))      # 2560,3072,3584,4096   step 512
    + list(range(5120, 8193, 1024))     # 5120,6144,7168,8192   step 1024
)


def _prefill_bucket(num_tokens: int) -> int:
    """Padded bucket for prefill num_tokens.

    Progressive granularity (SGLang-style): denser at low token counts
    (less padding waste), sparser at high counts (fewer buckets to capture).
    """
    for b in _PREFILL_BUCKETS:
        if num_tokens <= b:
            return b
    return 0


class _PrefillGraphEntry:
    """Per-bucket captured breakable graph plus its fixed-address static buffers."""

    __slots__ = (
        "num_tokens_pad",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
    )


class PrefillPiecewiseRunner:
    """Breakable piecewise CUDA graph for prefill/chunked-prefill.

    Captures the model forward as segmented CUDA graphs with eager break
    points at attention ops. All segments share one graph pool so intermediate
    tensor addresses are stable across replays.

    Bucketed by num_tokens (= total tokens in the prefill batch). Unlike
    decode (which is per-sequence, q_len=1), prefill processes variable-length
    prompts so the primary bucketing dimension is total token count.

    Runs on the CURRENT stream (no dedicated stream) — prefill is synchronous
    and doesn't overlap with other work, so stream switching would add pure
    overhead from event synchronization.
    """

    _MAX_PREFILL_TOKENS = 4096

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._graphs: dict[int, _PrefillGraphEntry] = {}
        self._pool: tuple[int, int] | None = None
        self._capture_stream: torch.cuda.Stream | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttentionMetadata,
        kv_caches: list,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        num_tokens_pad = _prefill_bucket(num_tokens)

        if num_tokens_pad == 0:
            self._model.plan_attention(meta, kv_caches)
            set_forward_context(meta, kv_caches)
            return self._model(input_ids, positions)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()

        entry = self._graphs.get(num_tokens_pad)
        first_capture = entry is None
        if first_capture:
            entry = self._alloc_entry(num_tokens_pad, input_ids)
            self._graphs[num_tokens_pad] = entry

        self._fill_buffers(entry, input_ids, positions, num_tokens)
        set_forward_context(meta, kv_caches)
        self._model.plan_attention(meta, kv_caches)

        if first_capture:
            self._capture(entry)

        entry.graph.replay()
        return entry.static_output[:num_tokens].clone()

    def _alloc_entry(
        self, num_tokens_pad: int, input_ids: torch.Tensor
    ) -> _PrefillGraphEntry:
        dev = input_ids.device
        e = _PrefillGraphEntry()
        e.num_tokens_pad = num_tokens_pad
        e.graph = None
        e.static_output = None
        e.static_input_ids = torch.zeros(
            num_tokens_pad, dtype=input_ids.dtype, device=dev
        )
        e.static_positions = torch.zeros(
            num_tokens_pad, dtype=torch.long, device=dev
        )
        return e

    def _fill_buffers(
        self,
        entry: _PrefillGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
    ) -> None:
        entry.static_input_ids[:num_tokens].copy_(input_ids)
        entry.static_positions[:num_tokens].copy_(positions)
        if num_tokens < entry.num_tokens_pad:
            entry.static_input_ids[num_tokens:].zero_()
            entry.static_positions[num_tokens:].zero_()

    def _capture(self, entry: _PrefillGraphEntry) -> None:
        from .breakable_cuda_graph import BreakableCUDAGraph, BreakableCUDAGraphCapture

        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(
                device=entry.static_input_ids.device
            )

        model = self._model
        ii, pp = entry.static_input_ids, entry.static_positions

        # Capture must happen on a non-default stream. Warmup + capture run
        # here once; the hot-path replay runs on the caller's stream directly.
        self._capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._capture_stream):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                model(ii, pp)

            graph = BreakableCUDAGraph()
            with BreakableCUDAGraphCapture(
                graph, pool=self._pool, stream=self._capture_stream
            ):
                entry.static_output = model(ii, pp)
            entry.graph = graph
        torch.cuda.current_stream().wait_stream(self._capture_stream)


class GraphRunner:
    """Selects eager / decode-full-graph / torch.compile."""

    def __init__(self, model: nn.Module, backend: str = "off", max_batch: int = 256):
        self._model = model
        self._backend = backend.strip().lower()
        self._compiled = None
        self._decode_runner = None
        self._warmed_up = False

        if self._backend in ("", "off", "none", "0"):
            self._mode = "off"
        elif self._backend == "cudagraphs":
            self._mode = "graph"
            self._decode_runner = DecodeFullGraphRunner(model, max_batch)
        else:
            from ..ops import fake_impls  # noqa: F401
            self._mode = "compile"
            self._compiled = torch.compile(model, backend=self._backend)

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttentionMetadata,
        kv_caches: list,
    ) -> torch.Tensor:
        if self._mode == "graph" and not self._warmed_up:
            self._warmup(input_ids.device, input_ids.dtype, kv_caches)

        if self._mode == "off":
            self._model.plan_attention(meta, kv_caches)
            set_forward_context(meta, kv_caches)
            return self._model(input_ids, positions)

        if self._mode == "graph":
            if meta.is_prefill or meta.is_chunked_prefill:
                self._model.plan_attention(meta, kv_caches)
                set_forward_context(meta, kv_caches)
                return self._model(input_ids, positions)
            return self._decode_runner(input_ids, positions, meta, kv_caches)

        # "compile": single torch.compile backend for all shapes.
        self._model.plan_attention(meta, kv_caches)
        set_forward_context(meta, kv_caches)
        return self._compiled(input_ids, positions)

    def _warmup(
        self, device: torch.device, dtype: torch.dtype, kv_caches: list
    ) -> None:
        """Pre-capture all decode buckets at service startup.

        Uses slot_mapping=0 (reserved padding block) so warmup writes don't
        corrupt real KV cache data.
        """
        self._warmed_up = True

        # --- Decode buckets ---
        decode_buckets = [1, 2, 4, 8]
        max_batch = self._decode_runner._max_batch
        b = 16
        while b <= max_batch:
            decode_buckets.append(b)
            b += 16
        for batch_size in decode_buckets:
            dummy_ids = torch.zeros(batch_size, dtype=dtype, device=device)
            dummy_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_meta = AttentionMetadata({
                "slot_mapping": torch.zeros(batch_size, dtype=torch.int32, device=device),
                "paged_kv_indptr": torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                "paged_kv_indices": torch.zeros(batch_size, dtype=torch.int32, device=device),
                "paged_kv_last_page_len": torch.ones(batch_size, dtype=torch.int32, device=device),
                "is_prefill": False,
                "is_chunked_prefill": False,
                "enable_cuda_graph": False,
                "use_tensor_core": False,
            })
            self._decode_runner(dummy_ids, dummy_pos, dummy_meta, kv_caches)

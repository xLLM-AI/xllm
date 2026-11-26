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


from __future__ import annotations

import json
import os
import re
import unicodedata
import weakref
from typing import TYPE_CHECKING, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed
from xllm.python.attention.dsa_metadata import last_ring_writes
from xllm.python.layers.moe_parallel import DpScatterState, dp_gather_tokens
from xllm.python.model_executor.forward_context import in_acl_graph

if TYPE_CHECKING:
    from xllm.python.distributed.collectives import EngramTpAllGather
    from xllm.python.models.deepseek_v41_engram_host import EngramHostTable

# Cache value for tokens that take no part in an n-gram (image spans).
ENGRAM_DEAD_ID = -1
# Image sentinel / padding token ids that break an n-gram (mm_preprocess.py).
ENGRAM_IMAGE_SENTINEL_BASE_ID = 129264
ENGRAM_IMAGE_PAD_ID = 129265
# Signed-sqrt clamp of the gate input (reference kernel clamp_value).
_ENGRAM_GATE_CLAMP = 1e-6
_QUAROT_BLOCK_SIZE = 32

_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
# Private-use char so a token that is exactly one space survives Strip().
_SENTINEL = "\ue000"


_UE8M0_ZERO_BITS = 0x00400000
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _validate_ue8m0_scale_bytes(raw: torch.Tensor) -> None:
    if raw.dtype != torch.uint8:
        raise ValueError(f"expected raw uint8 ue8m0 scale storage, got {raw.dtype}")
    # A reduction avoids a table-sized temporary for mmap-backed weights.
    if raw.numel() > 0 and bool(raw.amax() == 0xFF):
        raise ValueError("ue8m0 scale byte 0xFF (NaN) is invalid")


def ue8m0_to_scale(scale_u8: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """ue8m0: byte b means 2**(b-127); 0x00 is subnormal and 0xFF is invalid."""
    if scale_u8.dtype == torch.uint8:
        raw = scale_u8
    elif scale_u8.element_size() == 1:
        raw = scale_u8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte ue8m0 scale storage, got {scale_u8.dtype}")
    if validate:
        _validate_ue8m0_scale_bytes(raw)
    raw_i32 = raw.to(torch.int32)
    bits = torch.where(
        raw_i32 == 0,
        torch.full_like(raw_i32, _UE8M0_ZERO_BITS),
        raw_i32 << 23,
    )
    return bits.contiguous().view(torch.float32)


def _e4m3_to_float(x_e4m3: torch.Tensor) -> torch.Tensor:
    bits = x_e4m3.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 128, -1.0, 1.0)
    magnitude_bits = bits & 0x7F
    exponent = torch.div(magnitude_bits, 8, rounding_mode="floor")
    mantissa = magnitude_bits & 0x7
    mantissa_f = mantissa.to(torch.float32)
    normal = (1.0 + mantissa_f / 8.0) * torch.exp2(exponent.to(torch.float32) - 7.0)
    subnormal = mantissa_f * _E4M3_SUBNORMAL_STEP
    value = torch.where(exponent > 0, normal, subnormal)
    value = torch.where(magnitude_bits == 0x7F, torch.full_like(value, float("nan")), value)
    return value * sign


def image_sentinel_mask(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.reshape(-1)
    return (ids == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (ids == ENGRAM_IMAGE_PAD_ID)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


class EngramLayout:
    def __init__(
        self,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        self.layer_ids: tuple[int, ...] = tuple(int(i) for i in layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(int(n) for n in num_embeddings)
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.pad_token_id = int(pad_token_id)
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                f"engram_layer_ids {self.layer_ids} and engram_num_embeddings "
                f"{self.num_embeddings} must have the same length"
            )
        if self.max_ngram_size < 2:
            raise ValueError(f"engram_max_ngram_size must be >= 2, got {max_ngram_size}")

        primes: list[list[tuple[int, ...]]] = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                current = self.vocab_size - 1
                for _ in range(self.n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(per_ngram)
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(tuple(p) for p in primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        offsets = []
        for layer, layer_primes in enumerate(self.primes):
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            total = sum(flat)
            if total > self.num_embeddings[layer]:
                raise ValueError(
                    f"engram layer {self.layer_ids[layer]} prime buckets sum to {total} rows "
                    f"but engram_num_embeddings only provides {self.num_embeddings[layer]}"
                )
            prefix = [0]
            for size in flat[:-1]:
                prefix.append(prefix[-1] + size)
            offsets.append(prefix)
        self.offsets = torch.tensor(offsets, dtype=torch.int64)

    def head_sizes(self, layer_hash_index: int) -> tuple[int, ...]:
        return tuple(size for per_ngram in self.primes[layer_hash_index] for size in per_ngram)


def compute_hash_multipliers(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Building DeepSeek-V4.1 engram hash multipliers requires numpy (the "
            "reference RNG is numpy PCG64 and must be reproduced bit-exactly). "
            "Install numpy, or pre-generate an 'engram_hash_state.bin' sidecar "
            "with build_engram_hash_state() and place it next to "
            "the checkpoint."
        ) from exc
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


def normalize_token_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    if s == " ":
        s = _SENTINEL
    s = s.strip()
    return s.replace(_SENTINEL, " ")


def build_compressed_token_map_from_texts(
    token_texts: Sequence[str],
    id_to_token: Callable[[int], str] | None = None,
) -> tuple[list[int], int]:
    key_to_new: dict[str, int] = {}
    lookup: list[int] = []
    for token_id, text in enumerate(token_texts):
        if "\ufffd" in text:
            key = id_to_token(token_id) if id_to_token is not None else text
        else:
            normalized = normalize_token_text(text)
            key = normalized or text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup.append(new_id)
    return lookup, len(key_to_new)


def _load_engram_hash_state_sidecar(model_path: str) -> dict[str, torch.Tensor] | None:
    sidecar = os.path.join(model_path, "engram_hash_state.bin")
    if not os.path.isfile(sidecar):
        return None
    try:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load engram hash-state sidecar {sidecar}: {exc}") from exc


def build_engram_hash_state(
    model_path: str,
    tokenizer_path: str | None = None,
    output_path: str | None = None,
) -> str:
    model_dir = os.path.abspath(model_path)
    if not os.path.isdir(model_dir):
        model_dir = os.path.dirname(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        top = json.load(handle)
    config = {**top, **top.get("text_config", {})}

    layer_ids = config.get("engram_layer_ids") or []
    if not layer_ids:
        raise ValueError("config.json has no engram_layer_ids; this model does not use engram")
    vocab_size = int(config["vocab_size"])
    max_ngram_size = int(config["engram_max_ngram_size"])
    compressed_vocab = int(config["engram_compressed_vocab_size"])
    pad_token_id = int(config["engram_pad_token_id"])
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"engram_pad_token_id {pad_token_id} is out of vocab range {vocab_size}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "building the engram token map requires the 'tokenizers' package (pip install tokenizers)"
        ) from exc
    tokenizer_file = tokenizer_path or os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != vocab_size:
        raise ValueError(f"tokenizer vocab size {tokenizer_vocab} does not match config vocab_size {vocab_size}")

    token_texts = tokenizer.decode_batch([[token_id] for token_id in range(vocab_size)], skip_special_tokens=False)
    lookup, compressed = build_compressed_token_map_from_texts(token_texts, id_to_token=tokenizer.id_to_token)
    if compressed != compressed_vocab:
        raise ValueError(
            f"the tokenizer folds to {compressed} compressed ids but the config declares "
            f"engram_compressed_vocab_size={compressed_vocab}; every hash multiplier derives "
            "from this count, so the engram tables would be silently rehashed -- refusing to "
            "write the sidecar"
        )

    multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    output = output_path or os.path.join(model_dir, "engram_hash_state.bin")
    torch.save(
        {
            "token_map": torch.tensor(lookup, dtype=torch.int32),
            "multipliers": multipliers,
        },
        output,
    )
    return output


class EngramHashState(nn.Module):
    def __init__(
        self,
        layout: EngramLayout,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.pad_id = int(token_map[layout.pad_token_id].item())
        self.register_buffer("token_map", token_map.to(torch.int64), persistent=False)
        self.register_buffer(
            "primes",
            torch.tensor(layout.primes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("offsets", layout.offsets.to(torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

    @classmethod
    def from_config(cls, layout: EngramLayout, config: dict) -> EngramHashState:
        sidecar = _load_engram_hash_state_sidecar(str(config.get("model_path") or ""))
        if sidecar is None:
            raise RuntimeError(
                "DeepSeek-V4.1 engram requires the tokenizer-derived hash state. "
                "Call build_engram_hash_state(model_path) against the "
                "checkpoint's tokenizer to write 'engram_hash_state.bin'."
            )
        token_map = sidecar["token_map"]
        multipliers = sidecar["multipliers"]
        expected_shape = (len(layout.layer_ids), layout.max_ngram_size)
        if multipliers.shape != expected_shape:
            raise ValueError(
                f"engram hash multipliers have shape {tuple(multipliers.shape)} but config expects {expected_shape}"
            )
        # Validate against the tokenizer vocabulary, not the separate Engram bucket count.
        expected_vocab = int(config.get("vocab_size") or token_map.numel())
        if token_map.numel() != expected_vocab:
            raise ValueError(
                f"engram token map covers {token_map.numel()} ids but model vocab_size is {expected_vocab}"
            )
        compressed = int(token_map.max().item()) + 1
        if compressed != layout.compressed_vocab_size:
            raise ValueError(
                f"engram token map compresses to {compressed} ids but config expects "
                f"{layout.compressed_vocab_size}; every hash multiplier derives from "
                "it, so the engram tables would be silently rehashed"
            )
        return cls(layout, token_map, multipliers)

    def ensure_cache(self, swa_cache: torch.Tensor | None) -> bool:
        if swa_cache is None or swa_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is swa_cache:
            return True
        flat_rows = swa_cache.numel() // swa_cache.size(-1) if swa_cache.dim() > 1 else swa_cache.numel()
        self._cache = torch.zeros(flat_rows, dtype=torch.int32, device=swa_cache.device)
        self._kv_cache_ref = weakref.ref(swa_cache)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        swa_block_size: int,
        *,
        unit_query_rows: torch.Tensor | None = None,
        replay_history: torch.Tensor | None = None,
        replay_requests: tuple[int, ...] = (),
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        num_layers = len(self.layout.layer_ids)
        n_heads = self.layout.n_heads
        max_ngram = self.layout.max_ngram_size
        device = input_ids.device
        if num_tokens == 0:
            return torch.empty((0, num_layers, self.layout.n_hash_cols), dtype=torch.int32, device=device)
        pos = positions.reshape(-1)
        ids = input_ids.reshape(-1)
        on_npu = device.type in ("npu", "privateuseone")
        prepared_decode = (
            on_npu
            and unit_query_rows is not None
            and block_table is not None
            and slot_mapping is not None
            and replay_history is None
        )
        if prepared_decode:
            from xllm.python.kernels_npu.triton.engram import prepare_unit_query_hash

            req = unit_query_rows.contiguous()
            pos = pos.contiguous()
            comp, chunk_start = prepare_unit_query_hash(
                ids, pos, dead_mask, slot_mapping, block_table, swa_block_size, self.token_map, max_ngram
            )
        else:
            pos = pos.to(torch.int64)
            ids = ids.to(torch.int64)
            if unit_query_rows is None:
                rows = torch.arange(num_tokens, device=device)
                ends = query_start_loc[1:].to(torch.int64).contiguous()
                req = torch.searchsorted(ends, rows, right=True).clamp(0, max(ends.numel() - 1, 0))
            else:
                # Prepared unit-width offsets give each physical row its own table, including padding.
                rows = unit_query_rows.to(torch.int64)
                req = rows
            if query_start_loc.numel() == num_tokens + 1 and slot_mapping is not None and block_table is not None:
                # Speculative verification flattens a request into single-token
                # rows. Consecutive positions backed by the same physical history
                # still read earlier tokens from this invocation, before cache writeback.
                prior_pos = (pos - 1).clamp_min(0)
                column = torch.div(prior_pos, swa_block_size, rounding_mode="floor") % block_table.size(1)
                prior_slot = block_table.to(torch.int64)[req, column] * swa_block_size + prior_pos % swa_block_size
                previous = (rows - 1).clamp_min(0)
                slots = slot_mapping.reshape(-1)
                continues = (rows > 0) & (pos == pos[previous] + 1) & (slots[previous] >= 0)
                continues = continues & (slots[previous] == prior_slot)
                starts = torch.where(continues, 0, rows).cummax(0).values
                chunk_start = pos[starts]
            else:
                chunk_start = pos[query_start_loc.to(torch.int64)[req].clamp(0, num_tokens - 1)]

            comp = self.token_map[ids.clamp(0, self.token_map.numel() - 1)]
            comp = torch.where(dead_mask.reshape(-1), ENGRAM_DEAD_ID, comp).to(torch.int64)

        cache = self._cache
        mapped_history = None
        request_floors = None
        if replay_history is not None:
            history_sentinels = image_sentinel_mask(replay_history).reshape_as(replay_history)
            replay_valid = (replay_history >= 0) & ~history_sentinels
            mapped_history = self.token_map[replay_history.clamp(0, self.token_map.numel() - 1).to(torch.int64)]
            mapped_history = torch.where(replay_valid, mapped_history, ENGRAM_DEAD_ID)
            request_floors = positions.new_tensor(replay_requests)
        if on_npu:
            from xllm.python.kernels_npu.triton.engram import hash_rows

            output = hash_rows(
                comp,
                pos,
                req,
                chunk_start,
                cache,
                block_table,
                swa_block_size,
                mapped_history,
                request_floors,
                self.multipliers,
                self.primes,
                self.offsets,
                self.pad_id,
            )
            if cache is not None and slot_mapping is not None:
                if prepared_decode and in_acl_graph():
                    from xllm.python.kernels_npu.triton.engram import write_unit_query_history

                    write_unit_query_history(cache, comp, slot_mapping)
                else:
                    self._write_hash_cache(cache, comp, slot_mapping)
            return output

        output = torch.empty((num_tokens, num_layers, self.layout.n_hash_cols), dtype=torch.int32, device=device)
        rolling = torch.zeros(num_tokens, num_layers, dtype=torch.int64, device=device)
        blocked = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if mapped_history is not None:
            replay_rows = request_floors[req] > 0
            history_width = replay_history.size(1)
        for shift in range(max_ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            source = self._chunk_source(comp, rows, shift, in_batch, num_tokens)
            if cache is not None and block_table is not None:
                source = self._cache_source(cache, source, req, lookback, in_batch, block_table, swa_block_size)
            if replay_history is not None:
                column = (lookback - chunk_start + history_width).clamp(0, history_width - 1)
                source = torch.where(replay_rows & ~in_batch, mapped_history[req, column], source)
            blocked = blocked | (lookback < 0) | (source == ENGRAM_DEAD_ID)
            value = torch.where(blocked, self.pad_id, source)
            rolling ^= value.unsqueeze(1) * self.multipliers[:, shift].unsqueeze(0)
            if shift > 0:
                lo = (shift - 1) * n_heads
                prime = self.primes[:, shift - 1, :]
                offset = self.offsets[:, lo : lo + n_heads]
                hashed = rolling.unsqueeze(-1) % prime.unsqueeze(0) + offset.unsqueeze(0)
                output[:, :, lo : lo + n_heads] = hashed.to(torch.int32)
        # A wrapping chunk must consume its old history before overwriting the ring.
        if slot_mapping is not None and cache is not None:
            self._write_hash_cache(cache, comp, slot_mapping)
        return output

    @staticmethod
    def _write_hash_cache(
        cache: torch.Tensor,
        comp: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        comp = comp.reshape(-1).to(torch.int32)
        slots = slot_mapping.reshape(-1).to(torch.int64)
        if not in_acl_graph():
            slots, comp = last_ring_writes(slots, comp, slot_capacity=cache.numel())
        valid = slots >= 0
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        keep = cache.index_select(0, safe)
        cache.scatter_(0, safe, torch.where(valid, comp, keep))

    def _chunk_source(
        self,
        comp: torch.Tensor,
        rows: torch.Tensor,
        shift: int,
        in_batch: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        source_row = (rows - shift).clamp_min(0).clamp_max(num_tokens - 1)
        source = comp.index_select(0, source_row)
        return torch.where(in_batch, source, torch.zeros_like(source))

    @staticmethod
    def _cache_source(
        cache: torch.Tensor,
        source: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        block_table: torch.Tensor,
        swa_block_size: int,
    ) -> torch.Tensor:
        table_cols = block_table.shape[1]
        clamped = lookback.clamp_min(0)
        col = torch.div(clamped, swa_block_size, rounding_mode="floor") % table_cols
        req_safe = req.clamp(0, block_table.shape[0] - 1)
        block = block_table.to(torch.int64)[req_safe, col]
        slot = (block * swa_block_size + clamped % swa_block_size).clamp(0, cache.numel() - 1)
        fallback = cache.index_select(0, slot).to(torch.int64)
        return torch.where(in_batch, source, fallback)


# FP8 tables use raw E4M3/UE8M0 bytes; Ascend int8_sym tables use FP32 group scales.
ENGRAM_STORAGE_FP8 = "fp8"
ENGRAM_STORAGE_INT8 = "int8"


def load_quarot_gate_unrotate(model_path: str, device: torch.device) -> torch.Tensor | None:
    config_path = os.path.join(str(model_path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            rotation = json.load(handle).get("engram_rotation_config") or {}
    except OSError:
        return None
    if not rotation.get("value_projection_rotated"):
        return None
    quarot_path = os.path.join(str(model_path), "optional", "quarot.safetensors")
    from safetensors.torch import load_file

    q = load_file(quarot_path, device="cpu")["global_rotation"]
    width = _QUAROT_BLOCK_SIZE
    if q.ndim == 2 and q.size(0) == q.size(1) and q.size(0) > width and q.size(0) % width == 0:
        count = q.size(0) // width
        block = q[:width, :width]
        diagonal = q.reshape(count, width, count, width).diagonal(dim1=0, dim2=2).movedim(-1, 0)
        # Equal diagonal blocks account for every nonzero entry iff the off-diagonal blocks are all zero.
        if torch.equal(diagonal, block.expand_as(diagonal)) and q.count_nonzero() == count * block.count_nonzero():
            q = block
    return q.t().contiguous().to(device=device, dtype=torch.float32)


class EngramEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: Sequence[int],
        tp_size: int,
        tp_rank: int,
        quant_block: int = 32,
        storage: str = ENGRAM_STORAGE_FP8,
        device: torch.device | None = None,
        *,
        dp_size: int = 1,
        dp_rank: int = 0,
        host_offload: bool = False,
    ) -> None:
        super().__init__()
        if not head_sizes or any(size <= 0 for size in head_sizes):
            raise ValueError(f"engram head sizes must be positive, got {head_sizes}")
        if sum(head_sizes) > num_embeddings:
            raise ValueError(f"engram head sizes sum to {sum(head_sizes)} rows but the table only has {num_embeddings}")
        if dim % quant_block != 0:
            raise ValueError(f"engram head dim {dim} is not divisible by {quant_block}")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.storage = storage
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = (self.n_hash_cols + tp_size - 1) // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        tp_vocab_start = sum(head_sizes[: self.head_start])
        tp_vocab_rows = sum(head_sizes[self.head_start : head_end])
        # DP peers own disjoint rows of the same TP heads. Keep the head
        # ranges unchanged so the final TP gather preserves checkpoint order.
        if host_offload:
            self.vocab_start = tp_vocab_start
            self.vocab_end = tp_vocab_start + tp_vocab_rows
        else:
            self.vocab_start = tp_vocab_start + tp_vocab_rows * dp_rank // dp_size
            self.vocab_end = tp_vocab_start + tp_vocab_rows * (dp_rank + 1) // dp_size
        self.part_num_embeddings = self.vocab_end - self.vocab_start
        self.tp_size = int(tp_size)
        self.dp_size = 1 if host_offload else dp_size
        self.dp_rank = dp_rank
        self.host_offload = host_offload
        self.host_table: EngramHostTable | None = None
        if storage == ENGRAM_STORAGE_INT8:
            self.weight = nn.Parameter(
                torch.empty(0 if host_offload else self.part_num_embeddings, self.dim, dtype=torch.int8, device=device),
                requires_grad=False,
            )
            # The loader allocates the scale shard using the checkpoint dtype
            # so E8M0 scales do not expand to FP32 in persistent memory.
            self.weight_scale = nn.Parameter(torch.empty(0, dtype=torch.float32, device=device), requires_grad=False)
        else:
            # FP8-E4M3 payloads and ue8m0 scales are kept as raw uint8 bytes:
            # index_select stays on a universally supported dtype and the
            # bit-level decoders in deepseek_v41 do the rest.
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.uint8, device=device),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.quant_block,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )

    def _local_block(self, indices: torch.Tensor) -> torch.Tensor:
        num_tokens = indices.shape[0]
        if self.host_table is not None:
            return self.host_table.lookup(
                indices,
                self.head_start,
                self.part_n_hash_cols,
                self.vocab_start,
                self.vocab_end,
                self.dim,
                self.quant_block,
            )
        if self.storage == ENGRAM_STORAGE_INT8 and indices.device.type in ("npu", "privateuseone"):
            from xllm.python.kernels_npu.triton.engram import lookup_int8

            return lookup_int8(
                indices,
                self.weight,
                self.weight_scale,
                self.head_start,
                self.part_n_hash_cols,
                self.vocab_start,
                self.vocab_end,
                self.dim,
                self.quant_block,
                e8m0=self.weight_scale.dtype == torch.uint8,
            )
        if self.part_num_embeddings == 0:
            return torch.zeros(
                num_tokens,
                self.part_n_hash_cols,
                self.dim,
                dtype=torch.bfloat16,
                device=indices.device,
            )
        local_cols = indices[:, self.head_start : self.head_start + self.part_n_hash_cols]
        if local_cols.size(1) < self.part_n_hash_cols:
            # tp_size does not divide n_hash_cols: the last head-owning
            # rank's slice runs short of the uniform block width. Pad with
            # column 0 (id 0 predates every non-first rank's row range, so
            # the padding is never owned and is zeroed below) to keep the
            # gathered blocks shape-uniform.
            local_cols = F.pad(local_cols, (0, self.part_n_hash_cols - local_cols.size(1)))
        owned = (local_cols >= self.vocab_start) & (local_cols < self.vocab_end)
        local = (local_cols - self.vocab_start).clamp(0, self.part_num_embeddings - 1)
        flat = local.reshape(-1).to(torch.int64)
        payload = self.weight.index_select(0, flat)
        scale = self.weight_scale.index_select(0, flat)
        if self.storage == ENGRAM_STORAGE_INT8:
            values = payload.to(torch.float32)
        else:
            values = _e4m3_to_float(payload)
        if scale.dtype == torch.uint8:
            # Persistent scales were validated by the weight loader. Decode
            # only the selected rows without synchronizing back to the CPU.
            scale = ue8m0_to_scale(scale, validate=False)
        scale = scale.repeat_interleave(self.quant_block, dim=-1)
        rows = (values * scale).to(torch.bfloat16)
        rows = rows.view(num_tokens, self.part_n_hash_cols, self.dim)
        return torch.where(owned.unsqueeze(-1), rows, torch.zeros_like(rows))

    def lookup(
        self,
        indices: torch.Tensor,
        dp_scatter: DpScatterState | None = None,
        *,
        tp_gather: EngramTpAllGather | None = None,
    ) -> torch.Tensor:
        if self.dp_size > 1 and dp_scatter is None:
            indices, dp_scatter = dp_gather_tokens(indices, self.dp_size, self.dp_rank)
        rows = self._local_block(indices)
        if dp_scatter is not None:
            distributed.all_reduce_(rows, "dp")
            rows = dp_scatter.scatter(rows)
        return self.gather_heads(rows, tp_gather=tp_gather)

    def gather_heads(self, rows: torch.Tensor, *, tp_gather: EngramTpAllGather | None = None) -> torch.Tensor:
        if self.tp_size > 1:
            gathered = (
                tp_gather.gather(rows, dim=1)
                if tp_gather is not None
                else distributed.tp_all_gather(rows, dim=1, world_size=self.tp_size)
            )
            rows = gathered[:, : self.n_hash_cols]
        return rows


class Engram(nn.Module):
    gate_unrotate: torch.Tensor | None

    def __init__(
        self,
        layout: EngramLayout,
        layer_hash_index: int,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        storage: str = ENGRAM_STORAGE_FP8,
        gate_unrotate: torch.Tensor | None = None,
        *,
        dp_size: int = 1,
        dp_rank: int = 0,
        host_offload: bool = False,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.eps = rms_norm_eps
        self.embed_tokens = EngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            layout.head_sizes(layer_hash_index),
            tp_size,
            tp_rank,
            storage=storage,
            device=device,
            dp_size=dp_size,
            dp_rank=dp_rank,
            host_offload=host_offload,
        )
        n_hash_cols = layout.n_hash_cols
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            hidden_size * (hc_mult + 1),
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        # QuaRot gates need the original basis; unrotated checkpoints leave this unset.
        self.register_buffer("gate_unrotate", gate_unrotate, persistent=False)
        self._staged_rows: torch.Tensor | None = None

    def prepare_embeddings(self, hash_ids: torch.Tensor, dp_scatter: DpScatterState | None = None) -> None:
        self._staged_rows = self.embed_tokens.lookup(hash_ids, dp_scatter)

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if self._staged_rows is None or self._staged_rows.shape[0] < hash_ids.shape[0]:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)
        return self._staged_rows[: hash_ids.shape[0]]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        prefetched: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = self.embed(hash_ids) if prefetched is None else prefetched
        rows = rows.to(self.wkv.weight.dtype)
        kv = self.wkv(rows.flatten(-2))
        num_tokens, hc_mult, dim = hidden_states.shape
        hidden = hidden_states.float()
        # The gate needs the original basis; wkv already produces rotated values.
        gate_hidden = hidden
        if self.gate_unrotate is not None:
            width = self.gate_unrotate.size(0)
            gate_hidden = (hidden.reshape(-1, width) @ self.gate_unrotate).reshape_as(hidden)
        if (
            in_acl_graph()
            and hidden_states.device.type in ("npu", "privateuseone")
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and dim == 5120
            and num_tokens > 0
            and token_mask is not None
            and token_mask.is_contiguous()
        ):
            from xllm.python.kernels_npu.triton.engram import engram_gate

            return engram_gate(
                hidden_states, gate_hidden, kv, self.q_weight, self.k_weight, token_mask, self.eps, _ENGRAM_GATE_CLAMP
            )
        keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
        value = kv[:, hc_mult * dim :].float()
        hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + self.eps)
        key_rms = torch.rsqrt(keys.square().mean(-1) + self.eps)
        dot = (gate_hidden * self.q_weight.float() * self.k_weight.float() * keys).sum(-1)
        dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
        gate_input = dot.abs().clamp_min(_ENGRAM_GATE_CLAMP).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        if token_mask is not None:
            gate = torch.where(token_mask.reshape(-1, 1), gate, torch.zeros_like(gate))
        output = hidden + gate.unsqueeze(-1) * value.unsqueeze(1)
        return output.to(hidden_states.dtype)

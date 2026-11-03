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

from itertools import accumulate
from typing import TYPE_CHECKING, cast

import torch

from xllm.python.attention import backend as attention_backend
from xllm.python.attention.backend import ExecutionMetadataSource, LayerCache, resolve_linear_state_io_indices
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnDummyMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnStateCache,
    GdnVerifyMetadata,
)

if TYPE_CHECKING:
    from xllm.python.model_executor.input_batch import InputBatch
    from xllm.python.models.qwen3_5 import Qwen3_5Config

_MEGA_GDN_CHUNK_SIZE = 128


def _layout_value(layout: object, name: str, *aliases: str) -> int | None:
    """Read one validated shared-cache layout field without coupling to its class."""
    for field_name in (name, *aliases):
        value = getattr(layout, field_name, None)
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen3.5 shared cache layout field {field_name} must be positive")
            return value
    return None


def _detect_shared_layout(layer_caches: list[LayerCache]) -> object | None:
    detector = getattr(attention_backend, "detect_shared_kv_cache_layout", None)
    if detector is not None:
        return detector(layer_caches)
    for cache in layer_caches:
        layout = getattr(cache, "shared_layout", None)
        if layout is not None:
            return layout
    return None


def _build_mega_prefill_indices(
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    checkpoint_stride: int,
    conv_page_row_stride: int | None = None,
    ssm_page_row_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if checkpoint_stride <= 0:
        raise ValueError("Qwen3.5 SSM checkpoint stride must be positive")
    if has_initial_state.shape != read_state_indices.shape:
        raise ValueError("Qwen3.5 state validity must be sequence-scoped")
    valid_read = has_initial_state & (read_state_indices > 0)
    conv_read = torch.where(valid_read, read_state_indices, -1)
    if conv_page_row_stride is None or ssm_page_row_stride is None:
        if checkpoint_stride == 1:
            return conv_read, write_state_indices, conv_read, write_state_indices
        return (
            conv_read,
            write_state_indices,
            torch.where(valid_read, read_state_indices * checkpoint_stride, -1),
            write_state_indices * checkpoint_stride,
        )
    if conv_page_row_stride <= 0 or ssm_page_row_stride <= 0:
        raise ValueError("Qwen3.5 shared cache page row strides must be positive")
    conv_read = torch.where(valid_read, read_state_indices * conv_page_row_stride, -1)
    conv_write = torch.where(
        write_state_indices >= 0,
        write_state_indices * conv_page_row_stride,
        write_state_indices,
    )
    ssm_read = torch.where(valid_read, read_state_indices * ssm_page_row_stride, -1)
    ssm_write = torch.where(
        write_state_indices >= 0,
        write_state_indices * ssm_page_row_stride,
        write_state_indices,
    )
    return conv_read, conv_write, ssm_read, ssm_write


def _scale_state_indices(indices: torch.Tensor, row_stride: int) -> torch.Tensor:
    """Translate logical state slots while preserving negative padding sentinels."""
    if row_stride <= 0:
        raise ValueError("Qwen3.5 shared cache row stride must be positive")
    return torch.where(indices >= 0, indices * row_stride, indices)


def _compute_mega_prefill_num_matrices(query_lengths: list[int], num_value_heads: int) -> int:
    if not query_lengths or any(length <= 0 for length in query_lengths):
        raise ValueError("Qwen3.5 prefill query lengths must be positive")
    return (
        sum((length + _MEGA_GDN_CHUNK_SIZE - 1) // _MEGA_GDN_CHUNK_SIZE for length in query_lengths) * num_value_heads
    )


class Qwen3_5GdnMetadataBuilder:
    metadata_type: type[object] = GdnMetadata
    supports_linear_spec_verify_graph = True

    def __init__(self, cfg: Qwen3_5Config) -> None:
        if cfg.linear_conv_kernel_dim != 4:
            raise NotImplementedError("Qwen3.5 MegaGdn requires convolution width 4")
        if cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128:
            raise NotImplementedError("Qwen3.5 MegaGdn requires K/V head dimension 128")
        self._layer_ids = tuple(i for i, kind in enumerate(cfg.layer_types) if kind == "linear_attention")
        self._num_value_heads = cfg.linear_num_value_heads // cfg.tp_size
        self._conv_dim = (2 * cfg.linear_num_key_heads + cfg.linear_num_value_heads) // cfg.tp_size * 128
        self._state_caches: dict[int, GdnStateCache] | None = None
        self._checkpoint_stride: int | None = None
        self._device: torch.device | None = None
        self._shared_layout: object | None = None
        self._conv_page_row_stride: int | None = None
        self._ssm_page_row_stride: int | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        state_caches: dict[int, GdnStateCache] = {}
        stride: int | None = None
        device: torch.device | None = None
        shared_layout = _detect_shared_layout(layer_caches)
        conv_page_row_stride: int | None = None
        ssm_page_row_stride: int | None = None
        if shared_layout is not None:
            conv_page_row_stride = _layout_value(shared_layout, "conv_row_stride", "conv_page_row_stride")
            ssm_page_row_stride = _layout_value(shared_layout, "ssm_row_stride", "ssm_page_row_stride")
            stride = _layout_value(shared_layout, "checkpoint_stride", "linear_checkpoint_stride")
            if conv_page_row_stride is None or ssm_page_row_stride is None or stride is None:
                raise ValueError("Qwen3.5 shared cache layout is missing GDN row strides")
        for layer_id in self._layer_ids:
            if layer_id >= len(layer_caches):
                raise ValueError(f"GDN cache is missing for layer {layer_id}")
            conv, ssm = layer_caches[layer_id].conv, layer_caches[layer_id].ssm
            if conv is None or ssm is None:
                raise ValueError(f"GDN state cache is missing for layer {layer_id}")
            if conv.dim() != 3 or conv.shape[0] <= 0 or conv.shape[2] != self._conv_dim:
                raise ValueError(f"GDN Conv cache has invalid geometry for layer {layer_id}")
            if ssm.dim() != 4 or ssm.shape[0] <= 0 or ssm.shape[1:] != (self._num_value_heads, 128, 128):
                raise ValueError(f"GDN SSM cache has invalid geometry for layer {layer_id}")
            if conv.dtype != torch.bfloat16 or ssm.dtype != torch.float32:
                raise ValueError(f"GDN state cache has invalid dtype for layer {layer_id}")
            if conv.device != ssm.device or (device is not None and device != conv.device):
                raise ValueError("Qwen3.5 GDN state caches must use one device")
            if shared_layout is None:
                if ssm.shape[0] % conv.shape[0]:
                    raise ValueError(f"GDN SSM cache has invalid checkpoint rows for layer {layer_id}")
                layer_stride = ssm.shape[0] // conv.shape[0]
                if conv.shape[1] != layer_stride + 2 or (stride is not None and stride != layer_stride):
                    raise ValueError("Qwen3.5 GDN caches must use one valid checkpoint stride")
            else:
                layer_stride = stride
                if conv.shape[1] != layer_stride + 2:
                    raise ValueError(
                        f"GDN shared Conv cache checkpoint geometry is invalid for layer {layer_id}"
                    )
            stride, device = layer_stride, conv.device
            state_caches[layer_id] = cast(GdnStateCache, layer_caches[layer_id])
        self._state_caches = state_caches
        self._checkpoint_stride = stride
        self._device = device
        self._shared_layout = shared_layout
        self._conv_page_row_stride = conv_page_row_stride
        self._ssm_page_row_stride = ssm_page_row_stride

    def allocate_persistent(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> GdnMetadata:
        if (
            not input_batch.is_dummy
            and (metadata.is_prefill or metadata.is_chunked_prefill)
            and not getattr(metadata, "is_spec_verify", False)
            and getattr(metadata, "linear_verify_graph", None) is None
        ):
            raise NotImplementedError("Qwen3.5 GDN ACL Graph supports decode and verify only")
        return self.build(input_batch, metadata)

    def update_persistent(
        self, persistent_metadata: object, input_batch: InputBatch, metadata: ExecutionMetadataSource
    ) -> None:
        # Graph capture keeps the metadata object and its tensor addresses.  A
        # fresh build is only a source of values; copy each tensor into the
        # persistent alias rather than replacing the dataclass or its views.
        if not isinstance(persistent_metadata, GdnMetadata):
            raise TypeError("Qwen3.5 GDN persistent metadata has an invalid type")
        updated = self.build(input_batch, metadata)
        if type(updated) is not type(persistent_metadata):
            raise ValueError("Qwen3.5 GDN graph metadata changed execution mode during replay")
        for field_name in (
            "read_state_indices",
            "write_state_indices",
            "conv_read_state_indices",
            "conv_write_state_indices",
            "ssm_read_state_indices",
            "ssm_write_state_indices",
            "state_indices",
            "num_accepted_tokens",
            "packed_to_dense",
            "dense_to_packed",
            "cu_seqlens",
            "conv_read_indices",
            "conv_write_indices",
            "ssm_read_indices",
            "ssm_write_indices",
        ):
            current = getattr(persistent_metadata, field_name, None)
            replacement = getattr(updated, field_name, None)
            if current is None or replacement is None:
                continue
            if current.shape != replacement.shape or current.dtype != replacement.dtype:
                raise ValueError(f"Qwen3.5 GDN graph metadata shape changed for {field_name}")
            current.copy_(replacement)

    def build(self, input_batch: InputBatch, metadata: ExecutionMetadataSource) -> GdnMetadata:
        if self._state_caches is None or self._checkpoint_stride is None:
            raise RuntimeError("Qwen3.5 GDN state caches must be bound before execution")
        # A padded graph entry must capture the stateful path for later replays.
        if input_batch.is_dummy and input_batch.is_padding is None:
            return GdnDummyMetadata(self._state_caches)
        device = input_batch.input_ids.device
        if device != self._device:
            raise ValueError("Qwen3.5 GDN input and state caches must use one device")
        graph = getattr(metadata, "linear_verify_graph", None)
        if graph is not None:
            if self._shared_layout is not None:
                raise NotImplementedError(
                    "Qwen3.5 shared Conv/SSM pages do not support packed verify graphs yet"
                )
            if graph.query_width != self._checkpoint_stride:
                raise ValueError("Qwen3.5 packed graph width must match the GDN checkpoint stride")
            batch_size = graph.state_indices.numel()
            token_count = input_batch.num_tokens_after_padding
            if graph.cu_seqlens is not None and (
                graph.cu_seqlens.shape != (batch_size + 1,)
                or graph.cu_seqlens.dtype != torch.int32
                or graph.cu_seqlens.device != device
            ):
                raise ValueError("Qwen3.5 packed graph requires device int32 request endpoints")
            tensors = (
                (graph.state_indices, batch_size, torch.int32),
                (graph.num_accepted_tokens, batch_size, torch.int32),
            )
            if graph.cu_seqlens is None:
                tensors += (
                    (graph.packed_to_dense, token_count, torch.int64),
                    (graph.dense_to_packed, batch_size * graph.query_width, torch.int64),
                )
            for tensor, count, dtype in tensors:
                if tensor is None or tensor.shape != (count,) or tensor.dtype != dtype or tensor.device != device:
                    raise ValueError("Qwen3.5 packed graph requires fixed request indices and token maps")
            return GdnVerifyMetadata(
                self._state_caches,
                graph.state_indices,
                graph.num_accepted_tokens,
                graph.query_width,
                graph.packed_to_dense,
                graph.dense_to_packed,
                graph.cu_seqlens,
            )
        read_indices, write_indices = resolve_linear_state_io_indices(metadata)
        if read_indices is None or write_indices is None:
            raise RuntimeError("Qwen3.5 GDN requires read/write state indices")
        if read_indices.dim() != 1 or write_indices.shape != read_indices.shape:
            raise ValueError("Qwen3.5 GDN state indices must be one-dimensional and shape-aligned")
        read_indices = read_indices.to(device=device, dtype=torch.int32).contiguous()
        write_indices = write_indices.to(device=device, dtype=torch.int32).contiguous()
        # Verify is also marked chunked prefill by the speculative worker.
        # It must restore the last accepted checkpoint before processing rows.
        if getattr(metadata, "is_spec_verify", False):
            if self._shared_layout is not None:
                raise NotImplementedError(
                    "Qwen3.5 shared Conv/SSM pages do not support speculative verify yet"
                )
            return self._build_verify(input_batch, metadata, write_indices)
        if not (metadata.is_prefill or metadata.is_chunked_prefill):
            # Captured decode graphs index their whole padded bucket.
            if read_indices.numel() != input_batch.num_tokens_after_padding or read_indices.numel() == 0:
                raise ValueError("Qwen3.5 GDN decode requires one state slot per token")
            if self._checkpoint_stride != 1 and self._shared_layout is None:
                raise ValueError("Qwen3.5 MegaGdnDecode requires one SSM row per Conv slot")
            if self._shared_layout is None:
                return GdnDecodeMetadata(self._state_caches, read_indices, write_indices)
            assert self._conv_page_row_stride is not None and self._ssm_page_row_stride is not None
            conv_state_indices = _scale_state_indices(read_indices, self._conv_page_row_stride)
            conv_write_state_indices = _scale_state_indices(write_indices, self._conv_page_row_stride)
            ssm_state_indices = _scale_state_indices(read_indices, self._ssm_page_row_stride)
            ssm_write_state_indices = _scale_state_indices(write_indices, self._ssm_page_row_stride)
            return GdnDecodeMetadata(
                self._state_caches,
                read_indices,
                write_indices,
                conv_state_indices,
                conv_write_state_indices,
                ssm_state_indices,
                ssm_write_state_indices,
            )
        query_lengths_host = metadata.q_seq_lens_host
        if query_lengths_host is None:
            raise RuntimeError("Qwen3.5 MegaGdnPrefill requires host query lengths")
        if (
            query_lengths_host.device.type != "cpu"
            or query_lengths_host.dim() != 1
            or query_lengths_host.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Qwen3.5 host query lengths must be a CPU int32/int64 vector")
        query_lengths: list[int] = query_lengths_host.tolist()
        query_ends = list(accumulate(query_lengths, initial=0))
        if (
            len(query_lengths) != input_batch.num_reqs
            or tuple(int(value) for value in input_batch.num_scheduled_tokens) != tuple(query_lengths)
            or tuple(int(value) for value in input_batch.query_start_loc) != tuple(query_ends)
            or len(input_batch.num_computed_tokens) != input_batch.num_reqs
            or len(input_batch.is_prefilling) != input_batch.num_reqs
        ):
            raise ValueError("Qwen3.5 GDN request layout disagrees with the execution batch")
        if len(query_lengths) != write_indices.numel() or sum(query_lengths) != input_batch.num_tokens:
            raise ValueError("Qwen3.5 packed token count does not match host query lengths")
        has_initial_state = metadata.has_initial_state
        if has_initial_state is None:
            raise RuntimeError("has_initial_state is required by Qwen3.5 prefill")
        conv_read, conv_write, ssm_read, ssm_write = _build_mega_prefill_indices(
            read_indices,
            write_indices,
            has_initial_state.to(device=device, dtype=torch.bool),
            self._checkpoint_stride,
            self._conv_page_row_stride,
            self._ssm_page_row_stride,
        )
        cu_seqlens = metadata.q_cu_seq_lens
        host_query_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
        if host_query_ends and host_query_ends != query_ends and host_query_ends != query_ends[1:]:
            raise ValueError("Qwen3.5 cumulative query lengths do not match host query lengths")
        # Prepared Slots expose B cumulative ends for paged attention. GDN's
        # packed prefill contract instead includes the initial zero (B+1).
        native_end_only = (
            cu_seqlens is not None
            and cu_seqlens.dim() == 1
            and cu_seqlens.numel() == len(query_lengths)
            and host_query_ends == query_ends[1:]
        )
        if native_end_only and cu_seqlens.device.type == "cpu" and cu_seqlens.tolist() != query_ends[1:]:
            raise ValueError("Qwen3.5 cumulative query lengths do not match host query lengths")
        if cu_seqlens is None or native_end_only or (cu_seqlens.device.type != "cpu" and not host_query_ends):
            cu_seqlens = torch.tensor(query_ends, dtype=torch.int32, device=device)
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() != len(query_lengths) + 1:
            raise ValueError("Qwen3.5 cu_seqlens must contain B+1 entries")
        if cu_seqlens.device.type == "cpu" and cu_seqlens.tolist() != query_ends:
            raise ValueError("Qwen3.5 cumulative query lengths do not match host query lengths")
        return GdnPrefillMetadata(
            self._state_caches,
            conv_read,
            conv_write,
            ssm_read,
            ssm_write,
            cu_seqlens.to(device=device, dtype=torch.int32).contiguous(),
            _compute_mega_prefill_num_matrices(query_lengths, self._num_value_heads),
        )

    def _build_verify(
        self,
        input_batch: InputBatch,
        metadata: ExecutionMetadataSource,
        state_indices: torch.Tensor,
    ) -> GdnVerifyMetadata:
        assert self._state_caches is not None and self._checkpoint_stride is not None
        width = self._checkpoint_stride
        if not 2 <= width <= 17:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode requires 2 to 17 checkpoints per state slot")
        accepted = metadata.num_accepted_tokens
        accepted_host = tuple(getattr(metadata, "num_accepted_tokens_host_values", ()))
        # Prepared pipeline inputs keep attention at BS one-query rows and
        # recurrent state at B rows. Their accepted counts are updated on the
        # execution stream, so retain the final Device views without reading
        # them on the Host or allocating graph-time index tensors.
        if not accepted_host and state_indices.numel() * width == input_batch.num_tokens_after_padding:
            if (
                state_indices.numel() == 0
                or accepted is None
                or accepted.ndim != 1
                or accepted.numel() != state_indices.numel()
                or accepted.dtype != torch.int32
                or accepted.device != state_indices.device
                or not accepted.is_contiguous()
            ):
                raise ValueError("Qwen3.5 MTP requires contiguous int32 accepted lengths per logical sequence")
            return GdnVerifyMetadata(self._state_caches, state_indices, accepted, width, None)
        query_lengths_host = metadata.q_seq_lens_host
        if (
            query_lengths_host is None
            or query_lengths_host.device.type != "cpu"
            or query_lengths_host.ndim != 1
            or query_lengths_host.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Qwen3.5 MTP requires sequence-scoped CPU query lengths")
        lengths = query_lengths_host.tolist()
        if not lengths or any(length <= 0 or length > width for length in lengths):
            raise ValueError("Qwen3.5 MTP query lengths must be in [1, checkpoint stride]")
        token_count = sum(lengths)
        if token_count != input_batch.num_tokens:
            raise ValueError("Qwen3.5 MTP query lengths must cover every packed token")
        batch_size = len(lengths)
        if state_indices.numel() == token_count:
            # The C++ Python bridge can expand one logical state ID over all
            # tokens of its request. Graph entries own sequence-scoped storage.
            starts = torch.tensor(list(accumulate(lengths, initial=0))[:-1], device=state_indices.device)
            state_indices = state_indices.index_select(0, starts)
        elif state_indices.numel() != batch_size:
            raise ValueError("Qwen3.5 MTP requires one state slot per request or packed token")
        if accepted is None or accepted.ndim != 1 or accepted.numel() != batch_size:
            raise ValueError("Qwen3.5 MTP accepted counts must be sequence-scoped")
        if len(accepted_host) != batch_size or any(count < 1 or count > width for count in accepted_host):
            raise ValueError("Qwen3.5 MTP accepted counts must be in [1, checkpoint stride]")
        packed_to_dense = None
        if any(length != width for length in lengths):
            packed_to_dense = torch.tensor(
                [row * width + offset for row, length in enumerate(lengths) for offset in range(length)],
                dtype=torch.int64,
                device=state_indices.device,
            )
        return GdnVerifyMetadata(
            self._state_caches,
            state_indices,
            accepted.to(device=state_indices.device, dtype=torch.int32).contiguous(),
            width,
            packed_to_dense,
        )

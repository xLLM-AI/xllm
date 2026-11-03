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

"""NPU-native Qwen3.5 gated delta network composition."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.flash_comm1 import get_flash_comm1_context
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.npu.flash_comm1 import FlashComm1RowParallelLinear
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnDummyMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnStateCache,
    GdnVerifyMetadata,
)
from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.layers.qwen3_5.gated_delta_net import load_gdn_weights
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_execution_context,
    in_acl_graph,
)
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

_MEGA_GDN_MAX_DECODE_BATCH_SIZE = 32
_MEGA_GDN_MAX_VARLEN_BATCH_SIZE = 128
_MEGA_GDN_FIXED_RMS_NORM_EPS = 1e-6
_SUPPORTED_MEGA_PREFILL_HEAD_COUNTS = frozenset((1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))


@dataclass(frozen=True, slots=True)
class _StagedGdnState:
    """Compact fused-kernel state and its physical shared-arena destinations."""

    conv: torch.Tensor
    ssm: torch.Tensor
    indices: torch.Tensor
    conv_rows: torch.Tensor
    ssm_rows: torch.Tensor
    checkpoint_stride: int

    def write_back(self, conv_cache: torch.Tensor, ssm_cache: torch.Tensor) -> None:
        conv_cache.view(-1, conv_cache.shape[-1]).index_copy_(
            0, self.conv_rows, self.conv[1:].reshape(-1, conv_cache.shape[-1])
        )
        ssm_cache.index_copy_(0, self.ssm_rows, self.ssm[self.checkpoint_stride :])


class NpuQwen3_5GatedDeltaNet(nn.Module):
    """NPU graph whose nodes match the native fused operator boundaries."""

    def __init__(
        self,
        cfg: Qwen3_5GatedDeltaNetConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not math.isclose(
            cfg.rms_norm_eps,
            _MEGA_GDN_FIXED_RMS_NORM_EPS,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise NotImplementedError(
                "Qwen3.5 MegaGdn fuses RMSNorm with a fixed epsilon; "
                f"cfg.rms_norm_eps={cfg.rms_norm_eps} is unsupported"
            )
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_k_heads = cfg.linear_num_key_heads // cfg.tp_size
        self.num_v_heads = cfg.linear_num_value_heads // cfg.tp_size
        if self.num_v_heads not in _SUPPORTED_MEGA_PREFILL_HEAD_COUNTS:
            raise NotImplementedError(f"Qwen3.5 MegaGdnPrefill does not support {self.num_v_heads} local value heads")
        if self.num_k_heads <= 0 or self.num_k_heads > 16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
        if self.num_k_heads & (self.num_k_heads - 1):
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
        if self.num_v_heads % self.num_k_heads != 0:
            raise NotImplementedError("Qwen3.5 MegaGdn requires Nv divisible by Nk")
        if self.num_v_heads // self.num_k_heads > 4:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32, device=device))
        # MegaGdn consumes dt_bias as FP32. Keep it in the required dtype once
        # instead of allocating a converted tensor on every forward.
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32, device=device))
        self.norm_weight = nn.Parameter(torch.ones(self.value_head_dim, dtype=dtype, device=device))
        self.out_proj = FlashComm1RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
            use_checkpoint_layout=True,
        )
        # Prefill and Decode share rank-local QKVZ and BA projections. Weight
        # loading and Verify consume row views of the same storage.
        self.in_proj_qkvz = ColumnParallelLinear(
            cfg.hidden_size,
            self.conv_dim + self.value_dim,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_ba = ColumnParallelLinear(
            cfg.hidden_size,
            2 * self.num_v_heads,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_kernel_size,
                self.conv_dim,
                dtype=dtype,
                device=device,
            )
        )
        self._prefill_projection_sources: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prefill_projection_weights: list[torch.Tensor] = []

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> NpuQwen3_5GatedDeltaNet:
        # Module.to() can replace parameter storage without changing the
        # Parameter object. Rebuild our non-owning views after any conversion.
        self._prefill_projection_sources = None
        self._prefill_projection_weights = []
        return super()._apply(fn, recurse)

    def _input_projection_weight_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight = self.in_proj_qkvz.weight[: self.conv_dim]
        z_weight = self.in_proj_qkvz.weight[self.conv_dim :]
        b_weight = self.in_proj_ba.weight[: self.num_v_heads]
        a_weight = self.in_proj_ba.weight[self.num_v_heads :]
        return qkv_weight, z_weight, b_weight, a_weight

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        load_gdn_weights(
            self,
            state,
            context,
            self._input_projection_weight_views(),
            transpose_conv=True,
        )

    def _project_prefill_inputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inspect registered parameters directly: dynamic Module attribute
        # lookups otherwise dominate cache validation for small prefills.
        qkvz_weight = self._modules["in_proj_qkvz"]._parameters["weight"]
        ba_weight = self._modules["in_proj_ba"]._parameters["weight"]
        sources = self._prefill_projection_sources
        if sources is None or sources[0] is not qkvz_weight or sources[1] is not ba_weight:
            self._prefill_projection_weights = [weight.t() for weight in self._input_projection_weight_views()]
            self._prefill_projection_sources = (qkvz_weight, ba_weight)
        mixed_qkv, z, b, a = kernels.grouped_linear(hidden, self._prefill_projection_weights)
        return mixed_qkv, a, b, z.view(-1, self.num_v_heads, self.value_head_dim)

    def _project_verify_inputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight, z_weight, b_weight, a_weight = self._input_projection_weight_views()
        mixed_qkv = torch.nn.functional.linear(hidden, qkv_weight)
        z = torch.nn.functional.linear(hidden, z_weight).view(
            -1,
            self.num_v_heads,
            self.value_head_dim,
        )
        b = torch.nn.functional.linear(hidden, b_weight)
        a = torch.nn.functional.linear(hidden, a_weight)
        return mixed_qkv, a, b, z

    def _stage_shared_state(
        self,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        conv_indices: torch.Tensor,
        ssm_indices: torch.Tensor,
        *,
        checkpoint_stride: int,
        conv_rows: int,
    ) -> _StagedGdnState:
        """Adapt distinct physical row spaces to MegaGdn's common slot ABI."""
        batch_size = conv_indices.numel()
        conv_shape = (batch_size + 1, conv_rows, self.conv_dim)
        ssm_shape = ((batch_size + 1) * checkpoint_stride, *ssm_cache.shape[1:])
        if in_acl_graph():
            # Linear layers execute serially on the graph stream. Sharing only
            # this temporary state avoids retaining a second model-sized cache.
            conv = get_execution_buffer(
                ("GDN_SHARED_CONV", conv_shape, conv_cache.dtype),
                lambda: conv_cache.new_empty(conv_shape),
                shared=True,
            )
            ssm = get_execution_buffer(
                ("GDN_SHARED_SSM", ssm_shape, ssm_cache.dtype),
                lambda: ssm_cache.new_empty(ssm_shape),
                shared=True,
            )
            indices = get_execution_buffer(
                ("GDN_SHARED_STATE_INDICES", batch_size),
                lambda: torch.arange(1, batch_size + 1, dtype=torch.int32, device=conv_cache.device),
                shared=True,
            )
        else:
            conv = conv_cache.new_empty(conv_shape)
            ssm = ssm_cache.new_empty(ssm_shape)
            indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=conv_cache.device)
        conv_offsets = torch.arange(conv_rows, dtype=torch.int64, device=conv_cache.device)
        ssm_offsets = torch.arange(checkpoint_stride, dtype=torch.int64, device=ssm_cache.device)
        conv_physical_rows = (
            conv_indices.to(torch.int64).clamp_min(0)[:, None] * conv_cache.shape[1] + conv_offsets
        ).reshape(-1)
        ssm_physical_rows = (ssm_indices.to(torch.int64).clamp_min(0)[:, None] + ssm_offsets).reshape(-1)
        # The null slot is private to these buffers. Padded sequences also get
        # distinct dense slots, so fused kernels never share an update target.
        conv[0].zero_()
        ssm[:checkpoint_stride].zero_()
        torch.index_select(
            conv_cache.view(-1, self.conv_dim),
            0,
            conv_physical_rows,
            out=conv[1:].view(-1, self.conv_dim),
        )
        torch.index_select(ssm_cache, 0, ssm_physical_rows, out=ssm[checkpoint_stride:])
        return _StagedGdnState(conv, ssm, indices, conv_physical_rows, ssm_physical_rows, checkpoint_stride)

    def _decode(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
        conv_read_state_indices: torch.Tensor | None = None,
        conv_write_state_indices: torch.Tensor | None = None,
        ssm_read_state_indices: torch.Tensor | None = None,
        ssm_write_state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        mixed_qkvz = self.in_proj_qkvz(hidden)
        mixed_ba = self.in_proj_ba(hidden)
        mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
        z = z.view(batch_size, self.num_v_heads, self.value_head_dim)
        shared_indices = (
            conv_read_state_indices,
            conv_write_state_indices,
            ssm_read_state_indices,
            ssm_write_state_indices,
        )
        if any(index is not None for index in shared_indices):
            if any(index is None for index in shared_indices):
                raise ValueError("Qwen3.5 shared GDN decode requires all Conv/SSM state indices")
            assert (
                conv_read_state_indices is not None
                and conv_write_state_indices is not None
                and ssm_read_state_indices is not None
                and ssm_write_state_indices is not None
            )
            staged = self._stage_shared_state(
                conv_state,
                ssm_state,
                conv_read_state_indices,
                ssm_read_state_indices,
                checkpoint_stride=1,
                conv_rows=self.conv_kernel_size - 1,
            )
            conv_cache, ssm_cache = conv_state, ssm_state
            conv_state, ssm_state = staged.conv, staged.ssm
            read_state_indices = write_state_indices = staged.indices
        else:
            staged = None
        chunk_outputs = []
        for start in range(0, batch_size, _MEGA_GDN_MAX_DECODE_BATCH_SIZE):
            end = min(start + _MEGA_GDN_MAX_DECODE_BATCH_SIZE, batch_size)
            chunk_outputs.append(
                kernels.mega_gdn_decode(
                    mixed_qkv[start:end].contiguous(),
                    z[start:end].contiguous(),
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    self.conv1d_weight,
                    conv_state,
                    self.A_log,
                    self.dt_bias,
                    ssm_state,
                    read_state_indices[start:end],
                    write_state_indices[start:end],
                    self.norm_weight,
                    True,
                )
            )
        output = chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)
        if staged is not None:
            staged.write_back(conv_cache, ssm_cache)
        return output

    def _verify(self, hidden: torch.Tensor, metadata: GdnVerifyMetadata) -> torch.Tensor:
        batch_size = metadata.state_indices.numel()
        width = metadata.query_width
        state_cache = metadata.state_caches[self.layer_id]
        if metadata.conv_state_indices is not None or metadata.ssm_state_indices is not None:
            if metadata.conv_state_indices is None or metadata.ssm_state_indices is None:
                raise ValueError("Qwen3.5 shared verify requires both Conv/SSM physical state indices")
            staged = self._stage_shared_state(
                state_cache.conv,
                state_cache.ssm,
                metadata.conv_state_indices,
                metadata.ssm_state_indices,
                checkpoint_stride=width,
                conv_rows=width + self.conv_kernel_size - 2,
            )
            compact_cache: dict[int, GdnStateCache] = {self.layer_id: staged}
            output = self._verify(
                hidden,
                replace(
                    metadata,
                    state_caches=compact_cache,
                    state_indices=staged.indices,
                    conv_state_indices=None,
                    ssm_state_indices=None,
                ),
            )
            staged.write_back(state_cache.conv, state_cache.ssm)
            return output

        if metadata.cu_seqlens is not None:
            mixed_qkv, a, b, z = (tensor.contiguous() for tensor in self._project_prefill_inputs(hidden))
            output = torch.zeros_like(z)
            for start in range(0, batch_size, _MEGA_GDN_MAX_VARLEN_BATCH_SIZE):
                end = min(start + _MEGA_GDN_MAX_VARLEN_BATCH_SIZE, batch_size)
                indices = metadata.state_indices[start:end]
                kernels.mega_gdn_mtp_decode_varlen(
                    mixed_qkv,
                    z,
                    b,
                    a,
                    self.conv1d_weight,
                    state_cache.conv,
                    self.A_log,
                    self.dt_bias,
                    state_cache.ssm,
                    indices,
                    metadata.num_accepted_tokens[start:end],
                    self.norm_weight,
                    metadata.cu_seqlens[start : end + 1],
                    output,
                    True,
                )
            return output.reshape(-1, self.value_dim)

        def _pad(tensor: torch.Tensor) -> torch.Tensor:
            if metadata.dense_to_packed is not None:
                tensor = torch.cat((tensor, torch.zeros_like(tensor[:1])), dim=0)
                tensor = tensor.index_select(0, metadata.dense_to_packed)
            elif metadata.packed_to_dense is not None:
                dense = tensor.new_zeros((batch_size * width, *tensor.shape[1:]))
                tensor = dense.index_copy_(0, metadata.packed_to_dense, tensor)
            return tensor.reshape(batch_size, width, *tensor.shape[1:])

        # Match native project_split_inputs: project packed rows first, then
        # pad each request independently to the checkpoint stride.
        mixed_qkv, a, b, z = (_pad(tensor) for tensor in self._project_verify_inputs(hidden))
        chunk_outputs = []
        for start in range(0, batch_size, _MEGA_GDN_MAX_DECODE_BATCH_SIZE):
            end = min(start + _MEGA_GDN_MAX_DECODE_BATCH_SIZE, batch_size)
            indices = metadata.state_indices[start:end]
            chunk_outputs.append(
                kernels.mega_gdn_mtp_decode(
                    mixed_qkv[start:end].contiguous(),
                    z[start:end].contiguous(),
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    self.conv1d_weight,
                    state_cache.conv,
                    self.A_log,
                    self.dt_bias,
                    state_cache.ssm,
                    indices,
                    indices,
                    metadata.num_accepted_tokens[start:end],
                    self.norm_weight,
                    True,
                )
            )
        output = chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)
        output = output.reshape(batch_size * width, self.value_dim)
        if metadata.packed_to_dense is not None:
            if metadata.dense_to_packed is not None:
                output = torch.cat((output, torch.zeros_like(output[:1])), dim=0)
            output = output.index_select(0, metadata.packed_to_dense)
        return output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        flash_comm1 = get_flash_comm1_context()
        if flash_comm1 is not None:
            hidden = flash_comm1.gather(hidden)
        metadata = get_execution_context(GdnMetadata)
        if metadata is None:
            raise RuntimeError("Qwen3.5 GDN metadata is unavailable")
        if isinstance(metadata, GdnDummyMetadata):
            return torch.zeros_like(hidden)
        state_cache = metadata.state_caches[self.layer_id]
        if isinstance(metadata, GdnVerifyMetadata):
            output = self._verify(hidden, metadata)
        elif isinstance(metadata, GdnPrefillMetadata):
            mixed_qkv, a, b, z = self._project_prefill_inputs(hidden)
            output = kernels.mega_gdn_prefill(
                mixed_qkv,
                b,
                a,
                z,
                self.conv1d_weight,
                state_cache.conv,
                self.A_log,
                self.dt_bias,
                metadata.conv_read_indices,
                metadata.conv_write_indices,
                metadata.ssm_read_indices,
                metadata.ssm_write_indices,
                state_cache.ssm,
                metadata.cu_seqlens,
                self.norm_weight,
                metadata.num_matrices,
            )
        elif isinstance(metadata, GdnDecodeMetadata):
            output = self._decode(
                hidden,
                state_cache.conv,
                state_cache.ssm,
                metadata.read_state_indices,
                metadata.write_state_indices,
                metadata.conv_read_state_indices,
                metadata.conv_write_state_indices,
                metadata.ssm_read_state_indices,
                metadata.ssm_write_state_indices,
            )
        else:
            raise TypeError(f"Unsupported Qwen3.5 GDN metadata: {type(metadata).__name__}")
        return self.out_proj(output.reshape(-1, self.value_dim))

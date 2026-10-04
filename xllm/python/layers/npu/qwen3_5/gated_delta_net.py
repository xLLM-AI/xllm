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

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnDecodeMetadata, GdnMetadata, GdnPrefillMetadata
from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.layers.qwen3_5.gated_delta_net import load_gdn_weights
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

_MEGA_GDN_MAX_DECODE_BATCH_SIZE = 32
_MEGA_GDN_FIXED_RMS_NORM_EPS = 1e-6
_SUPPORTED_MEGA_PREFILL_HEAD_COUNTS = frozenset((1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))


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
        self.out_proj = RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
            use_checkpoint_layout=True,
        )
        # Keep one rank-local storage for each pair of projections. Prefill
        # consumes contiguous row views, while Decode runs two packed
        # projections and splits their outputs for the existing Mega operator.
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

    def _decode(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        mixed_qkvz = self.in_proj_qkvz(hidden)
        mixed_ba = self.in_proj_ba(hidden)
        mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
        z = z.view(batch_size, self.num_v_heads, self.value_head_dim)
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
        return chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        metadata = get_execution_context(GdnMetadata)
        if metadata is None:
            raise RuntimeError("Qwen3.5 GDN metadata is unavailable")
        state_cache = metadata.state_caches[self.layer_id]
        if isinstance(metadata, GdnPrefillMetadata):
            mixed_qkv, a, b, z = self._project_prefill_inputs(hidden)
            output = kernels.mega_gdn_prefill(
                mixed_qkv.contiguous(),
                b.contiguous(),
                a.contiguous(),
                z.contiguous(),
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
            )
        else:
            raise TypeError(f"Unsupported Qwen3.5 GDN metadata: {type(metadata).__name__}")
        return self.out_proj(output.reshape(-1, self.value_dim))

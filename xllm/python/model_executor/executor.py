# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, KVCache
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.runners.eager import EagerRunner


def _is_npu_device(device: torch.device) -> bool:
    return device.type in ("npu", "privateuseone")


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    if _is_npu_device(device):
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )
        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    if device.type == "cuda":
        from xllm.python.attention.flashinfer import FlashInferBackend
        return FlashInferBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    raise NotImplementedError(
        f"No attention backend available for device type '{device.type}'"
    )


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
    ) -> None:
        self.model = model
        self._kv_bound = False

        attention_layers = [
            module for module in model.modules() if isinstance(module, Attention)
        ]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")

        first_attention = attention_layers[0]
        expected_config = self._attention_config(first_attention)
        for layer in attention_layers[1:]:
            if self._attention_config(layer) != expected_config:
                raise ValueError(
                    "Attention backend requires identical attention configuration "
                    "across all layers"
                )

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        self.attention_backend = _create_attention_backend(
            first_attention, device, first_parameter.dtype
        )

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        self.decode_cuda_graph_runner = None
        self.inductor_runner = None

        graph_backend = str(config.get("python_graph_backend", "off")).lower()
        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )
            self.decode_cuda_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
            )
        else:
            from xllm.python.model_executor.runners.inductor import InductorRunner
            self.inductor_runner = InductorRunner(
                execution_model, self.attention_backend, device, graph_backend
            )

    @staticmethod
    def _attention_config(layer: Attention) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
        if len(kv_caches) != self._num_attention_layers:
            raise ValueError(
                "KV cache layer count does not match model attention layer count"
            )
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(kv_caches)
        self._kv_bound = True

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        graph_runner = self.decode_cuda_graph_runner
        if graph_runner is not None:
            graph_runner.warmup(input_ids.device, input_ids.dtype)
            if graph_runner.can_execute(input_ids, metadata):
                return graph_runner.execute(input_ids, positions, metadata)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(input_ids, positions, metadata)
        return self.eager_runner.execute(input_ids, positions, metadata)

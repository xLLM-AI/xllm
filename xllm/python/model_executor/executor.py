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

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform


def _resolve_graph_backend(config: dict) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False):
        if current_platform.is_npu():
            return "aclgraph"
    return graph_backend


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    if current_platform.is_npu():
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
    if current_platform.is_cuda():
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
    raise NotImplementedError(f"No attention backend available for device type '{device.type}'")


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
    ) -> None:
        self.model = model
        self._kv_bound = False

        attention_layers = [module for module in model.modules() if isinstance(module, Attention)]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")

        first_attention = attention_layers[0]
        expected_config = self._attention_config(first_attention)
        for layer in attention_layers[1:]:
            if self._attention_config(layer) != expected_config:
                raise ValueError("Attention backend requires identical attention configuration across all layers")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        self.attention_backend = _create_attention_backend(first_attention, device, first_parameter.dtype)

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        # Context-Parallel: shard prefill sequences across the CP group. Decode
        # stays on the non-CP path (CP is prefill-only, eager-only in v1).
        self.eager_runner.cp_size = int(config.get("cp_size", 1))
        self.eager_runner.cp_rank = int(config.get("cp_rank", 0))
        self.decode_graph_runner = None
        self.inductor_runner = None

        graph_backend = _resolve_graph_backend(config)
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        if dp_size > 1 and graph_backend not in (
            "",
            "off",
            "none",
            "0",
            "cudagraphs",
            "aclgraph",
        ):
            raise NotImplementedError("Python data parallel graph execution supports cudagraphs and aclgraph only")
        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            self.decode_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
            )
        elif graph_backend == "aclgraph":
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
            )
        else:
            if self.eager_runner.cp_size > 1:
                # CP is prefill-only and lives on eager_runner; a compile
                # backend serves prefill through InductorRunner, which carries
                # no cp_context, so CP would silently no-op. Reject rather than
                # run without the requested sharding.
                raise NotImplementedError(
                    "Context-Parallel (cp_size > 1) is not supported with the "
                    f"'{graph_backend}' graph backend; CP is eager-only. Use "
                    "graph_backend=off/aclgraph, or set cp_size=1."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

    @staticmethod
    def _attention_config(layer: Attention) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        required_layers = max(layer.layer_id for layer in self.model.modules() if isinstance(layer, Attention)) + 1
        if len(layer_caches) < required_layers:
            raise ValueError("cache layer count does not match the model layer layout")
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
        if self.inductor_runner is not None:
            self.inductor_runner.bind_layer_caches(layer_caches)
        self._kv_bound = True

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        graph_runner = self.decode_graph_runner
        if graph_runner is not None and graph_runner.can_execute(input_ids, metadata, input_embedding):
            graph_runner.warmup(input_ids.device, input_ids.dtype, input_embedding)
            return graph_runner.execute(input_ids, positions, metadata, input_embedding)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
        )

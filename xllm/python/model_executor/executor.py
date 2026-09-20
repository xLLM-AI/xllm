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

from xllm.python import distributed
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform


def _resolve_graph_backend(config: dict) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False):
        if current_platform.is_npu():
            return "aclgraph"
    return graph_backend


def _validate_npu_cp_model_config(config: dict, num_decoding_tokens: int) -> None:
    """Mirror model-side CP admission for callers without the C++ master."""
    model_type = config.get("model_type", "")
    if model_type not in ("qwen3", "glm_moe_dsa"):
        raise NotImplementedError(
            f"Python model-side CP does not support model_type={model_type!r}; "
            "supported models are qwen3 and glm_moe_dsa"
        )
    if config.get("task_type", "generate") != "generate":
        raise NotImplementedError("Python model-side CP supports only the generate task")
    role = config.get("instance_role", "DEFAULT")
    if role not in ("DEFAULT", "PREFILL"):
        raise NotImplementedError("Python model-side CP supports only DEFAULT or PREFILL roles")
    speculative = int(config.get("num_speculative_tokens", 0)) > 0 or num_decoding_tokens > 1
    algorithm = str(config.get("speculative_algorithm", "mtp")).lower()
    if speculative and algorithm in ("eagle3", "dflash", "dflash2", "dspark"):
        raise NotImplementedError("Python model-side CP does not support aux-hidden-capture speculative algorithms")
    if model_type == "glm_moe_dsa" and speculative and algorithm == "mtp":
        raise NotImplementedError("Python model-side CP does not support MTP speculative verification; use cp_size=1")
    kv_split = int(config.get("kv_split_size", 0)) or int(config["cp_size"])
    if model_type == "glm_moe_dsa" and kv_split > 1:
        if not config.get("enable_disagg_pd", False) or role != "PREFILL":
            raise NotImplementedError(
                "Python GLM CP with kv_split_size > 1 requires disaggregated PD with the PREFILL role"
            )


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
    config: dict | None = None,
    max_num_reqs: int = 1,
) -> AttentionBackend:
    config = config or {}
    model_type = config.get("model_type", "")
    if model_type == "deepseek_v4" and current_platform.is_npu():
        from xllm.python.attention.dsa_attention import DsaAttentionBackend

        return DsaAttentionBackend(
            compress_ratios=list(config.get("compress_ratios", [])),
            window_size=int(config.get("window_size", 128)),
            n_layers=int(config.get("n_layers", config.get("num_hidden_layers", 0))),
            num_heads=first_attention.num_heads,
            attn_head_dim=first_attention.head_dim,
            index_topk=int(config.get("index_topk", 512)),
            index_n_heads=int(config.get("index_n_heads", 64)),
            index_head_dim=int(config.get("index_head_dim", 128)),
            rope_head_dim=int(config.get("qk_rope_head_dim", 64)),
            device=device,
            dtype=dtype,
        )
    if current_platform.is_npu():
        dcp_group = distributed.dcp_group(device)
        if int(config.get("cp_size", 1)) == 1 and dcp_group is not None and dcp_group.size() > 1:
            from xllm.python.attention.sfa_dcp_backend import (
                SfaDcpAttentionBackend,
                dcp_layer_options,
            )

            index_topk = dcp_layer_options(first_attention)
            return SfaDcpAttentionBackend(
                num_heads=first_attention.num_heads,
                num_kv_heads=first_attention.num_kv_heads,
                head_dim=first_attention.head_dim,
                scale=first_attention.scale,
                sliding_window=first_attention.sliding_window,
                device=device,
                dtype=dtype,
                dcp_group=dcp_group,
                index_topk=index_topk,
                max_num_reqs=max(max_num_reqs, 1),
            )
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            is_mla=bool(config.get("enable_mla", False)),
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
        num_decoding_tokens: int = 1,
        acl_graph_decode_batch_size_limit: int | None = None,
    ) -> None:
        self.model = model
        self._kv_bound = False
        cp_size = int(config.get("cp_size", 1))
        cp_rank = int(config.get("cp_rank", 0))
        dp_size = int(config.get("dp_size", 1))
        if cp_size < 1:
            raise ValueError("cp_size must be greater than or equal to 1")
        if dp_size < 1:
            raise ValueError("dp_size must be greater than or equal to 1")
        if not 0 <= cp_rank < cp_size:
            raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
        if cp_size > 1 and dp_size > 1:
            raise NotImplementedError("Python CP requires dp_size == 1")
        if current_platform.is_npu():
            kv_split_size = int(config.get("kv_split_size", 0))
            if kv_split_size < 0:
                raise ValueError("kv_split_size must be nonnegative; 0 follows cp_size")
            effective_kv_split = kv_split_size or cp_size
            if cp_size > 1 and cp_size % effective_kv_split != 0:
                raise ValueError("Python CP requires effective kv_split_size to be a positive divisor of cp_size")
            if cp_size == 1 and effective_kv_split > 1 and dp_size > 1:
                raise NotImplementedError("Python DCP requires dp_size == 1 until DP-local KV groups are implemented")
        graph_backend = _resolve_graph_backend(config)
        if (
            current_platform.is_npu()
            and config.get("model_type") == "glm_moe_dsa_mtp"
            and graph_backend not in ("", "off", "none", "0", "aclgraph")
        ):
            # Cross-draft top-k and repair rows require the ACL runner's
            # persistent inputs. Other graph runners cannot retain that state.
            raise NotImplementedError(f"Python NPU GLM MTP requires graph_backend=off/aclgraph; got '{graph_backend}'")
        if cp_size > 1 and graph_backend not in ("", "off", "none", "0", "aclgraph"):
            # Only decode-only ACL graphs preserve CP prefill on EagerRunner.
            # Match the C++ admission gate before allocating backend state.
            raise NotImplementedError(
                f"Context-Parallel requires eager Prefill with graph_backend=off/aclgraph; got '{graph_backend}'"
            )
        if current_platform.is_npu() and cp_size > 1:
            _validate_npu_cp_model_config(config, int(num_decoding_tokens))

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
        num_decoding_tokens = max(1, int(num_decoding_tokens))
        # GLM MTP can prepend a repair row after all draft tokens are accepted.
        # Two requests then need up to four attention rows, even though later
        # draft steps still decode one token per request.
        max_decode_rows_per_request = num_decoding_tokens
        if config.get("model_type") == "glm_moe_dsa_mtp":
            max_decode_rows_per_request = max(max_decode_rows_per_request, 2)
        self._num_attention_layers = len(attention_layers)
        self.attention_backend = _create_attention_backend(
            first_attention,
            device,
            first_parameter.dtype,
            config,
            max(max_seqs_per_batch, 1) * max_decode_rows_per_request,
        )

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        # Context-Parallel: shard prefill sequences across the CP group. Decode
        # stays on the non-CP path (CP is prefill-only, eager-only in v1).
        self.eager_runner.cp_size = cp_size
        self.eager_runner.cp_rank = cp_rank
        self.layerwise_split_size = int(config.get("layerwise_split_size", 1))
        self.layerwise_split_rank = int(config.get("layerwise_split_rank", 0))
        if self.layerwise_split_size > 1 and config.get("model_type") != "glm_moe_dsa":
            raise NotImplementedError("Python layerwise split is supported only for GLM5.2")
        self.decode_graph_runner = None
        self.inductor_runner = None

        if self.layerwise_split_size > 1 and graph_backend not in ("", "off", "none", "0"):
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; "
                f"graph backend '{graph_backend}' is not supported."
            )
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size
        self._supports_prepared_metadata = (
            self.attention_backend.supports_prepared_metadata
            and config.get("model_type") in ("qwen3", "glm_moe_dsa")
            and int(config.get("kv_split_size", 1)) in (0, 1)
            and graph_backend in ("", "off", "none", "0")
            and all(int(config.get(key, 1)) == 1 for key in ("dp_size", "cp_size", "layerwise_split_size"))
        )
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

            decode_batch_size_limit = (
                None if acl_graph_decode_batch_size_limit is None else max(1, int(acl_graph_decode_batch_size_limit))
            )
            # The configured sequence budget is global, while the decode
            # limit is per DP rank. Round sequences before expanding MTP rows
            # so an uneven request split retains every row of the last request.
            local_graph_sequence_capacity = (max_seqs_per_batch + dp_size - 1) // dp_size
            if decode_batch_size_limit is not None:
                local_graph_sequence_capacity = min(
                    local_graph_sequence_capacity,
                    decode_batch_size_limit,
                )
            # The runner accepts a global token capacity and divides it by DP.
            max_graph_tokens = local_graph_sequence_capacity * max_decode_rows_per_request * dp_size
            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_graph_tokens,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
                decode_batch_size_limit,
                num_decoding_tokens,
            )
        else:
            if self.layerwise_split_size > 1:
                raise NotImplementedError(
                    "Python layerwise split requires eager execution; graph "
                    f"backend '{graph_backend}' is not supported."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

    @staticmethod
    def _attention_config(
        layer: Attention,
    ) -> tuple[int, int, int, float, int, bool, int | None, int, int, bool]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
            layer.causal,
            layer.fia_sparse_mode,
            layer.fia_pre_tokens,
            layer.fia_next_tokens,
            layer.fia_use_attention_mask,
        )

    @property
    def supports_prepared_metadata(self) -> bool:
        return self._supports_prepared_metadata

    def prepare_metadata(self, metadata: AttentionMetadata) -> None:
        if not self._supports_prepared_metadata or not self._kv_bound:
            raise RuntimeError("prepared metadata requires an initialized supported Qwen3 or GLM executor")
        metadata.prepared_attention_state = self.attention_backend.prepare_metadata(metadata)

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
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> ModelExecutionOutput:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")
        if self.layerwise_split_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("Python GLM5.2 layerwise split is decode-only")

        graph_runner = self.decode_graph_runner
        if getattr(metadata, "prepared_attention_state", None) is not None:
            if mtp_topk_indices is not None:
                raise ValueError("prepared metadata does not support MTP top-k state")
            return self.eager_runner.execute(input_ids, positions, metadata, input_embedding, layer_synchronizer)
        graph_kwargs = {}
        if mtp_topk_indices is not None:
            from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

            if isinstance(graph_runner, DecodeAclGraphRunner):
                graph_kwargs["mtp_topk_indices"] = mtp_topk_indices
            else:
                graph_runner = None
        if graph_runner is not None and graph_runner.can_execute(input_ids, metadata, input_embedding, **graph_kwargs):
            from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner

            graph_key = None
            # CUDA can_execute only admits previously captured buckets. ACL
            # captures lazily with scheduler metadata and MTP inputs.
            if not isinstance(graph_runner, DecodeCudaGraphRunner):
                graph_key = graph_runner.warmup(input_ids, positions, metadata, input_embedding, **graph_kwargs)
                if graph_key is not None:
                    graph_kwargs["graph_key"] = graph_key
            return graph_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                **graph_kwargs,
            )
        if mtp_topk_indices is None and self.inductor_runner is not None:
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
            mtp_topk_indices,
        )

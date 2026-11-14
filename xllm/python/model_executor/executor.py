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

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import (
    EplbRuntimeState,
    KVLoadContext,
    KVLoadSynchronizer,
    LayerSynchronizer,
)
from xllm.python.model_executor.input_batch import (
    InputBatch,
    InputBatchMetadata,
)
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.model_platform_support import (
    is_deepseek_v4_model,
    is_dspark_draft_model,
    uses_context_kv_projection,
)
from xllm.python.platform import current_platform
from xllm.python.registry import GLM5_NEXT_TARGET_FAMILY

if TYPE_CHECKING:
    from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

# Backend strings that mean "no graph execution" (matched case-insensitively).
_DISABLED_GRAPH_BACKENDS = ("", "off", "none", "0")


def _resolve_graph_backend(config: dict[str, Any]) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    if graph_backend in _DISABLED_GRAPH_BACKENDS and config.get("enable_graph", False) and current_platform.is_npu():
        return "aclgraph"
    return graph_backend


def _validate_npu_cp_model_config(config: dict[str, Any]) -> None:
    """Mirror model-side CP admission for callers without the C++ master."""
    model_type = config.get("model_type", "")
    # DSpark drafts do not run Python CP admission; only the dense target.
    is_glm5_next = model_type in GLM5_NEXT_TARGET_FAMILY
    is_glm_mtp_draft = (
        model_type == "glm_moe_dsa_mtp"
        and config.get("is_draft_model", False)
        and bool(config.get("is_mtp_algorithm", False))
    )
    if model_type not in ("qwen3", "glm_moe_dsa", "deepseek_v4") and not is_glm5_next and not is_glm_mtp_draft:
        raise NotImplementedError(
            f"Python model-side CP does not support model_type={model_type!r}; "
            "supported models are qwen3, glm_moe_dsa, deepseek_v4, "
            "glm5_next, glm5_3_flash, glm5_next_text and "
            "glm_moe_dsa_mtp draft models"
        )
    if config.get("task_type", "generate") != "generate":
        raise NotImplementedError("Python model-side CP supports only the generate task")
    role = config.get("instance_role", "DEFAULT")
    if role not in ("DEFAULT", "PREFILL"):
        raise NotImplementedError("Python model-side CP supports only DEFAULT or PREFILL roles")
    speculative = int(config.get("num_speculative_tokens", 0)) > 0
    if speculative and bool(config.get("requires_aux_hidden_capture", False)):
        raise NotImplementedError("Python model-side CP does not support aux-hidden-capture speculative algorithms")
    kv_split = int(config.get("kv_split_size", 0)) or int(config["cp_size"])
    # Only the dense target requires replicated KV under Python CP.
    if model_type == "deepseek_v4" and kv_split != 1:
        raise NotImplementedError("Python DeepSeek-V4 CP requires replicated KV caches; use kv_split_size=1")
    if is_glm5_next:
        if speculative:
            raise NotImplementedError(
                "Python GLM-5 Next CP does not support target-side speculative verification; "
                "run speculation on a cp_size=1 Decode instance"
            )
        if int(config.get("ep_size", 1)) < 1:
            raise NotImplementedError("Python GLM-5 Next CP requires ep_size >= 1")
        if kv_split != 1:
            raise NotImplementedError("Python GLM-5 Next CP initially requires kv_split_size == 1")
        if bool(config.get("enable_disagg_pd", False)):
            raise NotImplementedError("Python GLM-5 Next CP does not support disaggregated PD")
        if bool(config.get("enable_chunked_prefill", False)):
            raise NotImplementedError("Python GLM-5 Next CP initially requires enable_chunked_prefill=false")
        if bool(config.get("enable_prefix_cache", False)):
            raise NotImplementedError("Python GLM-5 Next CP initially requires enable_prefix_cache=false")
        if bool(config.get("enable_schedule_overlap", False)):
            raise NotImplementedError("Python GLM-5 Next CP initially requires enable_schedule_overlap=false")


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any] | None = None,
    max_num_reqs: int = 1,
) -> AttentionBackend:
    config = config or {}
    model_type = config.get("model_type", "")
    if is_deepseek_v4_model(config) and current_platform.is_npu():
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
            block_size=int(config.get("block_size", 0)),
            dspark_use_native_sas=bool(config.get("dspark_use_native_sas", False)),
        )
    if current_platform.is_npu():
        dcp_group = distributed.dcp_group(device)
        if (
            dcp_group is not None
            and dcp_group.size() > 1
            and (int(config.get("cp_size", 1)) == 1 or model_type in ("glm_moe_dsa", "glm_moe_dsa_mtp"))
        ):
            if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
                if current_platform.get_ascend_soc_generation() == "a5":
                    raise NotImplementedError(
                        "Qwen3.5 Python DCP is not supported on Ascend950 because "
                        "its TND attention fallback does not return softmax LSE"
                    )
                dcp_size = dcp_group.size()
                tp_size = int(config.get("tp_size", 1))
                if dcp_size > tp_size or tp_size % dcp_size:
                    raise ValueError(f"Qwen3.5 DCP size must divide TP size: dcp_size={dcp_size}, tp_size={tp_size}")
                num_kv_heads_value = config.get("n_kv_heads")
                if num_kv_heads_value is None:
                    num_kv_heads_value = config.get("num_key_value_heads")
                if num_kv_heads_value is None:
                    raise ValueError("Qwen3.5 DCP requires the global KV-head count")
                num_kv_heads = int(num_kv_heads_value)
                if num_kv_heads <= 0:
                    raise ValueError("Qwen3.5 DCP requires a positive global KV-head count")
                from xllm.python.model_loader.sharding import validate_gqa_kv_split

                validate_gqa_kv_split(num_kv_heads, tp_size)
                kv_head_replicas = 1 if num_kv_heads >= tp_size else tp_size // num_kv_heads
                if kv_head_replicas % dcp_size:
                    raise ValueError(
                        "Qwen3.5 DCP size must divide the replicated KV-head group: "
                        f"dcp_size={dcp_size}, kv_head_replicas={kv_head_replicas}"
                    )
                if bool(config.get("enable_disagg_pd", False)):
                    raise NotImplementedError("Qwen3.5 Python DCP does not yet support disaggregated P/D")
                from xllm.python.attention.dense_dcp_backend import (
                    DenseDcpAttentionBackend,
                )

                return DenseDcpAttentionBackend(
                    num_heads=first_attention.num_heads,
                    num_kv_heads=first_attention.num_kv_heads,
                    head_dim=first_attention.head_dim,
                    scale=first_attention.scale,
                    sliding_window=first_attention.sliding_window,
                    device=device,
                    dtype=dtype,
                    dcp_group=dcp_group,
                )
            if not bool(config.get("enable_mla", False)):
                raise NotImplementedError(f"Dense DCP attention is not supported for model type '{model_type}'")
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
            # Only the GLM5 DSpark draft carries the dense MLA block-draft
            # attention shape.
            mla_block_draft=model_type == "glm5_dspark",
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
            kv_cache_storage_head_dim=int(config.get("kv_cache_storage_head_dim", 0)),
        )
    raise NotImplementedError(f"No attention backend available for device type '{device.type}'")


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        max_seqs_per_batch: int,
        acl_graph_decode_batch_size_limit: int | None = None,
    ) -> None:
        self.model = model
        configure_runtime = getattr(model, "configure_runtime", None)
        if configure_runtime is not None:
            configure_runtime(config)
        num_speculative_tokens = int(config.get("num_speculative_tokens", 0))
        self.num_speculative_tokens = (
            0
            if config.get("is_draft_model", False) and not config.get("is_block_diffusion_algorithm", False)
            else num_speculative_tokens
        )
        self.execution_model = model.model
        self.supports_layerwise_kv_load = bool(
            getattr(self.execution_model, "supports_layerwise_kv_load", False)
        ) and not config.get("is_draft_model", False)
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
        if cp_size > 1 and dp_size > 1 and config.get("model_type") not in GLM5_NEXT_TARGET_FAMILY:
            # GLM-5.3-Flash CP restores DP-local rows before routed MoE and
            # supports the dp * cp * tp rank layout.
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
            and graph_backend not in (*_DISABLED_GRAPH_BACKENDS, "aclgraph")
        ):
            # Cross-draft top-k and repair rows require the ACL runner's
            # persistent inputs. Other graph runners cannot retain that state.
            raise NotImplementedError(f"Python NPU GLM MTP requires graph_backend=off/aclgraph; got '{graph_backend}'")
        if cp_size > 1 and graph_backend not in (*_DISABLED_GRAPH_BACKENDS, "aclgraph"):
            # Only decode-only ACL graphs preserve CP prefill on EagerRunner.
            # Match the C++ admission gate before allocating backend state.
            raise NotImplementedError(
                f"Context-Parallel requires eager Prefill with graph_backend=off/aclgraph; got '{graph_backend}'"
            )
        if current_platform.is_npu() and cp_size > 1:
            _validate_npu_cp_model_config(config)

        self._requires_kpool_tail_cache = bool(config.get("requires_kpool_tail_cache", False))

        attention_layers = [module for module in model.modules() if isinstance(module, Attention)]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")
        for layer in attention_layers:
            layer.cache_transfer_enabled = bool(config.get("enable_disagg_pd", False))

        # GLM-Next mixes DSA (MLA) and KDA (linear-attention) layers with
        # different head/dim configs; the paged backend only serves the DSA
        # layers, so it is built from the first DSA layer and the "identical
        # config across all layers" check is skipped. Non-GLM-Next models keep
        # the upstream behavior: backend from the first layer plus the
        # identical-config check. DSA layers are tagged with the
        # ``is_glm_next_mla`` class attribute so this dispatch does not have to
        # import glm5_next (that import pulls KDA kernel transitive deps and
        # fails on builds without them).
        dsa_layers = [layer for layer in attention_layers if getattr(layer, "is_glm_next_mla", False)]
        if dsa_layers:
            first_attention = dsa_layers[0]
        else:
            first_attention = attention_layers[0]
            expected_config = self._attention_config(first_attention)
            for layer in attention_layers[1:]:
                if self._attention_config(layer) != expected_config:
                    raise ValueError("Attention backend requires identical attention configuration across all layers")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        is_draft_tail_repair_required = (
            config.get("is_draft_model", False)
            and config.get("is_mtp_algorithm", False)
            and config.get("model_type") != "qwen3_5_mtp"
        )
        max_decode_rows_per_request = 2 if is_draft_tail_repair_required else num_speculative_tokens + 1
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
        self.decode_graph_runner: DecodeCudaGraphRunner | DecodeAclGraphRunner | None = None
        self._serving_acl_graph_logged = False
        self.context_kv_graph_runner = None
        self.inductor_runner = None

        if self.layerwise_split_size > 1 and graph_backend not in _DISABLED_GRAPH_BACKENDS:
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; "
                f"graph backend '{graph_backend}' is not supported."
            )
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size
        self._dspark_draft = is_dspark_draft_model(config)
        # DSpark drafts and DFlash2 write their own context-KV via the shared
        # context_projection_weight (see uses_context_kv_projection); DFlash
        # and DeepSeek-V4 DSpark do not.
        self._independent_block_draft = uses_context_kv_projection(config)
        kv_split_size = int(config.get("kv_split_size", 0)) or cp_size
        # Multimodal aliases own per-step embedding state outside decode graph inputs.
        self._qwen_gdn = config.get("model_type") in ("qwen3_5_text", "qwen3_5_moe_text")
        qwen_mtp_parallel = dp_size == cp_size == kv_split_size == self.layerwise_split_size == 1
        qwen_mtp_requested = bool(config.get("is_mtp_algorithm", False)) and (
            (self._qwen_gdn and self.num_speculative_tokens > 0) or config.get("model_type") == "qwen3_5_mtp"
        )
        if qwen_mtp_requested and self._qwen_gdn and not 1 <= self.num_speculative_tokens <= 16:
            raise NotImplementedError("Python Qwen MTP supports 1 to 16 speculative tokens")
        self._qwen_mtp_target = (
            config.get("model_type") == "qwen3_5_text"
            and bool(config.get("is_mtp_algorithm", False))
            and 1 <= self.num_speculative_tokens <= 16
            and not config.get("is_draft_model", False)
            and config.get("task_type", "generate") == "generate"
            and qwen_mtp_parallel
        )
        self._qwen_mtp_draft = (
            config.get("model_type") == "qwen3_5_mtp"
            and bool(config.get("is_mtp_algorithm", False))
            and config.get("is_draft_model", False)
            and self.num_speculative_tokens == 0
            and config.get("task_type", "generate") == "generate"
            and qwen_mtp_parallel
        )
        if dp_size > 1 and graph_backend not in (*_DISABLED_GRAPH_BACKENDS, "cudagraphs", "aclgraph"):
            raise NotImplementedError("Python data parallel graph execution supports cudagraphs and aclgraph only")
        if graph_backend in _DISABLED_GRAPH_BACKENDS:
            pass
        elif graph_backend == "cudagraphs":
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
            if self._independent_block_draft:
                from xllm.python.model_executor.runners.context_kv_acl_graph import ContextKvAclGraphRunner

                self.context_kv_graph_runner = ContextKvAclGraphRunner(model, device)
                self._context_kv_query_width = int(model.cfg.block_size)
                self._context_kv_draft_width = self._context_kv_query_width - int(not self._dspark_draft)
                if self._context_kv_draft_width <= 0:
                    raise ValueError("block-draft context graphs require a positive speculative width")
            from xllm.python.model_executor.runners.block_draft_acl_graph import (
                BlockDraftAclGraphRunner,
            )
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
            if self._independent_block_draft:
                self.decode_graph_runner = BlockDraftAclGraphRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    local_graph_sequence_capacity * dp_size,
                    int(config["max_position_embeddings"]),
                    dp_size,
                    dp_rank,
                    block_draft_graph_warmup_max_kv_len=int(config.get("block_draft_graph_warmup_max_kv_len", 0)),
                )
            else:
                self.decode_graph_runner = DecodeAclGraphRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    max_graph_tokens,
                    int(config["max_position_embeddings"]),
                    dp_size,
                    dp_rank,
                    num_speculative_tokens=self.num_speculative_tokens,
                    is_draft_model=bool(config.get("is_draft_model", False)),
                    enable_mega_moe_token_mask=bool(config.get("enable_mega_moe", False)),
                    enable_graph_mode_decode_no_padding=bool(config.get("enable_graph_mode_decode_no_padding", False)),
                    device_kv_lengths=self._qwen_mtp_target or self._qwen_mtp_draft,
                    max_capture_tokens=getattr(execution_model, "graph_capture_token_limit", None),
                )
        else:
            if self.layerwise_split_size > 1:
                raise NotImplementedError(
                    "Python layerwise split requires eager execution; graph "
                    f"backend '{graph_backend}' is not supported."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)
        self.requires_input_batch_metadata = any(
            runner is not None and runner.execution_metadata_builders
            for runner in (
                self.eager_runner,
                self.inductor_runner,
                self.decode_graph_runner,
            )
        )

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

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        required_layers = max(layer.layer_id for layer in self.model.modules() if isinstance(layer, Attention)) + 1
        if len(layer_caches) < required_layers:
            raise ValueError("cache layer count does not match the model layer layout")
        if self._kv_bound:
            return
        if self._requires_kpool_tail_cache:
            indexed_caches = [cache for cache in layer_caches if cache.index is not None]
            if not indexed_caches or any(cache.kpool_tail is None for cache in indexed_caches):
                raise ValueError("model requires a framework-managed kPool tail for every indexer layer")
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
            if isinstance(self.decode_graph_runner, DecodeCudaGraphRunner):
                # Capture before prefill populates live KV state. CUDA decode
                # admission only accepts buckets captured during startup.
                parameter = next(self.model.parameters())
                with torch.inference_mode():
                    self.decode_graph_runner.warmup(parameter.device, parameter.dtype)
        if self.inductor_runner is not None:
            self.inductor_runner.bind_layer_caches(layer_caches)
        if self.context_kv_graph_runner is not None:
            self.context_kv_graph_runner.bind_kv_caches(layer_caches)
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
        enable_graph: bool = False,
        expert_load_data: torch.Tensor | None = None,
        eplb_decode_token_mask: torch.Tensor | None = None,
        is_graph_warmup: bool = False,
        input_batch_metadata: InputBatchMetadata | None = None,
        *,
        kv_load_synchronizer: KVLoadSynchronizer | None = None,
        kv_load_layers_per_event: int = 1,
    ) -> ModelExecutionOutput:
        input_batch = None
        if self.requires_input_batch_metadata:
            if input_batch_metadata is None:
                raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
            input_batch = InputBatch.from_runtime(
                input_ids,
                positions,
                input_batch_metadata,
            )
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")
        if self.context_kv_graph_runner is not None and is_graph_warmup:
            # Startup draft inputs have N physical query rows per request.
            # Capture every accepted-row bucket up to the N+1 verify capacity,
            # including prefixes not encountered in the synthetic warmup.
            width = self._context_kv_draft_width
            query_width = self._context_kv_query_width
            sequences = (input_ids.numel() + query_width - 1) // query_width
            self.context_kv_graph_runner.warmup(sequences * (width + 1))
        if self.layerwise_split_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("Python GLM5.2 layerwise split is decode-only")

        eplb = None
        if expert_load_data is not None:
            eplb = EplbRuntimeState(expert_load_data, eplb_decode_token_mask, is_graph_warmup)
        if kv_load_synchronizer is not None:
            if (
                not self.supports_layerwise_kv_load
                or enable_graph
                or metadata.is_spec_verify
                or not (metadata.is_prefill or metadata.is_chunked_prefill)
            ):
                raise ValueError("Layerwise Host KV load requires a supported eager prefill model")
            kv_load = KVLoadContext(kv_load_synchronizer, kv_load_layers_per_event, len(self.execution_model.layers))
            return self.eager_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
                mtp_topk_indices,
                eplb=eplb,
                kv_load=kv_load,
                input_batch=input_batch,
            )
        # Transfer completion events belong to this step. Graph replay cannot
        # record them through Python, and compilation must not retain a prior
        # step's synchronizer. Decode steps without a transfer can use graphs.
        graph_runner = self.decode_graph_runner if layer_synchronizer is None else None
        if graph_runner is not None and (eplb is None or current_platform.is_npu()):
            from xllm.python.model_executor.runners.block_draft_acl_graph import BlockDraftAclGraphRunner
            from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

            if isinstance(graph_runner, DecodeAclGraphRunner):
                decode_metadata = None
                graph_key = None
                if isinstance(graph_runner, BlockDraftAclGraphRunner):
                    if graph_runner.can_execute(input_ids, metadata, input_embedding, mtp_topk_indices):
                        graph_key = graph_runner.warmup(
                            input_ids,
                            positions,
                            metadata,
                            input_embedding,
                            mtp_topk_indices,
                            eplb=eplb,
                            pre_capture_kv_buckets=is_graph_warmup,
                            input_batch=input_batch,
                        )
                else:
                    decode_metadata = graph_runner.prepare_execution(
                        input_ids, metadata, input_embedding, mtp_topk_indices, is_graph_warmup=is_graph_warmup
                    )
                    if decode_metadata is not None:
                        graph_key = graph_runner.warmup(
                            input_ids,
                            positions,
                            metadata,
                            input_embedding,
                            mtp_topk_indices,
                            eplb=eplb,
                            decode_metadata=decode_metadata,
                            is_graph_warmup=is_graph_warmup,
                            input_batch=input_batch,
                        )
                if graph_key is not None:
                    output = graph_runner.execute(
                        input_ids,
                        positions,
                        metadata,
                        input_embedding,
                        mtp_topk_indices,
                        graph_key=graph_key,
                        eplb=eplb,
                        decode_metadata=decode_metadata,
                        input_batch=input_batch,
                    )
                    if not is_graph_warmup and not self._serving_acl_graph_logged:
                        logger.info(
                            "Python serving ACL graph replay: model=%s runner=%s rows=%d",
                            type(self.model).__name__,
                            type(graph_runner).__name__,
                            input_ids.numel(),
                        )
                        self._serving_acl_graph_logged = True
                    return output
            elif mtp_topk_indices is None and graph_runner.can_execute(input_ids, metadata, input_embedding):
                # CUDA can_execute only admits previously captured buckets. ACL
                # captures lazily with scheduler metadata and MTP inputs.
                return graph_runner.execute(
                    input_ids,
                    positions,
                    metadata,
                    input_embedding,
                    eplb=eplb,
                    input_batch=input_batch,
                )
        if mtp_topk_indices is None and self.inductor_runner is not None and layer_synchronizer is None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
                eplb=eplb,
                input_batch=input_batch,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
            mtp_topk_indices,
            eplb=eplb,
            input_batch=input_batch,
        )

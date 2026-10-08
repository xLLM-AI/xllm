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

from collections.abc import Callable

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
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    EplbRuntimeState,
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.model_executor.runners.mtp_acl_graph import (
    ActivateFn,
    ForwardFn,
    GraphBackend,
    MtpAclGraphRunner,
    MtpGraphRecipe,
    MtpGraphVariantRegistry,
    MtpRoleAdapter,
    MtpSamplingPlan,
)
from xllm.python.model_executor.runners.mtp_sparse_metadata import (
    MtpSparseMetadataStorage,
    MtpSparsePositionStorage,
    SparseMetadataFactory,
)
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
    algorithm = str(config.get("speculative_algorithm", "mtp")).lower()
    is_glm_mtp_draft = model_type == "glm_moe_dsa_mtp" and config.get("is_draft_engine", False) and algorithm == "mtp"
    if model_type not in ("qwen3", "glm_moe_dsa", "deepseek_v4") and not is_glm_mtp_draft:
        raise NotImplementedError(
            f"Python model-side CP does not support model_type={model_type!r}; "
            "supported models are qwen3, glm_moe_dsa, deepseek_v4 and glm_moe_dsa_mtp draft engines"
        )
    if config.get("task_type", "generate") != "generate":
        raise NotImplementedError("Python model-side CP supports only the generate task")
    role = config.get("instance_role", "DEFAULT")
    if role not in ("DEFAULT", "PREFILL"):
        raise NotImplementedError("Python model-side CP supports only DEFAULT or PREFILL roles")
    speculative = int(config.get("num_speculative_tokens", 0)) > 0 or num_decoding_tokens > 1
    if speculative and algorithm in ("eagle3", "dflash", "dflash2", "dspark"):
        raise NotImplementedError("Python model-side CP does not support aux-hidden-capture speculative algorithms")
    kv_split = int(config.get("kv_split_size", 0)) or int(config["cp_size"])
    if model_type == "deepseek_v4" and kv_split != 1:
        raise NotImplementedError("Python DeepSeek-V4 CP requires replicated KV caches; use kv_split_size=1")


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
    config: dict | None = None,
    max_num_reqs: int = 1,
    num_decoding_tokens: int = 1,
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
        if (
            dcp_group is not None
            and dcp_group.size() > 1
            and (int(config.get("cp_size", 1)) == 1 or model_type in ("glm_moe_dsa", "glm_moe_dsa_mtp"))
        ):
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
            num_decoding_tokens=num_decoding_tokens,
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


def _configure_unified_mtp_graph(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        setter = getattr(module, "set_unified_mtp_graph_enabled", None)
        if setter is not None:
            setter(enabled)


def _unified_mtp_graph_enabled(config: dict) -> bool:
    """Return whether this executor is one side of GLM's Unified MTP pair.

    The draft executor deliberately keeps ``num_speculative_tokens == 0``;
    that is its existing runtime contract, so executor width alone cannot be
    used to decide whether the MTP model should receive the Unified setting.
    Ordinary non-MTP executors must remain unaffected by the process-wide
    Unified flag.
    """
    if not bool(config.get("enable_unified_mtp_graph", False)):
        return False
    if str(config.get("speculative_algorithm", "mtp")).lower() != "mtp":
        return False

    model_type = config.get("model_type")
    is_draft_engine = bool(config.get("is_draft_engine", False))
    if model_type == "glm_moe_dsa_mtp":
        return is_draft_engine
    return model_type == "glm_moe_dsa" and not is_draft_engine and int(config.get("num_speculative_tokens", 0)) > 0


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
        # MTP width is a process-level graph contract.  The runner may accept
        # fewer tokens after verification, but it must never change the
        # capture width for a live executor.
        self.num_speculative_tokens = int(config.get("num_speculative_tokens", 0))
        unified_mtp_enabled = _unified_mtp_graph_enabled(config)
        _configure_unified_mtp_graph(self.model, unified_mtp_enabled)
        # Keep the full CausalLM object next to the body handle.  Ordinary
        # runners intentionally receive ``model.model`` only, while the
        # speculative graph needs the role-owned lm_head/compute_logits entry
        # after every unrolled body step.
        self.execution_model = model.model
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
        num_decoding_tokens = max(1, int(num_decoding_tokens), int(config.get("num_speculative_tokens", 0)) + 1)
        # GLM MTP can prepend a repair row after all draft tokens are accepted.
        # Two requests then need up to four attention rows, even though later
        # draft steps still decode one token per request.
        max_decode_rows_per_request = num_decoding_tokens
        if config.get("model_type") == "glm_moe_dsa_mtp":
            max_decode_rows_per_request = max(max_decode_rows_per_request, 2)
        self._num_attention_layers = len(attention_layers)
        self._attention_backend_first_attention = first_attention
        self._attention_backend_device = device
        self._attention_backend_dtype = first_parameter.dtype
        self._attention_backend_config = dict(config)
        self._attention_backend_num_decoding_tokens = num_decoding_tokens
        self._attention_backend_max_num_reqs = max(max_seqs_per_batch, 1) * max_decode_rows_per_request
        self.attention_backend = _create_attention_backend(
            first_attention,
            device,
            first_parameter.dtype,
            config,
            self._attention_backend_max_num_reqs,
            num_decoding_tokens=num_decoding_tokens,
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
        self.prepared_graph_runner = None
        self.inductor_runner = None

        if self.layerwise_split_size > 1 and graph_backend not in ("", "off", "none", "0"):
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; "
                f"graph backend '{graph_backend}' is not supported."
            )
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size
        self._prepared_mtp = config.get("model_type") == "glm_moe_dsa_mtp"
        self._prepared_block_draft = config.get("model_type") in ("DFlashDraftModel", "DFlash2DraftModel")
        prepared_kv_split = int(config.get("kv_split_size", 0)) or cp_size
        dcp_group = distributed.dcp_group(device) if current_platform.is_npu() else None
        dcp_size = dcp_group.size() if dcp_group is not None else 1
        prepared_glm_parallel = (
            current_platform.is_npu()
            and config.get("model_type") in ("glm_moe_dsa", "glm_moe_dsa_mtp")
            and dp_size == 1
        )
        # Sharded KV requires the live group that selected the SFA backend.
        prepared_kv = dcp_size == prepared_kv_split and (
            prepared_kv_split == 1 or (prepared_kv_split > 1 and prepared_glm_parallel)
        )
        self._supports_prepared_metadata = (
            self.attention_backend.supports_prepared_metadata
            and (
                config.get("model_type") in ("qwen3", "glm_moe_dsa", "glm_moe_dsa_mtp")
                or (self._prepared_block_draft and graph_backend in ("", "off", "none", "0"))
            )
            and prepared_kv
            and graph_backend in ("", "off", "none", "0", "aclgraph")
            and (cp_size == 1 or prepared_glm_parallel)
            and self.layerwise_split_size == 1
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
            if config.get("enable_task_pipeline", False):
                from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner

                if not self._supports_prepared_metadata or self._prepared_block_draft:
                    raise ValueError("prepared ACL graphs require a supported target or MTP model")
                self.prepared_graph_runner = PreparedAclGraphRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    max_graph_tokens,
                    dp_size,
                    dp_rank,
                    enable_mega_moe_token_mask=bool(config.get("enable_mega_moe", False)),
                )
            else:
                if config.get("model_type") == "DSparkDraftModel":
                    if dp_size != 1:
                        raise NotImplementedError("DSpark ACL graphs currently require dp_size == 1")
                    self.decode_graph_runner = BlockDraftAclGraphRunner(
                        execution_model,
                        self.attention_backend,
                        device,
                        local_graph_sequence_capacity,
                        int(config["max_position_embeddings"]),
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
                        decode_batch_size_limit,
                        num_decoding_tokens=num_decoding_tokens,
                        is_spec_draft=bool(config.get("is_draft_engine", False)),
                        enable_mega_moe_token_mask=bool(config.get("enable_mega_moe", False)),
                    )
        else:
            if self.layerwise_split_size > 1:
                raise NotImplementedError(
                    "Python layerwise split requires eager execution; graph "
                    f"backend '{graph_backend}' is not supported."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

    @torch.inference_mode()
    def execute_mtp_role(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        acl_graph: AclGraphCaptureContext | None = None,
        execution_state: AclGraphExecutionState | None = None,
        attention_backend: AttentionBackend | None = None,
        skip_prepare: bool = False,
    ) -> ModelExecutionOutput:
        """Run one MTP role under the enclosing composite graph context."""
        if self.eager_runner.cp_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("MTP composite graph does not support CP prefill")
        role_backend = self.attention_backend if attention_backend is None else attention_backend
        if skip_prepare and execution_state is None:
            raise ValueError("skip_prepare requires a graph execution state")
        if skip_prepare and attention_backend is None:
            raise ValueError("graph MTP role requires a role-local attention backend")
        context = ForwardContext(
            role_backend,
            self.eager_runner.device,
            metadata,
            self.eager_runner.layer_caches,
            acl_graph=acl_graph,
            layer_synchronizer=layer_synchronizer,
            execution_state=execution_state,
        )
        with forward_context(context):
            if not skip_prepare:
                prepare_owned = getattr(role_backend, "prepare_owned_graph_metadata", None)
                if execution_state is not None and prepare_owned is not None:
                    prepare_owned(metadata)
                else:
                    role_backend.prepare(metadata, graph_mode=execution_state is not None)
            if mtp_topk_indices is not None:
                return self.execution_model(input_ids, positions, input_embedding, mtp_topk_indices)
            if input_embedding is None:
                return self.execution_model(input_ids, positions)
            return self.execution_model(input_ids, positions, input_embedding)

    def create_mtp_role_backends(self, count: int) -> tuple[AttentionBackend, ...]:
        """Create independently prepared backends for fixed MTP graph steps."""
        if count <= 0:
            raise ValueError("MTP role backend count must be positive")
        if not self._kv_bound:
            raise RuntimeError("KV caches must be bound before creating MTP role backends")
        backends: list[AttentionBackend] = []
        for _ in range(count):
            backend = _create_attention_backend(
                self._attention_backend_first_attention,
                self._attention_backend_device,
                self._attention_backend_dtype,
                self._attention_backend_config,
                self._attention_backend_max_num_reqs,
                num_decoding_tokens=self._attention_backend_num_decoding_tokens,
            )
            backend.bind_kv_caches(self.eager_runner.layer_caches)
            backends.append(backend)
        return tuple(backends)

    def create_mtp_graph_runner(
        self,
        draft_executor: ModelExecutor,
        *,
        draft_forward: ForwardFn,
        target_forward: ForwardFn,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        backend: GraphBackend = "aclgraph",
        kv_seq_lens: torch.Tensor | None = None,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        target_sampling: MtpSamplingPlan | None = None,
        position_storage: MtpSparsePositionStorage | None = None,
    ) -> MtpAclGraphRunner:
        """Create a paired MTP runner with explicit role adapters.

        ``draft_forward`` and ``target_forward`` install their own fixed
        attention metadata/``ForwardContext`` and return the body hidden
        states.  Requiring the adapters here prevents the composite runner
        from accidentally reusing one model's mutable attention backend for
        both roles.  The C++ MTP bridge will own these adapters when it wires
        the two ``PyExecutorImpl`` instances together.
        """
        if not isinstance(draft_executor, ModelExecutor):
            raise TypeError("draft_executor must be a ModelExecutor")
        for role, configured_tokens in (
            ("target", self.num_speculative_tokens),
            ("draft", draft_executor.num_speculative_tokens),
        ):
            if configured_tokens > 0 and configured_tokens != speculative_tokens:
                raise ValueError(
                    f"{role} executor was initialized with num_speculative_tokens="
                    f"{configured_tokens}, got graph K={speculative_tokens}"
                )

        def draft_logits(hidden: torch.Tensor) -> torch.Tensor:
            return draft_executor.model.compute_logits(hidden, None)

        def target_logits(hidden: torch.Tensor) -> torch.Tensor:
            return self.model.compute_logits(hidden, None)

        recipe = MtpGraphRecipe(
            draft_forward,
            draft_logits,
            target_forward,
            target_logits,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            device=next(self.model.parameters()).device,
            kv_seq_lens=kv_seq_lens,
            draft_activate=draft_activate,
            target_activate=target_activate,
            target_sampling=target_sampling,
            draft_greedy=getattr(draft_executor.model, "compute_greedy_tokens", None),
            target_greedy=getattr(self.model, "compute_greedy_tokens", None),
            position_storage=position_storage,
        )
        return MtpAclGraphRunner(recipe, backend=backend)

    def create_mtp_graph_variant_registry(
        self,
        draft_executor: ModelExecutor,
        *,
        max_variants: int = 8,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        capture_runner: Callable[..., None] | None = None,
        draft_metadata_factory: SparseMetadataFactory | None = None,
        target_metadata_factory: SparseMetadataFactory | None = None,
    ) -> MtpGraphVariantRegistry:
        """Create the Python owner for bounded MTP graph variants.

        The registry keeps sparse recipe construction, capture, fused replay
        updates, and FIFO eviction in Python.
        """
        if not isinstance(draft_executor, ModelExecutor):
            raise TypeError("draft_executor must be a ModelExecutor")
        return MtpGraphVariantRegistry(
            self,
            draft_executor,
            max_variants=max_variants,
            draft_activate=draft_activate,
            target_activate=target_activate,
            capture_runner=capture_runner,
            draft_metadata_factory=draft_metadata_factory,
            target_metadata_factory=target_metadata_factory,
        )

    def create_mtp_graph_runner_from_metadata(
        self,
        draft_executor: ModelExecutor,
        draft_metadata: tuple[object, ...],
        target_metadata: object,
        *,
        repair_token_ids: torch.Tensor,
        batch_size: int,
        speculative_tokens: int,
        vocab_size: int,
        kv_seq_lens: torch.Tensor | None = None,
        draft_activate: ActivateFn | None = None,
        target_activate: ActivateFn | None = None,
        target_sampling: MtpSamplingPlan | None = None,
        draft_metadata_storage: MtpSparseMetadataStorage | None = None,
        target_metadata_storage: MtpSparseMetadataStorage | None = None,
        position_storage: MtpSparsePositionStorage | None = None,
        target_step_major_layout: bool = False,
    ) -> MtpAclGraphRunner:
        """Construct both role adapters and their recipe in one Python call."""
        draft_forward = draft_executor.create_mtp_role_adapter(
            tuple(draft_metadata),
            speculative_tokens=speculative_tokens,
            repair_token_ids=repair_token_ids,
            metadata_storage=draft_metadata_storage,
            repair_positions=None if position_storage is None else position_storage.first_draft,
        )
        target_forward = self.create_mtp_role_adapter(
            (target_metadata,),
            speculative_tokens=speculative_tokens,
            target=True,
            step_major_layout=target_step_major_layout,
            metadata_storage=target_metadata_storage,
        )
        return self.create_mtp_graph_runner(
            draft_executor,
            draft_forward=draft_forward,
            target_forward=target_forward,
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            vocab_size=vocab_size,
            kv_seq_lens=kv_seq_lens,
            draft_activate=draft_activate,
            target_activate=target_activate,
            target_sampling=target_sampling,
            position_storage=position_storage,
        )

    def create_mtp_role_adapter(
        self,
        metadata_by_step: tuple[object, ...],
        *,
        speculative_tokens: int,
        target: bool = False,
        step_major_layout: bool = False,
        layer_synchronizer: LayerSynchronizer | None = None,
        repair_token_ids: torch.Tensor | None = None,
        metadata_storage: MtpSparseMetadataStorage | None = None,
        repair_positions: torch.Tensor | None = None,
    ) -> MtpRoleAdapter:
        """Bind fixed attention metadata to this executor's MTP role.

        The returned adapter always invokes this executor's eager body runner;
        the caller captures the enclosing ``MtpAclGraphRunner``.  This keeps
        target and draft attention ownership separate while avoiding nested
        decode graph capture.  C++ creates one adapter per ``PyExecutorImpl``
        and supplies K draft metadata entries plus one target K+1 entry.
        """
        return MtpRoleAdapter(
            self,
            metadata_by_step,
            speculative_tokens=speculative_tokens,
            target=target,
            step_major_layout=step_major_layout,
            layer_synchronizer=layer_synchronizer,
            repair_token_ids=repair_token_ids,
            metadata_storage=metadata_storage,
            repair_positions=repair_positions,
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

    @property
    def supports_prepared_metadata(self) -> bool:
        return self._supports_prepared_metadata

    def prepare_metadata(self, metadata: AttentionMetadata) -> None:
        if not self._supports_prepared_metadata or not self._kv_bound:
            raise RuntimeError("prepared metadata requires an initialized supported Qwen3 or GLM executor")
        if self._prepared_block_draft:
            metadata.prepared_attention_state = self.attention_backend.prepare_metadata(
                metadata, device_kv_lengths=True
            )
        else:
            metadata.prepared_attention_state = self.attention_backend.prepare_metadata(metadata)

    def warmup_prepared_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        expert_load_data: torch.Tensor | None = None,
        eplb_decode_token_mask: torch.Tensor | None = None,
        is_graph_warmup: bool = False,
    ) -> None:
        if self.prepared_graph_runner is None:
            raise RuntimeError("prepared ACL graph runner is not enabled")
        eplb = (
            None
            if expert_load_data is None
            else EplbRuntimeState(expert_load_data, eplb_decode_token_mask, is_graph_warmup)
        )
        self.prepared_graph_runner.warmup_prepared(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, eplb=eplb
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
        if self.prepared_graph_runner is not None:
            self.prepared_graph_runner.bind_layer_caches(layer_caches)
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
        enable_graph: bool = False,
        expert_load_data: torch.Tensor | None = None,
        eplb_decode_token_mask: torch.Tensor | None = None,
        is_graph_warmup: bool = False,
    ) -> ModelExecutionOutput:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")
        if self.layerwise_split_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("Python GLM5.2 layerwise split is decode-only")

        eplb = None
        if expert_load_data is not None:
            eplb = EplbRuntimeState(expert_load_data, eplb_decode_token_mask, is_graph_warmup)
        # Transfer completion events belong to this step. Graph replay cannot
        # record them through Python, and compilation must not retain a prior
        # step's synchronizer. Decode steps without a transfer can use graphs.
        graph_runner = self.decode_graph_runner if layer_synchronizer is None else None
        if getattr(metadata, "prepared_attention_state", None) is not None:
            if mtp_topk_indices is not None and not self._prepared_mtp:
                raise ValueError("prepared metadata does not support MTP top-k state")
            if enable_graph and layer_synchronizer is None:
                if self.prepared_graph_runner is None:
                    raise RuntimeError("prepared ACL graph runner is not enabled")
                return self.prepared_graph_runner.execute(
                    input_ids, positions, metadata, input_embedding, layer_synchronizer, mtp_topk_indices, eplb=eplb
                )
            if self._prepared_mtp:
                return self.eager_runner.execute(
                    input_ids, positions, metadata, input_embedding, layer_synchronizer, mtp_topk_indices, eplb=eplb
                )
            return self.eager_runner.execute(
                input_ids, positions, metadata, input_embedding, layer_synchronizer, eplb=eplb
            )
        graph_kwargs = {}
        if mtp_topk_indices is not None:
            from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

            if isinstance(graph_runner, DecodeAclGraphRunner):
                graph_kwargs["mtp_topk_indices"] = mtp_topk_indices
            else:
                graph_runner = None
        if (
            graph_runner is not None
            and (eplb is None or current_platform.is_npu())
            and graph_runner.can_execute(input_ids, metadata, input_embedding, **graph_kwargs)
        ):
            from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner

            graph_key = None
            # CUDA can_execute only admits previously captured buckets. ACL
            # captures lazily with scheduler metadata and MTP inputs.
            if not isinstance(graph_runner, DecodeCudaGraphRunner):
                graph_key = graph_runner.warmup(
                    input_ids,
                    positions,
                    metadata,
                    input_embedding,
                    **graph_kwargs,
                    eplb=eplb,
                )
                if graph_key is not None:
                    graph_kwargs["graph_key"] = graph_key
            return graph_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                **graph_kwargs,
                eplb=eplb,
            )
        if mtp_topk_indices is None and self.inductor_runner is not None and layer_synchronizer is None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
                eplb=eplb,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
            mtp_topk_indices,
            eplb=eplb,
        )

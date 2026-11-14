/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/mtp_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "common/metrics.h"
#include "core/common/constants.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/kv_cache/block/block_utils.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/mtp_async_input_builder.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "core/util/tensor_helper.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {
constexpr size_t kMaxPendingTargetContexts = 2;
torch::Tensor acquire_pinned_output_buffer(const torch::Tensor& source,
                                           std::vector<torch::Tensor>& pool) {
  // Every lease is a distinct TensorImpl/view. The pending context retains
  // it through event synchronization; serializers may retain it longer.
  // Unique storage therefore proves both producer and consumers retired.
  auto available =
      std::find_if(pool.begin(), pool.end(), [&source](const auto& buffer) {
        return buffer.storage().use_count() == 1 &&
               buffer.scalar_type() == source.scalar_type() &&
               buffer.numel() >= source.numel();
      });
  if (available == pool.end()) {
    available = std::find_if(pool.begin(), pool.end(), [](const auto& buffer) {
      return buffer.storage().use_count() == 1;
    });
  }
  torch::Tensor buffer;
  if (available != pool.end()) {
    buffer = *available;
  }
  if (!buffer.defined() || buffer.scalar_type() != source.scalar_type() ||
      buffer.numel() < source.numel()) {
    buffer =
        torch::empty({source.numel()},
                     source.options().device(torch::kCPU).pinned_memory(true));
    if (available != pool.end()) {
      *available = buffer;
    } else if (pool.size() < kMaxPendingTargetContexts + 2) {
      pool.emplace_back(buffer);
    }
  }
  // If all bounded pool slots are leased, this allocation belongs solely to
  // the current consumers. Never overwrite an older output to avoid a copy.
  return buffer.narrow(0, 0, source.numel()).view(source.sizes());
}

constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;

template <typename Input>
void clear_ready_events(Input& input) {
  input.runtime.metadata_ready_event.reset();
}

torch::Tensor clone_host_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU host tensor";
  return clone_contiguous_detached_tensor(tensor);
}

// Snapshot the host block tables without launching an ATen parallel copy
// team on every TP rank: a plain non-pinned CPU allocation plus memcpy.
// Other layouts fall back to the generic host clone.
template <typename Input>
void stabilize_decode_host_tensors(Input& input) {
  input.token_ids_host = clone_host_tensor(input.token_ids_host);
  input.positions_host = clone_host_tensor(input.positions_host);
  input.input_params.attention.host.shared_full_block_table = clone_host_tensor(
      input.input_params.attention.host.shared_full_block_table);
  input.input_params.attention.host.shared_full_slot_mapping =
      clone_host_tensor(
          input.input_params.attention.host.shared_full_slot_mapping);
  {
    const torch::Tensor& tensor =
        input.input_params.attention.host.block_tables;
    if (tensor.defined()) {
      if (tensor.device().is_cpu() && tensor.layout() == torch::kStrided &&
          tensor.dim() == 2 && tensor.scalar_type() == torch::kInt32 &&
          tensor.is_contiguous() && !tensor.is_neg() && !tensor.is_conj()) {
        torch::Tensor snapshot =
            torch::empty(tensor.sizes(), tensor.options().pinned_memory(false));
        if (tensor.numel() != 0) {
          std::memcpy(snapshot.data_ptr(), tensor.data_ptr(), tensor.nbytes());
        }
        input.input_params.attention.host.block_tables = std::move(snapshot);
      } else {
        input.input_params.attention.host.block_tables =
            clone_host_tensor(tensor);
      }
    }
  }
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    if (!block_table.defined()) {
      continue;
    }
    if (block_table.device().is_cpu() &&
        block_table.layout() == torch::kStrided && block_table.dim() == 2 &&
        block_table.scalar_type() == torch::kInt32 &&
        block_table.is_contiguous() && !block_table.is_neg() &&
        !block_table.is_conj()) {
      torch::Tensor snapshot = torch::empty(
          block_table.sizes(), block_table.options().pinned_memory(false));
      if (block_table.numel() != 0) {
        std::memcpy(
            snapshot.data_ptr(), block_table.data_ptr(), block_table.nbytes());
      }
      block_table = std::move(snapshot);
    } else {
      block_table = clone_host_tensor(block_table);
    }
  }
}

template <typename Input>
void set_token_ids_device_tensor(Input& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options,
                                 Stream& compute_stream) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  input.runtime.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids =
      safe_to(flat_token_ids, token_options, /*non_blocking=*/true);
  input.runtime.device_tensors_ready = true;
}

template <typename Input>
void replace_host_token_placeholders(Input& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

  input.runtime.device_tensors_ready = false;
  torch::Tensor replacement_cpu =
      to_cpu_contiguous(replacements.flatten(), torch::kInt);
  int32_t* token_ids = input.token_ids_host.template data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = tensor_slice(replacement_cpu);

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids =
        safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
    input.runtime.device_tensors_ready = true;
  }
}

ParallelArgs mtp_draft_parallel_args(const ParallelArgs& parallel_args,
                                     const runtime::Options& options) {
  if (!options.enable_mtp_draft_body_tp1()) {
    return parallel_args;
  }
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "MTP draft body TP1 requires a single-rank process group";
  ParallelArgs draft_args = parallel_args;
  draft_args.kv_transfer_groups_ = parallel_args.kv_transfer_groups();
  draft_args.rank(0)
      .world_size(1)
      .dp_size(1)
      .ep_size(1)
      .cp_size(1)
      .tp_size(1)
      .sp_size(1);
  draft_args.mapping_data(nlohmann::json{});
  draft_args.process_group_ = parallel_args.single_rank_group_;
  draft_args.dp_local_process_group_ = parallel_args.single_rank_group_;
  draft_args.lm_head_group_ = parallel_args.tp_group_;
  draft_args.tp_group_ = parallel_args.single_rank_group_;
  draft_args.cp_group_ = parallel_args.single_rank_group_;
  draft_args.moe_ep_group_ = parallel_args.single_rank_group_;
  draft_args.moe_tp_group_ = parallel_args.single_rank_group_;
  return draft_args;
}

void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = tensor_slice(token_ids_host);
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const EmbeddingCache::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
    CHECK(state.embedding.defined())
        << "MTP decode target state embedding is undefined, request_id="
        << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr) {
    return state;
  }
  const int64_t num_seqs = sampling_params.num_sample_seqs();
  if (num_seqs == 0 || state->num_rows() == num_seqs) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.logits_indices();
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

}  // namespace

template <typename TargetInput>
runtime::Options MTPWorkerImpl<TargetInput>::draft_options(
    const runtime::Options& options) {
  runtime::Options draft_options = options;
  if (KVCacheConfig::get_instance().enable_vmm_kvcache()) {
    draft_options.model_id(options.model_id() + "::mtp_draft");
  }
  draft_options.backend("llm")
      .enable_schedule_overlap(false)
      .is_draft_model(true)
      .enable_graph_aux_hidden_states(true);
  return draft_options;
}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::MTPWorkerImpl(const ParallelArgs& parallel_args,
                                          const torch::Device& device,
                                          const runtime::Options& options,
                                          WorkerType worker_type)
    : MTPWorkerImpl<TargetInput>(
          parallel_args,
          device,
          options,
          MTPWorkerImpl<TargetInput>::draft_options(options),
          worker_type) {}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::MTPWorkerImpl(const ParallelArgs& parallel_args,
                                          const torch::Device& device,
                                          const runtime::Options& options,
                                          const runtime::Options& draft_opts,
                                          WorkerType worker_type)
    : DraftModelSpecWorkerImpl<TargetInput>(
          parallel_args,
          device,
          options,
          worker_type,
          [&parallel_args, &device, &options, &draft_opts] {
            return std::make_unique<LLMWorkerImpl>(
                configure_draft_parallel_args(
                    mtp_draft_parallel_args(parallel_args, options)),
                device,
                draft_opts,
                parallel_args.dp_size() > 1);
          }) {}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::~MTPWorkerImpl() {
  drain_pending_execution();
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::drain_pending_execution() {
  if (execution_drained_) {
    return;
  }
  DraftModelSpecWorkerImpl<TargetInput>::drain_pending_execution();
  execution_drained_ = true;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  const bool result = DraftModelSpecWorkerImpl<TargetInput>::init_model(
      model_weights_path, random_seed, master_status);
  if (!result) {
    return false;
  }

  if (draft_impl_->get_status() != WorkerImpl::Status::UNINITIALIZED) {
    const std::string& draft_model_type =
        draft_impl_->context_.get_model_args().model_type();
    combined_draft_execution_path_ =
        mtp_async::classify_combined_draft_execution_path(draft_model_type);
    std::string resolved_draft_model_name;
    std::string error_message;
    CHECK(resolve_model_registration_name(
        draft_model_type, &resolved_draft_model_name, &error_message))
        << error_message;
    const MtpModelCapabilities draft_capabilities =
        ModelRegistry::get_mtp_capabilities(resolved_draft_model_name);
    const bool is_python_draft = ModelConfig::is_python_model_impl(
        draft_impl_->context_.get_model_impl());
    const bool use_tp1_draft_body =
        options_.enable_mtp_draft_body_tp1() &&
        combined_draft_execution_path_ ==
            mtp_async::CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
    const bool draft_owns_shared_weights =
        use_tp1_draft_body ||
        (is_python_draft
             ? draft_capabilities.python_draft_owns_embedding_and_lm_head
             : draft_capabilities.native_draft_owns_embedding_and_lm_head);
#if defined(USE_MLU)
    if (!is_python_draft && !use_tp1_draft_body) {
      if (!draft_impl_->context_.get_model_args().embedding_mode() &&
          share_target_lm_head_with_draft() &&
          !draft_impl_->get_lm_head()->is_weight_loaded()) {
        auto head = impl_->get_lm_head();
        draft_impl_->set_lm_head(head);
      }
      if (!draft_impl_->get_word_embedding()->is_weight_loaded()) {
        auto embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(embedding);
      }
    } else
#endif
        if (!draft_owns_shared_weights) {
      const bool python_weights_shared =
          draft_impl_->share_weights_from(*impl_);
      if (!python_weights_shared) {
        const bool share_lm_head = share_target_lm_head_with_draft();
        if (share_lm_head) {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
        }
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
      }
    }
    if (impl_->get_status() != WorkerImpl::Status::UNINITIALIZED) {
      // Resolve tail-extend vs accepted-span replay once both models have
      // loaded, before embedding/KV cache allocation.
      const bool is_python_target =
          ModelConfig::is_python_model_impl(context_.get_model_impl());
      const bool legacy_tail = draft_capabilities.allow_tail_fallback &&
                               (is_python_target || is_python_draft);
      const bool full_target_replay =
          draft_capabilities.native_draft_requires_accepted_span_replay &&
          !legacy_tail;
      if (full_target_replay) {
        std::string resolved_target_model_name;
        std::string error_message;
        CHECK(resolve_model_registration_name(
            context_.get_model_args().model_type(),
            &resolved_target_model_name,
            &error_message))
            << error_message;
        const MtpModelCapabilities target_capabilities =
            ModelRegistry::get_mtp_capabilities(resolved_target_model_name);
        const bool target_capable =
            target_verify_mode_ ==
                    mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL
                ? target_capabilities.supports_causal_replay
                : target_capabilities.supports_expanded_replay;
        const bool replay_pair =
            target_capable &&
            target_capabilities.model_family != MtpModelFamily::NONE &&
            target_capabilities.model_family == draft_capabilities.model_family;
        CHECK(replay_pair && !is_python_target && !is_python_draft &&
              (target_verify_mode_ ==
                   mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
               target_verify_mode_ ==
                   mtp_async::TargetSpecVerifyMode::EXPANDED_VERIFY))
            << "Incompatible draft context update: target="
            << context_.get_model_args().model_type()
            << ", draft=" << draft_model_type
            << ", verify_mode=" << static_cast<int32_t>(target_verify_mode_);
      }
      const DraftContextReplayPolicy replay_policy{
          /*full_target_replay=*/full_target_replay,
          /*num_speculative_tokens=*/
          full_target_replay ? options_.num_speculative_tokens() : 1,
          /*draft_position_offset=*/full_target_replay ? -1 : 0};
      if (draft_context_replay_policy_.has_value()) {
        CHECK_EQ(draft_context_replay_policy_->full_target_replay,
                 replay_policy.full_target_replay)
            << "Draft context update policy changed after initialization.";
      } else {
        draft_context_replay_policy_ = replay_policy;
      }
    }
  }
  draft_prefix_policy_ = mtp_async::classify_draft_prefix_policy(
      target_verify_mode_, combined_draft_execution_path_);
#if defined(USE_NPU)
  if (result && supports_expanded_spec_verify()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Qwen3.5 MTP only supports NPU Torch backend";
  }
#endif
  return result;
}

template <typename TargetInput>
int64_t MTPWorkerImpl<TargetInput>::get_embedding_placeholder_size() const {
  return mtp_hidden_state_width(impl_->context_.get_model_args());
}

template <typename TargetInput>
KVCacheShape MTPWorkerImpl<TargetInput>::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  if (options_.enable_mtp_draft_body_tp1()) {
    return build_draft_kv_cache_shape(target_kv_cache_shape,
                                      /*draft_world_size=*/1);
  }
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() ||
      target_kv_cache_shape.shared_draft_layers() > 0) {
    return build_draft_kv_cache_shape(target_kv_cache_shape);
  }
  return target_kv_cache_shape;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::supports_expanded_spec_verify() const {
  return target_verify_mode_ ==
         mtp_async::TargetSpecVerifyMode::EXPANDED_VERIFY;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::requires_full_target_replay() const {
  CHECK(draft_context_replay_policy_.has_value())
      << "Draft context replay policy is not initialized.";
  return draft_context_replay_policy_->full_target_replay;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::should_use_explicit_spec_verify_replay_update(
    const TargetInput& input) const {
#if defined(USE_NPU)
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  const int64_t num_sequences = input.input_params.meta.num_sequences;
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !::xllm::ExecutionConfig::get_instance()
           .enable_graph_mode_decode_no_padding() ||
      !supports_expanded_spec_verify() || options_.block_size() <= 0 ||
      !spec_verify::has_speculative_verify_block_table_layout(
          block_tables, input.input_params.multi_block_tables, num_sequences)) {
    return false;
  }
  const int64_t block_table_width =
      block_tables.defined()
          ? spec_verify_block_table_width(block_tables)
          : spec_verify::speculative_verify_block_table_capacity(
                impl_->context_.get_model_args().max_position_embeddings(),
                logical_block_size());
  if (block_table_width <= 0 ||
      block_table_width > kMaxSpecVerifyGraphUpdateBlockTableWidth) {
    return false;
  }
  const int64_t spec_width =
      static_cast<int64_t>(options_.num_speculative_tokens()) + 1;
  return kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
      spec_width, options_.block_size());
#else
  (void)input;
  return false;
#endif
}

template <typename TargetInput>
int64_t MTPWorkerImpl<TargetInput>::spec_verify_block_table_width(
    const torch::Tensor& block_tables) const {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  const int64_t declared_capacity =
      spec_verify::speculative_verify_block_table_capacity(
          impl_->context_.get_model_args().max_position_embeddings(),
          logical_block_size());
  CHECK_LE(block_tables.size(1), declared_capacity)
      << "block table width exceeds the model position capacity";
  return declared_capacity;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::ensure_spec_verify_control_block_table(
    TargetModelParams& input_params,
    int64_t num_sequences) {
  auto& block_tables = input_params.attention.host.block_tables;
  if (block_tables.defined()) {
    return;
  }
  CHECK(!input_params.multi_block_tables.empty())
      << "missing model-managed block tables for spec verify";
  const int64_t block_table_capacity =
      spec_verify::speculative_verify_block_table_capacity(
          impl_->context_.get_model_args().max_position_embeddings(),
          logical_block_size());
  CHECK_GT(num_sequences, 0);
  CHECK_GT(block_table_capacity, 0);

  const int64_t row_capacity = std::max(
      num_sequences, static_cast<int64_t>(options_.max_seqs_per_batch()));
  if (!spec_verify_control_block_table_buffer_.defined() ||
      spec_verify_control_block_table_buffer_.size(0) < row_capacity ||
      spec_verify_control_block_table_buffer_.size(1) < block_table_capacity) {
    spec_verify_control_block_table_buffer_ =
        spec_verify::make_speculative_verify_control_block_table(
            row_capacity, block_table_capacity);
  }

  block_tables = spec_verify_control_block_table_buffer_
                     .narrow(/*dim=*/0, /*start=*/0, num_sequences)
                     .narrow(/*dim=*/1, /*start=*/0, block_table_capacity);
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::owns_npu_parallel_input_prepare() const {
  return false;
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_empty(
    const TargetInput& input) {
  const bool use_prelaunched_first_draft =
      input.input_params.meta.batch_forward_type.is_decode() &&
      can_use_combined_first_draft() && pending_draft_context_matches(input);
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // The preceding validation may have speculatively submitted draft-0 before
    // the scheduler learned that the batch had finished.  Keep its graph/input
    // buffers alive until the queued work completes, then discard the result.
    // This is a batch-exit slow path and is never taken in steady decode.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain final MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  flush_pending_target_context();

  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    TargetInput target_prepared;
    LlmForwardInput draft_prepared;
    LlmForwardInput draft_input =
        make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
    auto output = run_worker_no_sync_impl(
        *impl_, input, *prepare_stream_, *compute_stream_, target_prepared);
    auto draft_output = run_worker_no_sync_impl(*draft_impl_,
                                                draft_input,
                                                *prepare_stream_,
                                                *compute_stream_,
                                                draft_prepared);
    if (draft_output.has_value()) {
      transfer_retained_inputs(*output, draft_output.value());
      append_failed_request_ids(*output,
                                std::move(draft_output->failed_request_ids));
    }
    clear_all_output_embeddings(*output);
    this->finalize_output_on_stream(
        *output, *compute_stream_, enable_schedule_overlap());
    return output;
  } else {
    LlmForwardInput draft_extend_prepared;
    std::vector<LlmForwardInput> draft_step_prepared(
        options_.num_speculative_tokens());
    TargetInput target_prepared;
    std::vector<ForwardOutput> draft_outputs;
    draft_outputs.reserve(options_.num_speculative_tokens());

    LlmForwardInput draft_extend_input =
        make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
    CHECK(draft_context_replay_policy_.has_value())
        << "Draft context replay policy is not initialized.";
    scale_speculative_parallel_token_counts(
        draft_extend_input.input_params,
        draft_context_replay_policy_->num_speculative_tokens);
    if (use_prelaunched_first_draft) {
      draft_outputs.emplace_back(
          std::move(pending_draft_context_.output.value()));
      draft_extend_prepared = std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_outputs.emplace_back(run_worker_no_sync_impl(*draft_impl_,
                                                         draft_extend_input,
                                                         *prepare_stream_,
                                                         *compute_stream_,
                                                         draft_extend_prepared)
                                     .value());
    }

    for (int32_t i = 1; i < options_.num_speculative_tokens(); ++i) {
      LlmForwardInput draft_input = make_llm_draft_input(
          input, draft_impl_->uses_shared_static_kv_cache());
      draft_outputs.emplace_back(run_worker_no_sync_impl(*draft_impl_,
                                                         draft_input,
                                                         *prepare_stream_,
                                                         *compute_stream_,
                                                         draft_step_prepared[i])
                                     .value());
    }

    TargetInput new_input = input.clone();
    scale_speculative_parallel_token_counts(new_input.input_params,
                                            options_.num_speculative_tokens());
    ForwardOutput output = run_worker_no_sync_impl(*impl_,
                                                   new_input,
                                                   *prepare_stream_,
                                                   *compute_stream_,
                                                   target_prepared)
                               .value();
    for (ForwardOutput& draft_output : draft_outputs) {
      transfer_retained_inputs(output, draft_output);
      append_failed_request_ids(output,
                                std::move(draft_output.failed_request_ids));
    }
    clear_all_output_embeddings(output);
    this->finalize_output_on_stream(
        output, *compute_stream_, enable_schedule_overlap());
    if (can_prelaunch_next_first_draft(input)) {
      LlmForwardInput next_first_draft_input = make_llm_draft_input(
          input, draft_impl_->uses_shared_static_kv_cache());
      scale_parallel_token_counts(
          next_first_draft_input.input_params.parallel,
          draft_context_replay_policy_->num_speculative_tokens + 1);
      submit_pending_first_draft(input, std::move(next_first_draft_input));
    }
    return output;
  }
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_prefill(
    const TargetInput& input) {
  flush_pending_target_context();
  if (uses_verifier_conditioned_draft_prefix()) {
    for (const int32_t embedding_id :
         input.input_params.embedding.embedding_ids) {
      draft_prefix_contexts_.erase(embedding_id);
    }
  }

  Timer target_timer;
  TargetInput target_prepared;
  LlmForwardInput draft_prepared;

  // run the target model to get first token and hidden states
  ForwardOutput output =
      run_worker_no_sync_impl(
          *impl_, input, *prepare_stream_, *compute_stream_, target_prepared)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  // MTP path that depends on hidden states.
  LlmForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  check_draft_input_embedding(embeddings, "prefill");

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill seeds the MTP decode cache. Under orthogonal CP x TP each
    // CP shard samples independently unless this token is synchronized across
    // both axes; caching divergent tokens makes the first decode input disagree
    // with the non-driver CP shard's DecodeState.
    maybe_broadcast_spec_tokens(output.sample_output.next_tokens,
                                input.sampling_params.sampling_mode());
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      replace_host_token_placeholders(prefill_input,
                                      -1,
                                      output.sample_output.next_tokens,
                                      prefill_input.token_ids.options());
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      CHECK(compute_stream_->wait_event(
          prefill_input.runtime.metadata_ready_event))
          << "failed to wait speculative metadata ready event";
      prefill_input.runtime.metadata_ready_event =
          compute_stream_->record_event_or_sync();
    }
  }
  // generate kv cache for draft model
  Timer draft_timer;
  ForwardOutput draft_output = run_worker_no_sync_impl(*draft_impl_,
                                                       prefill_input,
                                                       *prepare_stream_,
                                                       *compute_stream_,
                                                       draft_prepared)
                                   .value();
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    this->draft_vocab_.remap(draft_output.sample_output);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());

  if (input.sampling_params.num_sample_seqs() > 0) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    prepare_first_draft_inputs(*embedding_cache_, input, output);
  } else {
    clear_all_output_embeddings(output);
  }

  transfer_retained_inputs(output, draft_output);
  append_failed_request_ids(output, std::move(draft_output.failed_request_ids));
  this->finalize_output_on_stream(
      output, *compute_stream_, enable_schedule_overlap());

  return finalize_step_output(std::move(output));
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_prefill_inputs(
    const TargetInput& input,
    LlmForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  prefill_input =
      make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
  prepare_draft_sampling(options_.enable_probabilistic_draft(),
                         prefill_input.sampling_params);
  prefill_input = prefill_input.to(device_, dtype_);
  clear_ready_events(prefill_input);
  // Draft prefill consumes target hidden states, not vision inputs.
  auto& input_params = prefill_input.input_params;
  // The Qwen draft is a pure full-attention model; without this cleanup the
  // target's recurrent slot metadata makes MTP prefill enter a stateful path
  // it has neither a validity mask nor a recurrent cache for.
  // Full-attention KPool drafts still own per-request compression tails.
  if (!draft_impl_->has_request_state_cache()) {
    input_params.clear_linear_attention_state();
  }
  auto& extra_token_ids = input_params.embedding.extra_token_ids;

  const torch::Tensor& token_ids = input.token_ids_host;
  Slice<int32_t> tokens_ids_slice = tensor_slice(token_ids);

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(token_ids.numel());
  for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
    int32_t q_len = input_params.attention.host.q_seq_lens[i];
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.runtime.device_tensors_ready = false;
  prefill_input.token_ids_host = make_pinned_cpu_tensor(new_token_ids);
  prefill_input.token_ids = safe_to(prefill_input.token_ids_host,
                                    prefill_input.positions.options(),
                                    /*non_blocking=*/true);
  prefill_input.runtime.device_tensors_ready = true;
  prefill_input.runtime.metadata_ready_event =
      prepare_stream_->record_event_or_sync();
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_decode(
    const TargetInput& raw_input) {
  TargetInput input = raw_input.clone();
  if (this->uses_chunked_verify()) {
    stabilize_decode_host_tensors(input);
  }
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  // Reuse draft-0 prelaunched for this same batch.
  const bool use_prelaunched_first_draft =
      can_use_combined_first_draft() && pending_draft_context_matches(input);
  const bool matching_device_target_context =
      pending_target_context_matches(input);
  // Consume this batch's pending target context directly on device.
  const bool use_device_target_context =
      can_use_combined_first_draft() && matching_device_target_context &&
      device_target_context_ready_for_batch(input);
  // Keep the device-side accepted state alive across a first-transition Host
  // cache flush. The prelaunched draft can be valid before the batch is marked
  // device-context ready, while flush_pending_target_context() clears the
  // owning context below.
  const torch::Tensor accepted_tokens =
      latest_pending_target_context().accepted_tokens;
  const torch::Tensor accepted_embeddings =
      latest_pending_target_context().accepted_embeddings;
  const torch::Tensor target_base_positions =
      latest_pending_target_context().base_positions;
  const torch::Tensor target_base_kv_seq_lens =
      latest_pending_target_context().base_kv_seq_lens;
  const StreamEventPtr target_context_ready_event =
      latest_pending_target_context().ready_event;
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // A batch transition invalidates the speculative prelaunch.  Drain it
    // before releasing its graph/input buffers; this slow path is outside
    // steady decode and preserves cache/buffer lifetime correctness.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain stale MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  if (!use_device_target_context) {
    // Batch transitions are uncommon in steady decode.  Materialize the most
    // recent target state only for that fallback; the normal path below never
    // synchronizes the worker thread with the NPU.
    flush_pending_target_context();
    if (matching_device_target_context) {
      // The first target-context publication for a new batch establishes the
      // scheduler's corrected position/KV base. Subsequent publications can
      // derive that base fully on device without waiting for the scheduler.
      device_context_ready_embedding_ids_ =
          input.input_params.embedding.embedding_ids;
      device_context_ready_request_ids_ =
          input.input_params.embedding.request_ids;
    } else if (!device_target_context_ready_for_batch(input)) {
      device_context_ready_embedding_ids_.clear();
      device_context_ready_request_ids_.clear();
    }
  }
  std::vector<ForwardOutput> draft_outputs;
  LlmForwardInput current_draft_input, next_step_input;
  TargetInput validate_input;
  std::vector<LlmForwardInput> draft_prepared(num_speculative_tokens);
  // Records before the validate tail so its phases stay out of this window.
  Timer draft_timer;
  CHECK(embedding_cache_ != nullptr) << "MTP embedding cache is not allocated";

  prepare_draft_input(input);

  TargetInput metadata_template = input.clone();
  if (use_prelaunched_first_draft) {
    // The first draft was fully prepared and submitted by the preceding
    // run_validate(). Host metadata is corrected below after the accepted-token
    // update; continuous DSA drafts use device correction, while target
    // verification still consumes the exact Host metadata.
  } else if (use_device_target_context) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    if (uses_verifier_conditioned_draft_prefix()) {
      prepare_draft_prefix(input, {}, current_draft_input);
    } else {
      prepare_first_draft_template(metadata_template,
                                   current_draft_input,
                                   /*wait_for_compute_stream=*/true);
      CHECK(compute_stream_->wait_event(
          current_draft_input.runtime.metadata_ready_event))
          << "failed to wait speculative metadata ready event";
      clear_ready_events(current_draft_input);

      mtp_async::prepare_next_draft_from_accepted_state(
          current_draft_input,
          input,
          accepted_tokens,
          accepted_embeddings,
          embedding_cache_->embedding_placeholder(),
          target_base_positions,
          target_base_kv_seq_lens,
          /*rebuild_expanded_decode_metadata=*/true,
          logical_block_size());
    }
  } else {
    // First decode after prefill and batch transitions use the host cache.
    std::vector<EmbeddingCache::DecodeState> last_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    CHECK_EQ(last_states.size(),
             input.input_params.embedding.embedding_ids.size())
        << "decode target state count mismatch";
    // Synthetic graph warmup requests do not have a target decode history.
    // Draft preparation already falls back to the initialized placeholder for
    // invalid states, so reserve strict cache validation for real requests.
    if (!input.input_params.meta.is_graph_warmup) {
      check_mtp_decode_states(last_states,
                              input.input_params.embedding.request_ids,
                              input.token_ids_host,
                              enable_schedule_overlap());
    }
    update_decode_step_input(input, last_states);
    metadata_template = input.clone();
    if (uses_verifier_conditioned_draft_prefix()) {
      prepare_draft_prefix(input, last_states, current_draft_input);
    } else {
      prepare_draft_extend_inputs(input, last_states, current_draft_input);
    }
  }

  const auto materialize_pending_target_host_state = [&]() {
    flush_pending_target_context();
    std::vector<EmbeddingCache::DecodeState> resolved_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    // The scheduler input contains the conservative overlap placeholder.
    // Force cache correction before comparing it with the accepted target.
    input.token_ids_host = torch::full_like(input.token_ids_host, -1);
    update_decode_step_input(input, resolved_states);
    check_mtp_decode_states(resolved_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            /*allow_overlap_fake_token=*/false);
    metadata_template = input.clone();
  };

  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state;
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft &&
        prepare_static_mtp_graph_tasks_before_final_draft(input);
    if (reuse_mtp_topk_state) {
      current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
    }
    std::optional<ForwardOutput> draft_output_opt;
    if (use_prelaunched_first_draft && draft_idx == 0) {
      draft_output_opt = std::move(pending_draft_context_.output);
      draft_prepared[draft_idx] =
          std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_output_opt = run_worker_no_sync_impl(*draft_impl_,
                                                 current_draft_input,
                                                 *compute_stream_,
                                                 *compute_stream_,
                                                 draft_prepared[draft_idx]);
    }

    if ((use_device_target_context || use_prelaunched_first_draft) &&
        draft_idx == 0) {
      // The next draft forward is already queued behind target validation.
      // It can start immediately when rejection sampling finishes while the
      // worker materializes the accepted state for later draft/target metadata
      // on CPU.  This synchronization is therefore outside the NPU critical
      // path rather than sitting between target and draft launches.
      materialize_pending_target_host_state();
    }

    // Overlap next-step input preparation with async draft forward.
    if (is_final_draft) {
      prepare_validate_inputs(
          metadata_template, validate_input, static_graph_tasks_prepared);

    } else {
      // The verifier-conditioned first pass writes the bonus at anchor-1 and
      // produces proposal 0. Proposal 1 writes at anchor, not anchor+1.
      prepare_draft_inputs(
          metadata_template,
          next_step_input,
          uses_verifier_conditioned_draft_prefix() ? draft_idx : draft_idx + 1);
    }

    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.emplace_back(std::move(draft_output_opt.value()));
    const SamplingParameters& draft_sampling_params =
        draft_prepared[draft_idx].sampling_params;
    {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      if (reuse_mtp_topk_state) {
        mtp_topk_state = select_mtp_topk_state_for_next_step(
            draft_outputs.back().mtp_topk_state, draft_sampling_params);
      }
      // Keep draft tokens consistent across the consensus group.
      maybe_broadcast_spec_tokens(
          draft_outputs.back().sample_output.next_tokens,
          draft_sampling_params.sampling_mode());
      this->draft_vocab_.remap(draft_outputs.back().sample_output);
    }
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    check_draft_input_embedding(last_output.embeddings, "decode");
    current_draft_input = std::move(next_step_input);
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options(),
                                *compute_stream_);
    if (last_output.embeddings.defined()) {
      current_draft_input.input_params.embedding.input_embedding =
          last_output.embeddings;
      // input_embedding is produced and consumed on compute_stream_; FIFO
      // ordering replaces the same-stream EventRecord/EventWait pair.
    }
    CHECK(compute_stream_->wait_event(
        current_draft_input.runtime.metadata_ready_event))
        << "failed to wait speculative metadata ready event";
    current_draft_input.runtime.metadata_ready_event =
        compute_stream_->record_event_or_sync();
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return run_validate(
      input, draft_outputs, validate_input, num_speculative_tokens);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::fill_validate_input_from_draft_outputs(
    const TargetInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    TargetInput& validate_input,
    int32_t num_speculative_tokens,
    Stream& compute_stream) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";

  CHECK_EQ(validate_input.token_ids.numel(),
           static_cast<int64_t>(num_sequences) * num_val_tokens)
      << "MTP validation requires fixed-width rows";
  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  CHECK(compute_stream.wait_event(validate_input.runtime.metadata_ready_event))
      << "failed to wait speculative metadata ready event";

  if (!input.json_object_states.empty()) {
    CHECK_EQ(input.json_object_states.size(),
             static_cast<size_t>(num_sequences))
        << "JSON grammar state rows must match validation sequences";
  }

  std::vector<torch::Tensor> draft_token_steps;
  draft_token_steps.reserve(static_cast<size_t>(num_speculative_tokens));
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    CHECK(static_cast<size_t>(i) < draft_outputs.size())
        << "draft_outputs index out of range for step " << i;
    const torch::Tensor& next_tokens =
        draft_outputs[static_cast<size_t>(i)].sample_output.next_tokens;
    CHECK(next_tokens.defined())
        << "draft next_tokens must be defined for draft step " << i;
    draft_token_steps.push_back(next_tokens.flatten());
  }

  if (!input.json_object_states.empty()) {
    std::vector<std::vector<int32_t>> draft_token_ids;
    draft_token_ids.reserve(draft_token_steps.size());
    if (!draft_token_steps.empty()) {
      const std::vector<int32_t> flat_draft_tokens =
          tensor_to_vector<int32_t>(torch::stack(draft_token_steps, /*dim=*/0));
      for (size_t i = 0; i < draft_token_steps.size(); ++i) {
        const size_t row_begin = i * static_cast<size_t>(num_sequences);
        draft_token_ids.emplace_back(
            flat_draft_tokens.begin() + row_begin,
            flat_draft_tokens.begin() + row_begin + num_sequences);
      }
    }
    validate_input.json_object_states =
        detail::replay_draft_states(input.json_object_states, draft_token_ids);
    validate_input.sampling_params.filter_bitmask() =
        build_json_object_filter_bitmask(validate_input.json_object_states,
                                         device_);
  }

  validate_input.runtime.device_tensors_ready = false;
  auto& fused_draft_tokens =
      validate_input.input_params.graph.spec_verify_draft_token_sources;
  fused_draft_tokens.clear();
#if defined(USE_NPU)
  const bool use_fused_verify_token_update =
      validate_input.input_params.graph.spec_verify_source_addresses_stable &&
      validate_input.input_params.graph.input_tokens_override.defined() &&
      supports_expanded_spec_verify() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          num_val_tokens, options_.block_size());
  if (use_fused_verify_token_update) {
    fused_draft_tokens.reserve(draft_token_steps.size());
    for (const torch::Tensor& next_tokens : draft_token_steps) {
      CHECK(next_tokens.numel() == num_sequences &&
            next_tokens.scalar_type() == torch::kInt64 &&
            next_tokens.device() == validate_input.token_ids.device() &&
            next_tokens.is_contiguous())
          << "fused speculative verify input update requires one int64 token "
             "per sequence and draft step";
      fused_draft_tokens.emplace_back(next_tokens);
    }
    validate_input.runtime.device_tensors_ready = true;
    return;
  }
#endif
  // Stack the draft columns once and copy into the fixed-width rows.
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});
  if (!draft_token_steps.empty()) {
    std::vector<torch::Tensor> draft_token_columns;
    draft_token_columns.reserve(draft_token_steps.size());
    for (const torch::Tensor& draft_tokens : draft_token_steps) {
      draft_token_columns.push_back(
          safe_to(draft_tokens, token_options, /*non_blocking=*/true));
    }
    torch::Tensor packed_drafts = torch::stack(draft_token_columns, /*dim=*/1);
    validate_token_rows.slice(/*dim=*/1, /*start=*/1, /*end=*/num_val_tokens)
        .copy_(packed_drafts, /*non_blocking=*/true);
  }
  validate_input.runtime.device_tensors_ready = true;
  validate_input.runtime.metadata_ready_event =
      compute_stream.record_event_or_sync();
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::run_validate(
    const TargetInput& input,
    std::vector<ForwardOutput>& draft_outputs,
    TargetInput& validate_input,
    int32_t num_speculative_tokens) {
  Timer target_timer;
  TargetInput target_prepared;
  fill_validate_input_from_draft_outputs(input,
                                         draft_outputs,
                                         validate_input,
                                         num_speculative_tokens,
                                         *compute_stream_);
  ForwardOutput target_output = run_worker_no_sync_impl(*impl_,
                                                        validate_input,
                                                        *compute_stream_,
                                                        *compute_stream_,
                                                        target_prepared)
                                    .value();
  for (ForwardOutput& draft_output : draft_outputs) {
    append_failed_request_ids(target_output,
                              std::move(draft_output.failed_request_ids));
  }
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_EQ(target_output.logits.size(0),
           static_cast<int64_t>(batch_size) * (num_speculative_tokens + 1));

  const bool prelaunch_next_first_draft = can_prelaunch_next_first_draft(input);
  LlmForwardInput next_first_draft_input;
  if (prelaunch_next_first_draft) {
    // This input is independent of the accepted token.  Prepare it on the
    // auxiliary stream while target verification is still executing; the
    // compute stream consumes it through a device-side event after rejection
    // sampling, with no host synchronization.
    TargetInput next_metadata_template = input.clone();
    prepare_first_draft_template(next_metadata_template,
                                 next_first_draft_input,
                                 /*wait_for_compute_stream=*/false);
    next_first_draft_input.skip_sampling_for_logits_only = false;
  }

  // verify the proposals with target and update the batch
  SampleOutput val_output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_validation);
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Lift the JSON grammar masks (installed at full validate width) onto
    // the request params: the mask is the one per-token verify field (see
    // the verify transforms in rejection_sample), because its
    // per-sequence grammar state advances with each speculative token.
    SamplingParameters verify_params = input.sampling_params;
    if (verify_params.min_token_positions().defined()) {
      // Overlap can leave the scheduler's generated count behind accepted
      // tokens. Use the corrected target positions; rejection sampling applies
      // the candidate and bonus offsets within each verify row.
      // KV lengths count text and visual tokens uniformly. mRoPE coordinates
      // can have three axes and a prompt-dependent offset, so they cannot
      // determine how many output tokens have already been committed.
      torch::Tensor first_positions;
      if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
        const torch::Tensor& lengths =
            validate_input.input_params.attention.device.kv_seq_lens;
        const int64_t width = num_speculative_tokens + 1;
        if (lengths.numel() == batch_size) {
          first_positions = lengths - width;
        } else {
          first_positions =
              lengths.view({batch_size, width}).select(/*dim=*/1, /*index=*/0) -
              1;
        }
      } else {
        first_positions = validate_input.positions
                              .view({batch_size, num_speculative_tokens + 1})
                              .select(/*dim=*/1, /*index=*/0);
      }
      verify_params.min_tokens_remaining() =
          verify_params.min_token_positions() - first_positions;
    }
    if (validate_input.sampling_params.filter_bitmask().defined()) {
      verify_params.filter_bitmask() =
          validate_input.sampling_params.filter_bitmask();
    }
    val_output = validate(std::move(verify_params),
                          draft_outputs,
                          target_output,
                          num_speculative_tokens);
  }

  const int64_t num_val_tokens = options_.num_speculative_tokens() + 1;
  torch::Tensor validate_positions = validate_input.positions;
  CHECK_EQ(validate_positions.numel(), batch_size * num_val_tokens)
      << "validate positions must contain one row per speculative token";
  const torch::Tensor& validate_kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens;
  CHECK_GE(validate_kv_seq_lens.numel(), batch_size)
      << "validate KV lengths must be sequence-scoped";
  torch::Tensor accepted_tokens_host =
      acquire_accepted_tokens_host_buffer(val_output.next_tokens);
  torch::Tensor accepted_tokens_cpu_result = accepted_tokens_host;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    maybe_broadcast_spec_tokens(val_output.next_tokens,
                                input.sampling_params.sampling_mode());

    base_positions = validate_positions.view({batch_size, num_val_tokens})
                         .select(/*dim=*/1, /*index=*/0)
                         .contiguous();
    base_kv_seq_lens =
        mtp_async::extract_target_base_kv_seq_lens(validate_kv_seq_lens,
                                                   batch_size,
                                                   num_val_tokens,
                                                   this->uses_chunked_verify());

    if (uses_verifier_conditioned_draft_prefix()) {
      retain_draft_prefix(input, val_output, base_positions, base_kv_seq_lens);
    }

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers consensus, base-state derivation, and the D2H copy.
    target_context_ready_event = compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  std::vector<size_t> failed_sequence_rows;
  if (!input.json_object_states.empty()) {
    CHECK_EQ(accepted_tokens_host.dim(), 2)
        << "MTP JSON accepted output must be [sequence, token]";
    CHECK_EQ(input.sample_sequence_ids.size(), input.json_object_states.size())
        << "MTP JSON sequence ids must match grammar state rows";
    CHECK_EQ(input.input_params.embedding.embedding_ids.size(),
             input.json_object_states.size())
        << "MTP JSON embedding ids must match grammar state rows";
    CHECK(target_context_ready_event == nullptr ||
          target_context_ready_event->synchronize())
        << "failed to wait for accepted MTP tokens before JSON replay";

    const torch::Tensor accepted_tokens =
        to_cpu_contiguous(accepted_tokens_host, torch::kInt64);
    const std::vector<JsonTokenMismatch> mismatches =
        detail::find_json_token_mismatches(
            input.json_object_states,
            accepted_tokens.const_data_ptr<int64_t>(),
            static_cast<size_t>(accepted_tokens.size(0)),
            static_cast<size_t>(accepted_tokens.size(1)));
    std::unordered_set<std::string> failed_sequence_ids;
    failed_sequence_ids.reserve(input.json_object_errors.size() +
                                target_output.json_object_errors.size() +
                                mismatches.size());
    for (const JsonObjectOutputError& error : input.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON input error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const JsonObjectOutputError& error :
         target_output.json_object_errors) {
      CHECK(!error.sample_sequence_id.empty())
          << "MTP JSON output error requires a sampled sequence id";
      failed_sequence_ids.emplace(error.sample_sequence_id);
    }
    for (const JsonTokenMismatch& mismatch : mismatches) {
      const std::string& sample_sequence_id =
          input.sample_sequence_ids[mismatch.sequence_index];
      CHECK(!sample_sequence_id.empty())
          << "MTP JSON mismatch requires a sampled sequence id";
      LOG(ERROR) << "MTP JSON accepted output replay mismatch: sequence_id="
                 << sample_sequence_id
                 << ", token_offset=" << mismatch.token_offset
                 << ", token_id=" << mismatch.token_id
                 << ", committed_tokens=" << mismatch.committed_tokens
                 << ", state_fingerprint=" << mismatch.state_fingerprint;
      if (failed_sequence_ids.emplace(sample_sequence_id).second) {
        target_output.json_object_errors.push_back(
            {sample_sequence_id,
             "accepted MTP output violates json_object grammar, token_id=" +
                 std::to_string(mismatch.token_id)});
      }
    }
    failed_sequence_rows.reserve(failed_sequence_ids.size());
    for (size_t sequence_index = 0;
         sequence_index < input.sample_sequence_ids.size();
         ++sequence_index) {
      if (failed_sequence_ids.contains(
              input.sample_sequence_ids[sequence_index])) {
        failed_sequence_rows.emplace_back(sequence_index);
      }
    }
    CHECK_EQ(failed_sequence_rows.size(), failed_sequence_ids.size())
        << "MTP JSON errors must reference sampled rows in the current batch";
  }
  const bool has_failed_sequence_rows = !failed_sequence_rows.empty();
  if (uses_verifier_conditioned_draft_prefix()) {
    for (const size_t row : failed_sequence_rows) {
      draft_prefix_contexts_.erase(
          input.input_params.embedding.embedding_ids[row]);
    }
  }
  // Overlap mode hands the verify tokens back still on device (the device copy
  // feeds the next step's input update); expose the staged host snapshot so
  // metrics and serialization reuse it instead of issuing a second D2H.
  if (enable_schedule_overlap()) {
    target_output.next_tokens_host = accepted_tokens_host;
  }
  stage_target_context_write(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             target_context_ready_event,
                             std::move(accepted_tokens_host),
                             std::move(failed_sequence_rows));
  if (prelaunch_next_first_draft && !has_failed_sequence_rows) {
    // Submit the next iteration's first draft before returning to the
    // scheduler.  This is the actual asynchronous boundary: scheduler/host
    // accepted-state work can no longer sit between target validation and the
    // next draft launch.
    enqueue_next_first_draft(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             std::move(next_first_draft_input));
  }
  target_output.ready_event = target_context_ready_event;

  // Keep prepared inputs alive until the target-context event completes.
  for (ForwardOutput& draft_output : draft_outputs) {
    transfer_retained_inputs(target_output, draft_output);
  }

  if (!enable_schedule_overlap()) {
    flush_pending_target_context();
    target_output.retained_inputs.clear();
    target_output.ready_event.reset();
    val_output.next_tokens = std::move(accepted_tokens_cpu_result);
  }

  return finalize_step_output(std::move(target_output),
                              std::move(val_output),
                              input.json_object_states);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::stage_target_context_write(
    const TargetInput& input,
    const SampleOutput& validate_output,
    torch::Tensor base_positions,
    torch::Tensor base_kv_seq_lens,
    StreamEventPtr ready_event,
    torch::Tensor accepted_tokens_host,
    std::vector<size_t> failed_rows,
    torch::Tensor accepted_count_host,
    bool allow_immutable_compact_view) {
  PendingTargetContext context;
  context.generation = next_pending_target_generation_++;
  context.embedding_ids = input.input_params.embedding.embedding_ids;
  context.request_ids = input.input_params.embedding.request_ids;
  context.accepted_tokens = validate_output.next_tokens;
  context.accepted_tokens_host = std::move(accepted_tokens_host);
  context.accepted_count_host = std::move(accepted_count_host);
  context.accepted_embeddings = validate_output.embeddings;
  context.base_positions = std::move(base_positions);
  context.base_kv_seq_lens = std::move(base_kv_seq_lens);
  context.failed_rows = std::move(failed_rows);
  context.ready_event = std::move(ready_event);
  context.allow_immutable_compact_view = allow_immutable_compact_view;
  if (pending_target_context_queue_.size() >= kMaxPendingTargetContexts) {
    flush_pending_target_context(/*keep_latest=*/kMaxPendingTargetContexts - 1);
  }
  pending_target_context_queue_.emplace_back(std::move(context));
}

template <typename TargetInput>
torch::Tensor MTPWorkerImpl<TargetInput>::acquire_accepted_tokens_host_buffer(
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";
  const int64_t required_capacity = accepted_tokens.numel();
  const int64_t configured_capacity =
      static_cast<int64_t>(options_.max_seqs_per_batch()) *
      (static_cast<int64_t>(options_.num_speculative_tokens()) + 1);
  if (enable_schedule_overlap()) {
    return acquire_pinned_output_buffer(accepted_tokens,
                                        accepted_tokens_host_pool_);
  }
  const bool needs_allocation =
      !accepted_tokens_host_buffer_.defined() ||
      accepted_tokens_host_buffer_.scalar_type() !=
          accepted_tokens.scalar_type() ||
      accepted_tokens_host_buffer_.numel() < required_capacity;
  if (needs_allocation) {
    const int64_t capacity = std::max(required_capacity, configured_capacity);
    accepted_tokens_host_buffer_ = torch::empty(
        {capacity},
        accepted_tokens.options().device(torch::kCPU).pinned_memory(true));
  }

  return accepted_tokens_host_buffer_
      .narrow(/*dim=*/0, /*start=*/0, required_capacity)
      .view(accepted_tokens.sizes());
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::pending_target_context_matches(
    const TargetInput& input) const {
  if (pending_target_context_queue_.empty()) {
    return false;
  }
  const PendingTargetContext& context = pending_target_context_queue_.back();
  return context.failed_rows.empty() && context.accepted_tokens.defined() &&
         context.embedding_ids == input.input_params.embedding.embedding_ids &&
         context.request_ids == input.input_params.embedding.request_ids;
}

template <typename TargetInput>
const typename MTPWorkerImpl<TargetInput>::PendingTargetContext&
MTPWorkerImpl<TargetInput>::latest_pending_target_context() const {
  static const PendingTargetContext empty;
  return pending_target_context_queue_.empty()
             ? empty
             : pending_target_context_queue_.back();
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::device_target_context_ready_for_batch(
    const TargetInput& input) const {
  return device_context_ready_embedding_ids_ ==
             input.input_params.embedding.embedding_ids &&
         device_context_ready_request_ids_ ==
             input.input_params.embedding.request_ids;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::flush_pending_target_context(
    size_t keep_latest) {
  while (pending_target_context_queue_.size() > keep_latest) {
    PendingTargetContext pending =
        std::move(pending_target_context_queue_.front());
    pending_target_context_queue_.pop_front();
    if (!pending.accepted_tokens.defined()) {
      continue;
    }
    CHECK(pending.ready_event == nullptr || pending.ready_event->synchronize())
        << "failed to wait for pending MTP target context generation "
        << pending.generation;
    CHECK(embedding_cache_ != nullptr)
        << "embedding_cache_ must be initialized before target cache write";
    if (pending.failed_rows.empty()) {
      embedding_cache_->write_target_context(
          pending.embedding_ids,
          pending.request_ids,
          pending.accepted_tokens_host,
          pending.accepted_embeddings,
          pending.accepted_count_host,
          options_.num_speculative_tokens(),
          pending.allow_immutable_compact_view);
    } else {
      CHECK(pending.request_ids.empty() ||
            pending.request_ids.size() == pending.embedding_ids.size())
          << "target context request ids must match embedding ids";
      std::unordered_set<size_t> failed_row_set(pending.failed_rows.begin(),
                                                pending.failed_rows.end());
      CHECK_EQ(failed_row_set.size(), pending.failed_rows.size())
          << "target context failed rows must be unique";

      std::vector<int32_t> failed_embedding_ids;
      failed_embedding_ids.reserve(pending.failed_rows.size());
      for (const size_t failed_row : pending.failed_rows) {
        CHECK_LT(failed_row, pending.embedding_ids.size())
            << "target context failed row exceeds embedding ids";
        failed_embedding_ids.emplace_back(pending.embedding_ids[failed_row]);
      }
      embedding_cache_->clear(failed_embedding_ids);

      for (size_t sequence_index = 0;
           sequence_index < pending.embedding_ids.size();
           ++sequence_index) {
        if (failed_row_set.contains(sequence_index)) {
          continue;
        }
        const std::vector<int32_t> row_embedding_ids = {
            pending.embedding_ids[sequence_index]};
        const std::vector<std::string> row_request_ids =
            pending.request_ids.empty()
                ? std::vector<std::string>()
                : std::vector<std::string>{pending.request_ids[sequence_index]};
        embedding_cache_->write_target_context(
            row_embedding_ids,
            row_request_ids,
            pending.accepted_tokens_host.narrow(
                /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
            pending.accepted_embeddings.dim() == 2
                ? pending.accepted_embeddings.narrow(
                      /*dim=*/0, /*start=*/sequence_index * 2, /*length=*/2)
                : pending.accepted_embeddings.narrow(
                      /*dim=*/0, /*start=*/sequence_index, /*length=*/1),
            pending.accepted_count_host.defined()
                ? pending.accepted_count_host.narrow(
                      /*dim=*/0, /*start=*/sequence_index, /*length=*/1)
                : torch::Tensor(),
            options_.num_speculative_tokens(),
            pending.allow_immutable_compact_view);
      }
    }
  }
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::uses_verifier_conditioned_draft_prefix()
    const {
#if defined(USE_NPU)
  return draft_prefix_policy_ ==
             mtp_async::DraftPrefixPolicy::VERIFIER_CONDITIONED_FIRST_PASS &&
         ModelConfig::is_python_model_impl(
             ModelConfig::get_instance().model_impl()) &&
         KernelConfig::get_instance().npu_kernel_backend() == "TORCH" &&
         parallel_args_.dp_size() == 1 && parallel_args_.cp_size() == 1 &&
         parallel_args_.kv_split_size_effective() == 1 &&
         options_.num_speculative_tokens() >= 1 &&
         options_.num_speculative_tokens() <= 16;
#else
  return false;
#endif
}

template <typename TargetInput>
int32_t MTPWorkerImpl<TargetInput>::draft_prefix_kv_upper_bound(
    int32_t embedding_id,
    const std::string& request_id,
    int32_t raw_kv_seq_len,
    int32_t row_offset,
    int64_t kv_capacity) const {
  CHECK_GT(raw_kv_seq_len, 0);
  CHECK_GE(row_offset, 0);
  const int64_t raw_upper = static_cast<int64_t>(raw_kv_seq_len) + row_offset;
  int64_t upper = raw_upper;
  const auto found = draft_prefix_contexts_.find(embedding_id);
  if (found != draft_prefix_contexts_.end() &&
      found->second.request_id == request_id) {
    CHECK_GT(found->second.host_kv_upper_bound, 0);
    upper = std::max(upper,
                     static_cast<int64_t>(found->second.host_kv_upper_bound));
  }
  CHECK_LE(upper, kv_capacity)
      << "Verifier-conditioned draft prefix exceeds the reserved KV capacity";
  CHECK_LE(upper, std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(upper);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::retain_draft_prefix(
    const TargetInput& input,
    const SampleOutput& validate_output,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens) {
  const auto& embedding = input.input_params.embedding;
  const int64_t batch_size = embedding.embedding_ids.size();
  const int64_t width = options_.num_speculative_tokens() + 1;
  CHECK_EQ(embedding.request_ids.size(), embedding.embedding_ids.size());
  CHECK_EQ(validate_output.next_tokens.size(0), batch_size);
  CHECK_EQ(validate_output.next_tokens.size(1), width);
  CHECK_EQ(validate_output.embeddings.dim(), 3);
  CHECK_EQ(validate_output.embeddings.size(0), batch_size);
  CHECK_EQ(validate_output.embeddings.size(1), width);
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  CHECK_EQ(block_tables.dim(), 2);
  const int64_t kv_capacity = block_tables.size(1) * logical_block_size();
  // Snapshot before any following body can reuse the verifier output pool.
  // Views of these owned copies survive both Host materialization and reorder.
  const torch::Tensor tokens = validate_output.next_tokens.clone();
  const torch::Tensor hidden = validate_output.embeddings.clone();
  const torch::Tensor positions = base_positions.clone();
  const torch::Tensor kv_lengths = base_kv_seq_lens.clone();
  for (int64_t row = 0; row < batch_size; ++row) {
    DraftPrefixContext context;
    context.request_id = embedding.request_ids[row];
    context.tokens = tokens.select(/*dim=*/0, row);
    context.hidden = hidden.select(/*dim=*/0, row);
    context.base_position = positions.select(/*dim=*/0, row);
    context.base_kv_seq_len = kv_lengths.select(/*dim=*/0, row);
    // input was materialized before building this verifier. Preserve its
    // complete S-row bound rather than deriving capacity from next turn's
    // possibly unmaterialized scheduler placeholder or accepted-token count.
    const int64_t host_base_kv =
        static_cast<int64_t>(input.input_params.attention.host
                                 .kv_seq_lens[static_cast<size_t>(row)]);
    CHECK_GT(host_base_kv, 0);
    const int64_t host_upper = host_base_kv + width - 1;
    CHECK_LE(host_upper, kv_capacity)
        << "Verifier prefix exceeds the reserved KV capacity";
    CHECK_LE(host_upper, std::numeric_limits<int32_t>::max());
    context.host_kv_upper_bound = static_cast<int32_t>(host_upper);
    draft_prefix_contexts_.insert_or_assign(embedding.embedding_ids[row],
                                            std::move(context));
  }
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_draft_prefix(
    const TargetInput& input,
    const std::vector<EmbeddingCache::DecodeState>& bootstrap_states,
    LlmForwardInput& draft_input) {
  const auto& embedding = input.input_params.embedding;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t width = options_.num_speculative_tokens() + 1;
  CHECK_EQ(embedding.embedding_ids.size(), static_cast<size_t>(batch_size));
  CHECK_EQ(embedding.request_ids.size(), static_cast<size_t>(batch_size));
  CHECK_EQ(parallel_args_.dp_size(), 1);
  CHECK_EQ(parallel_args_.cp_size(), 1);
  CHECK_GE(width, 2);
  CHECK_LE(width, 17);
  CHECK(input.input_params.multi_block_tables.empty());
  {
    c10::StreamGuard guard = prepare_stream_->set_stream_guard();
    draft_input =
        make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
    prepare_draft_sampling(options_.enable_probabilistic_draft(),
                           draft_input.sampling_params);
    clear_ready_events(draft_input);
    draft_input.runtime.device_tensors_ready = false;
    auto& params = draft_input.input_params;
    params.clear_linear_attention_state();
    params.is_spec_verify = false;
    params.mtp_topk_state.reset();
    layer::ExpandedDecodeMetadataBuilder::reset_expanded_decode_metadata(
        ModelParamsView(params));
    const DecodeBatchContext batch_context = make_decode_batch_context(input);
    const int64_t kv_capacity =
        static_cast<int64_t>(batch_context.block_table_stride) *
        logical_block_size();
    DecodeBuildBuffers buffers;
    const int32_t token_count = batch_size * width;
    prepare_decode_buffers(buffers,
                           batch_context,
                           static_cast<size_t>(token_count),
                           /*emit_kv_len=*/true,
                           /*emit_block_tables=*/true);
    for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
      const int32_t raw_kv_seq_len =
          input.input_params.attention.host.kv_seq_lens[seq_id];
      for (int32_t row = 0; row < width; ++row) {
        const int32_t host_upper =
            draft_prefix_kv_upper_bound(embedding.embedding_ids[seq_id],
                                        embedding.request_ids[seq_id],
                                        raw_kv_seq_len,
                                        row,
                                        kv_capacity);
        DecodeTokenInput token_input;
        token_input.req_id = seq_id;
        token_input.position_offset = row;
        token_input.kv_len_offset = host_upper - raw_kv_seq_len;
        append_decode_token(
            batch_context, token_input, logical_block_size(), buffers);
      }
    }
    build_decode_input(draft_input, buffers, DecodeInputOptions{}, device_);
    scale_parallel_token_counts(params.parallel, width);
    draft_input.runtime.metadata_ready_event =
        prepare_stream_->record_event_or_sync();
  }

  c10::StreamGuard guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(draft_input.runtime.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
  clear_ready_events(draft_input);
  std::vector<torch::Tensor> tokens, hidden, positions, kv_lengths;
  tokens.reserve(batch_size);
  hidden.reserve(batch_size);
  positions.reserve(batch_size);
  kv_lengths.reserve(batch_size);
  const auto host_tokens = tensor_slice(input.token_ids_host);
  const auto host_positions = tensor_slice(input.positions_host);
  const auto token_options = input.token_ids.options();
  const auto& placeholder = embedding_cache_->embedding_placeholder();
  for (int32_t row = 0; row < batch_size; ++row) {
    const auto found =
        draft_prefix_contexts_.find(embedding.embedding_ids[row]);
    if (found != draft_prefix_contexts_.end() &&
        found->second.request_id == embedding.request_ids[row]) {
      const auto& context = found->second;
      tokens.emplace_back(context.tokens);
      hidden.emplace_back(context.hidden);
      positions.emplace_back(context.base_position);
      kv_lengths.emplace_back(context.base_kv_seq_len);
      continue;
    }
    CHECK_EQ(bootstrap_states.size(), static_cast<size_t>(batch_size))
        << "Verifier-conditioned draft prefix is missing verifier context";
    const auto& state = bootstrap_states[row];
    CHECK(input.input_params.meta.is_graph_warmup || host_positions[row] > 0)
        << "Draft prefix bootstrap requires the preceding target position";
    auto row_tokens = torch::full({width}, -1, token_options);
    row_tokens.select(/*dim=*/0, /*index=*/0)
        .fill_(std::max(host_tokens[row], 0));
    const torch::Tensor source_hidden =
        state.embedding.defined() ? state.embedding : placeholder;
    auto row_hidden = torch::zeros({width, source_hidden.numel()},
                                   source_hidden.options().device(device_));
    row_hidden.select(/*dim=*/0, /*index=*/0)
        .copy_(source_hidden.reshape({-1}), /*non_blocking=*/true);
    tokens.emplace_back(std::move(row_tokens));
    hidden.emplace_back(std::move(row_hidden));
    positions.emplace_back(torch::full(
        {}, std::max(host_positions[row] - 1, 0), input.positions.options()));
    kv_lengths.emplace_back(
        torch::full({},
                    std::max(input.input_params.attention.host
                                     .kv_seq_lens[static_cast<size_t>(row)] -
                                 1,
                             1),
                    input.positions.options()));
  }
  mtp_async::prepare_draft_prefix_from_target(draft_input,
                                              torch::stack(tokens),
                                              torch::stack(hidden),
                                              torch::stack(positions),
                                              torch::stack(kv_lengths),
                                              logical_block_size());
  draft_input.runtime.metadata_ready_event =
      compute_stream_->record_event_or_sync();
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::supports_combined_first_draft_execution()
    const {
#if defined(USE_NPU)
  if (draft_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }

  const std::string& npu_backend =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend();
  return device_.unwrap().is_privateuseone() &&
         mtp_async::supports_combined_draft_configuration(
             combined_draft_execution_path_,
             npu_backend,
             parallel_args_.dp_size());
#else
  return false;
#endif
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::can_use_combined_first_draft() const {
  return enable_schedule_overlap() && supports_combined_first_draft_execution();
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::can_prelaunch_next_first_draft(
    const TargetInput& input) const {
  if (uses_verifier_conditioned_draft_prefix()) {
    // A verifier-conditioned prefix is the next admitted draft-0 forward,
    // including its LM head and current sampling controls. Do not submit the
    // old two-row repair ahead of that admission or sample it twice.
    return false;
  }
  if (!can_use_combined_first_draft()) {
    return false;
  }
  // Prelaunch runs inside collectives, so every DP rank must agree. Missing
  // or malformed global JSON presence fails closed for DP > 1.
  const int32_t dp_size = parallel_args_.dp_size();
  if (dp_size == 1) {
    if (!input.json_object_states.empty() ||
        !input.json_object_state_snapshots.empty()) {
      return false;
    }
  } else {
    const auto& global_presence =
        input.input_params.parallel.dp_global_json_object_active;
    if (dp_size <= 0 ||
        global_presence.size() != static_cast<size_t>(dp_size) ||
        !std::all_of(global_presence.begin(),
                     global_presence.end(),
                     [](int32_t active) { return active == 0; })) {
      return false;
    }
  }
  return device_target_context_ready_for_batch(input);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_first_draft_template(
    TargetInput& metadata_template,
    LlmForwardInput& combined_input,
    bool wait_for_compute_stream) {
  CHECK(embedding_cache_ != nullptr);

  // Clone host tensors before mutating the shallow-copied template.
  metadata_template.token_ids_host =
      clone_host_tensor(metadata_template.token_ids_host);
  metadata_template.positions_host =
      clone_host_tensor(metadata_template.positions_host);
  const int32_t next_draft_token_advance =
      options_.num_speculative_tokens() + 1;
  int32_t* template_positions =
      metadata_template.positions_host.template data_ptr<int32_t>();
  int32_t* template_tokens =
      metadata_template.token_ids_host.template data_ptr<int32_t>();
  auto& template_kv_lens =
      metadata_template.input_params.attention.host.kv_seq_lens;
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_positions[seq_id] += next_draft_token_advance;
    template_kv_lens[seq_id] += next_draft_token_advance;
    if (template_tokens[seq_id] < 0) {
      template_tokens[seq_id] = 0;
    }
  }

  std::vector<EmbeddingCache::DecodeState> template_states(
      metadata_template.input_params.meta.num_sequences);
  const torch::Tensor& placeholder = embedding_cache_->embedding_placeholder();
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_states[seq_id].valid = true;
    template_states[seq_id].request_id =
        metadata_template.input_params.embedding.request_ids[seq_id];
    template_states[seq_id].token_id = template_tokens[seq_id];
    template_states[seq_id].embedding = placeholder;
  }

  prepare_draft_extend_inputs(metadata_template,
                              template_states,
                              combined_input,
                              /*force_two_rows=*/true,
                              wait_for_compute_stream);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::enqueue_next_first_draft(
    const TargetInput& input,
    const SampleOutput& validate_output,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    LlmForwardInput combined_input) {
  CHECK(validate_output.next_tokens.defined());
  CHECK(validate_output.embeddings.defined());
  CHECK(embedding_cache_ != nullptr);

  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(
      compute_stream_->wait_event(combined_input.runtime.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
  clear_ready_events(combined_input);

  // Interleave [repair, current] rows in one decode batch. Every transformer
  // layer projects both rows, writes both KV rows, and only then launches
  // PagedAttention. Same-stream ordering therefore makes repair KV visible to
  // the current row without a host wait or a separate repair forward.
  mtp_async::prepare_next_draft_from_accepted_state(
      combined_input,
      input,
      validate_output.next_tokens,
      validate_output.embeddings,
      embedding_cache_->embedding_placeholder(),
      base_positions,
      base_kv_seq_lens,
      /*rebuild_expanded_decode_metadata=*/false,
      logical_block_size());

  submit_pending_first_draft(input, std::move(combined_input));
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::submit_pending_first_draft(
    const TargetInput& batch_identity_input,
    LlmForwardInput draft_input) {
  CHECK(!pending_draft_context_.output.has_value())
      << "MTP first-draft prelaunch was not consumed";
  pending_draft_context_.embedding_ids =
      batch_identity_input.input_params.embedding.embedding_ids;
  pending_draft_context_.request_ids =
      batch_identity_input.input_params.embedding.request_ids;
  pending_draft_context_.dp_global_token_nums =
      batch_identity_input.input_params.parallel.dp_global_token_nums;
  pending_draft_context_.dp_global_sequence_nums =
      batch_identity_input.input_params.parallel.dp_global_sequence_nums;
  pending_draft_context_.raw_dp_global_token_nums =
      batch_identity_input.input_params.parallel.raw_dp_global_token_nums;
  pending_draft_context_.dp_global_batch_generations =
      batch_identity_input.input_params.parallel.dp_global_batch_generations;
  pending_draft_context_.output =
      run_worker_no_sync_impl(*draft_impl_,
                              draft_input,
                              *compute_stream_,
                              *compute_stream_,
                              pending_draft_context_.prepared_input);
  CHECK(pending_draft_context_.output.has_value())
      << "failed to prelaunch next MTP first draft";
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::pending_draft_context_matches(
    const TargetInput& input) const {
  return pending_draft_context_.output.has_value() &&
         pending_draft_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_draft_context_.request_ids ==
             input.input_params.embedding.request_ids &&
         pending_draft_context_.dp_global_token_nums ==
             input.input_params.parallel.dp_global_token_nums &&
         pending_draft_context_.dp_global_sequence_nums ==
             input.input_params.parallel.dp_global_sequence_nums &&
         pending_draft_context_.raw_dp_global_token_nums ==
             input.input_params.parallel.raw_dp_global_token_nums &&
         pending_draft_context_.dp_global_batch_generations ==
             input.input_params.parallel.dp_global_batch_generations;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_validate_inputs(
    const TargetInput& input,
    TargetInput& validate_input,
    bool static_graph_tasks_prepared,
    bool record_ready_event) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input.clone();
  clear_ready_events(validate_input);
  validate_input.runtime.device_tensors_ready = false;
  validate_input.skip_sampling_for_logits_only = true;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t logical_block_size = this->logical_block_size();
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      should_use_explicit_spec_verify_replay_update(input);
  const bool qwen_logical_state =
      target_verify_mode_ == mtp_async::TargetSpecVerifyMode::EXPANDED_VERIFY;
  const bool expand_python_mtp_linear_state_ids =
      num_val_tokens > 1 &&
      ModelConfig::is_python_model_impl(context_.get_model_impl()) &&
      !input_params.embedding.linear_state_ids.empty() && !qwen_logical_state;
  std::vector<int32_t> python_linear_state_ids;
  if (qwen_logical_state &&
      ModelConfig::is_python_model_impl(context_.get_model_impl())) {
    // Qwen's fused verifier consumes S tokens per logical recurrent state.
    // Attention is expanded independently, but state indices must remain B.
    python_linear_state_ids = input_params.embedding.linear_state_ids;
  } else if (expand_python_mtp_linear_state_ids) {
    CHECK_EQ(input_params.embedding.linear_state_ids.size(),
             static_cast<size_t>(num_sequences));
    python_linear_state_ids.reserve(static_cast<size_t>(total_num_val_tokens));
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      python_linear_state_ids.insert(
          python_linear_state_ids.end(),
          num_val_tokens,
          input_params.embedding.linear_state_ids[static_cast<size_t>(seq_id)]);
    }
  }
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  DecodeBatchContext batch_ctx = make_decode_batch_context(input);
  const bool chunked_verify = this->uses_chunked_verify();
  const DecodeExpansionOptions decode_options = make_verify_layout(
      options_.num_speculative_tokens(), logical_block_size, chunked_verify);
  DecodeBuildBuffers buf = expand_decode_inputs(batch_ctx, decode_options);

  if (impl_->has_request_state_cache()) {
    input_params.attention.host.kpool_query_lens =
        std::vector<int32_t>(num_sequences, num_val_tokens);
    input_params.is_spec_verify = true;
  }

  if (!use_explicit_spec_verify_replay_update) {
    set_token_position_tensors(
        validate_input, buf.out_token_ids, buf.out_positions);
  }
  build_decode_metadata(input_params, buf, decode_options);

  if (!input.json_object_states.empty()) {
    // Validation installs exact per-position JSON masks after the draft
    // tokens are known. Do not repeat the inherited one-row mask here.
    validate_input.sampling_params.filter_bitmask() = torch::Tensor();
  }
  // The validate forward is logits-only (see
  // expand_sampling_params_for_verify); the JSON bitmask installs later, from
  // fill_validate_input_from_draft_outputs.
  auto& validate_sampling_params = validate_input.sampling_params;
  torch::Tensor stable_indices;
#if defined(USE_NPU)
  // A single greedy sequence needs only the cached arange as
  // logits_indices; the generic builder's registry walk is pure
  // overhead on the final-draft -> target dependency chain.
  const bool use_stable_greedy_validate_sampling =
      use_explicit_spec_verify_replay_update &&
      validate_sampling_params.sampling_mode() == SamplingMode::GREEDY &&
      validate_sampling_params.num_sample_seqs() == 1;
  if (use_stable_greedy_validate_sampling) {
    stable_indices = mtp_validate_greedy_indices_.get(
        total_num_val_tokens,
        [&]() { return arange_indices(total_num_val_tokens, device_); });
  }
#endif
  expand_sampling_params_for_verify(validate_sampling_params,
                                    options_.num_speculative_tokens(),
                                    total_num_val_tokens,
                                    device_,
                                    stable_indices);

  scale_speculative_parallel_token_counts(input_params,
                                          options_.num_speculative_tokens());

  std::vector<int32_t> accepted_prefix_lengths;
  if (chunked_verify) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    accepted_prefix_lengths =
        num_accepted_tokens(embedding_cache_.get(),
                            static_cast<size_t>(num_sequences),
                            input.input_params.embedding.embedding_ids,
                            input.input_params.embedding.request_ids);
    // num_accepted_tokens must stay the true accepted count. The Qwen3.5 GDN
    // spec-verify kernel uses it to select which recurrent/conv checkpoint to
    // commit (checkpoint index = nat - 1); it is a logical checkpoint index,
    // not the conv_state physical history capacity. Clamping it commits a
    // stale checkpoint whenever 4+ tokens were accepted (see issue #2247).
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    if (!use_explicit_spec_verify_replay_update) {
      input_params.num_accepted_tokens = torch::tensor(
          accepted_prefix_lengths, validate_input.token_ids.options());
    }
  }

#if defined(USE_NPU)
  std::vector<int32_t> expanded_shared_full_pages;
  torch::Tensor expanded_shared_full_flat;
  int64_t shared_full_page_width = 0;
  const torch::Tensor& shared_full_source =
      input_params.attention.host.shared_full_block_table;
  if (supports_expanded_spec_verify() && shared_full_source.defined()) {
    CHECK_EQ(shared_full_source.size(0), num_sequences);
    const int64_t active_width = shared_full_source.size(1);
    shared_full_page_width = use_explicit_spec_verify_replay_update
                                 ? spec_verify_block_table_width(
                                       input_params.attention.host.block_tables)
                                 : active_width;
    expanded_shared_full_pages.assign(
        static_cast<size_t>(total_num_val_tokens * shared_full_page_width),
        input_params.attention.host.shared_full_read_padding_page);
    const int32_t* source = shared_full_source.template data_ptr<int32_t>();
    for (int64_t row = 0; row < total_num_val_tokens; ++row) {
      std::memcpy(
          expanded_shared_full_pages.data() + row * shared_full_page_width,
          source + (row / num_val_tokens) * active_width,
          static_cast<size_t>(active_width) * sizeof(int32_t));
    }
  }
  if (use_explicit_spec_verify_replay_update) {
    layer::ExpandedDecodeMetadataBuilder::update_expanded_decode_host_metadata(
        ModelParamsView(input_params));

    ensure_spec_verify_control_block_table(input_params, num_sequences);
    AttentionInputView attention(input_params.attention);
    CHECK(attention.host.block_tables.defined());
    CHECK_EQ(attention.host.block_tables.dim(), 2);
    CHECK_EQ(attention.host.block_tables.size(0), num_sequences);
    CHECK_EQ(attention.host.block_tables.scalar_type(), torch::kInt32);
    const int64_t active_block_table_width =
        attention.host.block_tables.size(1);
    const int64_t verify_block_table_width =
        spec_verify_block_table_width(attention.host.block_tables);
    if (active_block_table_width != verify_block_table_width) {
      torch::Tensor padded_block_tables = torch::zeros(
          {num_sequences, verify_block_table_width},
          attention.host.block_tables.options().device(torch::kCPU));
      padded_block_tables.narrow(1, 0, active_block_table_width)
          .copy_(attention.host.block_tables);
      attention.host.block_tables = std::move(padded_block_tables);
    }

    const bool initialize_stable_buffer =
        !spec_verify_attention_host_buffer_.defined();
    attention.attention_host_buffer = spec_verify_attention_host_buffer_;
    attention.attention_device_buffer = spec_verify_attention_device_buffer_;
    attention.attention_buffer_bytes = 0;
    attention.attention_buffer_capacity =
        spec_verify_attention_buffer_capacity_;
    attention.attention_buffer_owner = spec_verify_attention_buffer_owner_;

    const int64_t expanded_block_rows = static_cast<int64_t>(
        input_params.graph.expanded_kv_seq_lens_vec.size());
    CHECK_EQ(expanded_block_rows, total_num_val_tokens);
    torch::Tensor expanded_block_tables_flat;
    std::vector<PackedAttentionIntInput> extra_int_inputs;
    extra_int_inputs.reserve(6);
    extra_int_inputs.push_back({&buf.out_token_ids,
                                &validate_input.token_ids_host,
                                &validate_input.token_ids});
    extra_int_inputs.push_back({&buf.out_positions,
                                &validate_input.positions_host,
                                &validate_input.positions});
    if (!python_linear_state_ids.empty()) {
      extra_int_inputs.push_back(
          {&python_linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
    }
    if (!input_params.num_accepted_tokens_host.empty()) {
      extra_int_inputs.push_back({&accepted_prefix_lengths,
                                  nullptr,
                                  &input_params.num_accepted_tokens});
    }
    if (!input_params.graph.expanded_kv_seq_lens_vec.empty()) {
      extra_int_inputs.push_back({&input_params.graph.expanded_kv_seq_lens_vec,
                                  nullptr,
                                  &input_params.graph.expanded_kv_seq_lens});
    }
    extra_int_inputs.push_back({.device_view = &expanded_block_tables_flat,
                                .request_tensor = &attention.host.block_tables,
                                .num_tokens_per_request = num_val_tokens});
    if (!expanded_shared_full_pages.empty()) {
      extra_int_inputs.push_back(
          {&expanded_shared_full_pages, nullptr, &expanded_shared_full_flat});
    }
    attention.rebuild_device_buffer(
        device_,
        extra_int_inputs,
        initialize_stable_buffer ? AttentionBufferReusePolicy::GROWABLE
                                 : AttentionBufferReusePolicy::FIXED_CAPACITY);
    if (initialize_stable_buffer && num_sequences > 0) {
      // Source views are part of the explicit replay contract. Reserve enough
      // packed storage for any configured decode batch before the first graph
      // captures their addresses, so a later larger batch cannot relocate the
      // buffer and invalidate an already captured variant.
      const uint64_t max_sequences =
          static_cast<uint64_t>(options_.max_seqs_per_batch());
      const uint64_t batch_scale =
          (max_sequences + static_cast<uint64_t>(num_sequences) - 1) /
          static_cast<uint64_t>(num_sequences);
      const uint64_t reserve_capacity =
          attention.attention_buffer_bytes * batch_scale;
      if (reserve_capacity > attention.attention_buffer_capacity) {
        attention.reserve_device_buffer_capacity(reserve_capacity, device_);
        attention.rebuild_device_buffer(
            device_,
            extra_int_inputs,
            AttentionBufferReusePolicy::FIXED_CAPACITY);
      }
    }
    spec_verify_attention_host_buffer_ = attention.attention_host_buffer;
    spec_verify_attention_device_buffer_ = attention.attention_device_buffer;
    spec_verify_attention_buffer_capacity_ =
        attention.attention_buffer_capacity;
    input_params.graph.expanded_block_tables = expanded_block_tables_flat.view(
        {expanded_block_rows, verify_block_table_width});
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        ModelParamsView(input_params),
        input_params.graph.expanded_kv_seq_lens,
        input_params.graph.expanded_block_tables,
        input_params.graph.expanded_kv_seq_lens_vec,
        logical_block_size);
    input_params.graph.input_tokens_override = validate_input.token_ids;
    input_params.graph.spec_verify_source_addresses_stable = true;
  } else {
    if (supports_expanded_spec_verify()) {
      ensure_spec_verify_control_block_table(input_params, num_sequences);
    }
    std::vector<PackedAttentionIntInput> extra_int_inputs;
    if (!python_linear_state_ids.empty()) {
      extra_int_inputs.push_back(
          {&python_linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
    }
    if (!expanded_shared_full_pages.empty()) {
      extra_int_inputs.push_back(
          {&expanded_shared_full_pages, nullptr, &expanded_shared_full_flat});
    }
    AttentionInputView(input_params.attention)
        .rebuild_device_buffer(device_, extra_int_inputs);
    if (supports_expanded_spec_verify()) {
      layer::ExpandedDecodeMetadataBuilder::update_expanded_decode_metadata(
          ModelParamsView(input_params), device_, logical_block_size);
    }
  }
  if (!expanded_shared_full_pages.empty()) {
    input_params.attention.device.shared_full_block_table =
        expanded_shared_full_flat.view(
            {total_num_val_tokens, shared_full_page_width});
  }
#else
  input_params.attention.rebuild_device_buffer(device_);
#endif
  validate_input.runtime.device_tensors_ready = true;
  // This metadata is independent of the in-flight final draft. Keep it on the
  // auxiliary stream and hand it to the compute stream with a device event.
#if defined(USE_NPU)
  input_params.graph.spec_verify_static_graph_tasks_prepared =
      use_explicit_spec_verify_replay_update && static_graph_tasks_prepared;
#endif
  if (record_ready_event) {
    validate_input.runtime.metadata_ready_event =
        prepare_stream_->record_event_or_sync();
  }
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::
    prepare_static_mtp_graph_tasks_before_final_draft(
        const TargetInput& input) {
#if defined(USE_NPU)
  if (!should_use_explicit_spec_verify_replay_update(input) ||
      input.input_params.embedding.linear_state_ids.size() != 1 ||
      embedding_cache_ == nullptr ||
      input.input_params.embedding.embedding_ids.empty()) {
    return false;
  }
  const auto& block_tables = input.input_params.attention.host.block_tables;
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      block_tables.size(0) != 1) {
    return false;
  }
  const std::vector<int32_t> accepted_prefix_lengths =
      embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  if (accepted_prefix_lengths.size() != 1) {
    return false;
  }
  const int64_t verify_block_table_width =
      spec_verify_block_table_width(block_tables);
  const auto& kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  if (kv_seq_lens.size() != 1) {
    return false;
  }
  const int64_t spec_width = options_.num_speculative_tokens() + 1;
  const int64_t base_kv_seq_len = kv_seq_lens.front();
  const int64_t spec_verify_max_kv_seq_len = base_kv_seq_len + spec_width - 1;
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = input.input_params.embedding.linear_state_ids.front(),
      .num_accepted_tokens = accepted_prefix_lengths.front(),
      .spec_width = spec_width,
      .block_table_width = verify_block_table_width,
      .base_kv_seq_len = base_kv_seq_len,
      .max_kv_seq_len = spec_verify_max_kv_seq_len,
  };
  auto* llm_target = dynamic_cast<LLMWorkerImpl*>(impl_.get());
  if (llm_target == nullptr) {
    return false;
  }
  return llm_target->prepare_static_mtp_graph_tasks(signal, *compute_stream_);
#else
  (void)input;
  return false;
#endif
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_draft_extend_inputs(
    const TargetInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    LlmForwardInput& extend_input,
    bool force_two_rows,
    bool wait_for_compute_stream) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  // Regular draft preparation may consume tensors produced by the previous
  // compute. The placeholder-only first-draft prelaunch has no such dependency.
  if (wait_for_compute_stream) {
    prepare_stream_->wait_stream(*compute_stream_);
  }
  extend_input = make_llm_draft_input(
      base_input, draft_impl_->uses_shared_static_kv_cache());
  prepare_draft_sampling(options_.enable_probabilistic_draft(),
                         extend_input.sampling_params);
  clear_ready_events(extend_input);
  extend_input.runtime.device_tensors_ready = false;
  if (requires_full_target_replay()) {
    const int32_t uniform_width =
        draft_context_replay_policy_->num_speculative_tokens + 1;
    auto& input_params = extend_input.input_params;
    DecodeBatchContext batch_ctx = make_decode_batch_context(base_input);
    const torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
    const int32_t block_size = logical_block_size();
    // Replays the emitted target prefix at the positions of the target hidden
    // states, one position before the corresponding emitted tokens.
    const bool dp_enabled = parallel_args_.dp_size() > 1;
    auto& mask = input_params.expert.eplb_decode_token_mask;
    // Non-DP token-mask remap gathers each replay row back to its source
    // sequence and zeroes padding rows; skip the bookkeeping otherwise.
    const bool remap_token_mask = !dp_enabled && mask.defined();
    // MLU recurrent kernels skip writing KV when the slot is negative.
    const int32_t pad_slot_id = Platform::is_mlu() ? kPadSlotId : 0;
    CHECK_EQ(last_states.size(), static_cast<size_t>(batch_ctx.num_requests));
    DecodeBuildBuffers buffers;
    std::vector<torch::Tensor> embeddings;
    std::vector<int32_t> selected_rows;
    std::vector<int32_t> source_sequences;
    std::vector<int32_t> valid_rows;
    std::vector<int32_t> rotary;
    const int32_t capacity = batch_ctx.num_requests * uniform_width;
    prepare_decode_buffers(buffers,
                           batch_ctx,
                           static_cast<size_t>(capacity),
                           /*emit_kv_len=*/true,
                           /*emit_block_tables=*/true);
    embeddings.reserve(capacity);
    selected_rows.reserve(batch_ctx.num_requests);
    if (batch_ctx.num_rope_axes == 3) {
      rotary.resize(3 * capacity);
    }
    if (remap_token_mask) {
      source_sequences.reserve(capacity);
      valid_rows.reserve(capacity);
    }
    for (int32_t seq_id = 0; seq_id < batch_ctx.num_requests; ++seq_id) {
      const auto& state = last_states[seq_id];
      const int32_t span = static_cast<int32_t>(state.replay_token_ids.size());
      CHECK(state.valid || input_params.meta.is_graph_warmup)
          << "MTP replay requires target context";
      if (state.valid) {
        CHECK_GT(span, 0);
        CHECK_EQ(state.replay_token_ids.back(), batch_ctx.token_ids[seq_id]);
        CHECK(state.replay_embeddings.defined());
        CHECK_EQ(state.replay_embeddings.dim(), 2);
        CHECK_EQ(state.replay_embeddings.size(0), span);
        CHECK_GE(batch_ctx.positions[seq_id], span);
      }
      CHECK_GE(uniform_width, span);
      const int32_t padding = uniform_width - span;
      const auto stored_mrope =
          batch_ctx.num_rope_axes == 3 && state.replay_mrope_positions.defined()
              ? std::make_optional(
                    state.replay_mrope_positions.accessor<int32_t, 2>())
              : std::nullopt;
      for (int32_t row_id = 0; row_id < uniform_width; ++row_id) {
        const int32_t replay_id = row_id - padding;
        const bool valid = replay_id >= 0;
        DecodeTokenInput token_input;
        token_input.req_id = seq_id;
        token_input.token_id = valid ? state.replay_token_ids[replay_id] : 0;
        token_input.position_offset =
            valid ? replay_id - span : -batch_ctx.positions[seq_id];
        if (!valid) {
          token_input.cache_slot_override = pad_slot_id;
        }
        append_decode_token(batch_ctx, token_input, block_size, buffers);
        embeddings.emplace_back(
            valid ? state.replay_embeddings.select(0, replay_id) : placeholder);
        if (batch_ctx.num_rope_axes == 3 && valid) {
          const int32_t row_index = seq_id * uniform_width + row_id;
          if (stored_mrope) {
            const auto& stored = *stored_mrope;
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis * capacity + row_index] = stored[axis][replay_id];
            }
          } else {
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis * capacity + row_index] =
                  batch_ctx.positions[axis * batch_ctx.num_requests + seq_id] +
                  token_input.position_offset;
            }
          }
        }
        if (remap_token_mask) {
          source_sequences.emplace_back(seq_id);
          valid_rows.emplace_back(valid ? 1 : 0);
        }
      }
      selected_rows.emplace_back(static_cast<int32_t>(embeddings.size()) - 1);
    }
    const int32_t num_tokens =
        static_cast<int32_t>(buffers.out_positions.size());
    const DecodeInputOptions input_options;
    if (rotary.empty()) {
      build_decode_input(extend_input, buffers, input_options, device_);
    } else {
      set_token_position_tensors(extend_input, buffers.out_token_ids, rotary);
      extend_input.positions_host =
          extend_input.positions_host.view({3, num_tokens});
      extend_input.positions = extend_input.positions.view({3, num_tokens});
      build_decode_metadata(extend_input.input_params, buffers, input_options);
      extend_input.input_params.attention.rebuild_device_buffer(device_);
    }
    if (!draft_impl_->has_request_state_cache()) {
      input_params.clear_linear_attention_state();
      input_params.num_accepted_tokens = torch::Tensor();
      input_params.num_accepted_tokens_host.clear();
    }
    input_params.is_spec_verify = false;
    if (draft_impl_->has_request_state_cache()) {
      input_params.attention.host.kpool_query_lens.assign(
          batch_ctx.num_requests, uniform_width);
    }
    input_params.attn_metadata.reset();
    for (auto& embedding : embeddings) {
      if (embedding.device() != device_) {
        embedding = embedding.to(device_);
      }
    }
    input_params.embedding.input_embedding = torch::stack(embeddings);

    if (!input_params.parallel.dp_global_token_nums.empty()) {
      if (dp_enabled) {
        scale_parallel_token_counts(input_params.parallel, uniform_width);
      } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
        input_params.parallel.dp_global_token_nums[0] = num_tokens;
      }
    }
    if (dp_enabled) {
      mask = eplb::expand_decode_token_mask(mask, uniform_width);
    } else if (remap_token_mask) {
      auto source_rows =
          torch::tensor(source_sequences, torch::dtype(torch::kLong))
              .to(mask.device());
      auto valid_rows_tensor =
          torch::tensor(valid_rows, torch::dtype(torch::kInt))
              .to(mask.options());
      mask =
          mask.reshape({-1}).index_select(0, source_rows) * valid_rows_tensor;
    }
    auto& params = extend_input.sampling_params;
    torch::TensorOptions idx_options =
        params.logits_indices().defined()
            ? params.logits_indices().options()
            : torch::dtype(torch::kInt).device(device_);
    params.logits_indices() = async_h2d_tensor(selected_rows, idx_options);
    extend_input.runtime.device_tensors_ready = true;
    check_draft_input_embedding(input_params.embedding.input_embedding,
                                "target replay");
    extend_input.runtime.metadata_ready_event =
        prepare_stream_->record_event_or_sync();
    return;
  }
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";
  const std::string_view target_model_type =
      impl_->context_.get_model_args().model_type();
  const bool use_uniform_two_rows =
      force_two_rows || dp_enabled ||
      (Platform::is_npu() &&
       // Recurrent targets require a uniform verification width.
       (is_qwen3_5_target_model_type(target_model_type) ||
        is_glm5_next_target_model_type(target_model_type)) &&
       std::any_of(last_states.begin(),
                   last_states.end(),
                   [](const EmbeddingCache::DecodeState& state) {
                     return state.all_draft_accepted;
                   }));

  const int32_t logical_block_size = this->logical_block_size();
  DecodeBatchContext batch_ctx = make_decode_batch_context(base_input);
  DecodeBuildBuffers buf;
  prepare_decode_buffers(buf,
                         batch_ctx,
                         static_cast<size_t>(num_sequences) * 2,
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/true);
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);

  auto to_worker_device = [this](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device() == device_) {
      return tensor;
    }
    return tensor.to(device_);
  };

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = to_worker_device(placeholder);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding,
                       std::optional<int32_t> cache_slot_override =
                           std::nullopt) {
      DecodeTokenInput token_input;
      token_input.req_id = seq_id;
      token_input.token_id = token_id >= 0 ? token_id : 0;
      token_input.position_offset = position_offset;
      token_input.cache_slot_override = cache_slot_override;
      append_decode_token(batch_ctx, token_input, logical_block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(to_worker_device(embedding));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = batch_ctx.token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    const bool sequence_uses_two_rows =
        use_uniform_two_rows || state.all_draft_accepted;
    if (sequence_uses_two_rows) {
      int32_t prev_token_id = state.prev_token_id;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        prev_token_id = state.token_id;
        prev_embedding = state.embedding;
      }
      add_row(prev_token_id,
              /*position_offset=*/-1,
              prev_embedding,
              /*cache_slot_override=*/
              prev_is_placeholder ? std::make_optional(0) : std::nullopt);
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);
  }

  if (draft_impl_->has_request_state_cache()) {
    auto& spans = input_params.attention.host.kpool_query_lens;
    spans.clear();
    spans.reserve(num_sequences);
    int32_t begin = 0;
    for (const int32_t selected : selected_row_idx) {
      spans.emplace_back(selected + 1 - begin);
      begin = selected + 1;
    }
  }

  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  const int32_t num_query_tokens =
      static_cast<int32_t>(buf.out_positions.size());
  build_decode_input(extend_input, buf, DecodeInputOptions{}, device_);

  input_params.embedding.input_embedding = torch::stack(expanded_embeddings);
  check_draft_input_embedding(input_params.embedding.input_embedding,
                              "decode extend");

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (dp_enabled) {
      constexpr int32_t num_query_tokens_per_request = 2;
      scale_parallel_token_counts(input_params.parallel,
                                  num_query_tokens_per_request);
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] = num_query_tokens;
    }
  }
  input_params.expert.eplb_decode_token_mask =
      eplb::expand_decode_token_mask(input_params.expert.eplb_decode_token_mask,
                                     num_query_tokens / num_sequences);

#if defined(USE_NPU)
  // The extend layout is the 2B cache variant during steady overlap decode.
  this->prepare_draft_padding(input_params, prepare_stream_.get());
#endif

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.logits_indices().defined()
          ? params.logits_indices().options()
          : torch::dtype(torch::kInt).device(device_);
  if (use_uniform_two_rows) {
    params.logits_indices() = logits_indices_.get(num_sequences, [&]() {
      return torch::arange(/*start=*/1,
                           static_cast<int64_t>(num_sequences) * 2,
                           /*step=*/2,
                           idx_options);
    });
  } else {
    params.logits_indices() = async_h2d_tensor(selected_row_idx, idx_options);
  }
  extend_input.runtime.device_tensors_ready = true;
  extend_input.runtime.metadata_ready_event =
      prepare_stream_->record_event_or_sync();
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_draft_inputs(
    const TargetInput& input,
    LlmForwardInput& draft_input,
    int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input =
      make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
  if (requires_full_target_replay()) {
    // The first replay proposal consumed the target hidden at
    // current_position - 1; recursive proposals advance from that draft
    // position, while target validation keeps the unshifted template.
    position_offset += draft_context_replay_policy_->draft_position_offset;
    draft_input.input_params.attn_metadata.reset();
  }
  prepare_draft_sampling(options_.enable_probabilistic_draft(),
                         draft_input.sampling_params);
  clear_ready_events(draft_input);
  draft_input.runtime.device_tensors_ready = false;

  auto& input_params = draft_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  if (!draft_impl_->has_request_state_cache()) {
    input_params.clear_linear_attention_state();
    input_params.num_accepted_tokens = torch::Tensor();
    input_params.num_accepted_tokens_host.clear();
  }
  const int32_t num_sequences = input_params.meta.num_sequences;
  if (draft_impl_->has_request_state_cache()) {
    input_params.attention.host.kpool_query_lens.assign(num_sequences, 1);
    input_params.is_spec_verify = false;
    // The first draft already restored its request slot. Subsequent drafts
    // append to that live state instead of replaying a prefix restore.
    input_params.embedding.linear_state_read_ids =
        input_params.embedding.linear_state_ids;
    input_params.embedding.linear_state_read_indices =
        input_params.embedding.linear_state_indices;
  }
  const int32_t logical_block_size = this->logical_block_size();
  DecodeBatchContext batch_ctx = make_decode_batch_context(input);
  DecodeBuildBuffers buf;
  prepare_decode_buffers(buf,
                         batch_ctx,
                         static_cast<size_t>(num_sequences),
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/false);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    DecodeTokenInput token_input;
    token_input.req_id = seq_id;
    token_input.position_offset = position_offset;
    token_input.append_token = false;
    append_decode_token(batch_ctx, token_input, logical_block_size, buf);
  }

  draft_input.positions_host = make_pinned_cpu_tensor(buf.out_positions);
  draft_input.positions = safe_to(draft_input.positions_host,
                                  draft_input.positions.options(),
                                  /*non_blocking=*/true);
  update_kv_lengths_and_slots(input_params, buf);
  if (supports_expanded_spec_verify()) {
    input_params.attention.host.q_cu_seq_lens =
        util::cu_num_tokens(input_params.attention.host.q_seq_lens);
  }
  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  this->prepare_draft_padding(input_params, prepare_stream_.get());
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.runtime.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result. Prepare
  // them concurrently; token ids and embeddings are filled on compute_stream.
  draft_input.runtime.metadata_ready_event =
      prepare_stream_->record_event_or_sync();
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    SamplingParameters sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens) {
  CHECK(!draft_outputs.empty())
      << "MTP validate requires at least one draft step";
  // An all-greedy batch under probabilistic mode carries no draft logits.
  // The draft logits carry the DRAFT model's own vocabulary; size the buffer
  // from their last dim, never from the target logits.
  const torch::Tensor& first_step_logits =
      draft_outputs.front().sample_output.draft_logits;
  const torch::Tensor& first_tokens =
      draft_outputs.front().sample_output.next_tokens;
  const bool is_probabilistic_draft = first_step_logits.defined();
  DraftProposal proposal = DraftProposal::preallocate(
      first_tokens.size(0),
      static_cast<int64_t>(num_speculative_tokens),
      is_probabilistic_draft ? first_step_logits.size(-1) : 0,
      /*is_probabilistic_draft=*/is_probabilistic_draft,
      first_tokens.options(),
      is_probabilistic_draft ? first_step_logits.options()
                             : first_tokens.options());
  int64_t step = 0;
  for (const auto& draft_output : draft_outputs) {
    proposal.tokens_at(step).copy_(draft_output.sample_output.next_tokens);
    if (is_probabilistic_draft) {
      proposal.logits_at(step).copy_(draft_output.sample_output.draft_logits);
    }
    ++step;
  }

  return validate(std::move(sampling_params),
                  proposal,
                  target_output,
                  num_speculative_tokens);
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    SamplingParameters sampling_params,
    const DraftProposal& draft_proposal,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens) {
  return spec_verify::run_rejection_sampling(
      std::move(sampling_params),
      draft_proposal,
      target_output.logits,
      target_output.sample_output.embeddings);
}

template class MTPWorkerImpl<LlmForwardInput>;
template class MTPWorkerImpl<VlmForwardInput>;

}  // namespace xllm

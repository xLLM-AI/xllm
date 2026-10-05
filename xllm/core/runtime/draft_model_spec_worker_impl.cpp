/* Copyright 2026 The xLLM Authors.

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

#include "runtime/draft_model_spec_worker_impl.h"

#include <absl/strings/ascii.h>
#include <absl/strings/escaping.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <tuple>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"
#include "core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/embedding_cache.h"
#include "runtime/llm_worker_impl.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

template <typename MakeShape, typename Allocate>
bool allocate_pool_if_loaded(WorkerImpl& worker,
                             MakeShape&& make_shape,
                             Allocate&& allocate) {
  const WorkerImpl::Status status = worker.get_status();
  if (status == WorkerImpl::Status::LOADED) {
    return allocate(worker, make_shape());
  }
  CHECK_EQ(status, WorkerImpl::Status::READY);
  return true;
}

int64_t get_dp_local_tp_size(const ParallelArgs& parallel_args) {
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  return std::max<int64_t>(parallel_args.world_size() / dp_size / cp_size, 1);
}

std::string stable_path_digest(const std::string& path_string) {
  const std::filesystem::path path(path_string);
  std::error_code error;
  const std::filesystem::path canonical_path =
      std::filesystem::weakly_canonical(path, error);
  const std::string normalized_path =
      (error ? path.lexically_normal() : canonical_path).generic_string();
  const XXH3Key path_hash = hash_string(normalized_path);

  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(path_hash.data), sizeof(path_hash.data)));
}

std::string draft_store_key_component(const runtime::Options& options) {
  const std::string algorithm =
      absl::AsciiStrToLower(options.speculative_algorithm());

  const std::string draft_model_path = options.draft_model_path().value_or("");
  if (draft_model_path.empty()) {
    return "spec_draft::" + algorithm + "::embedded";
  }

  std::string draft_model_name = std::filesystem::path(draft_model_path)
                                     .lexically_normal()
                                     .filename()
                                     .generic_string();
  if (draft_model_name.empty()) {
    draft_model_name = "checkpoint";
  }
  return "spec_draft::" + algorithm + "::" + draft_model_name +
         "::" + stable_path_digest(draft_model_path);
}

KVCacheEstimateOptions make_kv_cache_estimate_options(
    const ModelArgs& model_args,
    const runtime::Options& options,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    int64_t cache_size_in_bytes) {
  const int64_t dp_local_tp_size = get_dp_local_tp_size(parallel_args);
  const int64_t n_heads = model_args.n_heads();
  const int64_t n_kv_heads = model_args.n_kv_heads().value_or(n_heads);

  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes = cache_size_in_bytes;
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = dp_local_tp_size;
  estimate_options.n_local_kv_heads =
      std::max<int64_t>(n_kv_heads / dp_local_tp_size, 1);
  if (has_linear_attention_layers(model_args)) {
    estimate_options.n_local_linear_k_heads = std::max<int64_t>(
        model_args.linear_num_key_heads() / dp_local_tp_size, 1);
    estimate_options.n_local_linear_v_heads = std::max<int64_t>(
        model_args.linear_num_value_heads() / dp_local_tp_size, 1);
  }
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.num_speculative_tokens =
      static_cast<int64_t>(options.num_speculative_tokens());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();
  estimate_options.enable_schedule_overlap = options.enable_schedule_overlap();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  estimate_options.enable_prefix_cache =
      kv_cache_config.enable_prefix_cache() &&
      !kv_cache_config.enable_xtensor();
  estimate_options.enable_disagg_pd = options.enable_disagg_pd();
  estimate_options.instance_role = options.instance_role();
  estimate_options.dp_size = options.dp_size();
  estimate_options.enable_dp_fair_token_budget =
      SchedulerConfig::get_instance().enable_dp_fair_token_budget();
  return estimate_options;
}

}  // namespace

template <typename TargetInput>
DraftModelSpecWorkerImpl<TargetInput>::DraftModelSpecWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type,
    const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory)
    : SpeculativeWorkerImpl<TargetInput>(parallel_args,
                                         device,
                                         options,
                                         target_options,
                                         worker_type),
      draft_sampling_mode_(
          parse_draft_sampling_mode(options.draft_sampling_mode())) {
  CHECK(draft_factory) << "Draft worker factory must not be empty.";
  draft_impl_ = draft_factory();
  CHECK(draft_impl_ != nullptr) << "Draft worker factory must return a worker.";
}

template <typename TargetInput>
DraftModelSpecWorkerImpl<TargetInput>::~DraftModelSpecWorkerImpl() {
  // Stop asynchronous transfers before draft_impl_ releases its KV cache.
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    hierarchy_kv_cache_transfer_->shutdown();
  }
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  bool result;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl<TargetInput>::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
  }
  return result;
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::task_models_loaded() const {
  return impl_ != nullptr && draft_impl_ != nullptr &&
         impl_->get_status() == WorkerImpl::Status::LOADED &&
         draft_impl_->get_status() == WorkerImpl::Status::LOADED;
}

template <typename TargetInput>
KVCacheShape DraftModelSpecWorkerImpl<TargetInput>::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  return build_draft_kv_cache_shape(target_kv_cache_shape);
}

template <typename TargetInput>
KVCacheShape DraftModelSpecWorkerImpl<TargetInput>::build_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape,
    int64_t draft_world_size) const {
  if (draft_world_size <= 0 &&
      !target_kv_cache_shape.has_grouped_cache_layout()) {
    draft_world_size =
        get_dp_local_tp_size(draft_impl_->context_.get_parallel_args());
  }
  KVCacheCapacity draft_capacity;
  draft_capacity.enable_indexer_cache_quant(
      KVCacheConfig::get_instance().indexer_cache_dtype() == "int8");
  return build_speculative_draft_kv_cache_shape(
      target_kv_cache_shape,
      draft_capacity,
      draft_impl_->context_.get_model_args(),
      options_.block_size(),
      draft_world_size,
      options_.kv_cache_dtype());
}

template <typename TargetInput>
std::tuple<int64_t, int64_t>
DraftModelSpecWorkerImpl<TargetInput>::estimate_kv_cache_capacity_with_draft(
    const runtime::Options& target_options,
    const runtime::Options& draft_options) {
  const auto [target_cache_bytes, target_total_bytes] =
      impl_->estimate_kv_cache_capacity();
  const auto [draft_cache_bytes, draft_total_bytes] =
      draft_impl_->estimate_kv_cache_capacity();
  const int64_t cache_size_in_bytes =
      std::min(target_cache_bytes, draft_cache_bytes);
  const int64_t total_memory = std::min(target_total_bytes, draft_total_bytes);

  const ModelArgs& target_model_args = impl_->context_.get_model_args();
  if (!util::is_deepseek_v4_model_type(target_model_args.model_type())) {
    return {cache_size_in_bytes, total_memory};
  }

  const ModelArgs& draft_model_args = draft_impl_->context_.get_model_args();
  KVCacheEstimateOptions target_estimate_options =
      make_kv_cache_estimate_options(target_model_args,
                                     target_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  const KVCacheEstimateOptions draft_estimate_options =
      make_kv_cache_estimate_options(draft_model_args,
                                     draft_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  target_estimate_options.draft_model_args = &draft_model_args;
  target_estimate_options.draft_options = &draft_estimate_options;

  const KVCacheCapacity capacity = ::xllm::estimate_kv_cache_capacity(
      target_model_args, target_estimate_options);
  return {capacity.cache_size_in_bytes(), total_memory};
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<
    TargetInput>::prepare_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(impl_ != nullptr);
  const auto target_transfer = impl_->get_hierarchy_kv_cache_transfer();
  const auto draft_transfer = draft_impl_->get_hierarchy_kv_cache_transfer();
  auto unified_transfer = hierarchy_kv_cache_transfer_
                              ? hierarchy_kv_cache_transfer_
                              : target_transfer;
  if (unified_transfer == nullptr) {
    unified_transfer = draft_transfer
                           ? draft_transfer
                           : impl_->create_hierarchy_kv_cache_transfer();
  }

  if (target_transfer == nullptr) {
    impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::TARGET,
        compute_stream_.get(),
        /*store_key_component=*/"main");
  } else {
    CHECK_EQ(target_transfer.get(), unified_transfer.get())
        << "Speculative target worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (draft_transfer == nullptr) {
    draft_impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::DRAFT,
        compute_stream_.get(),
        draft_store_key_component(options_));
  } else {
    CHECK_EQ(draft_transfer.get(), unified_transfer.get())
        << "Speculative draft worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (hierarchy_kv_cache_transfer_ == nullptr) {
    set_hierarchy_kv_cache_transfer(std::move(unified_transfer));
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<
    TargetInput>::finalize_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(hierarchy_kv_cache_transfer_ != nullptr)
      << "Speculative hierarchy KV cache transfer is not prepared.";
  if (!hierarchy_kv_cache_transfer_->registration_finalized()) {
    CHECK(hierarchy_kv_cache_transfer_->finalize_registration());
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::init_embedding_cache(
    int64_t num_blocks) {
  embedding_cache_ = std::make_unique<EmbeddingCache>(num_blocks);
  const int64_t placeholder_size = get_embedding_placeholder_size();
  if (placeholder_size > 0) {
    embedding_cache_->set_placeholder(
        torch::zeros({placeholder_size}, torch::dtype(dtype_).device(device_)));
  }
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_pools(
    const KVCacheShape& kv_cache_shape,
    const AllocateFn& allocate) {
  const bool target_allocated = allocate_pool_if_loaded(
      *impl_,
      [&]() -> const KVCacheShape& { return kv_cache_shape; },
      allocate);
  const bool draft_allocated = allocate_pool_if_loaded(
      *draft_impl_,
      [&] { return draft_kv_cache_shape(kv_cache_shape); },
      allocate);

  init_embedding_cache(kv_cache_shape.key_cache_shape()[0]);
  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();
  return allocate_pools(kv_cache_shape,
                        [](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache(shape);
                        });
}

#if defined(USE_NPU) || defined(USE_MLU)
template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();
  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
    kv_cache_transfer_->initialize(device_.index());
  }
  return allocate_pools(kv_cache_shape,
                        [this](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache_with_transfer(
                              kv_cache_transfer_, shape);
                        });
}
#endif

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::force_greedy_draft_sampling(
    SamplingParameters& sampling_params) {
  if (sampling_params.do_sample.defined()) {
    sampling_params.do_sample = torch::zeros_like(sampling_params.do_sample);
  }
  sampling_params.all_random_sample = false;
  sampling_params.all_greedy_sample = true;
  sampling_params.logprobs = false;
  sampling_params.max_top_logprobs = 0;
  sampling_params.return_probs = false;
}

template <typename TargetInput>
TargetInput
DraftModelSpecWorkerImpl<TargetInput>::update_input_by_last_step_output(
    TargetInput& inputs) {
  return inputs.clone();
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::prepare_draft_input(
    const TargetInput& input) {
  const auto& embedding = input.input_params.embedding;
  if (!embedding.mtp_bootstrap_embeddings.defined()) {
    return;
  }
  CHECK(input.token_ids_host.defined() &&
        input.token_ids_host.scalar_type() == torch::kInt)
      << "draft token_ids_host must be int32";
  const Slice<int32_t> token_ids = tensor_slice(input.token_ids_host);

  torch::Tensor bootstrap_embeddings = safe_to(
      embedding.mtp_bootstrap_embeddings, torch::dtype(dtype_).device(device_));
  const int32_t num_rows =
      static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
  CHECK_EQ(bootstrap_embeddings.size(0), static_cast<int64_t>(num_rows))
      << "bootstrap row count mismatch";
  CHECK_EQ(embedding.embedding_ids.size(), embedding.request_ids.size())
      << "bootstrap embedding/request id vectors must be parallel";
  const int32_t num_ids = static_cast<int32_t>(embedding.embedding_ids.size());
  for (int32_t i = 0; i < num_rows; ++i) {
    const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
    CHECK_GE(row_idx, 0) << "bootstrap row index should be valid";
    CHECK_LT(row_idx, num_ids) << "bootstrap row index exceeds embedding ids";
    CHECK_LT(static_cast<size_t>(row_idx), token_ids.size())
        << "bootstrap row index exceeds token ids";
    embedding_cache_->write_mtp_bootstrap_context(
        embedding.embedding_ids[row_idx],
        embedding.request_ids[row_idx],
        token_ids[row_idx],
        bootstrap_embeddings.select(/*dim=*/0, i));
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::
    sync_dp_global_token_nums_after_prune(TargetModelParams& input_params,
                                          int32_t local_total_val_tokens) {
  // Static speculative width already keeps the DP counts consistent.
  if (adaptive_spec_controller_ == nullptr ||
      !adaptive_spec_controller_->enabled()) {
    return;
  }
  ProcessGroup* dp_group = parallel_args_.dp_local_process_group_;
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return;
  }
  const int32_t dp_size = static_cast<int32_t>(dp_group->world_size());
  // Every DP rank must contribute its actual validation width after pruning.
  torch::Tensor local = torch::tensor(
      {local_total_val_tokens},
      torch::TensorOptions().dtype(torch::kInt32).device(device_.unwrap()));
  torch::Tensor gathered = dp_group->allgather_base_sync(local);

  std::vector<int32_t>& token_nums = input_params.parallel.dp_global_token_nums;
  std::vector<int32_t>& raw_token_nums =
      input_params.parallel.raw_dp_global_token_nums;
  CHECK_EQ(static_cast<int32_t>(token_nums.size()), dp_size)
      << "dp_global_token_nums size must match DP group world size";
  token_nums = tensor_to_vector<int32_t>(gathered.view({dp_size}));
  if (!raw_token_nums.empty()) {
    CHECK_EQ(static_cast<int32_t>(raw_token_nums.size()), dp_size)
        << "raw_dp_global_token_nums size must match DP group world size";
    raw_token_nums = token_nums;
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::
    sync_dp_global_token_nums_for_idle_rank(TargetModelParams& input_params) {
  if (adaptive_spec_controller_ == nullptr ||
      !adaptive_spec_controller_->enabled()) {
    return;
  }
  ProcessGroup* dp_group = parallel_args_.dp_local_process_group_;
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return;
  }
  // Idle ranks contribute the dummy validation width prepared by the caller.
  const int32_t dp_rank = static_cast<int32_t>(dp_group->rank());
  const std::vector<int32_t>& token_nums =
      input_params.parallel.dp_global_token_nums;
  CHECK_LT(dp_rank, static_cast<int32_t>(token_nums.size()))
      << "DP rank out of range for dp_global_token_nums";
  sync_dp_global_token_nums_after_prune(
      input_params, token_nums[static_cast<size_t>(dp_rank)]);
}

template class DraftModelSpecWorkerImpl<LlmForwardInput>;
template class DraftModelSpecWorkerImpl<VlmForwardInput>;

}  // namespace xllm

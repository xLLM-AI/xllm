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
#include <unordered_map>
#include <utility>

#include "core/common/types.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_type.h"
#include "core/framework/multimodal/content_hash.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_config.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"
#include "core/kv_cache/transfer/hierarchy_kv_cache_transfer.h"
#include "core/kv_cache/transfer/mooncake_kv_cache_transfer.h"
#include "core/platform/platform.h"
#include "core/util/tensor_helper.h"
#include "framework/model_loader/model_loader.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/worker_memory_resources.h"
#include "util/hash_util.h"
#include "util/utils.h"

namespace xllm {

ParallelArgs configure_draft_parallel_args(ParallelArgs parallel_args) {
  parallel_args.layerwise_split_size(1);
  auto& mapping = parallel_args.mapping_data();
  if (mapping.contains("attnLayerwiseSplit")) {
    auto& split = mapping["attnLayerwiseSplit"];
    split["group_size"] = 1;
    split["rank"] = 0;
    split["rankIds"] =
        std::vector<uint32_t>{static_cast<uint32_t>(parallel_args.rank())};
  }
  return parallel_args;
}

template <typename TargetInput>
ModelContext DraftModelSpecWorkerImpl<TargetInput>::load_model_config(
    ModelLoader& loader) {
  if (!draft_impl_->options().model_path().empty() &&
      draft_impl_->options().model_path() != loader.model_weights_path()) {
    loader.set_reference_model_weights_path(
        draft_impl_->options().model_path());
  }

  ModelArgs args = loader.model_args();
  ModelRegistry::configure_mtp_args(
      args,
      draft_impl_->options().speculative_algorithm(),
      true,
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl()));

#if !defined(USE_NPU)
  if (SpeculativeConfig::is_block_diffusion_algorithm(
          draft_impl_->options().speculative_algorithm())) {
    LOG(FATAL) << draft_impl_->options().speculative_algorithm()
               << " block-diffusion draft is not supported on "
               << Platform::type_str() << ".";
  }
  if (args.num_nextn_predict_layers() != 0) {
    static const std::unordered_map<std::string, std::string>
        kModelTypeToMtpType = {
            {"deepseek_v3", "deepseek_v3_mtp"},
            {"deepseek_v32", "deepseek_v32_mtp"},
            {"deepseek_v4", "deepseek_v4_mtp"},
            {"glm_moe_dsa", "glm_moe_dsa_mtp"},
            {"joyai_llm_flash", "joyai_llm_flash_mtp"},
            {"mimo", "mimo_mtp"},
            {"qwen3_5", "qwen3_5_mtp"},
            {"qwen3_5_moe", "qwen3_5_moe_mtp"},
            {"qwen3_5_text", "qwen3_5_mtp"},
            {"qwen3_5_moe_text", "qwen3_5_moe_mtp"},
        };
    const std::string& current_type = args.model_type();
    auto it = kModelTypeToMtpType.find(current_type);
    if (it != kModelTypeToMtpType.end()) {
      LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
                << it->second << " for speculative decoding";
      args.model_type(it->second);
    }
  }
#endif

  int32_t num_speculative_tokens = 0;
  if (configure_draft_config_) {
    num_speculative_tokens =
        configure_draft_config_(args, draft_impl_->options());
  }
  return draft_impl_->load_model_config(
      std::move(args), loader.quant_args(), num_speculative_tokens, nullptr);
}

template <typename TargetInput>
torch::Tensor DraftModelSpecWorkerImpl<TargetInput>::load_draft_vocab_mapping(
    ModelLoader& loader) {
  torch::Tensor draft_token_id_offsets;
  if (SpeculativeConfig::is_eagle3_algorithm(
          draft_impl_->options().speculative_algorithm()) ||
      SpeculativeConfig::is_dspark_algorithm(
          draft_impl_->options().speculative_algorithm())) {
    for (const auto& state_dict : loader.get_state_dicts()) {
      auto offsets = state_dict->get_tensor("d2t");
      if (offsets.defined()) {
        draft_token_id_offsets = std::move(offsets);
        break;
      }
    }
  }
  return draft_token_id_offsets;
}

template <typename TargetInput>
KVCacheCreateOptions
DraftModelSpecWorkerImpl<TargetInput>::build_draft_kv_cache_layout(
    const KVCacheShape& shape) const {
  auto layout = get_kv_cache_create_options(
      draft_impl_->context_.get_model_args(), shape, true);
  auto indexer_layers = resolve_indexer_cache_enabled_layers(
      draft_impl_->context_.get_model_args(), draft_num_layers());
  if (shape.has_kpool_tail_shape()) {
    indexer_layers.assign(draft_num_layers(), true);
  }
  layout.indexer_cache_enabled_layers(std::move(indexer_layers));
  layout.num_layers(draft_num_layers());
  if (shape.shared_draft_layers() > 0) {
    const auto shared = impl_->shared_kv_cache_layout();
    CHECK_EQ(shared.static_layout()->draft_layer_count(), layout.num_layers());
    CHECK(kv_cache_transfer_ == nullptr && !shape.has_grouped_cache_layout() &&
          !layout.enable_linear_attention() &&
          draft_impl_->context_.get_model_args().index_n_heads() == 0 &&
          draft_impl_->options().kv_cache_dtype() != "int8" &&
          KVCacheConfig::get_instance().indexer_cache_dtype() != "int8")
        << "Static shared draft storage requires ordinary Full-only layers";
    layout.static_layout(shared.static_layout())
        .shared_draft_cache(true)
        .full_block_slot_table(shared.full_block_slot_table())
        .tensor_allocator(create_static_draft_kv_tensor_allocator(
            std::static_pointer_cast<StaticKVCacheTensorAllocator>(
                shared.tensor_allocator())));
  }
  return layout;
}

template <typename TargetInput>
int64_t DraftModelSpecWorkerImpl<TargetInput>::draft_num_layers() const {
#if !defined(USE_NPU)
  const int32_t nextn_layers =
      draft_impl_->context_.get_model_args().num_nextn_predict_layers();
  if (nextn_layers > 0) {
    return nextn_layers;
  }
#endif
  return draft_impl_->context_.get_model_args().n_layers();
}

#if defined(USE_NPU)
template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::prepare_draft_padding(
    LlmModelParams& params,
    Stream* prepare_stream) {
  if (!draft_impl_->uses_npu_dp_ep_padding()) {
    return;
  }
  const bool is_prefill = params.meta.batch_forward_type.no_decode();
  const auto& tokens = params.parallel.dp_global_token_nums;
  const auto& raw_tokens = params.parallel.raw_dp_global_token_nums;
  if (!is_prefill) {
    if (const auto* cached = dp_ep_padding_cache_.find(tokens, raw_tokens)) {
      params.parallel.dp_ep_padding_data = *cached;
      return;
    }
  }
  if (prepare_stream != nullptr) {
    draft_impl_->prepare_dp_ep_padding_on_stream(params, *prepare_stream);
  } else {
    draft_impl_->prepare_dp_ep_padding(params);
  }
  if (!is_prefill) {
    dp_ep_padding_cache_.insert(
        tokens, raw_tokens, params.parallel.dp_ep_padding_data);
  }
}

#endif

namespace {

runtime::Options target_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_model(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
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
    const ModelContext& context,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t cache_size_in_bytes) {
  const ModelArgs& model_args = context.get_model_args();
  const int64_t dp_local_tp_size =
      get_dp_local_tp_size(context.get_parallel_args());
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
      static_cast<int64_t>(context.num_speculative_tokens());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  estimate_options.is_draft_model = options.is_draft_model();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();

  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  estimate_options.enable_prefix_cache =
      kv_cache_config.enable_prefix_cache() &&
      !kv_cache_config.enable_vmm_kvcache();
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
    WorkerType worker_type,
    absl::FunctionRef<std::unique_ptr<LLMWorkerImpl>()> draft_factory,
    int32_t (*configure_draft_config)(ModelArgs&, const runtime::Options&))
    : SpeculativeWorkerImpl<TargetInput>(parallel_args,
                                         device,
                                         options,
                                         target_options(options),
                                         worker_type),
      draft_impl_(draft_factory()),
      configure_draft_config_(configure_draft_config) {
  draft_impl_->set_input_preparer(
      std::function<void(LlmForwardInput&)>([this](LlmForwardInput& input) {
        ::xllm::prepare_dummy_input(input, draft_impl_->context_, device_);
#if defined(USE_NPU)
        prepare_draft_padding(input.input_params);
#endif
      }));
}

template <typename TargetInput>
DraftModelSpecWorkerImpl<TargetInput>::~DraftModelSpecWorkerImpl() {
  drain_pending_execution();
  // Stop asynchronous transfers before draft_impl_ releases its KV cache.
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    hierarchy_kv_cache_transfer_->shutdown();
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::drain_pending_execution() {
  SpeculativeWorkerImpl<TargetInput>::drain_pending_execution();
  draft_impl_->drain_pending_execution();
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  bool result;
  torch::Tensor draft_token_id_offsets;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl<TargetInput>::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    auto loader = ModelLoader::create(model_weights_path);
    auto context = load_model_config(*loader);
    draft_token_id_offsets = load_draft_vocab_mapping(*loader);
    result = draft_impl_->WorkerImpl::init_model(std::move(loader),
                                                 std::move(context),
                                                 nullptr,
                                                 random_seed,
                                                 master_status);
  }
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
    target_verify_mode_ = mtp_async::classify_target_spec_verify_mode(
        context_.get_model_args().model_type(),
        ModelConfig::is_python_model_impl(context_.get_model_impl()));
  }
  if (result && draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const int64_t draft_vocab_size = draft_args.draft_vocab_size() > 0
                                         ? draft_args.draft_vocab_size()
                                         : draft_args.vocab_size();
    draft_vocab_ =
        DraftVocabMapping(std::move(draft_token_id_offsets),
                          draft_vocab_size,
                          impl_->context_.get_model_args().vocab_size(),
                          device_);
  }
  return result;
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
DraftModelSpecWorkerImpl<TargetInput>::estimate_kv_cache_capacity() {
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
      make_kv_cache_estimate_options(
          impl_->context_, impl_->options(), dtype_, cache_size_in_bytes);
  const KVCacheEstimateOptions draft_estimate_options =
      make_kv_cache_estimate_options(draft_impl_->context_,
                                     draft_impl_->options(),
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
  embedding_cache_ = std::make_unique<EmbeddingCache>(
      num_blocks, requires_full_target_replay());
  const int64_t placeholder_size = get_embedding_placeholder_size();
  if (placeholder_size > 0) {
    embedding_cache_->set_placeholder(
        torch::zeros({placeholder_size}, torch::dtype(dtype_).device(device_)));
  }
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_pools(
    const KVCacheShape& kv_cache_shape,
    absl::FunctionRef<bool(WorkerImpl&, const KVCacheShape&)> allocate) {
  const WorkerImpl::Status target_status = impl_->get_status();
  const bool bind_vmm_group =
      KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      target_status == WorkerImpl::Status::LOADED &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED &&
      impl_->context_.get_model_id() != draft_impl_->context_.get_model_id();
  bool target_allocated = true;
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = allocate(*impl_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const auto shape = draft_kv_cache_shape(kv_cache_shape);
    draft_impl_->allocate_kv_cache(
        shape, build_draft_kv_cache_layout(shape), kv_cache_transfer_);
    if (kv_cache_transfer_) {
      kv_cache_transfer_->configure_cache_layout(
          draft_impl_->context_.get_parallel_args(),
          draft_impl_->context_.get_model_args(),
          draft_impl_->options().block_size(),
          true);
      kv_cache_transfer_->register_kv_cache_spec(
          draft_impl_->kv_caches(), shape, draft_impl_->dtype());
    }
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::READY);
  }

  init_embedding_cache(kv_cache_shape.key_cache_shape()[0]);
  if (target_allocated) {
    if (bind_vmm_group) {
      // The scheduler's block IDs address both caches. Bind their mappings
      // before the master reserves its first (padding) page group.
      CHECK(WorkerMemoryResources::get_instance()
                .kv_cache_memory()
                .bind_mapping_group(impl_->context_.get_model_id(),
                                    {impl_->context_.get_model_id(),
                                     draft_impl_->context_.get_model_id()}));
    }
    finalize_hierarchy_kv_cache_transfers();
  }
  return target_allocated;
}

template <typename TargetInput>
std::optional<ForwardOutput> DraftModelSpecWorkerImpl<TargetInput>::step(
    const TargetInput& input) {
  if (draft_impl_->uses_shared_static_kv_cache()) {
    const auto& embedding = input.input_params.embedding;
    // Decode can execute draft before target. Initialize their common leases
    // here so a newly admitted draft page cannot expose previous Linear state.
    // The compute stream orders this after preceding target/draft consumers.
    impl_->initialize_shared_full_blocks(embedding.full_block_slot_rows,
                                         embedding.full_block_lease_epochs,
                                         *compute_stream_);
  }
  return SpeculativeWorkerImpl<TargetInput>::step(input);
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
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
bool DraftModelSpecWorkerImpl<TargetInput>::prepare_model_decode_continuation(
    TargetInput& /*inputs*/,
    const ForwardOutput& /*prior_output*/) {
  return true;
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
        bootstrap_embeddings.select(/*dim=*/0, i),
        input.host_positions().dim() == 2
            ? input.host_positions().select(/*dim=*/1, row_idx)
            : torch::Tensor());
  }
}

template <typename TargetInput>
std::optional<ForwardOutput>
DraftModelSpecWorkerImpl<TargetInput>::finalize_step_output(
    ForwardOutput output,
    std::optional<SampleOutput> verify_output,
    const std::vector<JsonObjectGrammarState>& json_object_states) const {
  if (!this->enable_schedule_overlap() && !this->is_driver() &&
      !EPLBConfig::get_instance().enable_eplb()) {
    return make_transfer_failure_output(std::move(output.failed_request_ids));
  }
  if (verify_output) {
    return finalize_verify_output(
        std::move(output), std::move(*verify_output), json_object_states);
  }
  return output;
}

template <typename TargetInput>
int32_t DraftModelSpecWorkerImpl<TargetInput>::logical_block_size() const {
  return options_.block_size() * parallel_args_.kv_split_size_effective();
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::update_decode_step_input(
    TargetInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const bool enable_cache_correction = this->enable_schedule_overlap();
  const bool chunked_verify = uses_chunked_verify();
  const bool positions_decoupled =
      !impl_->context_.get_model_args().rope_scaling_mrope_section().empty();
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
  kv_seq_lens_vec.reserve(static_cast<size_t>(num_sequences));

  const Slice<int32_t> input_token_ids = tensor_slice(input.token_ids_host);
  const Slice<int32_t> input_positions = tensor_slice(input.positions_host);
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  const bool clamp_active =
      chunked_verify && block_tables.defined() && block_tables.dim() == 2;
  const int32_t verify_lens = options_.num_speculative_tokens() + 1;
  const int32_t allocated_kv_len =
      clamp_active
          ? static_cast<int32_t>(block_tables.size(1)) * logical_block_size()
          : 0;

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size());
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size());
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len =
        input.input_params.attention.host.kv_seq_lens[seq_id] + position_offset;
    int32_t expected_kv_len =
        positions_decoupled ? current_kv_len : current_position + 1;
    if (clamp_active && seq_id < block_tables.size(0)) {
      const int32_t max_valid_position = allocated_kv_len - verify_lens;
      if (current_position > max_valid_position) {
        CHECK_GT(allocated_kv_len, 0)
            << "decode context has empty block table, seq_id=" << seq_id;
        CHECK_GE(max_valid_position, 0)
            << "decode context block table is too small for validation, "
            << "seq_id=" << seq_id << ", allocated_kv_len=" << allocated_kv_len
            << ", verify_lens=" << verify_lens;
        CHECK_LE(current_position - max_valid_position, verify_lens)
            << "decode context position exceeds allocated blocks, seq_id="
            << seq_id << ", current_position=" << current_position
            << ", current_kv_len=" << current_kv_len
            << ", allocated_kv_len=" << allocated_kv_len;
        current_position = max_valid_position;
        expected_kv_len = current_position + 1;
        current_kv_len = std::min(current_kv_len, expected_kv_len);
      }
    }
    if (chunked_verify && current_kv_len < expected_kv_len) {
      CHECK_LE(expected_kv_len - current_kv_len, verify_lens)
          << "decode context kv_len lag is too large, seq_id=" << seq_id
          << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len;
      current_kv_len = expected_kv_len;
    }
    if (chunked_verify && current_kv_len > expected_kv_len) {
      // First decode step: the prompt KV length with a zero-initialized
      // position; align the position to the KV context.
      current_position = current_kv_len - 1;
      expected_kv_len = current_kv_len;
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back(rewrite_fake_token ? state.token_id
                                                  : input_token_id);
    positions_vec.emplace_back(current_position);
    kv_seq_lens_vec.emplace_back(current_kv_len);
  }

  input.token_ids_host = make_pinned_cpu_tensor(token_ids_vec);
  input.positions_host = make_pinned_cpu_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.runtime.input_host_buffer_has_layout = false;
  input.runtime.device_tensors_ready = false;
}

template class DraftModelSpecWorkerImpl<LlmForwardInput>;
template class DraftModelSpecWorkerImpl<VlmForwardInput>;

}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "llm_engine.h"

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/options.h"
#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_controller.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "core/platform/platform.h"
#include "core/util/model_config_utils.h"
#include "framework/block/block_utils.h"
#include "framework/block/kv_cache_manager_factory.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/speculative/mtp_utils.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/params_utils.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace {

bool contains_graph_warmup(const std::vector<LlmForwardInput>& inputs) {
  return std::any_of(
      inputs.begin(), inputs.end(), [](const LlmForwardInput& input) {
        return input.input_params.meta.is_graph_warmup;
      });
}

bool contains_graph_warmup(const BatchGroup& batches) {
  for (const Batch& batch : batches) {
    const std::vector<Sequence*> sequences = batch.get_sequences();
    if (std::any_of(sequences.begin(), sequences.end(), [](Sequence* sequence) {
          return sequence != nullptr && sequence->is_graph_warmup();
        })) {
      return true;
    }
  }
  return false;
}

}  // namespace

// Extra weight pages reserved for mapping/alignment overhead.
constexpr size_t kXTensorWeightPageSafetyMargin = 20;

LLMEngine::LLMEngine(
    const runtime::Options& options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(options),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {
  auto master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty())
      << " LLM need to set master node addr, Please set --master_node_addr.";
  const auto& devices = options_.devices();
  // initialize device monitor
  DeviceMonitor::get_instance().initialize(devices);
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
#if defined(USE_NPU)
    FLAGS_enable_atb_comm_multiprocess =
        options.enable_offline_inference() || (options.nnodes() > 1);
#endif
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);

  dp_size_ = options_.dp_size();
  cp_size_ = options_.cp_size();
  dp_batch_embedding_ids_.resize(dp_size_);
  dp_batch_request_ids_.resize(dp_size_);
  dp_batch_generations_.resize(dp_size_, 0);
  worker_clients_num_ = worker_clients_.size();
  dp_local_size_ = worker_clients_num_ / dp_size_;
  // MLU and NPU model-side CP both use orthogonal CP x attention-TP, so the
  // DP-local TP width is divided by CP.
  dp_local_tp_size_ = dp_local_size_ / cp_size_;

  // create ThreadPool for link cluster
  link_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/worker_clients_num_,
      /*cpu_binding=*/false,
      /*pool_name=*/"LLMEngine.link");

  process_group_test();

  // init thread pool
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"LLMEngine.forward_input");
}

LLMEngine::~LLMEngine() = default;

runtime::DecodeGraphExecutionShape LLMEngine::decode_graph_execution_shape()
    const {
  runtime::DecodeGraphExecutionShape execution_shape;
  execution_shape.num_decoding_tokens = options_.num_decoding_tokens();
  execution_shape.num_speculative_tokens = options_.num_speculative_tokens();
  execution_shape.enable_graph_mode_decode_no_padding =
      options_.enable_graph_mode_decode_no_padding();
  return execution_shape;
}

void LLMEngine::process_group_test() {
#if !defined(USE_NPU)
  // In multi-node serving mode, only driver engine
  // create worker_clients_.
  if (worker_clients_num_ > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(worker_clients_num_);
    for (auto& worker : worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // Wait for all futures to complete with a configurable timeout.
    // The timeout can be adjusted via the
    // XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS environment variable (default: 4
    // seconds). This is particularly important in multi-node multi-device
    // scenarios where network latency may require a longer timeout period.
    const int timeout_seconds = util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
#endif
}

bool LLMEngine::init(MasterStatus master_status) {
  if (!init_model(master_status)) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  eplb_controller_ = EplbController::create(
      args_, static_cast<int32_t>(worker_clients_num_), options_.ep_size());

  auto kv_cache_cap = estimate_kv_cache_capacity();

  if (!allocate_kv_cache(kv_cache_cap)) {
    LOG(ERROR) << "Failed to initialize kv cache";
    return false;
  } else {
    LOG(INFO) << "Successfully initialized kv cache";
  }

  // If master_status is not MasterStatus::WAKEUP, put the model to sleep after
  // initialization
  // This allows KV cache allocation to complete first, then releases resources
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor() &&
      master_status != MasterStatus::WAKEUP) {
    const std::string& model_id = options_.model_id();
    if (!PageAllocator::get_instance().sleep_model(
            model_id, /*skip_weight_release=*/true)) {
      LOG(ERROR) << "Failed to sleep model " << model_id << " after init";
      return false;
    }
    LOG(INFO) << "Model " << model_id
              << " put to sleep after init (master_status=" << master_status
              << ")";
  }

  return true;
}

bool LLMEngine::init_model(MasterStatus master_status) {
  const std::string& model_path = options_.model_path();
  auto model_loader = ModelLoader::create(model_path);
  LOG(INFO) << "Initializing model from: " << model_path;

  args_ = model_loader->model_args();
#if defined(USE_NPU)
  configure_glm5_next_mtp_args(
      args_, options_.speculative_algorithm(), options_.is_draft_engine());
#endif
  quant_args_ = model_loader->quant_args();
  if (!options_.is_draft_engine() &&
      SpeculativeConfig::is_block_diffusion_algorithm(
          options_.speculative_algorithm())) {
    CHECK(options_.draft_model_path().has_value())
        << "Block-diffusion Target requires a draft checkpoint.";
    // Scheduler warmup must use the same auxiliary-state width as the workers.
    args_.layers_to_capture(
        util::read_capture_layer_ids(options_.draft_model_path().value()));
  }

  // A draft engine is fed token ids and detokenized by the target, so it
  // shares the target vocabulary and loads no tokenizer of its own.
  if (!options_.is_draft_engine()) {
    tokenizer_ = model_loader->tokenizer();
    CHECK(tokenizer_ != nullptr);
    tokenizer_args_ = model_loader->tokenizer_args();
  }

  dtype_ = util::parse_dtype(args_.dtype(), options_.devices()[0]);
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_layers: " << args_.n_layers() << ", dtype: " << dtype_
            << ", kv_cache_dtype: " << options_.kv_cache_dtype();

  if (tokenizer_ != nullptr) {
    const int64_t tokenizer_vocab_size = tokenizer_->vocab_size();
    int64_t model_vocab_size = args_.vocab_size();
    if (tokenizer_vocab_size != model_vocab_size) {
      // use tokenizer vocab size if model vocab size is not set
      if (model_vocab_size <= 0) {
        LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab "
                        "size: "
                     << tokenizer_vocab_size;
        args_.vocab_size(tokenizer_vocab_size);
      } else if (tokenizer_vocab_size > model_vocab_size) {
        LOG(WARNING) << "Unsafe vocab mismatch: tokenizer: "
                     << tokenizer_vocab_size << ", model: " << model_vocab_size;
      } else {
        LOG(INFO) << "Tokenizer/model vocab differ: tokenizer="
                  << tokenizer_vocab_size << ", model=" << model_vocab_size;
      }
    }
  }

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;
  LOG(INFO) << "Initializing model with random seed: "
            << ::xllm::ExecutionConfig::get_instance().random_seed();

  // Initialize PageAllocator if using XTensor mode (before using it)
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    auto& page_allocator = PageAllocator::get_instance();
    if (!page_allocator.is_initialized()) {
      auto& phy_pool = PhyPagePool::get_instance();
      CHECK(phy_pool.is_initialized())
          << "PhyPagePool must be initialized before PageAllocator";
      size_t num_phy_pages = phy_pool.num_total();
      // max_world_size = dp_size * tp_size = worker_clients_num_
      int32_t max_world_size = worker_clients_num_;
      page_allocator.init(num_phy_pages,
                          dp_size_,
                          max_world_size,
                          /*enable_page_prealloc=*/true);
    }

    // Register model with model_id from options
    // Each model has its own logical page_list but shares physical pages
    const std::string& model_id = options_.model_id();
    page_allocator.register_model(model_id, args_.n_layers(), master_status);

    // Set model-specific parallel strategy for broadcast operations
    // This is important for fork master with different dp/tp than original
    // master (each model may have different dp_size/tp_size)
    page_allocator.set_model_parallel_strategy(
        model_id, dp_size_, dp_local_tp_size_);
    auto& xtensor_allocator = XTensorAllocator::get_instance();
    xtensor_allocator.set_model_parallel_strategy(
        model_id, dp_size_, dp_local_tp_size_);

    // Get weight size for XTensor page allocation.
    const int64_t total_weight_size =
        get_effective_xtensor_weight_size(*model_loader);
    if (total_weight_size < 0) {
      return false;
    }
    int64_t weight_size_per_tp =
        (total_weight_size + dp_local_tp_size_ - 1) / dp_local_tp_size_;

    size_t page_size =
        ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
    size_t num_pages = (weight_size_per_tp + page_size - 1) / page_size +
                       kXTensorWeightPageSafetyMargin;

    LOG(INFO) << "XTensor weight allocation: total_weight_size="
              << total_weight_size << ", tp_size=" << dp_local_tp_size_
              << ", weight_size_per_tp=" << weight_size_per_tp
              << ", num_pages=" << num_pages
              << ", master_status=" << master_status;

    if (master_status == MasterStatus::WAKEUP) {
      // Consume physical pages for weights (global xtensor handles mapping)
      if (!page_allocator.alloc_weight_pages(model_id, num_pages)) {
        LOG(ERROR) << "Failed to allocate weight pages";
        return false;
      }
      LOG(INFO)
          << "master_status=0 (MasterStatus::WAKEUP): Allocated weight pages, "
             "will load to device";
    } else if (master_status == MasterStatus::LIGHT_SLEEP ||
               master_status == MasterStatus::DEEP_SLEEP) {
      // Record num_pages for later wakeup
      page_allocator.set_weight_pages_count(model_id, num_pages);
      LOG(INFO) << "master_status=" << master_status
                << " (SLEEP): Recorded weight pages, num_pages=" << num_pages;
    }
  }

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->init_model_async(
        model_path,
        ::xllm::ExecutionConfig::get_instance().random_seed(),
        master_status));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  return true;
}

int64_t LLMEngine::get_effective_xtensor_weight_size(
    const ModelLoader& model_loader) const {
  constexpr int64_t kInvalidWeightSize = -1;
  const int64_t all_size = model_loader.get_total_weight_size();
  if (all_size <= 0) {
    LOG(ERROR)
        << "Invalid total model weight size: " << all_size
        << ". Ensure model .index.json exists and has metadata.total_size";
    return kInvalidWeightSize;
  }

  if (!::xllm::LoadConfig::get_instance().enable_rolling_load()) {
    return all_size;
  }

  const int64_t non_decoder_size = model_loader.get_non_decoder_weight_size();
  if (non_decoder_size <= 0) {
    LOG(ERROR) << "Invalid non-decoder weight size: " << non_decoder_size;
    return kInvalidWeightSize;
  }
  if (non_decoder_size > all_size) {
    LOG(ERROR) << "non_decoder_weight_size (" << non_decoder_size
               << ") exceeds total_weight_size (" << all_size << ")";
    return kInvalidWeightSize;
  }
  if (args_.n_layers() <= 0) {
    LOG(ERROR) << "Invalid layer count: " << args_.n_layers();
    return kInvalidWeightSize;
  }

  const int64_t all_decoder_size = all_size - non_decoder_size;
  int64_t max_layer_size = model_loader.get_max_decoder_layer_weight_size();
  if (max_layer_size <= 0) {
    LOG(ERROR) << "Failed to get max decoder layer size for rolling load.";
    return kInvalidWeightSize;
  }
  const int64_t rolling_buffer_size =
      ::xllm::LoadConfig::get_instance().rolling_load_num_cached_layers() *
      max_layer_size;
  const int64_t total_weight_size = non_decoder_size + rolling_buffer_size;

  LOG(INFO)
      << "XTensor rolling_load weight budget: total=" << all_size
      << ", non_decoder=" << non_decoder_size
      << ", all_decoder=" << all_decoder_size
      << ", max_layer=" << max_layer_size
      << ", rolling_buffer=" << rolling_buffer_size << " ("
      << ::xllm::LoadConfig::get_instance().rolling_load_num_cached_layers()
      << " slots x " << max_layer_size << " bytes/max-layer)"
      << ", effective=" << total_weight_size;
  return total_weight_size;
}

KVCacheCapacity LLMEngine::estimate_kv_cache_capacity() {
  return KVCacheManagerFactory::estimate_capacity(
      args_, options_, dtype_, dp_local_tp_size_, worker_clients_);
}

bool LLMEngine::allocate_kv_cache(const KVCacheCapacity& kv_cache_cap) {
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();

  LOG(INFO) << "kv cache capacity: "
            << readable_size(kv_cache_cap.cache_size_in_bytes())
            << ", blocks: " << kv_cache_cap.n_blocks()
            << ", slot_size: " << kv_cache_cap.slot_size()
            << ", index_slot_size: " << kv_cache_cap.index_slot_size()
            << ", indexer_layers: " << kv_cache_cap.num_indexer_layers()
            << ", scale_slot_size: " << kv_cache_cap.scale_slot_size()
            << ", linear_slot_size: " << kv_cache_cap.linear_slot_size()
            << ", linear_blocks: " << kv_cache_cap.num_linear_state_blocks()
            << ", linear_state_slots: "
            << (has_linear_attention_layers(args_)
                    ? kv_cache_cap.num_linear_state_blocks()
                    : 0)
            << ", max_linear_state_cache_slots: "
            << options_.max_linear_state_cache_slots()
            << ", reserved_linear_bytes: "
            << readable_size(kv_cache_cap.linear_cache_size_in_bytes())
            << ", n_layers: " << kv_cache_cap.n_layers()
            << ", kv_cache_dtype: " << options_.kv_cache_dtype();

  CHECK_GT(kv_cache_cap.n_blocks(), 0) << "no memory for kv cache";
  const int32_t block_size = static_cast<int32_t>(kv_cache_cap.block_size());
  const bool enable_gdn_attention = has_linear_attention_layers(args_);

  // Validate host offload before constructing block managers or asking workers
  // to allocate potentially large pinned host tensors. The factory fills the
  // shape-dependent validation fields.
  HostCacheValidationOptions host_cache_validation_options{
      .host_blocks_factor = options_.host_blocks_factor(),
      .device_block_count = kv_cache_cap.n_blocks(),
      .supports_host_kv_offload = Platform::supports_host_kv_offload(),
      .enable_prefix_cache = options_.enable_prefix_cache(),
      .enable_disagg_pd = options_.enable_disagg_pd(),
      .enable_kvcache_store = options_.enable_kvcache_store(),
      .instance_role = options_.instance_role(),
      .supports_grouped_cache_offload =
          util::is_deepseek_v4_model_type(args_.model_type()),
      .kv_cache_dtype = options_.kv_cache_dtype(),
      .indexer_cache_dtype = kv_cache_config.indexer_cache_dtype(),
      .model_type = args_.model_type(),
  };
  // DECODE-side skips LINEAR prefix cache by role (see
  // composite_block_manager.cpp::leaf_participates_in_prefix_cache), so the
  // chunked-prefill + chunk-stride guards below are only meaningful for
  // PREFILL / MIX. On DECODE the linear-state cache is disabled anyway and
  // pd_launch.sh legitimately sets --enable_chunked_prefill=false.
  const bool is_decode = options_.instance_role() == InstanceRole::DECODE;
  if (options_.enable_prefix_cache() && enable_gdn_attention && !is_decode) {
    const auto& scheduler_config = ::xllm::SchedulerConfig::get_instance();
    CHECK(scheduler_config.enable_chunked_prefill())
        << "Linear-attention prefix cache requires block-aligned chunked "
           "prefill to save matching linear states. Please set "
           "--enable_chunked_prefill=true in your config.";
    CHECK(scheduler_config.max_tokens_per_chunk_for_prefill() % block_size == 0)
        << "linear-attention prefix cache saves linear-state checkpoints at "
           "chunk-end boundaries, so max_tokens_per_chunk_for_prefill ("
        << scheduler_config.max_tokens_per_chunk_for_prefill()
        << ") must be a multiple of block_size (" << block_size << ").";
  }

  // initialize block manager
  // Logical block_size *= kv_split_size.
  const int32_t kv_split_size_eff =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks())
      .hasher_type(mtp_hasher_type(BlockHasherType::TEXT,
                                   options_.num_speculative_tokens(),
                                   options_.speculative_algorithm()))
      .block_size(kv_split_size_eff > 1 ? block_size * kv_split_size_eff
                                        : block_size)
      .host_num_blocks(kv_cache_cap.n_blocks() * options_.host_blocks_factor())
      .enable_linear_state(enable_gdn_attention)
      .enable_prefix_cache(kv_cache_config.enable_xtensor()
                               ? false
                               : options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_kvcache_store(options_.enable_kvcache_store())
      .prefetch_batch_size(options_.prefetch_batch_size())
      .enable_xtensor(kv_cache_config.enable_xtensor())
      .num_layers(args_.n_layers())
      .slot_size(kv_cache_cap.slot_size())
      .model_id(options_.model_id())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .num_speculative_tokens(options_.num_speculative_tokens())
      // DECODE-side prefix cache participation is per-leaf and gated by the
      // predicate in composite_block_manager.cpp. P and MIX are treated
      // identically (both admit prefix cache on every leaf).
      .instance_is_decode(options_.instance_role() == InstanceRole::DECODE);
  if (enable_gdn_attention) {
    // The unified linear-state slot pool spans all physical slots [0, N);
    // id 0 is reserved as padding and ids [1, N) serve live and checkpoint
    // rows interchangeably under reference counting.
    options.linear_state_num_slots(
        static_cast<int32_t>(kv_cache_cap.num_linear_state_blocks()));
  }
  if (util::is_deepseek_v4_model_type(args_.model_type())) {
    constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
    constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

    std::vector<uint32_t> manager_types{kManagerTypeSlidingWindowBlockManager};
    std::vector<uint32_t> manager_compress_ratios{
        0};  // unused for sliding window manager
    std::vector<uint32_t> token_manager_ratios;
    token_manager_ratios.reserve(2);
    for (const int32_t ratio : args_.compress_ratios()) {
      if (ratio == 4 || ratio == 128) {
        const uint32_t ratio_u32 = static_cast<uint32_t>(ratio);
        if (std::find(token_manager_ratios.begin(),
                      token_manager_ratios.end(),
                      ratio_u32) == token_manager_ratios.end()) {
          token_manager_ratios.push_back(ratio_u32);
        }
      }
    }
    for (const uint32_t ratio : token_manager_ratios) {
      manager_types.push_back(kManagerTypeBlockManagerImpl);
      manager_compress_ratios.push_back(ratio);
    }

    const int64_t semantic_window = std::max(args_.window_size(), 1);
    const int64_t max_model_len = args_.max_seq_len();
    const int64_t effective_window =
        max_model_len > 0 ? std::min<int64_t>(semantic_window, max_model_len)
                          : semantic_window;
    const uint32_t swa_blocks_per_seq = static_cast<uint32_t>(
        get_swa_blocks_per_seq(effective_window, block_size));
    CHECK_LE(kv_cache_cap.swa_count(),
             static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        << "DSV4 swa_count exceeds uint32_t range: "
        << kv_cache_cap.swa_count();

    options.sliding_window_size(static_cast<uint32_t>(effective_window))
        .swa_blocks_per_seq(swa_blocks_per_seq)
        .swa_num_blocks(static_cast<uint32_t>(kv_cache_cap.swa_count()))
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .manager_types(std::move(manager_types))
        .compress_ratios(std::move(manager_compress_ratios));
  }

  if (options_.enable_kvcache_store()) {
    CHECK_GT(options_.host_blocks_factor(), 1.0)
        << "KV cache Store requires Host cache blocks.";
  }

  const bool is_mtp_draft_engine =
      options_.is_draft_engine() &&
      SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm());
  if (is_mtp_draft_engine) {
    // The speculative scheduler owns only the target engine BlockManager.
    // Keep the draft engine's placeholder manager non-caching so a Draft
    // model with only SWA tensors does not pretend to own logical prefixes.
    options.enable_prefix_cache(false);
  }

  if (options_.host_blocks_factor() > 1.0 && !is_mtp_draft_engine) {
    // Translate a composite cache capacity into typed Host pools. The
    // hierarchy layer consumes only this BlockType map and does not need to
    // identify the model that produced the layout.
    if (!options.manager_types().empty()) {
      std::map<BlockType, uint32_t> host_capacities;
      if (kv_cache_cap.swa_count() > 0) {
        host_capacities.emplace(
            BlockType::SWA,
            static_cast<uint32_t>(scale_host_block_count(
                kv_cache_cap.swa_count(), options_.host_blocks_factor())));
      }
      if (kv_cache_cap.c4_count() > 0) {
        host_capacities.emplace(
            BlockType::C4,
            static_cast<uint32_t>(scale_host_block_count(
                kv_cache_cap.c4_count(), options_.host_blocks_factor())));
      }
      if (kv_cache_cap.c128_count() > 0) {
        host_capacities.emplace(
            BlockType::C128,
            static_cast<uint32_t>(scale_host_block_count(
                kv_cache_cap.c128_count(), options_.host_blocks_factor())));
      }
      options.host_num_blocks_by_type(std::move(host_capacities));
    }
    options.enable_host_offload(true);
  }

  auto factory_result =
      KVCacheManagerFactory::create(kv_cache_cap,
                                    args_,
                                    dp_local_tp_size_,
                                    std::move(options),
                                    this,
                                    dp_size_,
                                    std::move(host_cache_validation_options));
  factory_result.shape.print_shapes();
  KVCacheShape kv_cache_shape = std::move(factory_result.shape);
  kv_cache_manager_ = std::move(factory_result.manager);

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  if (options_.instance_role() == InstanceRole::DEFAULT) {
    for (auto& worker : worker_clients_) {
      futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
    }
  } else {
    for (auto& worker : worker_clients_) {
      futures.push_back(
          worker->allocate_kv_cache_with_transfer_async(kv_cache_shape));
    }
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  // XTensor mode: reserve padding blocks and start prealloc thread.
  kv_cache_manager_->reserve_xtensor_padding_blocks();

  return true;
}

bool LLMEngine::set_speculative_validate_time_predictor(
    const SpeculativeProfileRegistry::ValidateTimePredictor& predictor) {
  bool success = true;
  for (size_t i = 0; i < worker_clients_.size(); ++i) {
    if (!worker_clients_[i]->set_speculative_validate_time_predictor(
            predictor)) {
      LOG(ERROR) << "Failed to set speculative validate predictor for worker "
                 << i;
      success = false;
    }
  }
  return success;
}

bool LLMEngine::pull_kv_blocks(const int32_t src_dp_size,
                               const int32_t src_dp_rank,
                               const std::vector<uint64_t>& src_cluster_ids,
                               const std::vector<std::string>& src_addrs,
                               const int32_t dst_dp_rank,
                               const std::vector<KVTransferMapping>& mappings) {
  int32_t src_world_size = src_cluster_ids.size();
  int32_t src_tp_size = src_world_size / src_dp_size;
  int32_t dst_world_size = options_.nnodes();
  int32_t dst_tp_size = dst_world_size / dp_size_;

  std::vector<bool> results;
  results.reserve(dst_tp_size);
  // Pull the KV cache for all workers in the current DP rank.
  for (size_t tp_rank = 0; tp_rank < dst_tp_size; ++tp_rank) {
    int32_t dst_worker_rank = dst_dp_rank * dst_tp_size + tp_rank;
    // Determine the ranks of the remote workers connected to the current
    // worker.
    int32_t src_dp_worker_rank = dst_worker_rank % src_tp_size;
    int32_t src_worker_rank = src_dp_rank * src_tp_size + src_dp_worker_rank;
    results.push_back(worker_clients_[dst_worker_rank]->pull_kv_blocks(
        src_cluster_ids[src_worker_rank],
        src_addrs[src_worker_rank],
        mappings));
  }

  for (bool result : results) {
    if (!result) {
      return false;
    }
  }
  return true;
}

std::vector<folly::SemiFuture<uint32_t>> LLMEngine::transfer_kv_blocks(
    const uint32_t dp_rank,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::vector<folly::SemiFuture<uint32_t>> futures;
  futures.reserve(dp_local_tp_size_);

  for (uint32_t tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    futures.emplace_back(worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]
                             ->transfer_kv_blocks(block_transfer_info));
  }

  return std::move(futures);
}

void LLMEngine::transfer_kv_blocks(
    const uint32_t dp_rank,
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  for (uint32_t tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]->transfer_kv_blocks(
        batch_id, block_transfer_info);
  }
}

void LLMEngine::prefetch_from_storage(
    const uint32_t dp_rank,
    std::shared_ptr<const StoragePrefetchRequest> request,
    PrefetchResult::StopPredicate stop_requested,
    PrefetchResult::DoneCallback done) {
  CHECK(request != nullptr);
  CHECK(request->valid());
  const uint32_t configured_timeout_ms = options_.prefetch_timeout();
  const int64_t timeout_ms = configured_timeout_ms == 0
                                 ? -1
                                 : static_cast<int64_t>(configured_timeout_ms);
  auto result =
      std::make_shared<PrefetchResult>(dp_local_tp_size_,
                                       request->batch_end_unit_offsets,
                                       timeout_ms,
                                       std::move(stop_requested),
                                       std::move(done));
  for (uint32_t tp_rank = 0; tp_rank < dp_local_tp_size_; ++tp_rank) {
    worker_clients_[tp_rank + dp_local_tp_size_ * dp_rank]
        ->prefetch_from_storage(request, result, tp_rank);
  }
}

void LLMEngine::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<uint16_t>& ports) {
  cluster_ids.reserve(worker_clients_num_);
  addrs.reserve(worker_clients_num_);
  ports.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    uint64_t cluster_id = 0;
    std::string addr;
    uint16_t port = 0;
    worker_clients_[worker_rank]->get_cache_info(cluster_id, addr, port);
    cluster_ids.emplace_back(cluster_id);
    addrs.emplace_back(std::move(addr));
    ports.emplace_back(port);
  }
}

void LLMEngine::get_xtensor_info(
    std::vector<size_t>& worker_free_phy_pages,
    std::unordered_map<std::string, std::vector<WeightSegment>>&
        model_weight_segments) {
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    return;
  }

  // Worker 0 is in the same process as Master, no RPC needed.
  // Both PageAllocator and XTensorAllocator are singletons.

  // Get free phy pages from PageAllocator
  auto& page_allocator = PageAllocator::get_instance();
  if (page_allocator.is_initialized()) {
    worker_free_phy_pages = page_allocator.get_all_worker_free_pages();
  }

  // Get model weight segments from XTensorAllocator directly (no RPC)
  // Worker 0 is always in dp group 0, weights are duplicated across dp groups
  auto& xtensor_allocator = XTensorAllocator::get_instance();
  model_weight_segments = xtensor_allocator.get_all_model_weight_segments();
}

bool LLMEngine::link_cluster(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& addrs,
                             const std::vector<uint16_t>& ports,
                             const int32_t src_dp_size,
                             const int32_t src_kv_split_size) {
  const int32_t src_world_size = static_cast<int32_t>(cluster_ids.size());
  if (src_dp_size <= 0 || src_kv_split_size <= 0 || src_world_size <= 0 ||
      src_world_size % src_dp_size != 0 ||
      (src_world_size / src_dp_size) % src_kv_split_size != 0 ||
      addrs.size() != cluster_ids.size() ||
      ports.size() != cluster_ids.size()) {
    LOG(ERROR) << "Invalid source topology for cache layout negotiation.";
    return false;
  }

  // Every D worker negotiates with all P workers. Logical shard intersection
  // determines which edges carry bytes; modulo TP routing cannot represent
  // non-integer TP changes or KV-head replication.
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    std::vector<uint64_t> target_cluster_ids;
    std::vector<std::string> target_addrs;
    std::vector<uint16_t> target_ports;
    target_cluster_ids.reserve(static_cast<size_t>(src_world_size));
    target_addrs.reserve(static_cast<size_t>(src_world_size));
    target_ports.reserve(static_cast<size_t>(src_world_size));
    for (int32_t source_rank = 0; source_rank < src_world_size; ++source_rank) {
      target_cluster_ids.emplace_back(cluster_ids[source_rank]);
      target_addrs.emplace_back(addrs[source_rank]);
      target_ports.emplace_back(ports[source_rank]);
    }

    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule(
        [this,
         promise = std::move(promise),
         worker_rank,
         target_cluster_ids = std::move(target_cluster_ids),
         target_addrs = std::move(target_addrs),
         target_ports = std::move(target_ports)]() mutable {
          promise.setValue(worker_clients_[worker_rank]->link_cluster(
              target_cluster_ids, target_addrs, target_ports));
        });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Link cluster failed.";
      return false;
    }
  }
  return true;
}

bool LLMEngine::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<uint16_t>& ports,
                               const int32_t src_dp_size,
                               const int32_t src_kv_split_size) {
  const int32_t src_world_size = static_cast<int32_t>(cluster_ids.size());
  if (src_dp_size <= 0 || src_kv_split_size <= 0 || src_world_size <= 0 ||
      src_world_size % src_dp_size != 0 ||
      (src_world_size / src_dp_size) % src_kv_split_size != 0 ||
      addrs.size() != cluster_ids.size() ||
      ports.size() != cluster_ids.size()) {
    LOG(ERROR) << "Invalid source topology for cache unlink.";
    return false;
  }

  // Symmetric to link_cluster: close every negotiated source edge.
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    std::vector<uint64_t> target_cluster_ids;
    std::vector<std::string> target_addrs;
    std::vector<uint16_t> target_ports;
    target_cluster_ids.reserve(static_cast<size_t>(src_world_size));
    target_addrs.reserve(static_cast<size_t>(src_world_size));
    target_ports.reserve(static_cast<size_t>(src_world_size));
    for (int32_t source_rank = 0; source_rank < src_world_size; ++source_rank) {
      target_cluster_ids.emplace_back(cluster_ids[source_rank]);
      target_addrs.emplace_back(addrs[source_rank]);
      target_ports.emplace_back(ports[source_rank]);
    }

    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule(
        [this,
         promise = std::move(promise),
         worker_rank,
         target_cluster_ids = std::move(target_cluster_ids),
         target_addrs = std::move(target_addrs),
         target_ports = std::move(target_ports)]() mutable {
          promise.setValue(worker_clients_[worker_rank]->unlink_cluster(
              target_cluster_ids, target_addrs, target_ports));
        });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Unlink cluster failed.";
      return false;
    }
  }
  return true;
}

bool LLMEngine::link_p2p(const std::vector<std::string>& remote_addrs) {
  if (remote_addrs.size() != worker_clients_num_) {
    LOG(ERROR) << "remote_addrs size " << remote_addrs.size()
               << " != worker_clients_num " << worker_clients_num_;
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);

  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    std::string remote_addr = remote_addrs[worker_rank];
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                remote_addr]() mutable {
      promise.setValue(worker_clients_[worker_rank]->link_p2p(remote_addr));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Link P2P failed.";
      return false;
    }
  }
  return true;
}

bool LLMEngine::unlink_p2p(const std::vector<std::string>& remote_addrs) {
  if (remote_addrs.size() != worker_clients_num_) {
    LOG(ERROR) << "remote_addrs size " << remote_addrs.size()
               << " != worker_clients_num " << worker_clients_num_;
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);

  for (size_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    std::string remote_addr = remote_addrs[worker_rank];
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                remote_addr]() mutable {
      promise.setValue(worker_clients_[worker_rank]->unlink_p2p(remote_addr));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Unlink P2P failed.";
      return false;
    }
  }
  return true;
}

ForwardOutput LLMEngine::step(BatchGroup& batch) {
  if (worker_clients_.empty()) {
    // empty worker, return
    return {};
  }
  Timer timer;
  DCHECK(dp_size_ == batch.size())
      << "Split DP batch failed with dp_size as " << dp_size_
      << " and actual batch size as " << batch.size() << ".";

  auto forward_inputs = prepare_inputs(batch);
  const bool is_graph_warmup = contains_graph_warmup(forward_inputs);
  if (eplb_controller_ != nullptr) {
    eplb_controller_->on_step_dispatched(forward_inputs, is_graph_warmup);
  }
  DCHECK(dp_size_ == forward_inputs.size())
      << "The processed forward inputs size " << forward_inputs.size()
      << " is not equal to dp size " << dp_size_ << ".";

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);

  // Engine sends full global tokens; model-side CP shards inside the worker.
  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    const int32_t dp_rank = worker_rank / dp_local_size_;
    futures.emplace_back(worker_clients_[worker_rank]->step_remote_async(
        forward_inputs[dp_rank]));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  DCHECK_EQ(dp_size_, worker_clients_num_ / dp_local_size_);
  // Every worker must have produced a value before EPLB consumes all results
  // and before promotions are committed to the shared prefix cache below; a
  // failed worker must not reach those non-reversible steps.
  for (uint32_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    if (!results[worker_rank].hasValue() || !results[worker_rank].value()) {
      LOG(FATAL) << "Failed to execute model, result has no value";
    }
  }

  if (eplb_controller_ != nullptr && !options_.enable_schedule_overlap() &&
      !is_graph_warmup) {
    eplb_controller_->on_step_completed(results, is_graph_warmup);
  }

  size_t dp_rank = 0;
  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_size_) {
    auto result = results[worker_rank].value();
    // if src_seq_idxes is not empty, skip sample output processing and
    // process beam search output instead
    if (result.value().src_seq_idxes.size() == 0) {
      // set second input param enable_schedule_overlap to false,
      // if it's not enabled, process_sample_output will append the real
      // token, if it's enabled, this false here will append the fake token in
      // process_sample_output
      batch[dp_rank].process_sample_output(result.value(), false);
    } else {
      batch[dp_rank].process_beam_search_output(result.value(), false);
    }
    // Keep Batch::sequences_ aligned with SequencesGroup after beam updates.
    batch[dp_rank].refresh_sequences_from_groups();
    ++dp_rank;
  }

  COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());
  return {};
}

void LLMEngine::update_last_step_result(BatchGroup& last_batch) {
  const bool is_graph_warmup = contains_graph_warmup(last_batch);
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);
  std::vector<RawForwardOutput> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size_);

  // NOTE: We only need to get the output from the driver worker,
  // cause the output on other workers is the same as that on driver.
  // Under data parallelism (DP), we need to get dp_size outputs.
  // The `stride` means the workers num we can skip.
  // One retrievable result per DP group lives on its driver worker
  // (dp_driver_: rank % (tp_size*cp_size) == 0), spaced dp_local_size_
  // (= tp_size*cp_size) apart. dp_local_tp_size_ divides by cp_size, so
  // under cp_size>1 it lands on cp_rank=1 non-driver workers whose
  // get_last_step_result() blocks forever on cv_.wait(is_recorded_).
  uint32_t stride = dp_local_size_;
  // If EPLB is enabled, we need to get results from all workers,
  // because the experts on each worker are different,
  // and the controller needs the token load of every expert shard.
  // so we can not skip any worker.
  // Each pipeline worker retains its completed Slot until the result is taken,
  // including non-drivers. Retire every worker's Slot before the next submit.
  if ((eplb_controller_ != nullptr &&
       eplb_controller_->requires_all_worker_results()) ||
      options_.enable_task_pipeline()) {
    stride = 1;
  }

  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += stride) {
    futures.emplace_back(
        worker_clients_[worker_rank]->get_last_step_result_async());
  }
  // wait for the all future to complete
  auto last_step_results = folly::collectAll(futures).get();
  for (const auto& result : last_step_results) {
    CHECK(result.hasValue() && result.value().has_value())
        << "Failed to get last step results, result has no value";
  }

  if (eplb_controller_ != nullptr) {
    eplb_controller_->on_step_completed(last_step_results, is_graph_warmup);
  }

  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_size_) {
    auto& result = last_step_results[worker_rank / stride].value();
    raw_forward_outputs.emplace_back(std::move(result.value()));
  }

  for (auto i = 0; i < last_batch.size(); i++) {
    last_batch[i].process_sample_output(raw_forward_outputs[i],
                                        options_.enable_schedule_overlap());
    // Keep Batch::sequences_ aligned with SequencesGroup after beam updates.
    last_batch[i].refresh_sequences_from_groups();
  }
}

std::vector<int64_t> LLMEngine::get_active_activation_memory() const {
  // call worker to get active activation memory
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(worker_clients_num_);
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

void LLMEngine::setup_workers(const runtime::Options& options) {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ =
        std::make_shared<DistributedWorkerManager>(options);
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

std::vector<LlmForwardInput> LLMEngine::prepare_inputs(BatchGroup& batch) {
  std::vector<LlmForwardInput> batched_inputs;
  batched_inputs.reserve(dp_size_ * cp_size_);
  // some dp related variables
  std::vector<int32_t> dp_global_token_nums(dp_size_);
  std::vector<int32_t> dp_global_sequence_nums(dp_size_);
  std::vector<int32_t> dp_global_kv_max_seq_lens(dp_size_);
  std::vector<int32_t> dp_is_decode(dp_size_, 0);
  // when enable dp, we need to check the forward type of each batch
  // and set the empty forward type of each batch to the same value as the first
  // batch
  BatchForwardType batch_forward_type;
  bool has_non_empty_batch = false;
  bool all_non_empty_batches_are_decode = true;

  // build model input for every single micro batch
  for (auto dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    // Linear-state saves deferred from the previous step are executed by the
    // LINEAR leaf inside allocate_for_sequence (scheduler-side), so the builder
    // below already sees the rotated live slot -- no apply step is needed here.
    batched_inputs.emplace_back(std::move(batch[dp_rank].prepare_forward_input(
        args_, threadpool_.get(), cp_size_)));
    const BatchForwardType& current_batch_forward_type =
        batched_inputs[dp_rank].input_params.meta.batch_forward_type;
    dp_global_token_nums[dp_rank] =
        static_cast<int32_t>(batched_inputs[dp_rank].host_token_ids().numel());
    dp_global_sequence_nums[dp_rank] =
        batched_inputs[dp_rank].input_params.meta.num_sequences;
    dp_global_kv_max_seq_lens[dp_rank] =
        batched_inputs[dp_rank].input_params.meta.kv_max_seq_len;
    if (util::is_deepseek_v4_model_type(args_.model_type())) {
      const int64_t actual_scheduled_tokens = static_cast<int64_t>(
          batched_inputs[dp_rank].host_token_ids().numel());
      const int64_t max_tokens_per_batch =
          static_cast<int64_t>(options_.max_tokens_per_batch());
      CHECK_LE(actual_scheduled_tokens, max_tokens_per_batch)
          << "DSV4 actual scheduled tokens exceed max_tokens_per_batch used "
             "for SWA cache allocation. This can make the shared SWA burst "
             "pool smaller than the block/table consumer needs and may cause "
             "SWA KV rows to be overwritten or read from the wrong position. "
             "Please increase --max_tokens_per_batch, reduce scheduler token "
             "load, or check chunked-prefill padding. Details: dp_rank="
          << dp_rank << ", actual_scheduled_tokens=" << actual_scheduled_tokens
          << ", max_tokens_per_batch=" << max_tokens_per_batch
          << ", q_max_seq_len="
          << batched_inputs[dp_rank].input_params.meta.q_max_seq_len
          << ", kv_max_seq_len="
          << batched_inputs[dp_rank].input_params.meta.kv_max_seq_len
          << ", batch_forward_type=" << current_batch_forward_type.to_string();
    }
    if (batch_forward_type.is_empty() &&
        !current_batch_forward_type.is_empty()) {
      batch_forward_type = current_batch_forward_type;
    }
    if (!current_batch_forward_type.is_empty()) {
      has_non_empty_batch = true;
      all_non_empty_batches_are_decode = all_non_empty_batches_are_decode &&
                                         current_batch_forward_type.is_decode();
    }
    dp_is_decode[dp_rank] =
        current_batch_forward_type.is_decode() &&
        batched_inputs[dp_rank].input_params.meta.q_max_seq_len == 1;

    const LlmEmbeddingInput& embedding =
        batched_inputs[dp_rank].input_params.embedding;
    if (dp_batch_embedding_ids_[dp_rank] != embedding.embedding_ids ||
        dp_batch_request_ids_[dp_rank] != embedding.request_ids) {
      dp_batch_embedding_ids_[dp_rank] = embedding.embedding_ids;
      dp_batch_request_ids_[dp_rank] = embedding.request_ids;
      ++dp_batch_generations_[dp_rank];
    }
  }

  if (eplb_controller_ != nullptr) {
    eplb_controller_->annotate_inputs(
        batched_inputs,
        dp_global_token_nums,
        /*allow_eplb_command=*/has_non_empty_batch &&
            all_non_empty_batches_are_decode &&
            !contains_graph_warmup(batched_inputs));
  }

  // Empty DP ranks inherit decode below and use fake inputs in WorkerImpl.
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      batch_forward_type.is_decode()) {
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      if (batched_inputs[dp_rank]
              .input_params.meta.batch_forward_type.is_empty() &&
          dp_global_token_nums[dp_rank] == 0) {
        dp_is_decode[dp_rank] = 1;
      }
    }
  }

  // Empty ranks must participate in the same warmup capture as active ranks.
  const bool is_graph_warmup = contains_graph_warmup(batched_inputs);
  // update dp_global_token_nums and batch_forward_type
  for (auto dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    batched_inputs[dp_rank].input_params.meta.is_graph_warmup = is_graph_warmup;
    batched_inputs[dp_rank].input_params.parallel.dp_global_token_nums =
        dp_global_token_nums;
    batched_inputs[dp_rank].input_params.parallel.dp_global_sequence_nums =
        dp_global_sequence_nums;
    batched_inputs[dp_rank].input_params.parallel.raw_dp_global_token_nums =
        dp_global_token_nums;
    batched_inputs[dp_rank].input_params.parallel.dp_global_batch_generations =
        dp_batch_generations_;
    batched_inputs[dp_rank].input_params.parallel.dp_global_kv_max_seq_lens =
        dp_global_kv_max_seq_lens;
    batched_inputs[dp_rank].input_params.parallel.dp_is_decode = dp_is_decode;
    if (batched_inputs[dp_rank]
            .input_params.meta.batch_forward_type.is_empty()) {
      batched_inputs[dp_rank].input_params.meta.batch_forward_type =
          batch_forward_type;
    }
  }

  return batched_inputs;
}

bool LLMEngine::sleep(MasterStatus master_status) {
  return xtensor_sleep(master_status);
}

bool LLMEngine::xtensor_sleep(MasterStatus master_status) {
  // sleep/wakeup/fork_master (xtensor path) requires
  // ::xllm::KVCacheConfig::get_instance().enable_xtensor()
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    LOG(WARNING) << "sleep requires --enable_xtensor=true";
    return false;
  }

  LOG(INFO) << "Starting to sleep. Worker clients count: "
            << worker_clients_num_;
  if (worker_clients_.empty()) {
    LOG(ERROR) << "No worker clients available to sleep.";
    return false;
  }

  // Put the model to sleep in PageAllocator (xtensor path).
  // This releases both weight pages and KV cache pages.
  const std::string& model_id = options_.model_id();
  auto& page_allocator = PageAllocator::get_instance();
  if (!page_allocator.sleep_model(model_id)) {
    LOG(ERROR) << "PageAllocator sleep_model failed, aborting sleep flow";
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->sleep_async(master_status));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Sleep failed.";
      return false;
    }
  }
  return true;
}

bool LLMEngine::start_profile() {
  std::lock_guard<std::mutex> lock(profile_mutex_);
  LOG(INFO) << "Starting profiler on " << worker_clients_num_ << " worker(s).";
  if (!profile_workers(/*is_start=*/true)) {
    // A partially successful broadcast must not leave other workers tracing.
    if (!profile_workers(/*is_start=*/false)) {
      LOG(ERROR) << "Failed to clean up profiling on some workers. "
                    "Retry /stop_profile.";
    }
    return false;
  }
  return true;
}

bool LLMEngine::stop_profile() {
  std::lock_guard<std::mutex> lock(profile_mutex_);
  LOG(INFO) << "Stopping profiler on " << worker_clients_num_ << " worker(s).";
  return profile_workers(/*is_start=*/false);
}

bool LLMEngine::profile_workers(bool is_start) {
  if (worker_clients_.empty()) {
    LOG(ERROR) << "No worker clients available for profiling.";
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.emplace_back(is_start ? worker->start_profile_async()
                                  : worker->stop_profile_async());
  }

  auto results = folly::collectAll(futures).get();
  bool success = true;
  for (const auto& result : results) {
    if (result.hasException()) {
      LOG(ERROR) << "Profiling worker RPC failed: "
                 << result.exception().what();
      success = false;
      continue;
    }
    if (!result.value()) {
      LOG(ERROR) << "Profiling failed on a worker (start=" << is_start << ").";
      success = false;
    }
  }

  return success;
}

bool LLMEngine::xtensor_wakeup(const WakeupOptions& options) {
  // sleep/wakeup/fork_master (xtensor path) requires
  // ::xllm::KVCacheConfig::get_instance().enable_xtensor()
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    LOG(WARNING) << "wakeup requires --enable_xtensor=true";
    return false;
  }

  LOG(INFO) << "Starting to wakeup. Worker clients count: "
            << worker_clients_num_;
  if (worker_clients_.empty()) {
    LOG(ERROR) << "No worker clients available to wakeup.";
    return false;
  }

  // Wake up the model in PageAllocator (xtensor path).
  // This re-allocates both KV cache pages and weight pages.
  const std::string& model_id = options_.model_id();
  auto& page_allocator = PageAllocator::get_instance();
  if (!page_allocator.wakeup_model(model_id)) {
    LOG(ERROR) << "PageAllocator wakeup_model failed, aborting wakeup flow";
    return false;
  }

  LOG(INFO) << "Waking up LLM engine, remote_addrs.size()="
            << options.remote_addrs.size();
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);

  if (!options.remote_addrs.empty() &&
      options.remote_addrs.size() == worker_clients_num_) {
    // P2P mode with TP: each worker pulls only from its corresponding source
    for (size_t i = 0; i < worker_clients_num_; ++i) {
      WakeupOptions per_worker_options;
      per_worker_options.master_status = options.master_status;
      per_worker_options.remote_addrs = {options.remote_addrs[i]};
      if (i < options.src_weight_segments.size()) {
        per_worker_options.src_weight_segments = {
            options.src_weight_segments[i]};
      }
      futures.push_back(worker_clients_[i]->wakeup_async(per_worker_options));
    }
  } else {
    // H2D mode or non-TP: pass options as-is
    for (auto& worker : worker_clients_) {
      futures.push_back(worker->wakeup_async(options));
    }
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Wakeup failed.";
      return false;
    }
  }
  LOG(INFO) << "Wakeup finished for LLM engine.";
  return true;
}

bool LLMEngine::wakeup(const WakeupOptions& options) {
  return xtensor_wakeup(options);
}

bool LLMEngine::get_xtensor_offsets_for_blocks(
    int32_t dp_rank,
    const std::vector<int32_t>& block_ids,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) {
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    return false;
  }

  const std::string& model_id = options_.model_id();

  // Calculate block size in bytes: block_size * slot_size
  // slot_size is stored in kv_cache_manager (BlockManagerPool)
  auto* block_manager = block_manager_pool();
  if (!block_manager) {
    LOG(ERROR) << "BlockManagerPool not available";
    return false;
  }

  // Note: Currently, xtensor only supports the traditional attention mechanism,
  // meaning both K and V must be present and have identical shapes.
  uint64_t block_size_bytes =
      static_cast<uint64_t>(block_manager->options().slot_size()) *
      options_.block_size() / 2;

  // Use RPC to call worker in the specified DP group
  auto& allocator = XTensorAllocator::get_instance();
  bool success = allocator.get_xtensor_offsets(
      dp_rank, model_id, block_ids, block_size_bytes, layer_offsets);

  if (!success) {
    LOG(ERROR) << "get_xtensor_offsets_for_blocks via RPC failed for dp_rank="
               << dp_rank << ", model_id=" << model_id;
    return false;
  }

  VLOG(1) << "get_xtensor_offsets_for_blocks: dp_rank=" << dp_rank
          << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();
  return true;
}

}  // namespace xllm

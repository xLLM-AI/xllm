/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/xllm_build_info.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "dit_master.h"
#if defined(USE_NPU)
#include "framework/parallel_state/npu_rank_table_env.h"
#endif
#include "core/platform/device_name_utils.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "llm_engine.h"
#include "llm_master.h"
#include "models/model_registry.h"
#include "platform/platform.h"
#include "rec_engine.h"
#include "rec_master.h"
#include "speculative_engine.h"
#include "util/model_config_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"
#include "vlm_engine.h"
#include "vlm_master.h"

namespace brpc {
DECLARE_bool(graceful_quit_on_sigterm);
DECLARE_bool(graceful_quit_on_sighup);
}  // namespace brpc

namespace xllm {
namespace {

std::optional<std::string> validate_model_cp(const Options& options,
                                             EngineType engine_type,
                                             const std::string& model_type,
                                             int32_t global_world_size) {
  if (options.cp_size() < 1) {
    return "cp_size must be greater than or equal to 1";
  }
  const bool use_model_partition =
      options.cp_size() > 1 && Platform::uses_model_cp_partition();
  if (!use_model_partition) {
    return std::nullopt;
  }
  if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
    return "MLU CP supports only LLM text generation";
  }
  if (options.task_type() != "generate") {
    return "MLU CP supports only the generate task";
  }
  if (engine_type == EngineType::SSM &&
      options.speculative_algorithm() != "Suffix") {
    return "Current MLU model-side CP does not support MTPWorkerImpl-based "
           "speculative algorithms such as MTP or Eagle3; disable CP, disable "
           "the speculative algorithm, or wait for MLU worker-side CP";
  }
  if (model_type != "deepseek_v32" && model_type != "glm_moe_dsa") {
    return "MLU CP does not support model_type=" + model_type;
  }
  if (options.instance_role() != InstanceRole::DEFAULT &&
      options.instance_role() != InstanceRole::PREFILL) {
    return "MLU CP supports only DEFAULT or PREFILL roles";
  }
  if (options.dp_size() != 1) {
    return "MLU CP requires dp_size == 1";
  }
  if (options.cp_size() != global_world_size) {
    return "MLU CP requires cp_size == global world size";
  }
  if (ParallelConfig::get_instance().kv_split_size() != 1) {
    return "MLU CP requires kv_split_size == 1";
  }
  if (options.ep_size() != 1 && options.ep_size() != global_world_size) {
    return "MLU CP requires ep_size == 1 or global world size";
  }
  return std::nullopt;
}

void print_startup_banner(const std::filesystem::path& model_path,
                          const std::string& backend,
                          int32_t node_rank) {
  if (node_rank != 0) {
    return;
  }

  constexpr std::string_view kAnsiRed = "\033[31m";
  constexpr std::string_view kAnsiReset = "\033[0m";
  const bool use_color = ::isatty(::fileno(stderr));

  std::array<std::string_view, 4> x_logo = {
      "      ", "▀█▄ ▀ ", "  █▶  ", "▄█▀ ▄ "};
  std::array<std::string_view, 4> llm_logo = {"█     █     █▄   ▄█",
                                              "█     █     █ ▀▄▀ █",
                                              "█     █     █     █",
                                              "█▄▄▄▄ █▄▄▄▄ █     █"};

  LOG(INFO) << "";
  LOG(INFO) << x_logo[0] << llm_logo[0];
  if (use_color) {
    LOG(INFO) << kAnsiRed << x_logo[1] << kAnsiReset << llm_logo[1]
              << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << kAnsiRed << x_logo[2] << kAnsiReset << llm_logo[2]
              << "  model   " << model_path.string();
    LOG(INFO) << kAnsiRed << x_logo[3] << kAnsiReset << llm_logo[3]
              << "  backend " << backend;
  } else {
    LOG(INFO) << x_logo[1] << llm_logo[1] << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << x_logo[2] << llm_logo[2] << "  model   "
              << model_path.string();
    LOG(INFO) << x_logo[3] << llm_logo[3] << "  backend " << backend;
  }
  LOG(INFO) << "";
}

}  // namespace

namespace {

#if defined(USE_NPU)
void validate_rank_tablefile_backend() {
  const EPLBConfig& eplb_config = EPLBConfig::get_instance();
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (!eplb_config.rank_tablefile().empty() &&
      parallel_config.communication_backend() != "hccl") {
    LOG(FATAL) << "--rank_tablefile requires --communication_backend=hccl, "
               << "but got --communication_backend="
               << parallel_config.communication_backend();
  }
}

void resolve_npu_kernel_backend_for_options(Options* options) {
  CHECK(options != nullptr) << "options must not be null";
  if (options->backend() == "dit") {
    return;
  }

  const std::string model_type =
      util::get_model_type(options->model_path(), options->backend());
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration(model_type,
                                  options->npu_kernel_backend(),
                                  &effective_backend,
                                  &resolved_name,
                                  &error_message)) {
    LOG(FATAL) << error_message;
  }

  options->npu_kernel_backend(effective_backend);
  KernelConfig::get_instance().npu_kernel_backend(effective_backend);
  LOG(INFO) << "Resolved npu_kernel_backend=" << effective_backend
            << " for model_type=" << model_type;
}
#endif

}  // namespace

Master::Master(const Options& options, EngineType type)
    : options_(options),
      engine_type_(type),
      master_status_(options.master_status()) {
  const auto model_path =
      std::filesystem::path(options_.model_path()).lexically_normal();
  // Multi-process serving runs one worker per process. Select one runtime
  // logical device from the process-visible devices while keeping node_rank as
  // the global distributed identity.
  const int32_t visible_device_count = Platform::device_count();
  const int32_t device_idx = DeviceNameUtils::get_device_idx(
      options_.node_rank(), options_.nnodes(), visible_device_count);
  const auto visible_devices = DeviceNameUtils::parse_devices("auto");
  const std::vector<torch::Device> devices = {visible_devices[device_idx]};
  // World size is the node count (one worker per process).
  const int32_t global_world_size = options_.nnodes();
  std::string cp_model_type;
  if (options_.cp_size() > 1 && Platform::uses_model_cp_partition()) {
    cp_model_type = util::get_model_type(model_path, options_.backend());
  }
  const std::optional<std::string> cp_error =
      validate_model_cp(options_, type, cp_model_type, global_world_size);
  CHECK(!cp_error.has_value()) << cp_error.value();
  if (options_.enable_prefix_cache() && options_.backend() == "llm") {
    const std::string model_type = util::get_model_type(model_path);
    if (util::is_deepseek_v4_model_type(model_type)) {
      LOG(WARNING) << model_type
                   << " does not support prefix cache with "
                      "CompositeBlockManager yet, fallback to "
                      "enable_prefix_cache=false";
      options_.enable_prefix_cache(false);
      KVCacheConfig::get_instance().enable_prefix_cache(false);
    }
  }
  options_.enable_mla(util::should_enable_mla(model_path, options_.backend()));
  print_startup_banner(model_path, options_.backend(), options_.node_rank());
  LOG(INFO) << "Master init options: " << options_.to_string();
  ParallelConfig::get_instance().cp_size(options_.cp_size());
  const char* cp_partition_stage =
      options_.cp_size() <= 1
          ? "disabled"
          : (Platform::uses_model_cp_partition() ? "model" : "worker");
  LOG(INFO) << "Resolved CP config: cp_size=" << options_.cp_size()
            << ", world_size=" << global_world_size
            << ", dp_size=" << options_.dp_size()
            << ", ep_size=" << options_.ep_size()
            << ", cp_partition_stage=" << cp_partition_stage
            << ", instance_role=" << options_.instance_role().to_string();

  // Allow brpc receive SIGTREM and SIGINT signal.
  brpc::FLAGS_graceful_quit_on_sigterm = true;
  brpc::FLAGS_graceful_quit_on_sighup = true;

#if defined(USE_NPU)
  EPLBConfig& eplb_config = EPLBConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (options.rank_tablefile().has_value()) {
    eplb_config.rank_tablefile(options.rank_tablefile().value());
  }
  if (options.communication_backend().has_value()) {
    parallel_config.communication_backend(
        options.communication_backend().value());
  }
  validate_rank_tablefile_backend();
  parallel_state::sync_torch_npu_rank_table_file_env(
      eplb_config.rank_tablefile());
  if (options.expert_parallel_degree().has_value()) {
    eplb_config.expert_parallel_degree(
        options.expert_parallel_degree().value());
  }
  if (options.enable_eplb().has_value()) {
    eplb_config.enable_eplb(options.enable_eplb().value());
  }
  if (options.redundant_experts_num().has_value()) {
    eplb_config.redundant_experts_num(options.redundant_experts_num().value());
  }
  if (options.eplb_update_interval().has_value()) {
    eplb_config.eplb_update_interval(options.eplb_update_interval().value());
  }
  if (options.eplb_update_threshold().has_value()) {
    eplb_config.eplb_update_threshold(options.eplb_update_threshold().value());
  }
  resolve_npu_kernel_backend_for_options(&options_);
#endif
  ParallelConfig::get_instance().enable_multi_stream_parallel(
      options.enable_multi_stream_parallel() && (options.nnodes() > 1));
  if (ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    LOG(FATAL)
        << "Multi-stream parallel is refactoring now, will be supported later.";
  }
  // construct engine
  LOG(INFO) << "Creating engine with devices: "
            << DeviceNameUtils::to_string(devices);

  if (options_.enable_disagg_pd()) {
    // Enable service routing in disagg pd mode
    options_.enable_service_routing(true);
    if (options_.instance_role() == InstanceRole::PREFILL) {
      // Disable schedule overlap for prefill instance in disagg pd mode
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for prefill instance "
                      "in disagg pd mode.";
    }
  }

  if (type == EngineType::VLM) {
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options.backend())
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options.enable_prefix_cache())
        .max_encoder_cache_size(options.max_encoder_cache_size())
        .max_linear_state_cache_slots(options.max_linear_state_cache_slots())
        .task_type(options.task_type())
        .enable_mla(options_.enable_mla())
        .cp_size(options_.cp_size())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill());

    auto engine = std::make_unique<VLMEngine>(eng_options);
    engine_ = std::move(engine);
  } else if (type == EngineType::SSM) {
    // create a speculative engine if draft model path is provided
    const std::string draft_model_path =
        options_.draft_model_path().value_or("");
    const bool use_suffix_spec = options_.speculative_algorithm() == "Suffix";
    CHECK(use_suffix_spec || !draft_model_path.empty())
        << "draft model path is required unless --speculative_algorithm=Suffix";
    // Draft model shares the same devices as the target model.
    const auto& draft_devices = devices;
    LOG(INFO) << "Using draft devices: "
              << DeviceNameUtils::to_string(draft_devices);
    runtime::Options spec_options;
    spec_options.model_path(options_.model_path())
        .draft_model_path(draft_model_path)
        .devices(devices)
        .draft_devices(draft_devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .num_speculative_tokens(options_.num_speculative_tokens())
        .speculative_algorithm(options_.speculative_algorithm())
        .enable_mtp_draft_body_tp1(options_.enable_mtp_draft_body_tp1())
        .speculative_suffix_cache_max_depth(
            options_.speculative_suffix_cache_max_depth())
        .speculative_suffix_max_spec_factor(
            options_.speculative_suffix_max_spec_factor())
        .speculative_suffix_max_spec_offset(
            options_.speculative_suffix_max_spec_offset())
        .speculative_suffix_min_token_prob(
            options_.speculative_suffix_min_token_prob())
        .speculative_suffix_max_cached_requests(
            options_.speculative_suffix_max_cached_requests())
        .speculative_suffix_use_tree_spec(
            options_.speculative_suffix_use_tree_spec())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode());

    if (use_suffix_spec) {
      engine_ = std::make_unique<SuffixSpeculativeEngine>(spec_options);
    } else {
      engine_ = std::make_unique<SpeculativeEngine>(spec_options);
    }
  } else if (type == EngineType::LLM) {
    if (options_.task_type() == "embed" || options.task_type() == "mm_embed") {
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for embedding model, "
                      "avoiding performance degradation.";
    }
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .host_blocks_factor(options_.host_blocks_factor())
        .enable_kvcache_store(options_.enable_kvcache_store())
        .store_protocol(options_.store_protocol())
        .store_master_server_address(options_.store_master_server_address())
        .store_metadata_server(options_.store_metadata_server())
        .store_local_hostname(options_.store_local_hostname())
        .prefetch_batch_size(options_.prefetch_batch_size())
        .layers_wise_copy_batchs(options_.layers_wise_copy_batchs())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .server_idx(options_.server_idx())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .kv_cache_dtype(options_.kv_cache_dtype())
        .enable_sleep_mode(options_.enable_sleep_mode())
        .model_id(options_.model_id());

    engine_ = std::make_unique<LLMEngine>(eng_options);
  } else if (type == EngineType::REC) {
    options_.enable_schedule_overlap(false);
    LOG(WARNING) << "Force to disable schedule overlap for REC model, not "
                    "supported yet.";
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .task_type(options_.task_type())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_mla(options_.enable_mla())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .is_local(options_.is_local())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .cp_size(options_.cp_size())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .beam_width(options_.beam_width())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());

    engine_ = std::make_unique<RecEngine>(eng_options);
  } else if (type == EngineType::DIT) {
    // construct dit engine
    runtime::Options eng_options;
    eng_options.model_path(options.model_path())
        .model_id(options.model_id())
        .devices(devices)
        .backend(options.backend())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .max_memory_utilization(options_.max_memory_utilization())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .task_type(options_.task_type())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .node_rank(options_.node_rank())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .tp_size(options_.tp_size())
        .sp_size(options_.sp_size())
        .cfg_size(options_.cfg_size())
        .vae_size(options_.vae_size());

    auto dit_engine = std::make_unique<DiTEngine>(eng_options);
    engine_ = std::move(dit_engine);
  } else {
    LOG(WARNING) << "Not supported llm engine type: "
                 << static_cast<size_t>(type);
  }
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  } else if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  } else if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  } else if (backend == "rec") {
    LOG(INFO) << "creating rec master";
    return std::make_unique<RecMaster>(options);
  } else {
    LOG(FATAL) << "Failed to create master, backend is" << backend;
    return nullptr;
  }
}

std::unique_ptr<Master> fork_master(Master* master, const Options& options) {
  // sleep/wakeup/fork_master requires --enable_xtensor=true
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    LOG(WARNING) << "fork_master requires xtensor to be enabled";
    return nullptr;
  }

  static uint64_t server_idx = 1;
  CHECK(master != nullptr);

  Options new_options = master->options();

  if (!options.model_id().empty()) {
    new_options.model_id() = options.model_id();
  }
  if (!options.model_path().empty()) {
    new_options.model_path() = options.model_path();
  }
  new_options.master_node_addr() = options.master_node_addr();
  new_options.server_idx() = server_idx++;
  new_options.master_status() = options.master_status();
  // Set nnodes and dp_size from fork request (tp_size * dp_size = nnodes)
  if (options.nnodes() > 0 && new_options.nnodes() >= options.nnodes()) {
    new_options.nnodes() = options.nnodes();
  }
  if (options.dp_size() > 0 && new_options.dp_size() >= options.nnodes()) {
    new_options.dp_size() = options.dp_size();
  }
  std::unique_ptr<Master> new_master;
  if (new_options.node_rank() != 0) {
    new_master = std::make_unique<LLMAssistantMaster>(new_options);
  } else {
    new_master = create_master(new_options.backend(), new_options);
  }
  new_master->run();

  return new_master;
}
}  // namespace xllm

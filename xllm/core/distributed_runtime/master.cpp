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

#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>

#include "common/types.h"
#include "core/common/xllm_build_info.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#if defined(USE_NPU)
#include "framework/parallel_state/npu_rank_table_env.h"
#endif
#include "core/platform/device_name_utils.h"
#include "platform/platform.h"

namespace brpc {
DECLARE_bool(graceful_quit_on_sigterm);
DECLARE_bool(graceful_quit_on_sighup);
}  // namespace brpc

namespace xllm {
namespace {

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
#endif

}  // namespace

Master::Master(const Options& options) : options_(options) {
  // Multi-process serving runs one worker per process. Select one runtime
  // logical device from the process-visible devices while keeping node_rank as
  // the global distributed identity.
  const int32_t visible_device_count = Platform::device_count();
  const int32_t device_idx = DeviceNameUtils::get_device_idx(
      options_.node_rank(), options_.nnodes(), visible_device_count);
  const auto visible_devices = DeviceNameUtils::parse_devices("auto");
  devices_ = {visible_devices[device_idx]};
  print_startup_banner(
      std::filesystem::path(options_.model_path()).lexically_normal(),
      options_.backend(),
      options_.node_rank());
  LOG(INFO) << "Master init options: " << options_.to_string();
  ParallelConfig::get_instance().cp_size(options_.cp_size());
  // cp_size <= 1 -> "disabled", otherwise "model" (model-side CP).
  const char* cp_sharding_stage =
      options_.cp_size() <= 1 ? "disabled" : "model";
  LOG(INFO) << "Resolved CP config: cp_size=" << options_.cp_size()
            << ", world_size=" << options_.nnodes()
            << ", dp_size=" << options_.dp_size()
            << ", ep_size=" << options_.ep_size()
            << ", cp_sharding_stage=" << cp_sharding_stage
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
  if (options.eplb_min_peak_load_improvement().has_value()) {
    eplb_config.eplb_min_peak_load_improvement(
        options.eplb_min_peak_load_improvement().value());
  }
#endif

  ParallelConfig::get_instance().enable_multi_stream_parallel(
      options.enable_multi_stream_parallel() && (options.nnodes() > 1));
  if (ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    LOG(FATAL)
        << "Multi-stream parallel is refactoring now, will be supported later.";
  }
  LOG(INFO) << "Using devices: " << DeviceNameUtils::to_string(devices_);

  if (options_.task_type() == "mm_embed") {
    options_.enable_chunked_prefill(false);
    options_.enable_prefix_cache(false);
    SchedulerConfig::get_instance().enable_chunked_prefill(false);
    KVCacheConfig::get_instance().enable_prefix_cache(false);
    LOG(WARNING) << "Disabling chunked prefill and prefix cache for "
                    "task=mm_embed to process all multimodal inputs.";
  }

  if (!is_leader()) {
    const std::string master_node_addr =
        options_.master_node_addr().value_or("");
    if (master_node_addr.empty()) {
      LOG(FATAL) << "Multi-node serving requires --master_node_addr, "
                    "current value is empty.";
    }
  }
}

std::atomic<bool> Master::idle_running_{false};

void Master::handle_shutdown_signal(int /*signum*/) {
  idle_running_.store(false, std::memory_order_relaxed);
}

Master::~Master() {
  idle_running_.store(false, std::memory_order_relaxed);
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::wait() {
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::run() {
  idle_running_.store(true, std::memory_order_relaxed);
  signal(SIGINT, Master::handle_shutdown_signal);
  signal(SIGTERM, Master::handle_shutdown_signal);

  idle_thread_ = std::thread([]() {
    while (idle_running_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

}  // namespace xllm

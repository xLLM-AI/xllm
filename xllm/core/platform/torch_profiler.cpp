/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "platform/torch_profiler.h"

#include <glog/logging.h>
#include <torch/csrc/autograd/profiler.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <set>

namespace xllm {
namespace {

namespace tp = torch::profiler::impl;
namespace ap = torch::autograd::profiler;

// Build the destination path for a worker's Chrome trace. Mirrors the
// ".pt.trace.json" suffix produced by torch.profiler's tensorboard handler so
// the file is recognized by the usual viewers (Perfetto, chrome://tracing,
// TensorBoard).
std::string make_trace_path(const std::string& profile_dir, int32_t rank) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ts =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const std::string file_name = "xllm_rank" + std::to_string(rank) + "_" +
                                std::to_string(::getpid()) + "_" +
                                std::to_string(ts) + ".pt.trace.json";
  if (profile_dir.empty()) {
    return file_name;
  }
  return (std::filesystem::path(profile_dir) / file_name).string();
}

}  // namespace

TorchProfiler& TorchProfiler::get_instance() {
  static TorchProfiler instance;
  return instance;
}

TorchProfiler::~TorchProfiler() {
  if (running_) {
    // Best-effort: drop the trace rather than write it during teardown.
    try {
      ap::disableProfiler();
    } catch (const std::exception& e) {
      LOG(WARNING) << "disableProfiler during shutdown failed: " << e.what();
    }
    running_ = false;
  }
}

bool TorchProfiler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    LOG(WARNING)
        << "Torch profiler is already running, ignoring start request.";
    return true;
  }

  try {
    const std::set<tp::ActivityType> activities = {tp::ActivityType::CPU,
                                                   tp::ActivityType::CUDA};
    const tp::ProfilerConfig config(tp::ProfilerState::KINETO,
                                    /*report_input_shapes=*/false,
                                    /*profile_memory=*/false,
                                    /*with_stack=*/false,
                                    /*with_flops=*/false,
                                    /*with_modules=*/false);
    ap::prepareProfiler(config, activities);
    ap::enableProfiler(config, activities);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to start torch profiler: " << e.what();
    return false;
  }

  running_ = true;
  LOG(INFO) << "Torch profiler started (in-process Kineto). The trace will be "
               "written on /stop_profile; no external profiler is required.";
  return true;
}

bool TorchProfiler::stop(const std::string& profile_dir, int32_t rank) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    LOG(WARNING) << "Torch profiler is not running, ignoring stop request.";
    return true;
  }
  running_ = false;

  const std::string path = make_trace_path(profile_dir, rank);
  try {
    if (!profile_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(profile_dir, ec);
      if (ec) {
        LOG(WARNING) << "Failed to create profile dir '" << profile_dir
                     << "': " << ec.message();
      }
    }
    auto result = ap::disableProfiler();
    if (result == nullptr) {
      LOG(ERROR) << "disableProfiler returned no result; no trace written.";
      return false;
    }
    result->save(path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to stop torch profiler / save trace: " << e.what();
    return false;
  }

  LOG(INFO) << "Torch profiler stopped. Trace written to: "
            << std::filesystem::absolute(path).string();
  return true;
}

bool TorchProfiler::is_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

}  // namespace xllm

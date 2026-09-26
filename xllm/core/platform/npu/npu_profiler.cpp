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

#include "core/platform/npu/npu_profiler.h"

#include <glog/logging.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>

namespace xllm {
namespace {

constexpr uint64_t kProfilingActivities =
    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_HCCL_TRACE;

std::string create_session_dir(const std::string& profile_dir) {
  std::error_code error;
  const auto root =
      std::filesystem::absolute(profile_dir.empty() ? "." : profile_dir, error);
  if (error) {
    LOG(ERROR) << "Invalid NPU profile directory: " << error.message();
    return {};
  }
  std::filesystem::create_directories(root, error);
  if (error) {
    LOG(ERROR) << "Cannot create NPU profile directory: " << error.message();
    return {};
  }
  // mkdtemp also checks writability and avoids collisions across hosts,
  // processes, and repeated capture windows sharing the same directory.
  std::string path =
      (root / ("xllm_npu_" + std::to_string(getpid()) + "_XXXXXX")).string();
  if (mkdtemp(path.data()) == nullptr) {
    LOG(ERROR) << "Cannot create NPU profiling session: "
               << std::strerror(errno);
    return {};
  }
  return path;
}

}  // namespace

NpuProfiler& NpuProfiler::get_instance() {
  static NpuProfiler instance;
  return instance;
}

void NpuProfiler::ConfigDeleter::operator()(aclprofConfig* config) const {
  const aclError status = aclprofDestroyConfig(config);
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "aclprofDestroyConfig failed: " << status;
  }
}

bool NpuProfiler::start(const std::string& profile_dir, int32_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_id < 0) {
    LOG(ERROR) << "Invalid NPU profiling device: " << device_id;
    return false;
  }
  const auto existing = configs_.find(device_id);
  if (existing != configs_.end()) {
    return !existing->second.stopped;
  }
  if (initialized_ && configs_.empty()) {
    LOG(ERROR) << "Previous NPU profiling session has not finalized. "
                  "Retry /stop_profile before starting another session.";
    return false;
  }

  if (!initialized_) {
    output_dir_ = create_session_dir(profile_dir);
    if (output_dir_.empty()) {
      return false;
    }
    const aclError status =
        aclprofInit(output_dir_.c_str(), output_dir_.size());
    if (status != ACL_SUCCESS) {
      LOG(ERROR) << "aclprofInit failed: " << status;
      return false;
    }
    initialized_ = true;
  }

  uint32_t device = static_cast<uint32_t>(device_id);
  ConfigPtr config(aclprofCreateConfig(&device,
                                       /*deviceNums=*/1,
                                       ACL_AICORE_NONE,
                                       /*aicoreEvents=*/nullptr,
                                       kProfilingActivities));
  if (config == nullptr) {
    LOG(ERROR) << "aclprofCreateConfig failed for NPU " << device_id;
    finalize_if_idle();
    return false;
  }
  const aclError status = aclprofStart(config.get());
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "aclprofStart failed for NPU " << device_id << ": " << status;
    config.reset();
    finalize_if_idle();
    return false;
  }
  configs_.emplace(device_id, DeviceCapture{std::move(config)});
  LOG(INFO) << "NPU profiler started on device " << device_id
            << ". Output directory: " << output_dir_;
  return true;
}

bool NpuProfiler::stop(int32_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = configs_.find(device_id);
  if (it == configs_.end()) {
    return finalize_if_idle();
  }

  auto& capture = it->second;
  if (!capture.stopped) {
    const aclError status = aclprofStop(capture.config.get());
    if (status != ACL_SUCCESS) {
      // Retain ownership so a later stop can retry. Never finalize a live
      // device.
      LOG(ERROR) << "aclprofStop failed for NPU " << device_id << ": "
                 << status;
      return false;
    }
    capture.stopped = true;
  }
  const aclError status = aclprofDestroyConfig(capture.config.get());
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "aclprofDestroyConfig failed for NPU " << device_id << ": "
               << status;
    return false;
  }
  capture.config.release();
  configs_.erase(it);
  LOG(INFO) << "NPU profiler stopped on device " << device_id;
  return finalize_if_idle();
}

bool NpuProfiler::finalize_if_idle() {
  if (!initialized_ || !configs_.empty()) {
    return true;
  }
  const aclError status = aclprofFinalize();
  if (status != ACL_SUCCESS) {
    LOG(ERROR) << "aclprofFinalize failed: " << status;
    return false;
  }
  initialized_ = false;
  LOG(INFO) << "NPU profiling data flushed to: " << output_dir_
            << ". Export with: msprof --export=on --output=" << output_dir_;
  return true;
}

}  // namespace xllm

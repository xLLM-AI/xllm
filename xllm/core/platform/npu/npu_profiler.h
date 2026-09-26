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

#pragma once

#include <acl/acl_prof.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xllm {

// CANN initialization is process-wide, while capture is configured per device.
// Each worker starts/stops its own device. The last stop flushes the session;
// duplicate worker calls must not finalize another device's active capture.
class NpuProfiler final {
 public:
  static NpuProfiler& get_instance();

  // Call on the worker's compute thread, after draining outstanding work.
  bool start(const std::string& profile_dir, int32_t device_id);
  bool stop(int32_t device_id);

 private:
  class ConfigDeleter final {
   public:
    void operator()(aclprofConfig* config) const;
  };
  using ConfigPtr = std::unique_ptr<aclprofConfig, ConfigDeleter>;

  struct DeviceCapture {
    ConfigPtr config;
    bool stopped = false;
  };

  NpuProfiler() = default;
  NpuProfiler(const NpuProfiler&) = delete;
  NpuProfiler& operator=(const NpuProfiler&) = delete;

  bool finalize_if_idle();

  std::mutex mutex_;
  bool initialized_ = false;
  std::string output_dir_;
  std::unordered_map<int32_t, DeviceCapture> configs_;
};

}  // namespace xllm

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

#pragma once

#include <mutex>
#include <string>

namespace xllm {

// TorchProfiler drives online timeline collection through libtorch's in-process
// Kineto profiler (the C++ equivalent of torch.profiler.profile). Unlike the
// CUDA-profiler capture-range path (see CudaProfiler), this backend records CPU
// and CUDA activities itself and, on stop(), writes a Chrome trace JSON to disk
// without needing an external profiler such as nsys attached. This mirrors
// vLLM's default TorchProfilerWrapper so that simply launching the server with
// --enable_online_profile=true and driving /start_profile and /stop_profile
// produces a timeline.
//
// xLLM runs one worker thread per device inside a single process and Kineto /
// CUPTI is process-global, so the profiler is wrapped in a process-wide
// singleton: the first start() opens the collection window for the whole
// process and repeated / overlapping start()/stop() calls are made idempotent
// under a mutex.
//
// IMPORTANT: Kineto captures CPU operators via thread-local RecordFunction
// callbacks, so start() and stop() must be invoked on the same thread that runs
// the model forward pass for the host-side timeline to be complete. CUDA kernel
// activities are collected process-globally via CUPTI regardless of thread.
class TorchProfiler {
 public:
  static TorchProfiler& get_instance();

  // Open the Kineto collection window. Repeated calls while already running are
  // ignored. Returns true on success (or if already running).
  bool start();

  // Close the collection window and write the Chrome trace to `profile_dir`
  // (current directory when empty). `rank` is embedded in the file name so
  // per-worker traces in a multi-device process do not collide. Calling stop()
  // while not running is a no-op that returns true.
  bool stop(const std::string& profile_dir, int32_t rank);

  bool is_running() const;

 private:
  TorchProfiler() = default;
  ~TorchProfiler();
  TorchProfiler(const TorchProfiler&) = delete;
  TorchProfiler& operator=(const TorchProfiler&) = delete;

  mutable std::mutex mutex_;
  bool running_ = false;
};

}  // namespace xllm

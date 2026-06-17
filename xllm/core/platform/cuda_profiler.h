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

namespace xllm {

// CudaProfiler drives online timeline collection on CUDA GPUs by toggling the
// CUDA profiler capture range (cudaProfilerStart / cudaProfilerStop). This
// mirrors vLLM's CudaProfilerWrapper and is meant to be used together with
// NVIDIA Nsight Systems: the server is launched under
//   nsys profile --capture-range=cudaProfilerApi --capture-range-end repeat ...
// and /start_profile / /stop_profile open and close the capture range so that
// nsys records exactly the window of interest. The trace output location is
// controlled by nsys (its -o flag), not by xLLM.
//
// xLLM runs one worker thread per device inside a single process, so the CUDA
// profiler (which is process-global) is wrapped in a process-wide singleton:
// the first start() opens the capture range for the whole process and repeated
// / overlapping start()/stop() calls are made idempotent under a mutex.
class CudaProfiler {
 public:
  static CudaProfiler& get_instance();

  // Open the CUDA profiler capture range. Repeated calls while already running
  // are ignored. Returns true on success (or if already running).
  bool start();

  // Close the CUDA profiler capture range. Calling stop() while not running is
  // a no-op that returns true.
  bool stop();

  bool is_running() const;

 private:
  CudaProfiler() = default;
  ~CudaProfiler();
  CudaProfiler(const CudaProfiler&) = delete;
  CudaProfiler& operator=(const CudaProfiler&) = delete;

  mutable std::mutex mutex_;
  bool running_ = false;
};

}  // namespace xllm

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

#include "platform/cuda_profiler.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace xllm {

CudaProfiler& CudaProfiler::get_instance() {
  static CudaProfiler instance;
  return instance;
}

CudaProfiler::~CudaProfiler() {
  if (running_) {
    stop();
  }
}

bool CudaProfiler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    LOG(WARNING) << "CUDA profiler is already running, ignoring start request.";
    return true;
  }

  cudaError_t ret = cudaProfilerStart();
  if (ret != cudaSuccess) {
    LOG(ERROR) << "cudaProfilerStart failed: " << cudaGetErrorString(ret);
    return false;
  }

  running_ = true;
  LOG(INFO) << "CUDA profiler started. Make sure the server was launched under "
               "nsys with --capture-range=cudaProfilerApi for the trace to be "
               "recorded.";
  return true;
}

bool CudaProfiler::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    LOG(WARNING) << "CUDA profiler is not running, ignoring stop request.";
    return true;
  }

  cudaError_t ret = cudaProfilerStop();
  running_ = false;
  if (ret != cudaSuccess) {
    LOG(ERROR) << "cudaProfilerStop failed: " << cudaGetErrorString(ret);
    return false;
  }

  LOG(INFO) << "CUDA profiler stopped.";
  return true;
}

bool CudaProfiler::is_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

}  // namespace xllm

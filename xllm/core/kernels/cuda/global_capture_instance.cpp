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

#include "global_capture_instance.h"

#include <glog/logging.h>

#include "attention_runner.h"
#include "core/common/global_flags.h"
#include "piecewise_graphs.h"

namespace xllm::runtime::cuda {

// Define static mutex
std::mutex GlobalCaptureInstance::capture_mutex_;

GlobalCaptureInstance& GlobalCaptureInstance::get_instance() {
  thread_local GlobalCaptureInstance instance;
  return instance;
}

GlobalCaptureInstance::GlobalCaptureInstance() = default;
GlobalCaptureInstance::~GlobalCaptureInstance() = default;

void GlobalCaptureInstance::cleanup_capture_state() {
  is_capturing_ = false;
  current_graph_.reset();
  current_piecewise_graph_.reset();
}

void GlobalCaptureInstance::begin_capture(const at::cuda::MempoolId_t& pool) {
  CHECK(!is_capturing_) << "Already capturing, call end_capture() first";

  // Acquire global lock to ensure only one instance captures at a time
  capture_lock_ = std::make_unique<std::lock_guard<std::mutex>>(capture_mutex_);
  LOG(INFO) << "GlobalCaptureInstance::begin_capture()";
  is_capturing_ = true;
  graph_pool_ = pool;

  // Reset current_piecewise_graph_
  current_piecewise_graph_ = std::make_unique<PiecewiseGraphs>();

  // Create first graph and begin capture
  current_graph_ = std::make_unique<at::cuda::CUDAGraph>();
  current_graph_->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
}

std::unique_ptr<PiecewiseGraphs> GlobalCaptureInstance::end_capture() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_graph_)
      << "Current graph is null, cannot end without active graph. "
      << "Did you call temporarily_end_graph() without "
         "temporarily_begin_graph()?";

  // End last graph and add to piecewise_graph
  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  is_capturing_ = false;

  LOG(INFO) << "GlobalCaptureInstance::end_capture(), total graphs: "
            << current_piecewise_graph_->size()
            << ", total runners: " << current_piecewise_graph_->num_runners();

  // Move the result before releasing the lock
  auto result = std::move(current_piecewise_graph_);

  // Release global lock to allow next instance to capture
  capture_lock_.reset();

  return result;
}

void GlobalCaptureInstance::temporarily_end_graph() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_end_graph_locked();
}

void GlobalCaptureInstance::temporarily_begin_graph() {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_begin_graph_locked();
}

void GlobalCaptureInstance::temporarily_end_graph_locked() {
  CHECK(current_graph_) << "Current graph is null, cannot end. "
                        << "Did you call temporarily_end_graph() twice?";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  // End current graph capture and add to piecewise_graph
  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  VLOG(kGraphExecutorLogVerboseLevel)
      << "GlobalCaptureInstance::temporarily_end_graph(), total graphs: "
      << current_piecewise_graph_->size();
}

void GlobalCaptureInstance::temporarily_begin_graph_locked() {
  CHECK(!current_graph_)
      << "Current graph already exists, cannot begin new graph. "
      << "Did you call temporarily_begin_graph() twice?";

  // Create new graph and begin capture
  current_graph_ = std::make_unique<at::cuda::CUDAGraph>();
  current_graph_->capture_begin(graph_pool_, cudaStreamCaptureModeThreadLocal);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "GlobalCaptureInstance::temporarily_begin_graph()";
}

void GlobalCaptureInstance::register_attention_runner(
    ::xllm::kernel::cuda::AttentionRunner&& runner) {
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  current_piecewise_graph_->add_attention_runner(std::move(runner));
  VLOG(kGraphExecutorLogVerboseLevel)
      << "GlobalCaptureInstance::register_attention_runner(), total runners: "
      << current_piecewise_graph_->num_runners();
}

}  // namespace xllm::runtime::cuda

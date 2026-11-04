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

#include "core/distributed_runtime/shm_channel.h"

#include <glog/logging.h>

#include <exception>
#include <new>
#include <utility>

#include "runtime/vlm_forward_params.h"
#include "util/net.h"

namespace xllm {

ShmChannel::ShmChannel(int dp_group,
                       int rank,
                       bool is_driver,
                       const runtime::Options& options)
    : enable_shm_(options.enable_shm()),
      input_shm_size_(options.input_shm_size()) {
  if (options.backend() == "dit") {
    CHECK_GE(options.output_shm_size(),
             static_cast<int64_t>(
                 ForwardSharedMemoryTransport::minimum_dit_output_size()))
        << "DiT output shared memory cannot hold a failure status";
  }
  bool is_creator;

  std::string name_prefix =
      "xllm_" + net::extract_port(options.master_node_addr().value_or(""));
  if (is_driver) {
    auto name = ForwardSharedMemoryTransport::create_unique_name(
        name_prefix, dp_group, ForwardType::RAW_INPUT, rank);
    input_shm_manager_ = std::make_unique<ForwardSharedMemoryTransport>(
        name, options.input_shm_size(), is_creator, ForwardType::RAW_INPUT);
    LOG(INFO) << "Create input shared memory manager with name: " << name
              << " and size: " << options.input_shm_size();
  }

  auto name = ForwardSharedMemoryTransport::create_unique_name(
      name_prefix, dp_group, ForwardType::RAW_OUTPUT, rank);
  output_shm_manager_ = std::make_unique<ForwardSharedMemoryTransport>(
      name, options.output_shm_size(), is_creator, ForwardType::RAW_OUTPUT);
  LOG(INFO) << "Create output shared memory manager with name: " << name
            << " and size: " << options.output_shm_size();
}

void ShmChannel::execute_llm_model_async(
    const LlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    const bool input_fits = input_shm_manager_
                                ? input_shm_manager_->input_write(input)
                                : ForwardSharedMemoryTransport::can_write_input(
                                      input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "LLM shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      if (!output_shm_manager_->llm_output_read(output)) {
        LOG(ERROR) << "LLM shared-memory output read failed";
        // The model has already executed. Retrying with BRPC would execute it
        // twice and advance worker state again.
        promise.setValue(std::nullopt);
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  execute_llm_model_with_brpc(input, promise);
}

void ShmChannel::execute_dit_model_async(
    const DiTForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    const bool input_fits = input_shm_manager_
                                ? input_shm_manager_->input_write(input)
                                : ForwardSharedMemoryTransport::can_write_input(
                                      input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "DIT shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      // Input is already published. Decode failures, including exceptions, must
      // complete this future without retrying model execution through BRPC.
      bool decoded = false;
      try {
        decoded = output_shm_manager_->dit_output_read(output);
      } catch (const std::bad_alloc& error) {
        promise.setException(DiTForwardExecutionError(
            {StatusCode::RESOURCE_EXHAUSTED, error.what()}));
        return;
      } catch (const std::exception& error) {
        promise.setException(
            DiTForwardExecutionError({StatusCode::UNKNOWN, error.what()}));
        return;
      } catch (...) {
        promise.setException(DiTForwardExecutionError(
            {StatusCode::UNKNOWN,
             "Unknown exception decoding DiT SHM output"}));
        return;
      }
      if (!decoded) {
        LOG(ERROR) << "Invalid DiT shared-memory output";
        promise.setException(DiTForwardExecutionError(
            {StatusCode::UNKNOWN, "Invalid DiT shared-memory output"}));
        return;
      }
      if (!output.dit_execution_status.ok()) {
        promise.setException(
            DiTForwardExecutionError(output.dit_execution_status));
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  execute_dit_model_with_brpc(input, promise);
}

void ShmChannel::execute_vlm_model_async(
    const VlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    const bool input_fits = input_shm_manager_
                                ? input_shm_manager_->input_write(input)
                                : ForwardSharedMemoryTransport::can_write_input(
                                      input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "VLM shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      if (!output_shm_manager_->vlm_output_read(output)) {
        LOG(ERROR) << "VLM shared-memory output read failed";
        // The model has already executed. Retrying with BRPC would execute it
        // twice and advance worker state again.
        promise.setValue(std::nullopt);
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  execute_vlm_model_with_brpc(input, promise);
}

}  // namespace xllm

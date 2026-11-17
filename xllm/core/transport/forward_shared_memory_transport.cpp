/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.
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

#include "core/transport/forward_shared_memory_transport.h"

#include <cstring>

#include "core/framework/config/execution_config.h"
#include "core/platform/stream.h"
#include "core/runtime/vlm_forward_params.h"

namespace xllm {
ForwardSharedMemoryTransport::ForwardSharedMemoryTransport(
    const std::string& name,
    size_t size,
    bool& is_creator,
    ForwardType type)
    : segment_(name, size, is_creator), forward_type_(type) {
  control_ptr_ = static_cast<ControlMetadata*>(base_address());
  metadata_addr_ = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  if (::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer()) {
    stream_ = std::make_unique<Stream>();
  }
}

ForwardSharedMemoryTransport::~ForwardSharedMemoryTransport() = default;

std::string ForwardSharedMemoryTransport::create_unique_name(
    const std::string& prefix,
    int dp_group,
    ForwardType forward_type,
    int rank) {
  std::string filename = prefix;
  if (forward_type == ForwardType::PB_INPUT ||
      forward_type == ForwardType::RAW_INPUT) {
    filename += "_dpg_" + std::to_string(dp_group) + "_input";
  } else if (forward_type == ForwardType::PB_OUTPUT ||
             forward_type == ForwardType::RAW_OUTPUT) {
    filename += "_rank_" + std::to_string(rank) + "_output";
  } else {
    // TODO: support more type later
  }

  return filename;
}

void ForwardSharedMemoryTransport::stop_input_read() {
  input_read_stopped_.store(true, std::memory_order_relaxed);
}

bool ForwardSharedMemoryTransport::wait_for_input() {
  while (!input_read_stopped_.load(std::memory_order_relaxed)) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      std::atomic_thread_fence(std::memory_order_acquire);
      return true;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kNumWaitNanoseconds));
  }
  return false;
}

bool ForwardSharedMemoryTransport::can_write_input(const DiTForwardInput& input,
                                                   size_t segment_size) {
  return segment_size >= sizeof(ControlMetadata) &&
         PackedForwardCodec::can_write_input(
             input, segment_size - sizeof(ControlMetadata));
}
bool ForwardSharedMemoryTransport::input_write(const DiTForwardInput& input) {
  if (!codec().input_write(input)) {
    return false;
  }
  publish();
  return true;
}

bool ForwardSharedMemoryTransport::can_write_input(const LlmForwardInput& input,
                                                   size_t segment_size) {
  return segment_size >= sizeof(ControlMetadata) &&
         PackedForwardCodec::can_write_input(
             input, segment_size - sizeof(ControlMetadata));
}
bool ForwardSharedMemoryTransport::input_write(const LlmForwardInput& input) {
  if (!codec().input_write(input)) {
    return false;
  }
  publish();
  return true;
}

bool ForwardSharedMemoryTransport::can_write_input(const VlmForwardInput& input,
                                                   size_t segment_size) {
  return segment_size >= sizeof(ControlMetadata) &&
         PackedForwardCodec::can_write_input(
             input, segment_size - sizeof(ControlMetadata));
}
bool ForwardSharedMemoryTransport::input_write(const VlmForwardInput& input) {
  if (!codec().input_write(input)) {
    return false;
  }
  publish();
  return true;
}

bool ForwardSharedMemoryTransport::input_read(DiTForwardInput& input) {
  return wait_for_input() && codec().input_read(input);
}

bool ForwardSharedMemoryTransport::input_read(
    LlmForwardInput& input,
    const torch::Device& device,
    InputDeviceMaterializationPolicy policy) {
  return wait_for_input() && codec().input_read(input, device, policy);
}

bool ForwardSharedMemoryTransport::input_read(
    VlmForwardInput& input,
    const torch::Device& device,
    InputDeviceMaterializationPolicy policy) {
  return wait_for_input() && codec().input_read(input, device, policy);
}

void ForwardSharedMemoryTransport::publish() {
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;
}
void ForwardSharedMemoryTransport::wait_for_output() {
  while (control_ptr_->version == last_version_) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(kNumWaitNanoseconds));
  }
  last_version_ = control_ptr_->version;
  std::atomic_thread_fence(std::memory_order_acquire);
}
PackedForwardCodec ForwardSharedMemoryTransport::codec() const {
  return PackedForwardCodec(
      static_cast<char*>(base_address()) + sizeof(ControlMetadata),
      size() - sizeof(ControlMetadata),
      stream_.get());
}

bool ForwardSharedMemoryTransport::llm_output_write(
    const ForwardOutput& output,
    const std::vector<SpeculativeTokenStats>& stats) {
  if (!codec().write_token_output(output, stats, "LLM")) {
    return false;
  }
  publish();
  return true;
}
bool ForwardSharedMemoryTransport::llm_output_read(RawForwardOutput& output) {
  wait_for_output();
  return codec().read_token_output(output, "LLM");
}

bool ForwardSharedMemoryTransport::vlm_output_write(
    const ForwardOutput& output,
    const std::vector<SpeculativeTokenStats>& stats) {
  if (!codec().write_token_output(output, stats, "VLM")) {
    return false;
  }
  publish();
  return true;
}
bool ForwardSharedMemoryTransport::vlm_output_read(RawForwardOutput& output) {
  wait_for_output();
  return codec().read_token_output(output, "VLM");
}

bool ForwardSharedMemoryTransport::dit_output_write(
    const DiTForwardOutput& output) {
  if (!codec().dit_output_write(output)) {
    return false;
  }
  publish();
  return true;
}
bool ForwardSharedMemoryTransport::dit_output_read(RawForwardOutput& output) {
  wait_for_output();
  return codec().dit_output_read(output);
}
void ForwardSharedMemoryTransport::clear() {
  std::memset(base_address(), 0, size());
}
}  // namespace xllm

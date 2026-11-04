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

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/runtime/forward_params.h"
#include "core/util/shared_memory_manager.h"
#include "worker.pb.h"

namespace xllm {
class VlmForwardInput;

class Stream;

constexpr int64_t kNumWaitNanoseconds = 1000;  // 1us

struct ControlMetadata {
  volatile uint64_t version;
};

struct PbMetadata {
  uint64_t pb_size;
};

enum class ForwardType : int8_t {
  PB_INPUT = 1,
  PB_OUTPUT = 2,
  RAW_INPUT = 3,
  RAW_OUTPUT = 4,
};

enum class InputDeviceMaterializationPolicy : int8_t {
  MATERIALIZE_ON_READ = 0,
  DEFER_TO_WORKER_PREPARE,
};

// Owns a shared segment and transports forward messages in its wire format.
class ForwardSharedMemoryTransport final {
 public:
  explicit ForwardSharedMemoryTransport(const std::string& name,
                                        size_t size,
                                        bool& is_creator,
                                        ForwardType type);
  ~ForwardSharedMemoryTransport();

  void* base_address() const { return segment_.base_address(); }
  int64_t size() const { return segment_.size(); }
  std::string name() const { return segment_.name(); }
  static std::string create_unique_name(const std::string& prefix,
                                        int dp_group,
                                        ForwardType forward_type,
                                        int rank);

  template <typename PbType>
  bool pb_write(const PbType* pb_data) {
    size_t data_size = pb_data->ByteSizeLong();
    if (data_size + sizeof(ControlMetadata) + sizeof(PbMetadata) > size()) {
      LOG(ERROR) << "pb size overflow, data_size: " << data_size
                 << ", shm size: " << size();
      return false;
    }

    auto metadata = reinterpret_cast<PbMetadata*>(metadata_addr_);
    metadata->pb_size = data_size;

    auto data_ptr =
        reinterpret_cast<char*>(metadata_addr_) + sizeof(PbMetadata);
    if (!pb_data->SerializeToArray(data_ptr, data_size)) {
      LOG(ERROR) << "Failed to serialize protobuf data to shared memory";
      return false;
    }

    std::atomic_thread_fence(std::memory_order_release);
    control_ptr_->version = ++last_version_;

    return true;
  };

  template <typename PbType>
  bool pb_read(PbType& pb_data) {
    while (true) {
      if (control_ptr_->version != last_version_) {
        last_version_ = control_ptr_->version;
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(kNumWaitNanoseconds));
    }

    auto metadata = reinterpret_cast<PbMetadata*>(metadata_addr_);
    auto data_ptr =
        reinterpret_cast<char*>(metadata_addr_) + sizeof(PbMetadata);
    size_t pb_size = metadata->pb_size;
    if (!pb_data.ParseFromArray(data_ptr, pb_size)) {
      LOG(ERROR) << "Failed to parse pb data from shared memory";
      return false;
    }

    return true;
  };

  bool input_write(const DiTForwardInput& input);
  bool input_read(DiTForwardInput& input);

  // Check transport support and capacity without copying tensor payloads.
  // Every rank in a DP group must select the same transport before execution.
  static bool can_write_input(const DiTForwardInput& input,
                              size_t segment_size);
  static bool can_write_input(const VlmForwardInput& input,
                              size_t segment_size);
  static bool can_write_input(const LlmForwardInput& input,
                              size_t segment_size);

  bool input_write(const VlmForwardInput& input);
  bool input_read(VlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy =
                      InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);

  bool input_write(const LlmForwardInput& input);
  bool input_read(LlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy =
                      InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);

  bool llm_output_write(
      const ForwardOutput& output,
      const std::vector<SpeculativeTokenStats>& speculative_token_stats);
  bool vlm_output_write(
      const ForwardOutput& output,
      const std::vector<SpeculativeTokenStats>& speculative_token_stats);
  static size_t minimum_dit_output_size();
  bool dit_output_write(const DiTForwardOutput& output,
                        const Status& status = Status());

  // Read into owned storage and replace the destination only on success.
  bool llm_output_read(RawForwardOutput& output);
  bool vlm_output_read(RawForwardOutput& output);
  bool dit_output_read(RawForwardOutput& output);

  // Permanently cancel input reads on this transport without publishing
  // a synthetic request or changing the shared segment for other readers.
  void stop_input_read();

  void clear();

 private:
  bool write_token_output(
      const ForwardOutput& output,
      const std::vector<SpeculativeTokenStats>& speculative_token_stats,
      const char* domain);

  bool read_token_output(RawForwardOutput& output, const char* domain);

  template <typename Input>
  bool write_token_input(const Input& input);

  bool wait_for_input();

  template <typename Input>
  bool read_token_input(Input& input,
                        const torch::Device& device,
                        InputDeviceMaterializationPolicy policy);

  SharedMemoryManager segment_;
  std::atomic<bool> input_read_stopped_{false};
  ForwardType forward_type_;
  uint64_t last_version_ = 0;
  void* metadata_addr_ = nullptr;
  ControlMetadata* control_ptr_ = nullptr;
  std::unique_ptr<Stream> stream_ = nullptr;
};
}  // namespace xllm

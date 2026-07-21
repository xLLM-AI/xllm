/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.
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

#include <stddef.h>

#include <memory>

#include "dit_forward_params.h"
#include "forward_params.h"
#include "params_utils.h"
#include "util/shared_memory_manager.h"

namespace xllm {

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

class ForwardSharedMemoryManager : public SharedMemoryManager {
 public:
  explicit ForwardSharedMemoryManager(const std::string& name,
                                      size_t size,
                                      bool& is_creator,
                                      ForwardType type);
  ~ForwardSharedMemoryManager();
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

  bool input_write(const ForwardInput& input);
  void input_read(ForwardInput& input, const torch::Device& device);
  bool raw_output_write(const torch::Tensor& next_tokens,
                        const torch::Tensor& logprobs,
                        const torch::Tensor& top_tokens,
                        const torch::Tensor& top_logprobs,
                        const torch::Tensor& embeddings,
                        const std::vector<torch::Tensor>& mm_embeddings,
                        const std::vector<torch::Tensor>& dit_images,
                        const std::vector<std::string>& dit_text_output,
                        const torch::Tensor& expert_load_data,
                        int32_t prepared_layer_id,
                        const torch::Tensor& src_seq_idxes,
                        const torch::Tensor& out_tokens,
                        const torch::Tensor& out_logprobs);
  void raw_output_read(RawForwardOutput& outputs);

  void clear();

 private:
  ForwardType forward_type_;
  uint64_t last_version_ = 0;
  void* metadata_addr_ = nullptr;
  ControlMetadata* control_ptr_ = nullptr;
  std::unique_ptr<Stream> stream_ = nullptr;
};
}  // namespace xllm

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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/runtime/forward_params.h"

namespace xllm {
class Stream;
class VlmForwardInput;

enum class InputDeviceMaterializationPolicy : int8_t {
  MATERIALIZE_ON_READ = 0,
  DEFER_TO_WORKER_PREPARE,
};

// Encodes forward messages into a bounded payload region. Publication and
// shared-memory ownership belong to the transport, not this codec.
class PackedForwardCodec final {
 public:
  explicit PackedForwardCodec(void* data,
                              int64_t capacity,
                              Stream* stream = nullptr)
      : data_(data), capacity_(capacity), stream_(stream) {}

  static bool can_write_input(const LlmForwardInput& input, size_t capacity);
  static bool can_write_input(const VlmForwardInput& input, size_t capacity);
  static bool can_write_input(const DiTForwardInput& input, size_t capacity);
  bool input_write(const LlmForwardInput& input);
  bool input_write(const VlmForwardInput& input);
  bool input_write(const DiTForwardInput& input);
  bool input_read(LlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy);
  bool input_read(VlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy);
  bool input_read(DiTForwardInput& input);
  bool write_token_output(const ForwardOutput& output,
                          const std::vector<SpeculativeTokenStats>& stats,
                          const char* domain);
  bool read_token_output(RawForwardOutput& output, const char* domain);
  bool dit_output_write(const DiTForwardOutput& output);
  bool dit_output_read(RawForwardOutput& output);

 private:
  template <typename Input>
  bool write_token_input(const Input& input);
  template <typename Input>
  bool read_token_input(Input& input,
                        const torch::Device& device,
                        InputDeviceMaterializationPolicy policy);
  void* base_address() const { return data_; }
  int64_t size() const { return capacity_; }

  void* data_;
  int64_t capacity_;
  Stream* stream_;
};
}  // namespace xllm

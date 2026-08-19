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

#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "stream.h"

namespace xllm {

class Device {
 public:
  explicit Device(const torch::Device& device);
  explicit Device(const int32_t device_index);
  ~Device() = default;
  operator torch::Device() const;

  void set_device() const;

  void set_seed(uint64_t seed = 42) const;

  const torch::Device& unwrap() const;
  int32_t index() const;

  void init_device_context() const;

  int64_t total_memory();
  int64_t free_memory();
  static void empty_cache(int32_t device_index);

  int synchronize_default_stream();
  std::unique_ptr<Stream> get_stream_from_pool(const int32_t timeout = -1);

  // get the current stream
  std::unique_ptr<Stream> current_stream() const;

 private:
  struct DeviceMem {
    int64_t total_memory;
    int64_t free_memory;
  };

  DeviceMem get_device_mem() const;

 private:
  torch::Device device_;
};

}  // namespace xllm

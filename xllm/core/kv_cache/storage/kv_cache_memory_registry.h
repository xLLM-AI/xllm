/* Copyright 2026 The xLLM Authors.

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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/kv_cache/storage/kv_cache_memory_regions.h"

namespace xllm {

// Owns and synchronizes worker-local model KV mappings. The runtime supplies
// device/page geometry and must stop all tensor users before erasing mappings.
class KVCacheMemoryRegistry final {
 public:
  void init(const torch::Device& device, size_t page_size);

  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  bool map(const std::string& model_id, const std::vector<offset_t>& offsets);
  bool unmap(const std::string& model_id, const std::vector<offset_t>& offsets);
  std::optional<int64_t> num_layers(const std::string& model_id) const;

  // Returns offsets relative to the shared physical-page mapping, or
  // {UINT64_MAX, UINT64_MAX} when the model/block has no valid mapping.
  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id,
      size_t block_size) const;

  void erase(const std::string& model_id);
  void clear();

 private:
  mutable std::mutex mutex_;
  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  size_t page_size_ = 0;
  std::unordered_map<std::string, KVCacheMemoryRegions> regions_;
};

}  // namespace xllm

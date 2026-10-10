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
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/model_loader/weight/model_weight_store.h"
#include "core/framework/model_loader/weight/weight_page_coordinator.h"

namespace xllm {

class GlobalMemoryRegion;
class PhysicalPagePool;

// Local weight allocation policy and ownership. Distributed adapters use the
// same contiguous-first reservation and fragmented mapping path.
class WeightMemoryManager final {
 public:
  WeightMemoryManager(PhysicalPagePool& pool,
                      GlobalMemoryRegion& global_region);
  ~WeightMemoryManager();

  void init(const torch::Device& device);
  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);
  size_t free_weight(const std::string& model_id);
  std::optional<WeightAllocationInfo> get_weight_allocation_info(
      const std::string& model_id) const;
  std::vector<WeightSegment> get_model_weight_segments(
      const std::string& model_id) const;
  std::unordered_map<std::string, std::vector<WeightSegment>>
  get_all_model_weight_segments() const;
  void clear();

 private:
  bool record_weight_allocation_locked(const std::string& model_id,
                                       page_id_t start_page_id,
                                       size_t num_pages);
  bool record_weight_fallback_allocation_locked(
      const std::string& model_id,
      const std::vector<page_id_t>& page_ids);

  PhysicalPagePool& pool_;
  GlobalMemoryRegion& global_region_;
  torch::Device device_{torch::kCPU};
  mutable std::mutex mutex_;
  ModelWeightStore store_;
  WeightPageCoordinator page_coordinator_;
};

}  // namespace xllm

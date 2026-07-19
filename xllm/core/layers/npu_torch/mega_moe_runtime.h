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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace xllm::layer {

struct MegaMoeWeightBudgetStatus {
  bool allowed = false;
  int64_t estimated_bytes = 0;
  int64_t current_bytes = 0;
  int64_t limit_bytes = 0;
};

enum class MegaMoeWeightCacheAction : int32_t {
  SKIP = 0,
  WAIT_FOR_WEIGHTS,
  BUILD,
  FAIL_MISSING_WEIGHTS,
};

MegaMoeWeightCacheAction plan_mega_moe_weight_cache(
    bool mega_moe_enabled,
    bool cache_ready,
    bool w13_loaded,
    bool w2_loaded,
    bool require_complete_weights);

struct MegaMoeWeightCacheDestructionObserver {
  std::function<void()> on_tensor_fields_released;
  std::function<void()> on_reservation_released;
};

struct MegaMoeWeightCache {
  MegaMoeWeightCache() = default;
  MegaMoeWeightCache(const MegaMoeWeightCache&) = delete;
  MegaMoeWeightCache& operator=(const MegaMoeWeightCache&) = delete;
  MegaMoeWeightCache(MegaMoeWeightCache&&) noexcept = default;
  MegaMoeWeightCache& operator=(MegaMoeWeightCache&&) noexcept = default;

 private:
  friend std::optional<MegaMoeWeightCache>
  try_build_mega_moe_weight_cache(
      const torch::Tensor&,
      const torch::Tensor&,
      int64_t,
      int64_t,
      int64_t,
      MegaMoeWeightBudgetStatus*,
      MegaMoeWeightCacheDestructionObserver*);
  // Members are destroyed in reverse declaration order. Keep the accounting
  // reservation before the release probe and tensor fields so device storage
  // is released before the process-wide budget is returned.
  std::shared_ptr<void> budget_reservation_;
  std::shared_ptr<void> tensor_fields_release_probe_;

 public:
  torch::Tensor weight1_storage;
  torch::Tensor weight2_storage;
  std::vector<torch::Tensor> weight1;
  std::vector<torch::Tensor> weight2;
  int64_t memory_bytes = 0;

  bool ready() const {
    return weight1_storage.defined() && weight2_storage.defined() &&
           !weight1.empty() && weight1.size() == weight2.size();
  }

};

MegaMoeWeightCache build_mega_moe_weight_cache(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2,
    int64_t hidden_size,
    int64_t intermediate_size);

int64_t estimate_mega_moe_weight_cache_bytes(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2);

int64_t current_mega_moe_weight_cache_bytes();

MegaMoeWeightBudgetStatus inspect_mega_moe_weight_budget(
    int64_t estimated_bytes,
    int64_t limit_bytes);

// Reserves the process-wide budget atomically before contiguous allocation.
// A denied reservation returns nullopt without allocating either weight copy.
std::optional<MegaMoeWeightCache> try_build_mega_moe_weight_cache(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t budget_limit_bytes,
    MegaMoeWeightBudgetStatus* status = nullptr,
    MegaMoeWeightCacheDestructionObserver* observer = nullptr);

}  // namespace xllm::layer

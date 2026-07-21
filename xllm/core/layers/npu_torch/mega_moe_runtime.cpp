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

#include "layers/npu_torch/mega_moe_runtime.h"

#include <atomic>
#include <limits>
#include <memory>
#include <utility>

namespace xllm::layer {
namespace {

std::atomic<int64_t> g_weight_cache_bytes{0};

class BudgetReservation final {
 public:
  BudgetReservation(int64_t bytes, std::function<void()> on_release)
      : bytes_(bytes), on_release_(std::move(on_release)) {}
  ~BudgetReservation() {
    g_weight_cache_bytes.fetch_sub(bytes_, std::memory_order_acq_rel);
    if (on_release_) {
      on_release_();
    }
  }

 private:
  int64_t bytes_;
  std::function<void()> on_release_;
};

class TensorFieldsReleaseProbe final {
 public:
  explicit TensorFieldsReleaseProbe(std::function<void()> on_release)
      : on_release_(std::move(on_release)) {}
  ~TensorFieldsReleaseProbe() {
    if (on_release_) {
      on_release_();
    }
  }

 private:
  std::function<void()> on_release_;
};

void validate_canonical_weights(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2,
    int64_t hidden_size,
    int64_t intermediate_size) {
  TORCH_CHECK(canonical_w13.defined() && canonical_w2.defined(),
              "MegaMoe canonical weights must be defined.");
  TORCH_CHECK(canonical_w13.dim() == 3 && canonical_w2.dim() == 3,
              "MegaMoe canonical weights must be 3D.");
  TORCH_CHECK(canonical_w13.scalar_type() == at::kBFloat16 &&
                  canonical_w2.scalar_type() == at::kBFloat16,
              "MegaMoe weight cache requires bf16 canonical weights.");
  TORCH_CHECK(canonical_w13.size(0) == canonical_w2.size(0),
              "MegaMoe canonical expert count mismatch: ",
              canonical_w13.size(0),
              " vs ",
              canonical_w2.size(0));
  TORCH_CHECK(canonical_w13.size(1) == 2 * intermediate_size &&
                  canonical_w13.size(2) == hidden_size,
              "MegaMoe canonical W13 must be [expert, 2I, H], got ",
              canonical_w13.sizes());
  TORCH_CHECK(canonical_w2.size(1) == hidden_size &&
                  canonical_w2.size(2) == intermediate_size,
              "MegaMoe canonical W2 must be [expert, H, I], got ",
              canonical_w2.sizes());
}

std::shared_ptr<void> try_reserve_budget(
    int64_t estimated_bytes,
    int64_t limit_bytes,
    MegaMoeWeightBudgetStatus& status,
    std::function<void()> on_release) {
  int64_t current = g_weight_cache_bytes.load(std::memory_order_acquire);
  while (true) {
    status = inspect_mega_moe_weight_budget(estimated_bytes, limit_bytes);
    status.current_bytes = current;
    status.allowed = estimated_bytes > 0 && limit_bytes >= estimated_bytes &&
                     current <= limit_bytes - estimated_bytes;
    if (!status.allowed) {
      return nullptr;
    }
    if (g_weight_cache_bytes.compare_exchange_weak(
            current,
            current + estimated_bytes,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      break;
    }
  }

  try {
    return std::static_pointer_cast<void>(
        std::make_shared<BudgetReservation>(estimated_bytes,
                                            std::move(on_release)));
  } catch (...) {
    g_weight_cache_bytes.fetch_sub(estimated_bytes,
                                   std::memory_order_acq_rel);
    throw;
  }
}

void materialize_weight_cache(const torch::Tensor& canonical_w13,
                              const torch::Tensor& canonical_w2,
                              MegaMoeWeightCache& cache) {
  // contiguous() materializes independent storage; canonical registered
  // parameters remain in the legacy loader layout and stay readable.
  cache.weight1_storage = canonical_w13.transpose(1, 2).contiguous();
  cache.weight2_storage = canonical_w2.transpose(1, 2).contiguous();
  const int64_t expert_count = canonical_w13.size(0);
  cache.weight1.reserve(static_cast<size_t>(expert_count));
  cache.weight2.reserve(static_cast<size_t>(expert_count));
  for (int64_t expert = 0; expert < expert_count; ++expert) {
    cache.weight1.push_back(cache.weight1_storage.select(0, expert));
    cache.weight2.push_back(cache.weight2_storage.select(0, expert));
  }
}

}  // namespace

MegaMoeWeightCacheAction plan_mega_moe_weight_cache(
    bool mega_moe_enabled,
    bool cache_ready,
    bool w13_loaded,
    bool w2_loaded,
    bool require_complete_weights) {
  if (!mega_moe_enabled || cache_ready) {
    return MegaMoeWeightCacheAction::SKIP;
  }
  if (!w13_loaded || !w2_loaded) {
    return require_complete_weights
               ? MegaMoeWeightCacheAction::FAIL_MISSING_WEIGHTS
               : MegaMoeWeightCacheAction::WAIT_FOR_WEIGHTS;
  }
  return MegaMoeWeightCacheAction::BUILD;
}

torch::Tensor prepare_mega_moe_topk_weights(
    const torch::Tensor& topk_weights) {
  TORCH_CHECK(topk_weights.defined(),
              "MegaMoe top-k weights must be defined.");
  TORCH_CHECK(topk_weights.dim() == 2,
              "MegaMoe top-k weights must be 2D.");
  return topk_weights.to(torch::kFloat32).contiguous();
}

int64_t estimate_mega_moe_weight_cache_bytes(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2) {
  TORCH_CHECK(canonical_w13.defined() && canonical_w2.defined(),
              "MegaMoe canonical weights must be defined for estimation.");
  return canonical_w13.numel() *
             static_cast<int64_t>(canonical_w13.element_size()) +
         canonical_w2.numel() *
             static_cast<int64_t>(canonical_w2.element_size());
}

int64_t current_mega_moe_weight_cache_bytes() {
  return g_weight_cache_bytes.load(std::memory_order_acquire);
}

MegaMoeWeightBudgetStatus inspect_mega_moe_weight_budget(
    int64_t estimated_bytes,
    int64_t limit_bytes) {
  const int64_t current = current_mega_moe_weight_cache_bytes();
  const bool allowed =
      estimated_bytes > 0 && limit_bytes >= estimated_bytes &&
      current <= limit_bytes - estimated_bytes;
  return {allowed, estimated_bytes, current, limit_bytes};
}

std::optional<MegaMoeWeightCache> try_build_mega_moe_weight_cache(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t budget_limit_bytes,
    MegaMoeWeightBudgetStatus* status,
    MegaMoeWeightCacheDestructionObserver* observer) {
  validate_canonical_weights(
      canonical_w13, canonical_w2, hidden_size, intermediate_size);
  const int64_t estimated_bytes =
      estimate_mega_moe_weight_cache_bytes(canonical_w13, canonical_w2);
  MegaMoeWeightBudgetStatus reservation_status;
  std::function<void()> on_reservation_released;
  if (observer != nullptr) {
    on_reservation_released = observer->on_reservation_released;
  }
  std::shared_ptr<void> reservation = try_reserve_budget(
      estimated_bytes,
      budget_limit_bytes,
      reservation_status,
      std::move(on_reservation_released));
  if (status != nullptr) {
    *status = reservation_status;
  }
  if (reservation == nullptr) {
    return std::nullopt;
  }

  MegaMoeWeightCache cache;
  cache.memory_bytes = estimated_bytes;
  cache.budget_reservation_ = std::move(reservation);
  if (observer != nullptr && observer->on_tensor_fields_released) {
    cache.tensor_fields_release_probe_ = std::static_pointer_cast<void>(
        std::make_shared<TensorFieldsReleaseProbe>(
            observer->on_tensor_fields_released));
  }
  materialize_weight_cache(canonical_w13, canonical_w2, cache);
  return cache;
}

MegaMoeWeightCache build_mega_moe_weight_cache(
    const torch::Tensor& canonical_w13,
    const torch::Tensor& canonical_w2,
    int64_t hidden_size,
    int64_t intermediate_size) {
  auto cache = try_build_mega_moe_weight_cache(
      canonical_w13,
      canonical_w2,
      hidden_size,
      intermediate_size,
      std::numeric_limits<int64_t>::max());
  TORCH_CHECK(cache.has_value(),
              "MegaMoe weight cache accounting overflow.");
  return std::move(cache.value());
}

}  // namespace xllm::layer

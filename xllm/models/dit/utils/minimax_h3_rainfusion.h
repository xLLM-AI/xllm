/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string_view>

#include "models/dit/utils/minimax_h3_rainfusion_policy.h"

#if defined(USE_NPU)
#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"
#endif

namespace xllm::minimax_h3 {

#if defined(USE_NPU)
namespace rainfusion = kernel::npu::tilelang;

// Borrowed by executed blocks on the model's stream for this request only.
struct H3RainfusionRequest {
  H3RainfusionPolicy policy;
  rainfusion::H3RainfusionGeometry geometry;
  rainfusion::H3RainfusionWorkspace& workspace;
  torch::Tensor& output;
  rainfusion::H3RainfusionLaunchCounters launches;
  int64_t rank;

  torch::Tensor attention(const torch::Tensor& query,
                          const torch::Tensor& key,
                          const torch::Tensor& value,
                          int64_t live_rows) {
    const int64_t sp_rows = query.size(1);
    const int64_t heads = rainfusion::kH3RainfusionHeads;
    const int64_t dim = rainfusion::kH3RainfusionHeadDim;
    CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes() &&
          query.size(0) == 1 && query.size(2) == heads &&
          query.size(3) == dim && live_rows == geometry.valid_rows &&
          sp_rows >= live_rows && sp_rows <= geometry.total_rows)
        << "MiniMax-H3 RainFusion received incompatible post-Ulysses Q/K/V";
    CHECK(query.scalar_type() == torch::kBFloat16 &&
          key.scalar_type() == torch::kBFloat16 &&
          value.scalar_type() == torch::kBFloat16)
        << "MiniMax-H3 RainFusion requires BF16 Q/K/V";
    const auto pack = [&](const torch::Tensor& input) {
      if (sp_rows == geometry.total_rows && input.is_contiguous()) {
        return input.view({sp_rows, heads * dim});
      }
      // The kernel receives only real rows plus its own storage padding.
      torch::Tensor storage =
          torch::empty({geometry.total_rows, heads, dim}, input.options());
      storage.narrow(0, 0, live_rows)
          .copy_(input.squeeze(0).narrow(0, 0, live_rows));
      storage.slice(0, live_rows).zero_();
      return storage.view({geometry.total_rows, heads * dim});
    };
    const torch::Tensor q = pack(query);
    const torch::Tensor k = pack(key);
    const torch::Tensor v = pack(value);
    rainfusion::h3_rainfusion_attention_out(
        q, k, v, geometry, workspace, output, &launches);

    // Restore the Ulysses sequence extent before the inverse exchange.
    return output.narrow(0, 0, sp_rows).view(query.sizes());
  }
};

class H3RainfusionResources final {
 public:
  H3RainfusionResources(ProcessGroup* world,
                        const torch::Device& device,
                        H3RainfusionPolicy policy)
      : world_(world),
        device_(device),
        policy_(policy),
        allocation_failed_(torch::zeros(
            {1},
            torch::TensorOptions().device(device).dtype(torch::kInt32))) {}

  std::optional<H3RainfusionRequest> prepare(int64_t valid_rows,
                                             int64_t video_start,
                                             int64_t grid_t,
                                             int64_t grid_h,
                                             int64_t grid_w,
                                             int64_t steps) {
    const rainfusion::H3RainfusionGeometry geometry{
        .total_rows = rainfusion::h3_align_up(
            valid_rows, rainfusion::kH3RainfusionKvPackRows),
        .valid_rows = valid_rows,
        .video_start = video_start,
        .grid_t = grid_t,
        .grid_h = grid_h,
        .grid_w = grid_w,
        .topk_r =
            rainfusion::h3_rainfusion_derive_topk_r(rainfusion::h3_ceildiv(
                valid_rows, rainfusion::kH3RainfusionBlock))};
    const char* reason = "unsupported_native_geometry";
    if (steps != 50) {
      reason = "unsupported_native_step_count";
    } else if (geometry.total_rows >
               rainfusion::kH3RainfusionQualifiedCapacity.total_rows) {
      reason = "logical_geometry_exceeds_capacity";
    } else if (rainfusion::is_qualified_h3_rainfusion_profile(geometry) &&
               video_start + grid_t * grid_h * grid_w == valid_rows) {
      reason = rainfusion::h3_rainfusion_unavailability_reason(geometry);
    }
    const std::string_view unavailable(reason);
    const bool available = unavailable.empty();
    const bool missing_objects =
        unavailable == "rainfusion_disabled_at_build_time" ||
        unavailable == "capacity_object_absent";
    const int64_t decision = missing_objects ? -1 : (available ? 1 : 0);
    const torch::Tensor admission = torch::tensor({valid_rows,
                                                   video_start,
                                                   grid_t,
                                                   grid_h,
                                                   grid_w,
                                                   steps,
                                                   policy_.layer_begin,
                                                   decision},
                                                  torch::dtype(torch::kInt64))
                                        .to(device_);
    const torch::Tensor admitted = world_->allgather_base_sync(admission).cpu();
    CHECK(admitted.eq(admitted[0]).all().item<bool>())
        << "MiniMax-H3 RainFusion geometry, policy or admission differs across "
           "ranks";
    CHECK(!missing_objects) << "MiniMax-H3 RainFusion setup failed: " << reason;
    if (!available) {
      LOG(INFO) << "H3_RF_ADMISSION rank=" << world_->rank()
                << " decision=DENSE_FALLBACK policy=" << policy_.name
                << " reason=" << reason << " valid_rows=" << valid_rows
                << " storage_rows=" << geometry.total_rows << " grid=" << grid_t
                << "x" << grid_h << "x" << grid_w;
      return std::nullopt;
    }
    const bool changed = !workspace_ ||
                         geometry.total_rows != geometry_.total_rows ||
                         geometry.valid_rows != geometry_.valid_rows ||
                         geometry.video_start != geometry_.video_start ||
                         geometry.grid_t != geometry_.grid_t ||
                         geometry.grid_h != geometry_.grid_h ||
                         geometry.grid_w != geometry_.grid_w;
    std::unique_ptr<rainfusion::H3RainfusionWorkspace> new_workspace;
    torch::Tensor new_output;
    std::exception_ptr failure;
    try {
      if (changed) {
        new_workspace = std::make_unique<rainfusion::H3RainfusionWorkspace>(
            rainfusion::allocate_h3_rainfusion_workspace(geometry, device_));
        new_output = torch::zeros(
            {geometry.total_rows,
             rainfusion::kH3RainfusionHeads * rainfusion::kH3RainfusionHeadDim},
            torch::TensorOptions().device(device_).dtype(torch::kBFloat16));
      }
    } catch (const c10::OutOfMemoryError&) {
      failure = std::current_exception();
    } catch (const std::bad_alloc&) {
      failure = std::current_exception();
    }
    allocation_failed_.fill_(failure ? 1 : 0);
    world_->allreduce(allocation_failed_);
    if (allocation_failed_[0].item<int32_t>() != 0) {
      if (failure) {
        std::rethrow_exception(failure);
      }
      TORCH_CHECK(
          false,
          "MiniMax-H3 RainFusion workspace allocation failed on a peer rank");
    }
    if (changed) {
      workspace_ = std::move(new_workspace);
      output_ = std::move(new_output);
      geometry_ = geometry;
    }
    LOG(INFO) << "H3_RF_ADMISSION rank=" << world_->rank()
              << " decision=SPARSE policy=" << policy_.name << " selector_key="
              << rainfusion::detail::h3_rainfusion_selected_selector_key()
              << " attention_key="
              << rainfusion::detail::h3_rainfusion_selected_attention_key()
              << " valid_rows=" << valid_rows
              << " storage_rows=" << geometry.total_rows
              << " video_start=" << video_start << " grid=" << grid_t << "x"
              << grid_h << "x" << grid_w << " topk_r=" << geometry.topk_r;
    return H3RainfusionRequest{
        policy_, geometry_, *workspace_, output_, {}, world_->rank()};
  }

 private:
  ProcessGroup* world_;
  torch::Device device_;
  H3RainfusionPolicy policy_;
  torch::Tensor allocation_failed_;
  rainfusion::H3RainfusionGeometry geometry_;
  std::unique_ptr<rainfusion::H3RainfusionWorkspace> workspace_;
  torch::Tensor output_;
};
#else
struct H3RainfusionRequest;
#endif

}  // namespace xllm::minimax_h3

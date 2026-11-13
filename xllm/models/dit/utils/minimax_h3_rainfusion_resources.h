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

#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#include "models/dit/utils/minimax_h3_qualified_execution_context.h"

namespace xllm {

#if defined(USE_NPU)
// One instance per serialized model execution slot. Construct only when the
// explicit RF option is enabled, before request-time workspace allocations.
class H3QualifiedRainfusionResources final {
 public:
  explicit H3QualifiedRainfusionResources(const torch::Device& device)
      : device_(device),
        setup_status_(torch::zeros(
            {2},
            torch::TensorOptions().device(device).dtype(torch::kInt32))) {}

  void prepare(const H3QualifiedRequest& request,
               H3QualifiedExecutionContext& execution) {
    H3RainfusionRequestState& state = execution.rainfusion;
    state.ready = false;
    execution.metrics.rf_ready = false;
    execution.metrics.fallback_reason.clear();
    state.borrowed_kernel_context = {};
    if (!state.policy.enabled()) {
      return;
    }
    ProcessGroup* world = execution.consensus_group;
    if (world == nullptr || world->world_size() != 16 ||
        world->device() != device_ || execution.metrics.infer_steps != 49) {
      throw std::invalid_argument(
          "H3 RainFusion requires world16 and the qualified 49-step schedule");
    }
    const H3PackedLayout& layout = request.layout;
    const kernel::npu::tilelang::H3RainfusionGeometry geometry{
        .total_rows = layout.aligned_length,
        .valid_rows = layout.used_length,
        .video_start = layout.target_video_slice.start,
        .grid_t = layout.target_latent_grid.temporal,
        .grid_h = layout.target_latent_grid.height,
        .grid_w = layout.target_latent_grid.width,
        // Derived from the nominal 0.8 sparsity, never a literal: the fixed-K
        // SELECTION STRATEGY is what must survive a shape change, not the
        // value 94.
        .topk_r = kernel::npu::tilelang::h3_rainfusion_derive_topk_r(
            (layout.used_length + kernel::npu::tilelang::kH3RainfusionBlock -
             1) /
            kernel::npu::tilelang::kH3RainfusionBlock)};
    // Invalid explicit input is a setup error, not a missing-specialization
    // fallback. Geometry comes from validated typed rows, never CLI literals.
    if (!kernel::npu::tilelang::is_qualified_h3_rainfusion_profile(geometry) ||
        layout.target_video_slice.stop != geometry.valid_rows) {
      throw std::invalid_argument(
          "H3 RainFusion request geometry is outside the qualified profile");
    }
    const bool available =
        kernel::npu::tilelang::has_h3_rainfusion_specialization(geometry);
    setup_status_[0].fill_(available ? 1 : 0);
    setup_status_[1].fill_(state.policy.layer_begin != 0 ? 1 : 0);
    world->allreduce(setup_status_);
    const auto admission = setup_status_.cpu();
    const int32_t available_count = admission[0].item<int32_t>();
    const int32_t middle_layers_count = admission[1].item<int32_t>();
    if (middle_layers_count != 0 &&
        middle_layers_count != world->world_size()) {
      throw std::invalid_argument("H3 RainFusion policy differs across ranks");
    }
    if (available_count == 0) {
      // Name the reason, so a LEGAL geometry that merely has no sparse video
      // query is distinguishable from one that exceeds the compiled capacity
      // and from a genuinely absent object. The classifier depends only on the
      // geometry, which is identical on every rank, so the reason is
      // rank-symmetric.
      const char* local_reason =
          kernel::npu::tilelang::h3_rainfusion_unavailability_reason(geometry);
      state.fallback_reason =
          (local_reason != nullptr && local_reason[0] != '\0')
              ? local_reason
              : "no_complete_specialization_on_all_ranks";
      execution.metrics.fallback_reason = state.fallback_reason;
      const int64_t total_blocks =
          (geometry.valid_rows + kernel::npu::tilelang::kH3RainfusionBlock -
           1) /
          kernel::npu::tilelang::kH3RainfusionBlock;
      // Make the fallback OBSERVABLE per rank: a legal-but-unsupported geometry
      // must be reported as dense and never silently mistaken for sparse.
      LOG(INFO) << "H3_RF_ADMISSION rank=" << world->rank()
                << " decision=DENSE_FALLBACK reason=" << state.fallback_reason
                << " total_rows=" << geometry.total_rows
                << " valid_rows=" << geometry.valid_rows
                << " total_blocks=" << total_blocks << " n_tiles="
                << (total_blocks + kernel::npu::tilelang::kH3RainfusionNTile -
                    1) /
                       kernel::npu::tilelang::kH3RainfusionNTile
                << " video_start=" << geometry.video_start
                << " grid=" << geometry.grid_t << "x" << geometry.grid_h << "x"
                << geometry.grid_w << " topk_r=" << geometry.topk_r;
      return;
    }
    if (available_count != 16) {
      throw std::runtime_error(
          "H3 RainFusion specialization deployment differs across ranks");
    }
    // Per-rank admission receipt: the capacity keys the runtime actually
    // selected, the derived fixed-K budget, and the storage/valid split.
    // Emitted once per request per rank, after the pair admission succeeded on
    // all 16 ranks.
    {
      const kernel::npu::tilelang::H3RainfusionDerived d =
          kernel::npu::tilelang::h3_rainfusion_derive(geometry);
      LOG(INFO)
          << "H3_RF_ADMISSION rank=" << world->rank() << " decision=SPARSE"
          << " selector_key="
          << kernel::npu::tilelang::detail::
                 h3_rainfusion_selected_selector_key()
          << " attention_key="
          << kernel::npu::tilelang::detail::
                 h3_rainfusion_selected_attention_key()
          << " storage_rows=" << geometry.total_rows
          << " valid_rows=" << geometry.valid_rows << " storage_minus_valid="
          << (geometry.total_rows - geometry.valid_rows)
          << " storage_256_aligned=" << (geometry.total_rows % 256 == 0)
          << " total_blocks=" << d.total_blocks << " topk_r=" << geometry.topk_r
          << " sparsity=0.8"
          << " kv_tail_rows=" << d.tail_rows << " ff_blocks=" << d.ff_blocks
          << " video_full_blocks=" << d.video_full_blocks
          << " num_sq=" << d.num_sq << " plan_width=" << d.plan_width
          << " grid=" << geometry.grid_t << "x" << geometry.grid_h << "x"
          << geometry.grid_w << " video_start=" << geometry.video_start
          << " cap_total_rows="
          << kernel::npu::tilelang::kH3RainfusionQualifiedCapacity.total_rows
          << " cap_num_blocks="
          << kernel::npu::tilelang::kH3RainfusionQualifiedCapacity.num_blocks
          << " cap_topk_r="
          << kernel::npu::tilelang::kH3RainfusionQualifiedCapacity.topk_r
          // The ACTUAL logical tile count and the compiled capacity, not a
          // boolean. This field used to log (ceil(total_blocks/128) ==
          // cap_n_tiles), which was meaningful only while the scope limit
          // required equality; after the limit was relaxed to <= it printed 0
          // for a perfectly valid 3-tile geometry and read like a failure. Both
          // numbers are logged so a receipt shows the tile count that was
          // served and the headroom that remained.
          << " n_tiles=" << (d.total_blocks + 127) / 128
          << " cap_n_tiles=" << kernel::npu::tilelang::kH3RainfusionCapNTiles;
    }

    const bool changed = !workspace_ ||
                         geometry.total_rows != geometry_.total_rows ||
                         geometry.valid_rows != geometry_.valid_rows ||
                         geometry.video_start != geometry_.video_start ||
                         geometry.grid_t != geometry_.grid_t ||
                         geometry.grid_h != geometry_.grid_h ||
                         geometry.grid_w != geometry_.grid_w ||
                         geometry.topk_r != geometry_.topk_r;
    std::unique_ptr<kernel::npu::tilelang::H3RainfusionWorkspace>
        pending_workspace;
    torch::Tensor pending_output;
    std::exception_ptr allocation_error;
    try {
      if (changed) {
        pending_workspace =
            std::make_unique<kernel::npu::tilelang::H3RainfusionWorkspace>(
                kernel::npu::tilelang::allocate_h3_rainfusion_workspace(
                    geometry, device_));
        pending_output = torch::zeros(
            {geometry.total_rows, 512},
            torch::TensorOptions().device(device_).dtype(torch::kBFloat16));
      }
    } catch (const c10::OutOfMemoryError&) {
      allocation_error = std::current_exception();
    } catch (const std::bad_alloc&) {
      allocation_error = std::current_exception();
    }
    // Only typed allocation failures enter this second SUM. Unexpected
    // device/driver/communication failures propagate to worker cancellation;
    // an unusable device is never assumed capable of another collective.
    setup_status_.fill_(allocation_error ? 1 : 0);
    world->allreduce(setup_status_);
    if (setup_status_[0].item<int32_t>() != 0) {
      if (allocation_error) {
        std::rethrow_exception(allocation_error);
      }
      throw std::runtime_error(
          "H3 RainFusion workspace allocation failed on a peer rank");
    }
    if (changed) {
      workspace_ = std::move(pending_workspace);
      output_ = std::move(pending_output);
      geometry_ = geometry;
    }
    state.borrowed_kernel_context.policy = state.policy;
    state.borrowed_kernel_context.geometry = &geometry_;
    state.borrowed_kernel_context.workspace = workspace_.get();
    state.borrowed_kernel_context.output = &output_;
    state.borrowed_kernel_context.counters = &state.kernel_launches;
    state.fallback_reason.clear();
    state.ready = true;
    execution.metrics.rf_ready = true;
  }

 private:
  torch::Device device_;
  torch::Tensor setup_status_;
  kernel::npu::tilelang::H3RainfusionGeometry geometry_;
  std::unique_ptr<kernel::npu::tilelang::H3RainfusionWorkspace> workspace_;
  torch::Tensor output_;
};
#endif

}  // namespace xllm

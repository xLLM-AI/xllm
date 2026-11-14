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

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/framework/dit_cache/similarity_residual_cache.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "models/dit/utils/minimax_h3_execution_metrics.h"
#include "models/dit/utils/minimax_h3_qualified_input.h"
#include "models/dit/utils/minimax_h3_rainfusion_context.h"

namespace xllm {

struct H3QualifiedCacheProbeRows {
  int64_t used_rows = 0;
  torch::Tensor target_indices;
};

// Request-owned, deliberately not a resident model member. The pipeline owns
// the sole zero-based step clock; this context never increments another one.
struct H3QualifiedExecutionContext {
  H3QualifiedExecutionMetrics metrics;
  H3QualifiedLaunchSnapshot qkv_launches;
  H3RainfusionRequestState rainfusion;
  std::unique_ptr<SimilarityResidualCacheState> cache;
  H3QualifiedCacheProbeRows probe_rows;
  ProcessGroup* consensus_group = nullptr;
};

inline H3QualifiedCacheProbeRows h3_qualified_cache_probe_rows(
    const H3PackedLayout& layout,
    int32_t sp_rank,
    int32_t sp_size,
    const torch::Device& device) {
  if (sp_size <= 0 || sp_rank < 0 || sp_rank >= sp_size ||
      layout.aligned_length <= 0 || layout.aligned_length % sp_size != 0 ||
      layout.used_length <= 0 || layout.used_length > layout.aligned_length) {
    throw std::invalid_argument("H3 CacheDiT sequence shard is invalid");
  }
  const int64_t local_rows = layout.aligned_length / sp_size;
  const int64_t start = static_cast<int64_t>(sp_rank) * local_rows;
  H3QualifiedCacheProbeRows rows;
  rows.used_rows =
      std::clamp(layout.used_length - start, int64_t{0}, local_rows);
  if (rows.used_rows == 0) {
    throw std::invalid_argument(
        "H3 CacheDiT does not support an all-padding shard");
  }
  std::vector<int64_t> local_targets;
  int64_t global_targets = 0;
  const auto append = [&](const torch::Tensor& positions,
                          const torch::Tensor& mask) {
    if (!positions.defined() || !mask.defined() || positions.dim() != 1 ||
        mask.dim() != 1 || positions.scalar_type() != torch::kInt64 ||
        mask.scalar_type() != torch::kBool ||
        positions.numel() != mask.numel()) {
      throw std::invalid_argument("H3 CacheDiT target metadata is invalid");
    }
    const torch::Tensor host_positions = positions.to(torch::kCPU).contiguous();
    const torch::Tensor host_mask = mask.to(torch::kCPU).contiguous();
    const int64_t* values = host_positions.data_ptr<int64_t>();
    const bool* selected = host_mask.data_ptr<bool>();
    for (int64_t index = 0; index < positions.numel(); ++index) {
      if (values[index] < 0 || values[index] >= layout.used_length) {
        throw std::invalid_argument(
            "H3 CacheDiT target position is outside valid rows");
      }
      if (selected[index]) {
        ++global_targets;
      }
      if (selected[index] && values[index] >= start &&
          values[index] < start + local_rows) {
        local_targets.push_back(values[index] - start);
      }
    }
  };
  append(layout.img_pos, layout.update_mask);
  append(layout.audio_pos, layout.audio_update_mask);
  if (global_targets == 0) {
    throw std::invalid_argument("H3 CacheDiT request contains no target rows");
  }
  // A defined empty index tensor contributes five zeros to the world SUM;
  // undefined indices would instead select every used row.
  rows.target_indices = torch::tensor(local_targets, torch::kInt64).to(device);
  return rows;
}

inline H3QualifiedExecutionContext make_h3_qualified_execution_context(
    const H3QualifiedRequest& request,
    const DiTCacheConfig& cache_config,
    const ParallelArgs& parallel_args,
    int64_t infer_steps,
    const torch::Device& device) {
  if (infer_steps <= 0) {
    throw std::invalid_argument("H3 execution requires a positive step count");
  }
  H3QualifiedExecutionContext execution;
  execution.metrics.infer_steps = infer_steps;
  // N12 also needs this group with CacheDiT disabled. Borrow without creating
  // a tensor or launching a collective on the default dense path.
  execution.consensus_group = parallel_args.process_group_;
  if (cache_config.selected_policy != PolicyType::CACHE_DIT) {
    return execution;
  }
  if (parallel_args.world_size() != 16 || parallel_args.tp_size() != 2 ||
      parallel_args.sp_size() != 8 || parallel_args.dp_size() != 1 ||
      parallel_args.cfg_size() != 1 || execution.consensus_group == nullptr ||
      execution.consensus_group->world_size() != 16 ||
      parallel_args.dit_sp_group_ == nullptr ||
      parallel_args.dit_sp_group_->world_size() != 8) {
    throw std::invalid_argument(
        "H3 CacheDiT requires the qualified TP2/SP8 world16 topology");
  }
  const CacheDiTOptions& options = cache_config.cache_dit;
  if (options.front_blocks <= 0 || options.back_blocks < 0 ||
      options.front_blocks >= 50 ||
      options.back_blocks >= 50 - options.front_blocks) {
    throw std::invalid_argument(
        "H3 CacheDiT must retain a nonempty middle block stack");
  }
  execution.metrics.front_blocks = options.front_blocks;
  execution.metrics.back_blocks = options.back_blocks;
  execution.cache = std::make_unique<SimilarityResidualCacheState>(options);
  execution.probe_rows = h3_qualified_cache_probe_rows(
      request.layout, parallel_args.dit_sp_group_->rank(), 8, device);
  return execution;
}

}  // namespace xllm

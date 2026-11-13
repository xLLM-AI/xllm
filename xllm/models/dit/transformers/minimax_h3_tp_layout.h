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
#include <vector>

#include "models/dit/utils/minimax_h3_qualified_vae.h"

namespace xllm {

// The framework Ulysses all-to-all (parallel_state::all_to_all_4D) scatters the
// head dimension and hard-requires head_num % sp_size == 0. MiniMax-H3 has 56
// global heads, so each of the 2 TP ranks owns 28 logical heads and 28 % 8 = 4.
// Padding to 32 physical heads satisfies 32 % 8 == 0 and leaves 4 local heads
// per rank. The 4 dummy heads carry Q=K=V=0, so attention yields mean(V)=0 for
// them and they stay exactly zero through the inverse all-to-all; they are
// verified zero and trimmed away before the output projection.
inline constexpr int64_t kMiniMaxH3GlobalHeads = 56;
inline constexpr int64_t kMiniMaxH3TpSize = 2;
inline constexpr int64_t kMiniMaxH3SpSize = 8;
inline constexpr int64_t kMiniMaxH3HeadDim = 128;
inline constexpr int64_t kMiniMaxH3LogicalHeadsPerTp = 28;
inline constexpr int64_t kMiniMaxH3PhysicalHeadsPerTp = 32;
inline constexpr int64_t kMiniMaxH3DummyHeads = 4;
inline constexpr int64_t kMiniMaxH3LocalHeadsPerRank = 4;

inline constexpr int64_t kMiniMaxH3WorldSize = 16;
inline constexpr int64_t kMiniMaxH3DpSize = 1;
inline constexpr int64_t kMiniMaxH3CfgSize = 1;
inline constexpr int64_t kMiniMaxH3TextEncoderTpSize = 1;

struct MiniMaxH3Topology {
  int32_t world_size = 0;
  int32_t dp_size = 0;
  int32_t tp_size = 0;
  int32_t sp_size = 0;
  int32_t cfg_size = 0;
  int32_t vae_size = 0;
  int32_t text_encoder_tp_size = 0;
};

inline bool minimax_h3_topology_is_valid(const MiniMaxH3Topology& topology) {
  if (topology.world_size <= 0 || topology.dp_size <= 0 ||
      topology.tp_size <= 0 || topology.sp_size <= 0 ||
      topology.cfg_size <= 0 || topology.vae_size <= 0 ||
      topology.text_encoder_tp_size <= 0) {
    return false;
  }
  if (topology.cfg_size * topology.tp_size * topology.sp_size *
          topology.dp_size !=
      topology.world_size) {
    return false;
  }
  if (topology.cfg_size > 2 || topology.vae_size > topology.world_size ||
      topology.world_size % topology.vae_size != 0 ||
      topology.text_encoder_tp_size > topology.world_size ||
      topology.world_size % topology.text_encoder_tp_size != 0) {
    return false;
  }
  return topology.world_size == kMiniMaxH3WorldSize &&
         topology.dp_size == kMiniMaxH3DpSize &&
         topology.tp_size == kMiniMaxH3TpSize &&
         topology.sp_size == kMiniMaxH3SpSize &&
         topology.cfg_size == kMiniMaxH3CfgSize &&
         minimax_h3_qualified_vae_size_is_valid(topology.world_size,
                                                topology.vae_size) &&
         topology.text_encoder_tp_size == kMiniMaxH3TextEncoderTpSize;
}

inline bool minimax_h3_head_layout_is_valid() {
  return kMiniMaxH3PhysicalHeadsPerTp % kMiniMaxH3SpSize == 0 &&
         kMiniMaxH3LogicalHeadsPerTp % kMiniMaxH3SpSize != 0 &&
         kMiniMaxH3LogicalHeadsPerTp + kMiniMaxH3DummyHeads ==
             kMiniMaxH3PhysicalHeadsPerTp &&
         kMiniMaxH3LocalHeadsPerRank ==
             kMiniMaxH3PhysicalHeadsPerTp / kMiniMaxH3SpSize;
}

inline torch::Tensor minimax_h3_pad_heads_to_physical(const torch::Tensor& x) {
  CHECK(x.dim() >= 2)
      << "MiniMax-H3 head padding needs at least 2 dimensions, got " << x.dim();
  CHECK(x.size(-1) == kMiniMaxH3HeadDim)
      << "MiniMax-H3 head padding expects head_dim " << kMiniMaxH3HeadDim
      << ", got " << x.size(-1);
  CHECK(x.size(-2) == kMiniMaxH3LogicalHeadsPerTp)
      << "MiniMax-H3 head padding expects " << kMiniMaxH3LogicalHeadsPerTp
      << " logical heads, got " << x.size(-2);
  std::vector<int64_t> dummy_sizes(x.sizes().begin(), x.sizes().end());
  dummy_sizes[dummy_sizes.size() - 2] = kMiniMaxH3DummyHeads;
  return torch::cat({x, torch::zeros(dummy_sizes, x.options())}, /*dim=*/-2);
}

inline torch::Tensor minimax_h3_trim_heads_to_logical(const torch::Tensor& x) {
  CHECK(x.size(-1) == kMiniMaxH3HeadDim)
      << "MiniMax-H3 head trimming expects head_dim " << kMiniMaxH3HeadDim
      << ", got " << x.size(-1);
  CHECK(x.size(-2) == kMiniMaxH3PhysicalHeadsPerTp)
      << "MiniMax-H3 head trimming expects " << kMiniMaxH3PhysicalHeadsPerTp
      << " physical heads, got " << x.size(-2);
  return x.narrow(-2, 0, kMiniMaxH3LogicalHeadsPerTp);
}

inline bool minimax_h3_dummy_heads_are_zero(const torch::Tensor& x) {
  const torch::Tensor dummy_heads =
      x.narrow(-2, kMiniMaxH3LogicalHeadsPerTp, kMiniMaxH3DummyHeads);
  return torch::count_nonzero(dummy_heads).item<int64_t>() == 0;
}

}  // namespace xllm

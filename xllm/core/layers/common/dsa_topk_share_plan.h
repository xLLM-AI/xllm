/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm::layer {

struct DsaTopkShareDecision {
  bool reuse_topk = false;
  bool output_topk = false;
};

inline bool has_dsa_indexer(const ModelArgs& args) {
  return args.index_n_heads() > 0 && args.index_head_dim() > 0 &&
         args.index_topk() > 0;
}

// Cross-step MTP reuse is a backend-independent DSA policy. Backends own the
// carrier representation used to transport their top-k state between steps.
inline bool is_mtp_dsa_topk_reuse_enabled(const ModelArgs& args) {
  return args.model_type().ends_with("_mtp") &&
         args.index_share_for_mtp_iteration() && has_dsa_indexer(args);
}

// Resolves and validates the complete cross-layer plan once. Consumers use the
// resolved decisions instead of independently interpreting the raw config.
class DsaTopkSharePlan final {
 public:
  explicit DsaTopkSharePlan(const ModelArgs& args) {
    const int64_t num_layers = args.n_layers();
    decisions_.resize(static_cast<size_t>(num_layers));
    has_indexer_ = has_dsa_indexer(args);

    if (!has_indexer_ ||
        (args.index_topk_pattern().empty() && args.index_topk_freq() <= 1)) {
      return;
    }

    if (!args.index_topk_pattern().empty()) {
      const std::string& pattern = args.index_topk_pattern();
      CHECK_EQ(static_cast<int64_t>(pattern.size()), num_layers)
          << "DSA top-k sharing pattern length must equal num_hidden_layers.";
      for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        decisions_[static_cast<size_t>(layer_id)].reuse_topk =
            should_reuse_from_pattern(pattern, layer_id);
      }
    } else {
      const int32_t freq = args.index_topk_freq();
      const int32_t offset = args.index_skip_topk_offset();
      CHECK_GT(freq, 1) << "DSA top-k sharing freq must be greater than 1.";
      CHECK_GE(offset, 0) << "DSA top-k sharing offset must be non-negative.";
      for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        decisions_[static_cast<size_t>(layer_id)].reuse_topk =
            should_reuse_from_freq(freq, offset, layer_id);
      }
    }

    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      DsaTopkShareDecision& decision =
          decisions_[static_cast<size_t>(layer_id)];
      has_reuse_ = has_reuse_ || decision.reuse_topk;
      const bool next_layer_reuses =
          layer_id + 1 < num_layers &&
          decisions_[static_cast<size_t>(layer_id + 1)].reuse_topk;
      decision.output_topk = !decision.reuse_topk && next_layer_reuses;
    }
  }

  const DsaTopkShareDecision& decision_for(int32_t layer_id) const {
    CHECK_GE(layer_id, 0) << "DSA top-k sharing layer id must be non-negative.";
    CHECK_LT(static_cast<size_t>(layer_id), decisions_.size())
        << "DSA top-k sharing layer id exceeds num_hidden_layers.";
    return decisions_[static_cast<size_t>(layer_id)];
  }

  bool has_reuse() const { return has_reuse_; }

  bool layer_uses_indexer(int32_t layer_id) const {
    return has_indexer_ && !decision_for(layer_id).reuse_topk;
  }

  int64_t num_indexer_layers() const {
    if (!has_indexer_) {
      return 0;
    }
    return static_cast<int64_t>(
        std::count_if(decisions_.begin(),
                      decisions_.end(),
                      [](const DsaTopkShareDecision& decision) {
                        return !decision.reuse_topk;
                      }));
  }

 private:
  static bool should_reuse_from_pattern(const std::string& pattern,
                                        int32_t layer_id) {
    const char symbol = static_cast<char>(std::toupper(
        static_cast<unsigned char>(pattern[static_cast<size_t>(layer_id)])));
    CHECK(symbol == 'F' || symbol == 'S')
        << "DSA top-k sharing pattern only supports F/S, got " << symbol;
    return symbol == 'S';
  }

  static bool should_reuse_from_freq(int32_t freq,
                                     int32_t offset,
                                     int32_t layer_id) {
    if (offset > 0) {
      return std::max<int32_t>(layer_id - offset + 1, 0) % freq != 0;
    }
    return std::max<int32_t>(layer_id - 1, 0) % freq != 0;
  }

  std::vector<DsaTopkShareDecision> decisions_;
  bool has_indexer_ = false;
  bool has_reuse_ = false;
};

inline std::vector<bool> get_dsa_indexer_layer_mask(const ModelArgs& args,
                                                    int64_t num_cache_layers) {
  CHECK_GE(num_cache_layers, 0)
      << "DSA indexer cache layer count must be non-negative.";
  CHECK_LE(num_cache_layers, std::numeric_limits<int32_t>::max())
      << "DSA indexer cache layer count exceeds int32 range.";
  if (!has_dsa_indexer(args)) {
    return {};
  }

  std::vector<bool> layer_mask(static_cast<size_t>(num_cache_layers), true);
  if (args.model_type().ends_with("_mtp")) {
    return layer_mask;
  }

  const DsaTopkSharePlan topk_share_plan(args);
  for (int32_t layer_id = 0; layer_id < num_cache_layers; ++layer_id) {
    layer_mask[static_cast<size_t>(layer_id)] =
        topk_share_plan.layer_uses_indexer(layer_id);
  }
  return layer_mask;
}

inline DsaTopkShareDecision get_dsa_topk_share_decision(const ModelArgs& args,
                                                        int32_t layer_id) {
  const DsaTopkSharePlan topk_share_plan(args);
  return topk_share_plan.decision_for(layer_id);
}

// Prefill context parallelism combined with a DSA cross-layer top-k share plan
// is not supported yet: CP shards the sparse block table per local shard, which
// is incompatible with cross-layer top-k reuse.
inline bool cp_conflicts_with_dsa_topk_share(
    bool enable_prefill_cp,
    const DsaTopkSharePlan& topk_share_plan) {
  return enable_prefill_cp && topk_share_plan.has_reuse();
}

}  // namespace xllm::layer

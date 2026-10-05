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

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/util/utils.h"

namespace xllm {

inline bool configure_glm5_next_mtp_args(ModelArgs& model_args,
                                         std::string_view speculative_algorithm,
                                         bool is_draft_engine) {
  if (!is_draft_engine ||
      !SpeculativeConfig::is_mtp_algorithm(speculative_algorithm) ||
      model_args.model_type() != "glm5_next") {
    return false;
  }
  CHECK_EQ(model_args.num_nextn_predict_layers(), 1)
      << "GLM MTP requires exactly one appended prediction layer.";
  CHECK_GT(model_args.n_layers(), 0);
  model_args.mtp_start_layer_idx(model_args.n_layers());
  model_args.model_type("glm5_next_mtp");
  model_args.n_layers(1);
  model_args.first_k_dense_replace(0);
  model_args.layer_types({"deepseek_sparse_attention"});
  model_args.full_attn_layers({0});
  model_args.mlp_layer_types({"sparse"});
  model_args.indexer_types({"full"});
  if (model_args.index_share_for_mtp_iteration()) {
    model_args.index_topk_freq(std::max(model_args.index_topk_freq(), 2));
    model_args.index_skip_topk_offset(0);
    model_args.index_topk_pattern("S");
  }
  return true;
}

inline int64_t mtp_hidden_state_width(const ModelArgs& model_args) {
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    return model_args.hc_mult() * model_args.hidden_size();
  }
  return model_args.hidden_size();
}

inline bool uses_embedded_eagle3_draft(std::string_view algorithm,
                                       const ModelArgs& target_model_args) {
  return algorithm == "Eagle3" &&
         target_model_args.enable_embedded_eagle3_draft();
}

}  // namespace xllm

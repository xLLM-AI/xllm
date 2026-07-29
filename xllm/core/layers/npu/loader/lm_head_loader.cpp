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

#include "lm_head_loader.h"

namespace xllm {
namespace layer {

LmHeadLoader::LmHeadLoader(uint64_t weight_count,
                           const ModelContext& context,
                           LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  if (load_to_host()) {
    working_tensors()[0] = torch::zeros({1});
  } else {
    working_tensors()[0] = torch::zeros({1}).to(options);
  }
  vocab_size_ = context.get_model_args().vocab_size();
  padded_vocab_size_ = get_padded_vocab_size(context);
}

void LmHeadLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  if (cp_size_ > 1 || dp_size_ > 1) {
    // CP-unaware, DP-aware: shard the lm_head weight across the dp-local-TP
    // group (world / dp_size = cp_size * tp_size), matching the LM_HEAD_TP
    // comm group built by MappingNPU. Using the CP-local TP
    // (dp_local_tp_size_, size tp) here would load the SAME shard on every CP
    // rank and replicate the weight across CP. When cp_size == 1 this collapses
    // to tp_size, so non-CP runs are unchanged.
    const int32_t cp_unaware_tp_size = parallel_args_.world_size() / dp_size_;
    const int32_t cp_unaware_tp_rank =
        parallel_args_.rank() % cp_unaware_tp_size;
    set_weight_with_padding(state_dict,
                            "weight",
                            0,
                            0,
                            cp_unaware_tp_rank,
                            cp_unaware_tp_size,
                            padded_vocab_size_,
                            to_host);
  } else if (parallel_args_.world_size() > 1) {
    set_weight_with_padding(state_dict,
                            "weight",
                            0,
                            0,
                            parallel_args_.rank(),
                            parallel_args_.world_size(),
                            padded_vocab_size_,
                            to_host);
  } else {
    set_weight_with_padding(
        state_dict, "weight", 0, 0, padded_vocab_size_, to_host);
  }
}

void LmHeadLoader::verify_loaded_weights(const std::string& weight_str) const {
  CHECK(working_tensors()[0].sizes() != std::vector<int64_t>({1}))
      << "final lm_head weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace xllm

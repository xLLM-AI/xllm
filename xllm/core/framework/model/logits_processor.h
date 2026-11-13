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

#include "core/framework/model/causal_lm.h"

namespace xllm {

// Logits stage: selected hidden rows → head projection. Pre-head norm is the
// model's (CausalLM::project_lm_head); post-projection transforms belong to
// the sampling layer.
class LogitsProcessor {
 public:
  // hidden_states: [num_tokens, hidden_size]
  // selected_idxes: [num_selected] (undefined = all rows)
  // returns: [num_selected, vocab_size]
  torch::Tensor forward(CausalLM& model,
                        const torch::Tensor& hidden_states,
                        const torch::Tensor& selected_idxes) {
    torch::Tensor out_hidden;
    return forward(model, hidden_states, selected_idxes, out_hidden);
  }

  torch::Tensor forward(CausalLM& model,
                        const torch::Tensor& hidden_states,
                        const torch::Tensor& selected_idxes,
                        torch::Tensor& out_hidden) {
    torch::Tensor logits;
    model.project_lm_head(hidden_states, selected_idxes, out_hidden, logits);
    return logits;
  }
};

}  // namespace xllm

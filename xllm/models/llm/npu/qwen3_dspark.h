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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/dspark_markov_head.h"
#include "models/llm/npu/qwen3_dflash.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

// DSpark draft model = DFlash block-diffusion backbone (context-K/V injection,
// prefill, weight loading all inherited unchanged) + a low-rank Markov head
// held by the ForCausalLM layer. The backbone remains independent from
// sampling.
class DSparkQwen3ModelImpl final : public DFlashQwen3ModelImpl {
 public:
  explicit DSparkQwen3ModelImpl(const ModelContext& context)
      : DFlashQwen3ModelImpl(context) {}
};
TORCH_MODULE(DSparkQwen3Model);

class DSparkQwen3ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DSparkQwen3Model> {
 public:
  explicit DSparkQwen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DSparkQwen3Model>(context) {
    const ModelArgs& model_args = context.get_model_args();
    markov_head_.initialize(context.get_tensor_options(),
                            model_args.markov_rank());
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) const {
    return markov_head_.bias(previous_token_ids);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
      markov_head_.load_state_dict(
          sub_dict.get_dict_with_prefix("markov_head."));
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
    markov_head_.verify_loaded_weights("markov_head.");
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

 private:
  DSparkMarkovHead markov_head_;
};
TORCH_MODULE(DSparkQwen3ForCausalLM);

// Draft config carries model_type="qwen3"; worker_impl overwrites
// args.model_type to "DSparkDraftModel" so this factory builds the draft body.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(dspark_draft_model,
                                   DSparkDraftModel,
                                   DSparkQwen3ForCausalLM);

}  // namespace xllm::npu::model

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
#include <memory>
#include <string>
#include <vector>

#include "framework/config/speculative_config.h"
#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/dspark_confidence_head.h"
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
      : LlmForCausalLMImplBase<DSparkQwen3Model>(context),
        markov_head_(context.get_tensor_options(),
                     context.get_model_args().markov_rank()) {
    const ModelArgs& model_args = context.get_model_args();
    // Only stand up the ConfidenceHead when adaptive speculative decode is on:
    // it is consumed solely by the adaptive pruning controller, so under static
    // decoding its weights would occupy device memory and never be read.
    if (model_args.enable_confidence_head() &&
        SpeculativeConfig::get_instance()
            .enable_adaptive_speculative_decode()) {
      confidence_head_.initialize(context.get_tensor_options(),
                                  model_args.hidden_size(),
                                  model_args.markov_rank(),
                                  model_args.confidence_head_with_markov());
    }
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) const {
    return markov_head_.bias(previous_token_ids);
  }

  // Compute per-request acceptance probability using the trained ConfidenceHead
  // over the draft-step hidden state and the previous token embedding.
  // hidden: [num_reqs, hidden_size], prev_token_ids: [num_reqs].
  // Returns [num_reqs] fp32 in [0, 1]. Defined only when
  // enable_confidence_head.
  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous_token_ids) const {
    CHECK(confidence_head_.defined())
        << "DSpark ConfidenceHead is not initialized (enable_confidence_head?)";
    torch::Tensor markov_embed;
    if (previous_token_ids.defined()) {
      markov_embed = markov_head_.markov_embed(previous_token_ids);
    }
    return confidence_head_.forward(hidden, markov_embed);
  }

  // Batched variant of dspark_confidence_probs over the whole draft block.
  //   hidden_all:  [num_reqs, num_spec, hidden_size]
  //   prev_matrix: [num_reqs, num_spec] int64 — column k is step k's "prev"
  //                token (col 0 = anchor, col k = draft token sampled at k-1).
  // Returns [num_reqs, num_spec] fp32 in [0, 1]. Defined only when
  // enable_confidence_head.
  torch::Tensor dspark_confidence_probs_batched(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) const {
    CHECK(confidence_head_.defined())
        << "DSpark ConfidenceHead is not initialized (enable_confidence_head?)";
    torch::Tensor markov_embed_all;
    if (prev_matrix.defined()) {
      markov_embed_all = markov_head_.markov_embed(prev_matrix);
    }
    return confidence_head_.forward_batched(hidden_all, markov_embed_all);
  }

  bool has_dspark_confidence_head() const { return confidence_head_.defined(); }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
      // The shared DSparkMarkovHead reads unprefixed markov_w1/markov_w2 keys,
      // so strip the "markov_head." prefix the checkpoint stores them under.
      markov_head_.load_state_dict(
          sub_dict.get_dict_with_prefix("markov_head."));
      confidence_head_.load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
    markov_head_.verify_loaded_weights("");
    if (confidence_head_.defined()) {
      confidence_head_.verify_loaded_weights("");
    }
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
  DSparkConfidenceHead confidence_head_;
};
TORCH_MODULE(DSparkQwen3ForCausalLM);

// Draft config carries model_type="qwen3"; worker_impl overwrites
// args.model_type to "DSparkDraftModel" so this factory builds the draft body.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(dspark_draft_model,
                                   DSparkDraftModel,
                                   DSparkQwen3ForCausalLM);

}  // namespace xllm::npu::model

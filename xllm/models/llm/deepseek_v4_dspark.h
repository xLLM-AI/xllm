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

#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "models/llm/deepseek_v4.h"
#include "models/llm/dspark_confidence_head.h"
#include "models/llm/dspark_markov_head.h"
#include "models/llm/dspark_weight_source.h"
#include "models/model_registry.h"

namespace xllm {
namespace dspark_detail {

template <typename Module>
void load_vocab_weight(VocabularyWeightSelector& selector,
                       const StateDict& dedicated_state_dict,
                       const StateDict& fallback_state_dict,
                       Module& module) {
  if (dedicated_state_dict.size() > 0 &&
      selector.should_load(/*dedicated=*/true)) {
    module->load_state_dict(dedicated_state_dict);
    selector.mark_loaded(/*dedicated=*/true);
  }
  if (fallback_state_dict.size() > 0 &&
      selector.should_load(/*dedicated=*/false)) {
    module->load_state_dict(fallback_state_dict);
    selector.mark_loaded(/*dedicated=*/false);
  }
}

}  // namespace dspark_detail

// DeepSeek-V4-0731 stores the DSpark draft in mtp.0/1/2. Unlike the preview
// model's serial MTP layer, these are ordinary V4 decoder layers: target layer
// outputs are combined once and written as shared context KV for every draft
// layer, then a non-causal draft block is evaluated from token embeddings.
class DeepseekV4DSparkModelImpl final : public DeepseekV4ModelImpl {
 public:
  explicit DeepseekV4DSparkModelImpl(const ModelContext& context)
      : DeepseekV4ModelImpl(context),
        markov_head_(context.get_tensor_options(),
                     context.get_model_args().markov_rank()) {
    const ModelArgs& args = context.get_model_args();
    const int64_t capture_count = args.dspark_num_layers();
    CHECK_GT(capture_count, 0)
        << "DeepSeek-V4 DSpark requires dspark_num_layers > 0.";
    CHECK_EQ(args.n_layers(), args.dspark_num_layers())
        << "DeepSeek-V4 DSpark draft layer count mismatch.";
    CHECK_GT(args.markov_rank(), 0)
        << "DeepSeek-V4 DSpark requires dspark_markov_rank > 0.";

    // DeepSeek-V4 DSpark checkpoints carry a confidence head (with_markov, no
    // bias) under mtp.<last>.confidence_head. Its config has no explicit enable
    // flag, so presence is decided by whether the weight loads. Only stand it
    // up under adaptive speculative decode, which is its sole consumer;
    // otherwise its weights would occupy device memory unused.
    if (SpeculativeConfig::get_instance()
            .enable_adaptive_speculative_decode()) {
      confidence_head_.initialize(context.get_tensor_options(),
                                  args.hidden_size(),
                                  args.markov_rank(),
                                  /*with_markov=*/true);
    }

    const auto options = context.get_tensor_options();
    main_proj_ = register_module(
        "main_proj",
        layer::ReplicatedLinear(args.hidden_size() * capture_count,
                                args.hidden_size(),
                                /*bias=*/false,
                                context.get_quant_args(),
                                options));
    main_norm_ = register_module(
        "main_norm",
        layer::RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  void load_state_dict(const StateDict& state_dict) override {
    const int32_t last_layer = static_cast<int32_t>(layers_.size()) - 1;
    CHECK_GE(last_layer, 0);

    for (int32_t i = 0; i <= last_layer; ++i) {
      StateDict layer_dict =
          state_dict.get_dict_with_prefix("mtp." + std::to_string(i) + ".");
      if (layer_dict.size() > 0) {
        layers_[static_cast<size_t>(i)]->load_state_dict(layer_dict);
      }
    }

    StateDict first = state_dict.get_dict_with_prefix("mtp.0.");
    if (first.size() > 0) {
      main_proj_->load_state_dict(first.get_dict_with_prefix("main_proj."));
      main_norm_->load_state_dict(first.get_dict_with_prefix("main_norm."));
    }
    // Original FP checkpoints share the top-level vocabulary basis, while
    // QuaRot quantized checkpoints carry a dedicated mtp.0 basis. Accept both
    // layouts, but make the dedicated tensor win regardless of shard order.
    StateDict dedicated_embed = first.get_dict_with_prefix("embed.");
    StateDict fallback_embed = state_dict.get_dict_with_prefix("embed.");
    dspark_detail::load_vocab_weight(
        embedding_source_, dedicated_embed, fallback_embed, embed_tokens_);

    StateDict last = state_dict.get_dict_with_prefix(
        "mtp." + std::to_string(last_layer) + ".");
    if (last.size() > 0) {
      norm_->load_state_dict(last.get_dict_with_prefix("norm."));
      load_hc_head_state_dict(last);
      markov_head_.load_state_dict(last.get_dict_with_prefix("markov_head."));
      // confidence_head reads confidence_head.proj.{weight,bias}; DeepSeek-V4
      // ships a bias-less weight. Loads only if present.
      confidence_head_.load_state_dict(last);
    }
  }

  void verify_dspark_weights() const {
    CHECK(embedding_source_.loaded())
        << "Failed to load DeepSeek-V4 DSpark vocabulary embedding from "
           "mtp.0.embed.weight or embed.weight.";
    LOG(INFO) << "Loaded DeepSeek-V4 DSpark embedding from "
              << embedding_source_.source_name() << ".";
    markov_head_.verify_loaded_weights(
        "mtp." + std::to_string(layers_.size() - 1) + ".markov_head.");
    if (confidence_head_.defined()) {
      confidence_head_.verify_loaded_weights(
          "mtp." + std::to_string(layers_.size() - 1) + ".confidence_head.");
    }
  }

  int32_t last_dspark_layer_index() const {
    return static_cast<int32_t>(layers_.size()) - 1;
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) const {
    return markov_head_.bias(previous_token_ids);
  }

  bool has_dspark_confidence_head() const { return confidence_head_.defined(); }

  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous_token_ids) const {
    CHECK(confidence_head_.defined())
        << "DeepSeek-V4 DSpark ConfidenceHead is not loaded.";
    torch::Tensor markov_embed;
    if (previous_token_ids.defined()) {
      markov_embed = markov_head_.markov_embed(previous_token_ids);
    }
    return confidence_head_.forward(hidden, markov_embed);
  }

  torch::Tensor dspark_confidence_probs_batched(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) const {
    CHECK(confidence_head_.defined())
        << "DeepSeek-V4 DSpark ConfidenceHead is not loaded.";
    torch::Tensor markov_embed_all;
    if (prev_matrix.defined()) {
      markov_embed_all = markov_head_.markov_embed(prev_matrix);
    }
    return confidence_head_.forward_batched(hidden_all, markov_embed_all);
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    CHECK_EQ(kv_caches.size(), layers_.size())
        << "DeepSeek-V4 DSpark cache/layer count mismatch.";
    CHECK_EQ(device_cache_slots.numel(), target_hidden.size(0))
        << "DeepSeek-V4 DSpark context slot count mismatch.";

    torch::Tensor projected = main_proj_->forward(target_hidden);
    projected = std::get<0>(main_norm_->forward(projected));
    auto [cos, sin] = build_default_rope(positions);
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->write_context_kv(
          projected, cos, sin, device_cache_slots, kv_caches[i]);
#if defined(USE_NPU)
      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->record_event(
              static_cast<int64_t>(i), projected.device().index())) {
        return ModelOutput();
      }
#endif
    }
    return ModelOutput(projected);
  }

 private:
  layer::ReplicatedLinear main_proj_{nullptr};
  layer::RMSNorm main_norm_{nullptr};
  DSparkMarkovHead markov_head_;
  DSparkConfidenceHead confidence_head_;
  dspark_detail::VocabularyWeightSelector embedding_source_;
};
TORCH_MODULE(DeepseekV4DSparkModel);

class DeepseekV4DSparkForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4DSparkModel> {
 public:
  explicit DeepseekV4DSparkForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4DSparkModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "") override {
    UNUSED_PARAMETER(prefix);
    const int32_t last_layer = model_->last_dspark_layer_index();
    CHECK_GE(last_layer, 0);
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(*state_dict);
      StateDict dedicated_head = state_dict->get_dict_with_prefix(
          "mtp." + std::to_string(last_layer) + ".head.");
      StateDict fallback_head = state_dict->get_dict_with_prefix("head.");
      dspark_detail::load_vocab_weight(
          head_source_, dedicated_head, fallback_head, lm_head_);
    }
    model_->verify_dspark_weights();
    CHECK(head_source_.loaded() && lm_head_->is_weight_loaded())
        << "Failed to load DeepSeek-V4 DSpark vocabulary head from mtp."
        << last_layer << ".head.weight or head.weight.";
    LOG(INFO) << "Loaded DeepSeek-V4 DSpark LM head from "
              << head_source_.source_name() << ".";
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) const {
    return model_->dspark_markov_bias(previous_token_ids);
  }

  bool has_dspark_confidence_head() const {
    return model_->has_dspark_confidence_head();
  }

  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous_token_ids) const {
    return model_->dspark_confidence_probs(hidden, previous_token_ids);
  }

  torch::Tensor dspark_confidence_probs_batched(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) const {
    return model_->dspark_confidence_probs_batched(hidden_all, prev_matrix);
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

  bool requires_graph_forward_metadata() {
    return model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    model_->prepare_graph_forward_metadata(state, positions, input_params);
  }

 private:
  dspark_detail::VocabularyWeightSelector head_source_;
};
TORCH_MODULE(DeepseekV4DSparkForCausalLM);

REGISTER_CAUSAL_MODEL_WITH_VARNAME(deepseek_v4_dspark_draft_model,
                                   deepseek_v4_dspark,
                                   DeepseekV4DSparkForCausalLM);

}  // namespace xllm

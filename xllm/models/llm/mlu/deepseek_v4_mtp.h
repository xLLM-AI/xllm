/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"
#include "layers/mlu/deepseek_v4/dsa_cache_mapping.h"
#include "layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/deepseek_v4.h"
#include "models/llm/mlu/deepseek_v4_base.h"
#include "models/llm/mtp_model_base.h"

namespace xllm::mlu::model {

// Make xllm namespace types available unqualified inside this namespace,
// matching the style used in mlu/deepseek_v4.h.
using ::xllm::DSACacheInfo;
using ::xllm::DSACacheMapping;
using ::xllm::DSACacheType;
using ::xllm::DSAGroupInfo;
using ::xllm::JsonReader;
using ::xllm::KVCache;
using ::xllm::LlmForCausalLMImplBase;
using ::xllm::ModelArgs;
using ::xllm::ModelContext;
using ::xllm::ModelGraphMetadataState;
using ::xllm::ModelInputParams;
using ::xllm::ModelLoader;
using ::xllm::ModelOutput;
using ::xllm::MtpDecoderLayerImplBase;
using ::xllm::ParallelArgs;
using ::xllm::StateDict;
using ::xllm::layer::AttentionMetadata;
using ::xllm::layer::DeepseekV4DecoderLayer;
using ::xllm::layer::DSAMetadata;
using ::xllm::layer::DSAMetadataBuilderMlu;
using ::xllm::layer::RMSNorm;
using ::xllm::layer::WordEmbedding;

class DeepseekV4MultiTokenPredictorLayerImpl
    : public MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  DeepseekV4MultiTokenPredictorLayerImpl(const ModelContext& context,
                                         int32_t layer_index)
      : MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>(context,
                                                               layer_index) {}

  torch::Tensor forward(torch::Tensor inputs_embeds,
                        torch::Tensor previous_hidden_states,
                        torch::Tensor positions,
                        layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor tokens,
                        torch::Tensor* aux_hidden_states) {
    ModelInputParams modified_input_params = input_params;
    modified_input_params.embedding.input_embedding = previous_hidden_states;
    std::optional<torch::Tensor> residual;
    return MtpDecoderLayerImplBase<layer::DeepseekV4DecoderLayer>::forward(
        inputs_embeds,
        residual,
        positions,
        attn_metadata,
        kv_cache,
        modified_input_params,
        tokens,
        aux_hidden_states);
  }
};
TORCH_MODULE(DeepseekV4MultiTokenPredictorLayer);

class DeepseekV4MtpModelImpl final : public torch::nn::Module,
                                     private DeepseekV4Base {
 public:
  explicit DeepseekV4MtpModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    device_ = options.device();

    const int32_t mtp_n_layers = model_args_.n_layers();
    CHECK_GT(mtp_n_layers, 0)
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires n_layers > 0";
    CHECK_GE(model_args_.num_nextn_predict_layers(), 0)
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires "
           "num_nextn_predict_layers >= 0";

    const int32_t n_heads = model_args_.n_heads();
    const int64_t head_dim =
        model_args_.o_lora_rank() + model_args_.qk_rope_head_dim();
    const int64_t dp_local_tp_size =
        std::max<int64_t>(parallel_args.world_size() /
                              std::max<int64_t>(parallel_args.dp_size(), 1),
                          1);
    CHECK_EQ(n_heads % dp_local_tp_size, 0)
        << "[DeepseekV4Mtp] n_heads must be divisible by local tp "
           "size. n_heads="
        << n_heads << ", local_tp_size=" << dp_local_tp_size;
    hc_mult_ = model_args_.hc_mult();
    window_size_ = model_args_.window_size_;

    init_rope(model_args_, options);
    init_hadamard(model_args_, options);
    max_position_embeddings_ = model_args_.max_position_embeddings();

    mtp_layers_.reserve(mtp_n_layers);
    for (int32_t i = 0; i < mtp_n_layers; ++i) {
      const int32_t layer_index = i;
      mtp_layers_.emplace_back(
          DeepseekV4MultiTokenPredictorLayer(context, layer_index));
      register_module("layer_" + std::to_string(i), mtp_layers_.back());
    }

    build_dsa_cache_info(model_args_);

    for (int32_t layer_id = 0; layer_id < mtp_n_layers; ++layer_id) {
      mtp_layers_[static_cast<size_t>(layer_id)]->set_cache_mapping(
          cache_mappings_[static_cast<size_t>(layer_id)]);
    }

    final_norm_ = register_module("final_norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      mtp_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.norm."));
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("layers.0.emb.tok_emb."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    UNUSED_PARAMETER(prefix);
    for (const auto& layer : mtp_layers_) {
      layer->verify_loaded_weights();
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;

    const bool is_empty_dp_rank = !tokens.defined() || tokens.numel() == 0;
    if (is_empty_dp_rank) {
      tokens = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::tensor(
          {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    const torch::Device runtime_device = tokens.device();

    auto modified_input_params = input_params;
    if (is_empty_dp_rank) {
      layer::fill_dsv4_empty_dp_params(
          modified_input_params, group_infos_, window_size_);
    }

    torch::Tensor previous_hidden_states =
        modified_input_params.embedding.input_embedding;
    CHECK(previous_hidden_states.defined())
        << "[DeepseekV4Mtp] input_params.embedding.input_embedding must be "
           "defined for MTP model";

    torch::Tensor hidden_states = embed_tokens_(tokens);

    // Zero out embeddings at position 0
    auto mask = (positions == 0);
    if (mask.any().item<bool>()) {
      hidden_states.index_put_({mask},
                               torch::zeros_like(hidden_states.index({mask})));
    }

    tokens = maybe_to_device(tokens, runtime_device);
    positions = maybe_to_device(positions, runtime_device);

    const bool mlu_graph_forward = deepseek_v4_uses_mlu_graph(input_params);

    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    // Build DSA metadata if not already present
    if (!modified_input_params.attn_metadata ||
        !modified_input_params.attn_metadata->dsa_metadata) {
      CHECK(!mlu_graph_forward)
          << "[DeepseekV4Mtp] MLU graph requires prebuilt DSA metadata";
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::DSAMetadataBuilderMlu::build(modified_input_params,
                                                  positions,
                                                  caches_info_,
                                                  group_infos_,
                                                  window_size_));
    }
    layer::AttentionMetadata& attn_metadata =
        *(modified_input_params.attn_metadata);

    // Non-graph mode: prepare DSA metadata to device
    if (!mlu_graph_forward) {
      prepare_dsa_metadata(attn_metadata, runtime_device);
    }

    CHECK_GE(static_cast<int32_t>(kv_caches.size()),
             static_cast<int32_t>(mtp_layers_.size()))
        << "[DeepseekV4Mtp] deepseek_v4_mtp requires kv_caches size >= "
           "mtp layer count";

    torch::Tensor aux_hidden_states;
    for (size_t i = 0; i < mtp_layers_.size(); ++i) {
      const int32_t layer_id = static_cast<int32_t>(i);
      prepare_layer_metadata(attn_metadata, layer_id);
      hidden_states = mtp_layers_[i](hidden_states,
                                     previous_hidden_states,
                                     positions,
                                     attn_metadata,
                                     kv_caches[i],
                                     modified_input_params,
                                     tokens,
                                     &aux_hidden_states);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }

    // Apply final normalization
    auto [output, _] = final_norm_(hidden_states, std::nullopt);
    return ModelOutput(output, torch::Tensor(), aux_hidden_states.flatten(1));
  }

  // Re-export base class graph metadata interface as public.
  using DeepseekV4Base::create_graph_forward_metadata_state;
  using DeepseekV4Base::prepare_graph_forward_metadata;
  using DeepseekV4Base::requires_graph_forward_metadata;

 private:
  ModelArgs model_args_;
  torch::Device device_{torch::kCPU};

  layer::RMSNorm final_norm_{nullptr};
  layer::WordEmbedding embed_tokens_{nullptr};
  std::vector<DeepseekV4MultiTokenPredictorLayer> mtp_layers_;
};
TORCH_MODULE(DeepseekV4MtpModel);

class DeepseekV4MtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV4MtpModel> {
 public:
  explicit DeepseekV4MtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4MtpModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      std::unordered_map<std::string, torch::Tensor> remapped_dict;
      std::unordered_map<std::string, torch::Tensor> lm_head_dict;
      for (auto it = state_dict->begin(); it != state_dict->end(); ++it) {
        remapped_dict[detail::normalize_model_parameter_name(
            it->first, prefix)] = it->second;
        std::optional<std::string> lm_head_name =
            detail::normalize_lm_head_parameter_name(it->first);
        if (lm_head_name.has_value()) {
          lm_head_dict[lm_head_name.value()] = it->second;
        }
      }

      StateDict remapped_state_dict(remapped_dict);
      model_->load_state_dict(remapped_state_dict);
      if (!lm_head_dict.empty()) {
        lm_head_->load_state_dict(StateDict(lm_head_dict));
      } else {
        lm_head_->load_state_dict(
            remapped_state_dict.get_dict_with_prefix("layers.0.head."));
      }
    }
    model_->verify_loaded_weights(prefix);
  }

  bool requires_graph_forward_metadata() {
    return this->model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return this->model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input_params) {
    this->model_->prepare_graph_forward_metadata(
        state, positions, input_params);
  }
};
TORCH_MODULE(DeepseekV4MtpForCausalLM);

inline void load_deepseek_v4_mtp_model_args(const JsonReader& json,
                                            ModelArgs* args) {
  load_deepseek_v4_model_args(json, args);
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4_mtp");
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
  SET_ARG(n_hash_layers, 0);
}

REGISTER_CAUSAL_MODEL(deepseek_v4_mtp, DeepseekV4MtpForCausalLM);

REGISTER_MODEL_ARGS(deepseek_v4_mtp, [&] {
  const DeepseekV4ArgsPolicy args_policy = build_deepseek_v4_args_policy();
  load_deepseek_v4_mtp_model_args(json, args);
  process_deepseek_v4_args(args, args_policy);
  validate_deepseek_v4_args(*args, args_policy);
});

}  // namespace xllm::mlu::model
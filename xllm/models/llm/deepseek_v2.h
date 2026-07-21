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
#include <torch/torch.h>

#include <functional>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/model/model_output.h"
#if defined(USE_DCU)
#include "core/layers/dcu/deepseek_v2_decoder_layer_impl.h"
#elif defined(USE_MLU)
#include "core/layers/mlu/deepseek_v2_decoder_layer_impl.h"
#endif
#include "llm_model_base.h"

namespace xllm {

class DeepseekV2ModelImpl : public torch::nn::Module {
 protected:
  using DecoderLayerFactory =
      std::function<layer::DeepseekV2DecoderLayer(const ModelContext&,
                                                  int32_t)>;

  static DecoderLayerFactory default_decoder_layer_factory() {
    return [](const ModelContext& context, int32_t layer_id) {
      return layer::DeepseekV2DecoderLayer(context, layer_id);
    };
  }

 public:
  explicit DeepseekV2ModelImpl(const ModelContext& context)
      : DeepseekV2ModelImpl(context, default_decoder_layer_factory()) {}

 protected:
  DeepseekV2ModelImpl(const ModelContext& context,
                      const DecoderLayerFactory& decoder_layer_factory)
      : model_args_(context.get_model_args()),
        device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args_.n_layers());

    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args_.vocab_size(),
                                             model_args_.hidden_size(),
                                             context.get_parallel_args(),
                                             options));
    norm_ = register_module(
        "norm",
        layer::RMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));

    // create decoder layers
    for (int32_t i = 0; i < model_args_.n_layers(); ++i) {
      auto block = decoder_layer_factory(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    for (int i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

 public:
  ModelOutput forward_native(torch::Tensor tokens,
                             torch::Tensor positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& input_params) {
    // for dp, if tokens is empty, set tokens to 1 and positions to 0
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({1}).to(torch::kInt32).to(device_);
      }
      auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla(),
                                                     /*compute_dtype=*/"half",
                                                     /*attn_mask=*/std::nullopt,
                                                     /*device=*/device_));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    torch::Tensor hidden_states = embed_tokens_(tokens);
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      // NOTE: we will remove this until refactor flashinfer API
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = layers_[i];
      prepare_decoder_layer_for_forward(i, layer, attn_metadata);

      hidden_states = forward_decoder_layer(i,
                                            layer,
                                            hidden_states,
                                            residual,
                                            positions,
                                            attn_metadata,
                                            kv_caches[i],
                                            modified_input_params);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }
    auto [h, res] = norm_(hidden_states, residual);
    return ModelOutput(h, res);
  }

  // Provide batched signature to satisfy callers that pass vectors
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return forward_native(tokens, positions, kv_caches, input_params);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights() const {
    for (const auto& layer : layers_) {
      layer->verify_loaded_weights();
    }
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 protected:
  virtual void prepare_decoder_layer_for_forward(
      size_t /*layer_id*/,
      layer::DeepseekV2DecoderLayer& /*layer*/,
      const layer::AttentionMetadata& /*attn_metadata*/) {}

  // Single extension point for the per-layer invocation so subclasses can
  // attach model-specific per-layer plumbing (e.g. GLM5.2 DSA cross-layer
  // top-k sharing in glm52.h) without forking the whole forward loop.
  virtual torch::Tensor forward_decoder_layer(
      size_t layer_id,
      layer::DeepseekV2DecoderLayer& layer,
      torch::Tensor& hidden_states,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params) {
    return layer(hidden_states,
                 residual,
                 positions,
                 attn_metadata,
                 kv_cache,
                 input_params);
  }

  layer::WordEmbedding& embed_mod() { return embed_tokens_; }

  std::vector<layer::DeepseekV2DecoderLayer>& layers_ref() { return layers_; }

  layer::RMSNorm& norm_mod() { return norm_; }

  ModelArgs model_args_;

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::DeepseekV2DecoderLayer> layers_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
};
TORCH_MODULE(DeepseekV2Model);

class DeepseekV2ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV2Model> {
 public:
  DeepseekV2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV2Model>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(DeepseekV2ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v2, DeepseekV2ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
REGISTER_MODEL_ARGS(deepseek_v2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 102400);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 27);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 16);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 10944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 100001);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 100000);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 27);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "greedy");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(scoring_func, "scoring_func", "softmax");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 64);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 2);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 6);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1408);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", false);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 0);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({100001}));
});
}  // namespace xllm

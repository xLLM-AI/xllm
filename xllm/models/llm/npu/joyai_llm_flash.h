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

#include "core/framework/config/kv_cache_config.h"
#include "deepseek_v2.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm::npu::model {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class JoyAILLMFlashModelImpl : public torch::nn::Module {
 public:
  JoyAILLMFlashModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();

    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args.qk_rope_head_dim(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    max_seq_len_ = model_args.max_position_embeddings();
    int32_t mask_value = model_args.dtype() == "bfloat16" ? 1 : -9984;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = DeepseekV2DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    num_experts_per_tok_ = model_args.num_experts_per_tok();
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    auto h = npu_embed_tokens_(tokens, 0);
    auto cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto cos_sin_chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = cos_sin_chunks[0].contiguous();
    auto sin_pos = cos_sin_chunks[1].contiguous();

    torch::Tensor attn_mask;
    if (::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
        !input_params.meta.batch_forward_type.is_decode()) {
      attn_mask = attn_mask_.get_attn_mask(512, dtype_, device_);
    } else if (input_params.meta.batch_forward_type.is_prefill()) {
      attn_mask = attn_mask_.get_attn_mask(128, dtype_, device_);
    } else if (num_speculative_tokens_ > 0) {
      // TODO :the judgement of gen_free_mask need more check
      attn_mask = attn_mask_.gen_free_mask(
          num_speculative_tokens_ + 1, dtype_, device_);
    }

    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);
      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params,
            event,
            event_flag);
      rolling_guard.after_layer(layer_index);
    }
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    npu_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    npu_embed_tokens_->merge_loaded_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  void merge_and_move_pinned_host() {
    npu_embed_tokens_->merge_and_move_pinned_host();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_and_move_pinned_host();
    }
    norm_->merge_and_move_pinned_host();
  }

  void free_weights() {
    npu_embed_tokens_->free_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->free_weights();
    }
    norm_->free_weights();
  }

  void reload_weights() {
    npu_embed_tokens_->reload_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights();
    }
    norm_->reload_weights();
  }

  void reload_non_decoder_weights() {
    npu_embed_tokens_->reload_weights();
    norm_->reload_weights();
  }
  void reload_weights_from_device() {
    npu_embed_tokens_->reload_weights_from_device();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights_from_device();
    }
    norm_->reload_weights_from_device();
  }

  void refresh_rolling_weights() {
    for (auto& layer : layers_) {
      layer->refresh_rolling_weights();
    }
  }

  std::vector<layer::BaseLoader*> get_decoder_loaders() {
    std::vector<layer::BaseLoader*> loaders;
    loaders.reserve(layers_.size());
    for (auto& layer : layers_) {
      loaders.push_back(layer->get_loader());
    }
    return loaders;
  }

  void set_rolling_load_manager(RollingLoadManager* mgr) { rolling_mgr_ = mgr; }
  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    layers_[layer_id]->prepare_expert_weight(expert_ids);
  }

  void update_expert_weight(int32_t layer_id) {
    layers_[layer_id]->update_expert_weight();
  }

  layer::NpuWordEmbedding get_npu_word_embedding() { return npu_embed_tokens_; }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<DeepseekV2DecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::NpuRMSNorm norm_{nullptr};
  RollingLoadManager* rolling_mgr_ = nullptr;
};
TORCH_MODULE(JoyAILLMFlashModel);

class JoyAILLMFlashForCausalLMImpl
    : public LlmForCausalLMImplBase<JoyAILLMFlashModel> {
 public:
  JoyAILLMFlashForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<JoyAILLMFlashModel>(context),
        first_k_dense_replace_(
            context.get_model_args().first_k_dense_replace()) {}

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    model_->prepare_expert_weight(layer_id + first_k_dense_replace_,
                                  expert_ids);
  }

  void update_expert_weight(int32_t layer_id) override {
    model_->update_expert_weight(layer_id + first_k_dense_replace_);
  }

 private:
  int32_t first_k_dense_replace_;
};
TORCH_MODULE(JoyAILLMFlashForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(joyai_llm_flash, JoyAILLMFlashForCausalLM);
// register the model args
// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
REGISTER_MODEL_ARGS(joyai_llm_flash, [&] {
  LOAD_ARG_OR(model_type, "model_type", "joyai_llm_flash");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 32);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 7168);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 32000000.0f);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  // uses default rope_type, no deepseek_yarn scaling
  SET_ARG(rope_scaling_rope_type, "default");
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});
}  // namespace xllm::npu::model

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

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "models/model_registry.h"
#include "models/rec/npu/onerec_npu_impl.h"
#include "models/rec/rec_model_base.h"

namespace xllm {

class OneRecModelImpl final : public torch::nn::Module {
 public:
  explicit OneRecModelImpl(const ModelContext& context) {
    hidden_size_ = context.get_model_args().hidden_size();
    options_ = context.get_tensor_options();
    shared_ = register_module("shared", layer::WordEmbedding(context));

    encoder_ = register_module(
        "encoder", OneRecStack(context, /*is_decode=*/false, shared_));
    decoder_ = register_module(
        "decoder", OneRecStack(context, /*is_decode=*/true, shared_));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    if (!tokens.defined()) {
      return ModelOutput();
    }
    (void)positions;
    (void)kv_caches;

    const auto* onerec_params = input_params.onerec_params();

    if (onerec_params != nullptr) {
      if (onerec_params->is_encoder_forward) {
        std::vector<KVCache> encoder_kv_caches;
        auto encoder_output =
            encoder_(tokens, positions, encoder_kv_caches, input_params);

        torch::Tensor cached_encoder_output;
        if (encoder_output.defined() &&
            onerec_params->encoder_max_seq_len > 0 &&
            !onerec_params->encoder_seq_lens.empty()) {
          cached_encoder_output =
              pad_encoder_output(encoder_output, input_params);
        } else {
          cached_encoder_output = encoder_output;
        }
        {
          std::lock_guard<std::mutex> lock(encoder_output_mutex_);
          encoder_output_ = cached_encoder_output;
        }
        return ModelOutput(cached_encoder_output);
      }

      torch::Tensor cached_encoder_output;
      if (onerec_params->has_encoder_output) {
        std::lock_guard<std::mutex> lock(encoder_output_mutex_);
        cached_encoder_output = encoder_output_;
      }

      const torch::Tensor& decoder_context =
          onerec_params->decoder_context_embedding;

      if (!decoder_context.defined() && !cached_encoder_output.defined()) {
        LOG(ERROR)
            << "OneRec decoder requires decoder_context_embedding or encoder "
               "output.";
        return ModelOutput();
      }

      auto decoder_output = decoder_(
          tokens, positions, kv_caches, input_params, cached_encoder_output);
      return ModelOutput(decoder_output);
    }

    const bool is_encoder_forward =
        (onerec_params != nullptr) && onerec_params->is_encoder_forward;

    auto hidden_states =
        build_hidden_states(tokens, onerec_params, is_encoder_forward);
    if (!hidden_states.defined()) {
      return ModelOutput();
    }

    if (is_encoder_forward) {
      return ModelOutput(hidden_states);
    }

    auto cross_context = resolve_cross_context(onerec_params);
    if (cross_context.defined()) {
      auto enriched_hidden_states =
          add_cross_context_bias(hidden_states, cross_context);
      if (enriched_hidden_states.defined()) {
        hidden_states = std::move(enriched_hidden_states);
      }
    }

    return ModelOutput(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    auto shared_dict = state_dict.get_dict_with_prefix("shared.");
    if (shared_dict.size() > 0) {
      shared_->load_state_dict(shared_dict);
    }

    auto encoder_dict = state_dict.get_dict_with_prefix("encoder.");
    if (encoder_dict.size() > 0) {
      encoder_->load_state_dict(encoder_dict);
    }
    auto decoder_dict = state_dict.get_dict_with_prefix("decoder.");
    if (decoder_dict.size() > 0) {
      decoder_->load_state_dict(decoder_dict);
    }
  }

  void verify_loaded_weights() const {
    encoder_->verify_loaded_weights("encoder.");
    decoder_->verify_loaded_weights("decoder.");
  }

  void merge_loaded_weights() {
    encoder_->merge_loaded_weights();
    decoder_->merge_loaded_weights();
  }

  bool supports_onerec_graph() const { return true; }

  torch::Tensor get_onerec_graph_encoder_output() const {
    std::lock_guard<std::mutex> lock(encoder_output_mutex_);
    return encoder_output_;
  }

  void bind_onerec_prefill_graph_buffers(
      const torch::Tensor& encoder_output,
      const std::vector<torch::Tensor>& cross_k_caches,
      const std::vector<torch::Tensor>& cross_v_caches) {
    if (encoder_output.defined()) {
      std::lock_guard<std::mutex> lock(encoder_output_mutex_);
      encoder_output_ = encoder_output;
    }
    decoder_->bind_onerec_prefill_graph_cross_kv_caches(cross_k_caches,
                                                        cross_v_caches);
  }

  layer::WordEmbedding get_word_embedding() { return shared_; }

  void set_word_embedding(layer::WordEmbedding& embedding) {
    shared_ = embedding;
    encoder_->set_word_embedding(shared_);
    decoder_->set_word_embedding(shared_);
  }

 private:
  static bool is_token_id_tensor(const torch::Tensor& tokens) {
    return tokens.scalar_type() == torch::kLong ||
           tokens.scalar_type() == torch::kInt;
  }

  torch::Tensor build_hidden_states(const torch::Tensor& tokens,
                                    const OneRecModelInputParams* onerec_params,
                                    bool is_encoder_forward) {
    if (tokens.numel() == 0) {
      return torch::empty({0, hidden_size_}, options_);
    }

    if (is_token_id_tensor(tokens)) {
      return shared_(tokens);
    }

    if (tokens.dim() == 2 && tokens.size(-1) == hidden_size_) {
      if (onerec_params != nullptr) {
        if (onerec_params->is_hybrid_mode || is_encoder_forward) {
          return tokens;
        }
        if (onerec_params->decoder_context_embedding.defined()) {
          return tokens;
        }
      }
      return tokens;
    }

    LOG(ERROR) << "Invalid OneRec token tensor shape for non-id path: "
               << tokens.sizes();
    return torch::Tensor();
  }

  torch::Tensor resolve_cross_context(
      const OneRecModelInputParams* onerec_params) const {
    if (onerec_params == nullptr) {
      return torch::Tensor();
    }
    if (onerec_params->decoder_context_embedding.defined()) {
      return onerec_params->decoder_context_embedding;
    }
    return torch::Tensor();
  }

  torch::Tensor add_cross_context_bias(
      const torch::Tensor& hidden_states,
      const torch::Tensor& cross_context) const {
    if (!hidden_states.defined() || !cross_context.defined()) {
      return hidden_states;
    }

    if (hidden_states.dim() != 2 || hidden_states.size(-1) != hidden_size_) {
      LOG(ERROR) << "Unexpected hidden_states shape in OneRec decoder: "
                 << hidden_states.sizes();
      return hidden_states;
    }

    auto context = cross_context;
    if (context.device() != hidden_states.device()) {
      context = context.to(hidden_states.device());
    }
    if (context.dtype() != hidden_states.dtype()) {
      context = context.to(hidden_states.dtype());
    }

    if (context.dim() == 1 && context.size(0) == hidden_size_) {
      context = context.unsqueeze(0);
    } else if (context.dim() > 2 && context.size(-1) == hidden_size_) {
      context = context.reshape({-1, hidden_size_});
    }

    if (context.dim() != 2 || context.size(-1) != hidden_size_) {
      LOG(ERROR) << "Unexpected OneRec cross context shape: "
                 << context.sizes();
      return hidden_states;
    }

    auto pooled_context = context.mean(/*dim=*/0, /*keepdim=*/true);
    return hidden_states + pooled_context.expand(
                               {hidden_states.size(0), pooled_context.size(1)});
  }

  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  layer::WordEmbedding shared_{nullptr};

  OneRecStack encoder_{nullptr};
  OneRecStack decoder_{nullptr};
  torch::Tensor encoder_output_;
  mutable std::mutex encoder_output_mutex_;
};
TORCH_MODULE(OneRecModel);

class OneRecForConditionalGenerationImpl final
    : public RecForCausalLMImplBase<OneRecModel> {
 public:
  explicit OneRecForConditionalGenerationImpl(const ModelContext& context)
      : RecForCausalLMImplBase<OneRecModel>(context) {}

  bool supports_onerec_graph() const {
    return this->model_->supports_onerec_graph();
  }

  torch::Tensor get_onerec_graph_encoder_output() const {
    return this->model_->get_onerec_graph_encoder_output();
  }

  void bind_onerec_prefill_graph_buffers(
      const torch::Tensor& encoder_output,
      const std::vector<torch::Tensor>& cross_k_caches,
      const std::vector<torch::Tensor>& cross_v_caches) {
    this->model_->bind_onerec_prefill_graph_buffers(
        encoder_output, cross_k_caches, cross_v_caches);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      StateDict prefixed_state_dict =
          state_dict->get_dict_with_prefix("module.module3.t5_model.");
      StateDict model_state_dict =
          prefixed_state_dict.size() > 0
              ? prefixed_state_dict
              : state_dict->get_dict_with_prefix(prefix);
      if (prefixed_state_dict.size() > 0) {
        LOG(INFO) << "Detected temporary OneRec checkpoint prefix "
                  << "`module.module3.t5_model.`; loading weights via the "
                     "compatibility path.";
      } else if (model_state_dict.size() == 0) {
        model_state_dict = *state_dict;
      }
      model_->load_state_dict(model_state_dict);

      if (tie_word_embeddings_) {
        auto shared_dict = model_state_dict.get_dict_with_prefix("shared.");
        if (shared_dict.size() == 0) {
          shared_dict = state_dict->get_dict_with_prefix("shared.");
        }
        if (shared_dict.size() != 0) {
          lm_head_->load_state_dict(shared_dict);
        }
      } else {
        auto lm_head_dict = model_state_dict.get_dict_with_prefix("lm_head.");
        if (lm_head_dict.size() == 0) {
          lm_head_dict = state_dict->get_dict_with_prefix("lm_head.");
        }
        if (lm_head_dict.size() != 0) {
          lm_head_->load_state_dict(lm_head_dict);
        }
      }
    }

    model_->verify_loaded_weights();
    model_->merge_loaded_weights();
  }
};
TORCH_MODULE(OneRecForConditionalGeneration);

using OneRecCausalLM = CausalLMImpl<OneRecForConditionalGeneration>;
static_assert(std::is_base_of_v<CausalLM, OneRecCausalLM>,
              "OneRec must satisfy CausalLM contract.");

REGISTER_REC_MODEL(onerec, OneRecForConditionalGeneration);

REGISTER_MODEL_ARGS(onerec, [&] {
  LOAD_ARG_OR(model_type, "model_type", "onerec");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");

  LOAD_ARG_OR(hidden_size, "d_model", 128);
  LOAD_ARG_OR(intermediate_size, "d_ff", 256);

  LOAD_ARG_OR(n_layers, "num_decoder_layers", 4);
  LOAD_ARG_OR(n_encoder_layers, "num_layers", 12);

  LOAD_ARG_OR(n_heads, "num_heads", 4);
  LOAD_ARG_OR(head_dim, "d_kv", 32);
  LOAD_ARG_OR_FUNC(
      decoder_n_heads, "decoder_num_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      decoder_head_dim, "decoder_d_kv", [&] { return args->head_dim(); });

  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG(decoder_n_kv_heads, "decoder_num_key_value_heads");

  LOAD_ARG_OR(vocab_size, "vocab_size", 8200);
  LOAD_ARG_OR(rms_norm_eps, "layer_norm_epsilon", 1e-6);
  LOAD_ARG_OR(max_position_embeddings, "max_length", 500);
  LOAD_ARG_OR(use_absolute_position_embedding,
              "use_absolute_position_embedding",
              false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", true);

  LOAD_ARG_OR(use_moe, "use_moe", false);
  LOAD_ARG_OR(moe_score_func, "moe_score_func", "softmax");
  LOAD_ARG_OR(moe_route_scale, "moe_route_scale", 1.0f);
  LOAD_ARG_OR(n_routed_experts, "moe_num_experts", 8);
  LOAD_ARG_OR(moe_use_shared_experts, "moe_use_shared_experts", false);
  LOAD_ARG_OR(n_shared_experts, "moe_num_shared_experts", 0);
  LOAD_ARG_OR(num_experts_per_tok, "moe_topk", 2);
  LOAD_ARG_OR(moe_intermediate_size, "moe_inter_dim", 1024);

  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

REGISTER_TOKENIZER_ARGS(onerec, [&] {
  SET_ARG(tokenizer_type, "rec");
  LOAD_ARG_OR(vocab_file, "vocab_file", "");
});

}  // namespace xllm

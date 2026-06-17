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

// UMT5 encoder model compatible with HuggingFace weights.
// ref:
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/umt5
//
// Key differences from T5:
//   - Every block has its own relative_attention_bias (not just block 0).
//   - position_bias is NOT carried across blocks; each block recomputes it.
//   - UMT5Attention.forward does not accept or return position_bias.
//   - Weight key for embed_tokens is "encoder.embed_tokens.weight" (tied to
//     "shared.weight" in the full model).

#pragma once
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "framework/model_context.h"
#include "models/dit/encoders/t5_encoder.h"  // reuse T5LayerNorm, T5DenseInterface, T5LayerFFN
#include "models/model_registry.h"

namespace xllm {

// ---------------------------------------------------------------------------
// UMT5LayerSelfAttention
// ---------------------------------------------------------------------------
// Unlike T5LayerSelfAttention:
//   - has_relative_attention_bias is always true (every block owns a bias).
//   - forward() does not accept a position_bias argument.
//   - forward() does not return position_bias in its outputs.
// ---------------------------------------------------------------------------
class UMT5LayerSelfAttentionImpl : public torch::nn::Module {
 public:
  explicit UMT5LayerSelfAttentionImpl(const ModelContext& context) {
    // UMT5: every self-attention layer has its own relative_attention_bias.
    self_attention_ = register_module(
        "SelfAttention",
        T5Attention(context, /*has_relative_attention_bias=*/true));
    layer_norm_ = register_module("layer_norm", T5LayerNorm(context));
  }

  // Returns {hidden_states} — no position_bias in output, matching UMT5.
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt) {
    torch::Tensor normed = layer_norm_->forward(hidden_states);
    // Pass empty position_bias so T5Attention recomputes it from its own bias.
    auto attn_outputs = self_attention_->forward(
        normed, attention_mask, /*position_bias=*/std::nullopt);
    // attn_outputs[0] = attn_output, attn_outputs[1] = position_bias (unused)
    return hidden_states + attn_outputs[0];
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("SelfAttention."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attention_->verify_loaded_weights(prefix + "SelfAttention.");
    layer_norm_->verify_loaded_weights(prefix + "layer_norm.");
  }

 private:
  T5Attention self_attention_{nullptr};
  T5LayerNorm layer_norm_{nullptr};
};
TORCH_MODULE(UMT5LayerSelfAttention);

// ---------------------------------------------------------------------------
// UMT5Block
// ---------------------------------------------------------------------------
// Unlike T5Block:
//   - forward() does not accept or return position_bias.
//   - Internally calls UMT5LayerSelfAttention which handles bias independently.
// ---------------------------------------------------------------------------
class UMT5BlockImpl : public torch::nn::Module {
 public:
  explicit UMT5BlockImpl(const ModelContext& context) {
    self_attention_ =
        register_module("layer_0", UMT5LayerSelfAttention(context));
    ff_layer_ = register_module("layer_1", T5LayerFFN(context));
  }

  // Returns {hidden_states} — no position_bias, matching UMT5Block.forward.
  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt) {
    torch::Tensor curr =
        self_attention_->forward(hidden_states, attention_mask);
    if (curr.dtype() == torch::kFloat16) {
      curr = clamp_inf_values(curr);
    }
    curr = ff_layer_->forward(curr);
    if (curr.dtype() == torch::kFloat16) {
      curr = clamp_inf_values(curr);
    }
    return {curr};
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("layer.0."));
    ff_layer_->load_state_dict(state_dict.get_dict_with_prefix("layer.1."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attention_->verify_loaded_weights(prefix + "layer.0.");
    ff_layer_->verify_loaded_weights(prefix + "layer.1.");
  }

 private:
  torch::Tensor clamp_inf_values(const torch::Tensor& x) const {
    const float max_val = 65504.0f;  // fp16 max
    torch::Tensor clamp_val =
        torch::where(torch::isinf(x).any(),
                     torch::tensor(max_val - 1000.0f, x.options()),
                     torch::tensor(max_val, x.options()));
    return torch::clamp(x, -clamp_val, clamp_val);
  }

  UMT5LayerSelfAttention self_attention_{nullptr};
  T5LayerFFN ff_layer_{nullptr};
};
TORCH_MODULE(UMT5Block);

// ---------------------------------------------------------------------------
// UMT5EncoderModel
// ---------------------------------------------------------------------------
// Matches HuggingFace UMT5Stack (encoder only):
//   - All blocks have their own relative_attention_bias.
//   - position_bias is never passed between blocks.
//   - embed_tokens weight key: "encoder.embed_tokens.weight".
// ---------------------------------------------------------------------------
class UMT5EncoderModelImpl final : public torch::nn::Module {
 public:
  explicit UMT5EncoderModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    embed_tokens_ = register_module(
        "embed_tokens",
        torch::nn::Embedding(model_args.vocab_size(), model_args.d_model()));
    embed_tokens_->weight.set_data(embed_tokens_->weight.to(options));

    blocks_ = register_module("blocks", torch::nn::ModuleList{});
    layers_.reserve(model_args.num_layers());
    for (int64_t i = 0; i < model_args.num_layers(); ++i) {
      auto block = UMT5Block(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }

    final_layer_norm_ =
        register_module("final_layer_norm", T5LayerNorm(context));
  }

  torch::nn::Embedding& get_input_embeddings() { return embed_tokens_; }

  void set_input_embeddings(const torch::nn::Embedding& new_embeddings) {
    embed_tokens_ = new_embeddings;
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    torch::Tensor hidden_states = embed_tokens_->forward(input_ids);
    // UMT5: no position_bias passed between blocks — each block recomputes it.
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto layer_outputs = layers_[i]->forward(hidden_states);
      hidden_states = layer_outputs[0];
    }
    hidden_states = final_layer_norm_->forward(hidden_states);
    return hidden_states;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_from_state_dicts(loader->get_state_dicts(), "");
  }

  // Load from state dicts with an optional key_prefix (e.g. "text_encoder."
  // when all components share a single flat safetensors file).
  void load_from_state_dicts(
      std::vector<std::unique_ptr<StateDict>>& state_dicts,
      const std::string& key_prefix) {
    for (const auto& state_dict : state_dicts) {
      StateDict sd = key_prefix.empty()
                         ? state_dict->get_dict_with_prefix("")
                         : state_dict->get_dict_with_prefix(key_prefix);
      weight::load_weight(sd,
                          "encoder.embed_tokens.weight",
                          embed_tokens_->weight,
                          is_embed_tokens_loaded_);
      final_layer_norm_->load_state_dict(
          sd.get_dict_with_prefix("encoder.final_layer_norm."));
      for (int64_t i = 0; i < static_cast<int64_t>(layers_.size()); ++i) {
        const std::string block_prefix =
            "encoder.block." + std::to_string(i) + ".";
        layers_[i]->load_state_dict(sd.get_dict_with_prefix(block_prefix));
      }
    }
    verify_loaded_weights();
    LOG(INFO) << "UMT5EncoderModel loaded successfully.";
  }

  void verify_loaded_weights() const {
    CHECK(is_embed_tokens_loaded_)
        << "weight is not loaded for encoder.embed_tokens.weight";
    final_layer_norm_->verify_loaded_weights("encoder.final_layer_norm.");
    for (int64_t i = 0; i < static_cast<int64_t>(layers_.size()); ++i) {
      const std::string block_prefix =
          "encoder.block." + std::to_string(i) + ".";
      layers_[i]->verify_loaded_weights(block_prefix);
    }
  }

 private:
  T5LayerNorm final_layer_norm_{nullptr};
  torch::nn::Embedding embed_tokens_{nullptr};
  bool is_embed_tokens_loaded_ = false;
  std::vector<UMT5Block> layers_;
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(UMT5EncoderModel);

// Model args for UMT5-base (google/umt5-base defaults).
REGISTER_MODEL_ARGS(UMT5EncoderModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "float32");
  LOAD_ARG_OR(model_type, "model_type", "umt5");
  LOAD_ARG_OR(vocab_size, "vocab_size", 256384);
  LOAD_ARG_OR(d_model, "d_model", 768);
  LOAD_ARG_OR(num_layers, "num_layers", 12);
  LOAD_ARG_OR(d_kv, "d_kv", 64);
  LOAD_ARG_OR(n_heads, "num_heads", 12);
  LOAD_ARG_OR(d_ff, "d_ff", 2048);
  LOAD_ARG_OR(act_fn, "dense_act_fn", "gelu_new");
  LOAD_ARG_OR(is_gated_act, "is_gated_act", true);
  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-6f);
});

}  // namespace xllm

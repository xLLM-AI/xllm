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
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "framework/model_context.h"
#include "models/dit/t5_encoder.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "xllm/core/layers/common/add_matmul.h"

namespace xllm {
class UMT5LayerNormImpl : public T5LayerNormImpl {
 public:
  using T5LayerNormImpl::T5LayerNormImpl;

  torch::Tensor forward(torch::Tensor hidden_states) {
    auto variance = hidden_states.to(torch::kFloat32).pow(2).mean(-1, true);
    hidden_states = hidden_states * torch::rsqrt(variance + variance_epsilon_);
    if (weight_.dtype() == torch::kFloat16 ||
        weight_.dtype() == torch::kBFloat16) {
      hidden_states = hidden_states.to(weight_.dtype());
    }
    return weight_ * hidden_states;
  }
};
TORCH_MODULE(UMT5LayerNorm);

class UMT5LayerFFNImpl final : public torch::nn::Module {
 public:
  explicit UMT5LayerFFNImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layer_norm_ = register_module("layer_norm", UMT5LayerNorm(context));
    if (model_args.is_gated_act()) {
      dense_relu_dense_ =
          register_module("DenseReluDense",
                          std::make_shared<T5DenseGatedActDenseImpl>(context));
    } else {
      dense_relu_dense_ = register_module(
          "DenseReluDense", std::make_shared<T5DenseActDenseImpl>(context));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor forwarded_states = layer_norm_->forward(hidden_states);
    forwarded_states = dense_relu_dense_->forward(forwarded_states);
    torch::Tensor output = hidden_states + forwarded_states;
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    dense_relu_dense_->load_state_dict(
        state_dict.get_dict_with_prefix("DenseReluDense."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    dense_relu_dense_->verify_loaded_weights(prefix + "DenseReluDense.");
    layer_norm_->verify_loaded_weights(prefix + "layer_norm.");
  }

 private:
  std::shared_ptr<T5DenseInterface> dense_relu_dense_ = nullptr;
  UMT5LayerNorm layer_norm_ = nullptr;
};
TORCH_MODULE(UMT5LayerFFN);

class UMT5AttentionImpl : public T5AttentionImpl {
 public:
  using T5AttentionImpl::T5AttentionImpl;

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& mask = std::nullopt,
      const std::optional<torch::Tensor>& key_value_states = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt) {
    int64_t batch_size = hidden_states.size(0);
    int64_t seq_length = hidden_states.size(1);
    bool is_cross_attention = key_value_states.has_value();
    torch::Tensor query_states = q_->forward(hidden_states);
    query_states =
        query_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);

    torch::Tensor current_states =
        is_cross_attention ? key_value_states.value() : hidden_states;
    torch::Tensor key_states = k_->forward(current_states);
    torch::Tensor value_states = v_->forward(current_states);
    key_states =
        key_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);
    value_states =
        value_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);
    torch::Tensor scores =
        torch::matmul(query_states, key_states.transpose(3, 2));
    torch::Tensor curr_position_bias;
    if (position_bias.has_value() && position_bias.value().numel() > 0) {
      curr_position_bias = position_bias.value();
    } else {
      int64_t key_length = key_states.size(-2);
      if (!has_relative_attention_bias_) {
        curr_position_bias =
            torch::zeros({1, n_heads_, seq_length, key_length},
                         torch::dtype(scores.dtype()).device(scores.device()));
      } else {
        torch::Tensor bias =
            compute_bias(seq_length, key_length, scores.device());
        curr_position_bias = bias.index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(-seq_length, torch::indexing::None),
             torch::indexing::Slice()});
      }
      if (mask.has_value() && mask.value().numel() > 0) {
        torch::Tensor causal_mask = mask.value().index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(0, key_states.size(-2))});
        curr_position_bias = curr_position_bias + causal_mask;
      }
    }
    if (!pruned_heads_.empty()) {
      torch::Tensor head_mask =
          torch::ones(n_heads_ + pruned_heads_.size(), torch::kBool)
              .to(scores.device());
      for (int64_t pruned : pruned_heads_) {
        head_mask[pruned] = false;
      }
      curr_position_bias = curr_position_bias.index({torch::indexing::Slice(),
                                                     head_mask,
                                                     torch::indexing::Slice(),
                                                     torch::indexing::Slice()});
    }
    scores += curr_position_bias;
    torch::Tensor attn_weights =
        torch::softmax(scores.to(torch::kFloat), -1).to(scores.dtype());
    if (layer_head_mask.has_value() && layer_head_mask.value().numel() > 0) {
      attn_weights = attn_weights * layer_head_mask.value();
    }
    torch::Tensor attn_output = torch::matmul(attn_weights, value_states);
    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.view({batch_size, -1, inner_dim_});
    attn_output = o_->forward(attn_output);
    return {attn_output, attn_weights};
  }

 private:
  std::unordered_set<int64_t> pruned_heads_;
};
TORCH_MODULE(UMT5Attention);

class UMT5LayerSelfAttentionImpl final : public torch::nn::Module {
 public:
  UMT5LayerSelfAttentionImpl(const ModelContext& context,
                             bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention", UMT5Attention(context, has_relative_attention_bias));
    layer_norm_ = register_module("layer_norm", UMT5LayerNorm(context));
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt) {
    torch::Tensor normed_hidden_states = layer_norm_->forward(hidden_states);
    auto attention_output = self_attention_->forward(normed_hidden_states,
                                                     attention_mask,
                                                     std::nullopt,
                                                     position_bias,
                                                     layer_head_mask);

    torch::Tensor updated_hidden_states = hidden_states + attention_output[0];

    std::vector<torch::Tensor> outputs = {updated_hidden_states};
    outputs.emplace_back(attention_output[1]);
    return outputs;
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
  UMT5Attention self_attention_ = nullptr;
  UMT5LayerNorm layer_norm_ = nullptr;
};
TORCH_MODULE(UMT5LayerSelfAttention);

class UMT5BlockImpl final : public torch::nn::Module {
 public:
  UMT5BlockImpl(const ModelContext& context, bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention",
        UMT5LayerSelfAttention(context, has_relative_attention_bias));
    ff_layer_ = register_module("LayerFFN", UMT5LayerFFN(context));
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt) {
    auto self_attention_outputs = self_attention_->forward(
        hidden_states, attention_mask, layer_head_mask);
    torch::Tensor curr_hidden_states = self_attention_outputs[0];
    std::vector<torch::Tensor> attention_outputs;
    for (size_t i = 1; i < self_attention_outputs.size(); ++i) {
      attention_outputs.emplace_back(self_attention_outputs[i]);
    }

    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }

    curr_hidden_states = ff_layer_->forward(curr_hidden_states);

    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }

    std::vector<torch::Tensor> outputs = {curr_hidden_states};
    return outputs;
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
    float max_val;
    if (x.scalar_type() == torch::kFloat16) {
      max_val = 65504.0f;
    } else if (x.scalar_type() == torch::kFloat32) {
      max_val = std::numeric_limits<float>::max();
    } else if (x.scalar_type() == torch::kBFloat16) {
      max_val = 3.3895313892515355e+38f;
    } else {
      max_val = std::numeric_limits<float>::max();
    }
    torch::Tensor clamp_value =
        torch::where(torch::isinf(x).any(),
                     torch::tensor(max_val - 1000.0f, x.options()),
                     torch::tensor(max_val, x.options()));

    return torch::clamp(x, -clamp_value, clamp_value);
  }

  UMT5LayerSelfAttention self_attention_ = nullptr;
  UMT5LayerFFN ff_layer_ = nullptr;
};
TORCH_MODULE(UMT5Block);

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
      bool has_relative_bias = true;
      auto block = UMT5Block(context, has_relative_bias);
      blocks_->push_back(block);
      layers_.emplace_back(block);
    }
    final_layer_norm_ =
        register_module("final_layer_norm", UMT5LayerNorm(context));
  }

  torch::Tensor forward(
      const torch::Tensor& input_ids,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt) {
    auto options = torch::TensorOptions()
                       .dtype(torch::typeMetaToScalarType(input_ids.dtype()))
                       .device(input_ids.device());

    torch::Tensor hidden_states = embed_tokens_->forward(input_ids);
    auto input_shape = hidden_states.sizes();
    int64_t batch_size = input_shape[0];
    int64_t seq_length = input_shape[1];
    torch::Tensor causal_mask;
    if (attention_mask.has_value() && attention_mask.value().numel() > 0) {
      causal_mask = attention_mask.value();
    } else {
      causal_mask =
          (1.0 -
           (input_ids > 0).to(options.dtype()).unsqueeze(1).unsqueeze(2)) *
          (-1e4);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
      torch::Tensor layer_head_mask = torch::Tensor();
      auto layer_outputs =
          layers_[i]->forward(hidden_states, causal_mask, layer_head_mask);
      hidden_states = layer_outputs[0];
      layer_outputs.clear();
    }
    hidden_states = final_layer_norm_->forward(hidden_states);

    return hidden_states;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      weight::load_weight(*state_dict,
                          "shared.weight",
                          embed_tokens_->weight,
                          is_embed_tokens_loaded_);
      final_layer_norm_->load_state_dict(
          state_dict->get_dict_with_prefix("encoder.final_layer_norm."));
      for (int64_t i = 0; i < layers_.size(); ++i) {
        const auto block_prefix = "encoder.block." + std::to_string(i) + ".";
        layers_[i]->load_state_dict(
            state_dict->get_dict_with_prefix(block_prefix));
      }
    }
    verify_loaded_weights();
    LOG(INFO) << "UMT5EncoderModel loaded successfully.";
  }

  void verify_loaded_weights() const {
    CHECK(is_embed_tokens_loaded_)
        << "weight is not loaded for embed_tokens.weight";
    final_layer_norm_->verify_loaded_weights("encoder.final_layer_norm.");
    for (int64_t i = 0; i < layers_.size(); ++i) {
      const auto block_prefix = "encoder.block." + std::to_string(i) + ".";
      layers_[i]->verify_loaded_weights(block_prefix);
    }
  }

 private:
  UMT5LayerNorm final_layer_norm_ = nullptr;
  torch::nn::Embedding embed_tokens_ = nullptr;
  bool is_embed_tokens_loaded_ = false;
  std::vector<UMT5Block> layers_;
  torch::nn::ModuleList blocks_ = nullptr;
};
TORCH_MODULE(UMT5EncoderModel);

REGISTER_MODEL_ARGS(UMT5EncoderModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(model_type, "model_type", "umt5encoder");
  LOAD_ARG_OR(vocab_size, "vocab_size", 256384);
  LOAD_ARG_OR(d_model, "d_model", 4096);
  LOAD_ARG_OR(num_layers, "num_layers", 24);
  LOAD_ARG_OR(d_kv, "d_kv", 64);
  LOAD_ARG_OR(n_heads, "num_heads", 64);
  LOAD_ARG_OR(d_ff, "d_ff", 10240);
  LOAD_ARG_OR(act_fn, "dense_act_fn", "gelu_new");
  LOAD_ARG_OR(is_gated_act, "is_gated_act", true);
  LOAD_ARG_OR(
      relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(
      relative_attention_max_distance, "relative_attention_max_distance", 128);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-6f);
});

}  // namespace xllm

/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "xllm/core/layers/common/add_matmul.h"

namespace xllm {
// T5 model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5
class T5LayerNormImpl : public torch::nn::Module {
 public:
  explicit T5LayerNormImpl(ModelContext context)
      : device_(context.get_tensor_options().device()),
        dtype_(context.get_tensor_options().dtype().toScalarType()) {
    ModelArgs model_args = context.get_model_args();
    int64_t hidden_size = model_args.d_model();
    variance_epsilon_ = model_args.layer_norm_eps();
    weight_ = register_parameter(
        "weight", torch::ones({hidden_size}).to(device_).to(dtype_));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    auto variance = hidden_states.to(dtype_).pow(2).mean(-1, true);
    hidden_states = hidden_states * torch::rsqrt(variance + variance_epsilon_);
    if (weight_.dtype() == torch::kFloat16 ||
        weight_.dtype() == torch::kBFloat16) {
      hidden_states = hidden_states.to(weight_.dtype());
    }
    return weight_ * hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) { LOAD_WEIGHT(weight); }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 protected:
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
  double variance_epsilon_;
  torch::Device device_;
  torch::ScalarType dtype_;
};
TORCH_MODULE(T5LayerNorm);

torch::Tensor gelu_new(const torch::Tensor& x) {
  // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
  return 0.5 * x *
         (1.0 +
          torch::tanh(sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3))));
}

class T5DenseInterface : public torch::nn::Module {
 public:
  virtual torch::Tensor forward(const torch::Tensor& hidden_states) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;
};

class T5DenseActDenseImpl : public T5DenseInterface {
 public:
  explicit T5DenseActDenseImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    wi_ = register_module("wi",
                          layer::AddMatmul(model_args.d_model(),
                                           model_args.d_ff(),
                                           /*with_bias=*/false,
                                           options));
    wo_ = register_module("wo",
                          layer::AddMatmul(model_args.d_ff(),
                                           model_args.d_model(),
                                           /*with_bias=*/false,
                                           options));

    if (model_args.act_fn() == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (model_args.act_fn() == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      LOG(FATAL) << "Unsupported activation function: " << model_args.act_fn();
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden = wi_->forward(hidden_states);
    hidden = act_(hidden);
    hidden = wo_->forward(hidden);
    return hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    // wi
    wi_->load_state_dict(state_dict.get_dict_with_prefix("wi."));
    // wo
    wo_->load_state_dict(state_dict.get_dict_with_prefix("wo."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wi_->verify_loaded_weights(prefix + "wi.");
    wo_->verify_loaded_weights(prefix + "wo.");
  }

 private:
  layer::AddMatmul wi_ = nullptr;
  layer::AddMatmul wo_ = nullptr;
  torch::nn::Functional act_ = nullptr;
};

class T5DenseGatedActDenseImpl : public T5DenseInterface {
 public:
  explicit T5DenseGatedActDenseImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    wi_0_ = register_module("wi_0",
                            layer::AddMatmul(model_args.d_model(),
                                             model_args.d_ff(),
                                             /*with_bias=*/false,
                                             options));
    wi_1_ = register_module("wi_1",
                            layer::AddMatmul(model_args.d_model(),
                                             model_args.d_ff(),
                                             /*with_bias=*/false,
                                             options));
    wo_ = register_module("wo",
                          layer::AddMatmul(model_args.d_ff(),
                                           model_args.d_model(),
                                           /*with_bias=*/false,
                                           options));

    if (model_args.act_fn() == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (model_args.act_fn() == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      LOG(FATAL) << "Unsupported activation function: " << model_args.act_fn();
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden_gelu = act_(wi_0_->forward(hidden_states));
    torch::Tensor hidden_linear = wi_1_->forward(hidden_states);
    torch::Tensor new_hidden_states = hidden_gelu * hidden_linear;
    new_hidden_states = wo_->forward(new_hidden_states);
    return new_hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // wi_0
    wi_0_->load_state_dict(state_dict.get_dict_with_prefix("wi_0."));
    // wi_1
    wi_1_->load_state_dict(state_dict.get_dict_with_prefix("wi_1."));
    // wo
    wo_->load_state_dict(state_dict.get_dict_with_prefix("wo."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wi_0_->verify_loaded_weights(prefix + "wi_0.");
    wi_1_->verify_loaded_weights(prefix + "wi_1.");
    wo_->verify_loaded_weights(prefix + "wo.");
  }

 private:
  layer::AddMatmul wi_0_ = nullptr;
  layer::AddMatmul wi_1_ = nullptr;
  layer::AddMatmul wo_ = nullptr;
  torch::nn::Functional act_ = nullptr;
};

class T5LayerFFNImpl : public torch::nn::Module {
 public:
  explicit T5LayerFFNImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layer_norm_ = register_module("layer_norm", T5LayerNorm(context));
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
  T5LayerNorm layer_norm_ = nullptr;
};
TORCH_MODULE(T5LayerFFN);

std::pair<std::unordered_set<int64_t>, torch::Tensor>
find_pruneable_heads_and_indices(
    const std::vector<int64_t>& heads,
    int64_t n_heads,
    int64_t head_size,
    const std::unordered_set<int64_t>& already_pruned_heads,
    torch::ScalarType dtype = torch::kBFloat16) {
  std::unordered_set<int64_t> heads_to_prune;
  for (int64_t h : heads) {
    if (already_pruned_heads.find(h) == already_pruned_heads.end()) {
      heads_to_prune.insert(h);
    }
  }
  torch::Tensor mask = torch::ones({n_heads, head_size}, dtype);

  for (int64_t head : heads_to_prune) {
    int64_t adjusted_head = head;
    for (int64_t pruned : already_pruned_heads) {
      if (pruned < head) {
        adjusted_head--;
      }
    }
    mask[adjusted_head] = 0.0f;
  }
  mask = mask.view(-1).contiguous().eq(1.0f);
  torch::Tensor index = torch::arange(mask.numel(), torch::kLong).index({mask});

  return {heads_to_prune, index};
}

class T5AttentionImpl : public torch::nn::Module {
 public:
  T5AttentionImpl(const ModelContext& context,
                  bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    has_relative_attention_bias_ = has_relative_attention_bias;
    inner_dim_ = model_args.n_heads() * model_args.d_kv();

    n_heads_ = model_args.n_heads();
    key_value_proj_dim_ = model_args.d_kv();
    d_model_ = model_args.d_model();
    relative_attention_num_buckets_ =
        model_args.relative_attention_num_buckets();
    relative_attention_max_distance_ =
        model_args.relative_attention_max_distance();

    inner_dim_ = n_heads_ * key_value_proj_dim_;
    q_ = register_module(
        "q",
        layer::AddMatmul(d_model_, inner_dim_, /*with_bias=*/false, options));
    k_ = register_module(
        "k",
        layer::AddMatmul(d_model_, inner_dim_, /*with_bias=*/false, options));
    v_ = register_module(
        "v",
        layer::AddMatmul(d_model_, inner_dim_, /*with_bias=*/false, options));
    o_ = register_module(
        "o",
        layer::AddMatmul(inner_dim_, d_model_, /*with_bias=*/false, options));

    if (has_relative_attention_bias_) {
      relative_attention_bias_ = register_module(
          "relative_attention_bias",
          torch::nn::Embedding(relative_attention_num_buckets_, n_heads_));
    }
  }

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
            .transpose(1, 2);  // (batch_size, n_heads, seq_len, head_dim)

    torch::Tensor current_states =
        is_cross_attention ? key_value_states.value() : hidden_states;
    torch::Tensor key_states = k_->forward(current_states);
    torch::Tensor value_states = v_->forward(current_states);
    key_states =
        key_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    value_states =
        value_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    torch::Tensor scores = torch::matmul(
        query_states,
        key_states.transpose(3, 2));  // (batch, n_heads, seq_len, key_len)
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

    scores += curr_position_bias;
    torch::Tensor attn_weights =
        torch::softmax(scores.to(torch::kFloat), -1).to(scores.dtype());
    if (layer_head_mask.has_value() && layer_head_mask.value().numel() > 0) {
      attn_weights = attn_weights * layer_head_mask.value();
    }
    torch::Tensor attn_output = torch::matmul(
        attn_weights, value_states);  // (batch, n_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2)
                      .contiguous();  // (batch, seq_len, n_heads, head_dim)
    attn_output = attn_output.view({batch_size, -1, inner_dim_});
    attn_output = o_->forward(attn_output);
    std::vector<torch::Tensor> outputs = {attn_output, curr_position_bias};
    return outputs;
  }

  void load_state_dict(const StateDict& state_dict) {
    q_->load_state_dict(state_dict.get_dict_with_prefix("q."));
    k_->load_state_dict(state_dict.get_dict_with_prefix("k."));
    v_->load_state_dict(state_dict.get_dict_with_prefix("v."));
    o_->load_state_dict(state_dict.get_dict_with_prefix("o."));
    if (has_relative_attention_bias_) {
      weight::load_weight(state_dict,
                          "relative_attention_bias.weight",
                          relative_attention_bias_->weight,
                          is_relative_attention_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    q_->verify_loaded_weights(prefix + "q.");
    k_->verify_loaded_weights(prefix + "k.");
    v_->verify_loaded_weights(prefix + "v.");
    o_->verify_loaded_weights(prefix + "o.");
    if (has_relative_attention_bias_) {
      CHECK(is_relative_attention_bias_loaded_)
          << "weight is not loaded for "
          << prefix + "relative_attention_bias.weight";
    }
  }

 protected:
  torch::Tensor _relative_position_bucket(torch::Tensor& relative_position,
                                          bool bidirectional = true,
                                          int64_t num_buckets = 32,
                                          int64_t max_distance = 128) const {
    torch::Tensor relative_buckets =
        torch::zeros_like(relative_position, torch::kLong);
    if (bidirectional) {
      num_buckets /= 2;
      relative_buckets +=
          (relative_position > 0).to(torch::kLong) * num_buckets;
      auto abs_relative_position = torch::abs(relative_position);
      relative_position = abs_relative_position;
    } else {
      relative_position =
          -torch::min(relative_position, torch::zeros_like(relative_position));
    }
    int64_t max_exact = num_buckets / 2;
    torch::Tensor is_small = relative_position < max_exact;
    auto relative_position_float = relative_position.to(torch::kFloat);
    auto max_exact_float = static_cast<float>(max_exact);
    auto max_distance_float = static_cast<float>(max_distance);
    torch::Tensor relative_position_if_large =
        max_exact + (torch::log(relative_position_float / max_exact_float) /
                     std::log(max_distance_float / max_exact_float) *
                     (num_buckets - max_exact))
                        .to(torch::kLong);
    relative_position_if_large = torch::min(
        relative_position_if_large,
        torch::full_like(
            relative_position_if_large, num_buckets - 1, torch::kLong));
    relative_buckets +=
        torch::where(is_small, relative_position, relative_position_if_large);
    return relative_buckets;
  }

  torch::Tensor compute_bias(
      int64_t query_length,
      int64_t key_length,
      std::optional<torch::Device> device = std::nullopt) const {
    if (!has_relative_attention_bias_) {
      return torch::zeros(
          {1, n_heads_, query_length, key_length},
          torch::dtype(torch::kFloat).device(device.value_or(torch::kCPU)));
    }

    torch::Device dev =
        device.value_or(relative_attention_bias_->weight.device());

    torch::Tensor context_position;
    context_position =
        torch::arange(query_length, torch::dtype(torch::kLong).device(dev))
            .unsqueeze(1);

    torch::Tensor memory_position =
        torch::arange(key_length, torch::dtype(torch::kLong).device(dev))
            .unsqueeze(0);
    torch::Tensor relative_position = memory_position - context_position;
    torch::Tensor relative_position_bucket =
        _relative_position_bucket(relative_position,
                                  true,
                                  relative_attention_num_buckets_,
                                  relative_attention_max_distance_);
    torch::Tensor values =
        const_cast<torch::nn::EmbeddingImpl*>(relative_attention_bias_.get())
            ->forward(relative_position_bucket);
    values = values.permute({2, 0, 1}).unsqueeze(0);

    return values;
  }

 protected:
  bool has_relative_attention_bias_;
  int64_t relative_attention_num_buckets_;
  int64_t relative_attention_max_distance_;
  int64_t d_model_;
  int64_t key_value_proj_dim_;
  int64_t n_heads_;
  int64_t inner_dim_;
  std::optional<int64_t> layer_idx_;
  layer::AddMatmul q_ = nullptr;
  layer::AddMatmul k_ = nullptr;
  layer::AddMatmul v_ = nullptr;
  layer::AddMatmul o_ = nullptr;
  torch::nn::Embedding relative_attention_bias_ = nullptr;
  bool is_relative_attention_bias_loaded_ = false;
};
TORCH_MODULE(T5Attention);

class T5LayerSelfAttentionImpl : public torch::nn::Module {
 public:
  T5LayerSelfAttentionImpl(const ModelContext& context,
                           bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention", T5Attention(context, has_relative_attention_bias));
    layer_norm_ = register_module("layer_norm", T5LayerNorm(context));
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
    // hidden_states, position_bias, [attn_weights])
    std::vector<torch::Tensor> outputs = {updated_hidden_states};
    outputs.push_back(attention_output[1]);
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
  T5Attention self_attention_ = nullptr;
  T5LayerNorm layer_norm_ = nullptr;
};
TORCH_MODULE(T5LayerSelfAttention);

class T5BlockImpl : public torch::nn::Module {
 public:
  T5BlockImpl(const ModelContext& context, bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention",
        T5LayerSelfAttention(context, has_relative_attention_bias));
    ff_layer_ = register_module("LayerFFN", T5LayerFFN(context));
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt) {
    auto self_attention_outputs = self_attention_->forward(
        hidden_states, attention_mask, position_bias, layer_head_mask);
    torch::Tensor curr_hidden_states = self_attention_outputs[0];
    std::vector<torch::Tensor> attention_outputs;
    for (size_t i = 1; i < self_attention_outputs.size(); ++i) {
      attention_outputs.push_back(self_attention_outputs[i]);
    }
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    curr_hidden_states = ff_layer_->forward(curr_hidden_states);
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    std::vector<torch::Tensor> outputs = {curr_hidden_states};
    outputs.insert(
        outputs.end(), attention_outputs.begin(), attention_outputs.end());
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

 private:
  T5LayerSelfAttention self_attention_ = nullptr;
  T5LayerFFN ff_layer_ = nullptr;
};
TORCH_MODULE(T5Block);

class T5EncoderModelImpl : public torch::nn::Module {
 public:
  explicit T5EncoderModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_tokens_ = register_module(
        "embed_tokens",
        torch::nn::Embedding(model_args.vocab_size(), model_args.d_model()));
    embed_tokens_->weight.set_data(embed_tokens_->weight.to(options));
    blocks_ = register_module("blocks", torch::nn::ModuleList{});
    layers_.reserve(model_args.num_layers());
    for (int64_t i = 0; i < model_args.num_layers(); ++i) {
      bool has_relative_bias = (i == 0);
      auto block = T5Block(context, has_relative_bias);
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
    auto options = torch::TensorOptions()
                       .dtype(torch::typeMetaToScalarType(input_ids.dtype()))
                       .device(input_ids.device());
    torch::Tensor hidden_states = embed_tokens_->forward(input_ids);
    auto input_shape =
        hidden_states.sizes();  // (batch_size, seq_length, d_model)
    int64_t batch_size = input_shape[0];
    int64_t seq_length = input_shape[1];
    torch::Tensor causal_mask;
    causal_mask = torch::Tensor();
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    torch::Tensor position_bias = torch::Tensor();
    for (size_t i = 0; i < layers_.size(); ++i) {
      torch::Tensor layer_head_mask = torch::Tensor();
      auto layer_outputs = layers_[i]->forward(
          hidden_states, causal_mask, position_bias, layer_head_mask);
      hidden_states = layer_outputs[0];
      position_bias = layer_outputs[1];
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
    LOG(INFO) << "T5EncoderModel loaded successfully.";
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
  T5LayerNorm final_layer_norm_ = nullptr;
  torch::nn::Embedding embed_tokens_ = nullptr;
  bool is_embed_tokens_loaded_ = false;
  std::vector<T5Block> layers_;
  torch::nn::ModuleList blocks_ = nullptr;
};
TORCH_MODULE(T5EncoderModel);

REGISTER_MODEL_ARGS(T5EncoderModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(model_type, "model_type", "t5encoder");
  LOAD_ARG_OR(vocab_size, "vocab_size", 32128);
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

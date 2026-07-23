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
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {
// FLUX/DiT transformer model compatible with Hugging Face weights.
// Ref:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
inline torch::Tensor apply_rotary_emb(const torch::Tensor& x,
                                      const torch::Tensor& freqs_cis,
                                      bool head_first = true) {
  // assume freqs_cis is [2, S, D]，[0] is cos，[1] is sin
  torch::Tensor cos;
  torch::Tensor sin;
  if (head_first) {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(1);  // [1, 1, 6542, 128]
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(1);  // [1, 1, 6542, 128]
  } else {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(2);  // [1, 6542, 1, 128]
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(2);  // [1, 6542, 1, 128]
  }

#if defined(USE_NPU)
  return at_npu::native::custom_ops::npu_rotary_mul(x, cos, sin, "interleave");
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  std::vector<int64_t> reshape_shape;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    reshape_shape.push_back(x.size(i));
  }
  reshape_shape.push_back(-1);
  reshape_shape.push_back(2);
  torch::Tensor reshaped = x.reshape(reshape_shape);
  torch::Tensor x_real = reshaped.select(-1, 0);
  torch::Tensor x_imag = reshaped.select(-1, 1);
  // x_rotated = [-x_imag, x_real]
  torch::Tensor neg_x_imag = -x_imag;
  auto x_rotated = torch::stack({neg_x_imag, x_real}, -1).flatten(3);
  return (x.to(torch::kFloat32) * cos.to(torch::kFloat32) +
          x_rotated.to(torch::kFloat32) * sin.to(torch::kFloat32))
      .to(x.dtype());
#else
  NOT_IMPLEMENTED();
#endif
}

// Helper function for rotary position embedding
torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  CHECK_EQ(dim % 2, 0) << "Dimension must be even";

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);  // [D/2]

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);  // [S, D/2]
#if defined(USE_NPU)
  freqs_outer = freqs_outer.to(torch::kFloat32);
#endif
  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);  // [S, D/2]
    auto sin_vals = torch::sin(freqs_outer);  // [S, D/2]

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);  // [2, S, D]
  }
  // This case should not happen in practice, but required for compilation
  LOG(FATAL) << "get_1d_rotary_pos_embed returned empty tensor, which should "
                "not happen. use_real: "
             << use_real
             << " repeat_interleave_real: " << repeat_interleave_real;
  return torch::Tensor();
}

class DiTRMSNormImpl : public torch::nn::Module {
 public:
  // Constructor: dim (normalization dimension), eps (stabilization term)
  // elementwise_affine (enable affine transform), bias (enable bias term)
  DiTRMSNormImpl(int64_t dim,
                 float eps,
                 bool elementwise_affine,
                 bool bias,
                 const torch::TensorOptions& options)
      : eps_(eps),
        elementwise_affine_(elementwise_affine),
        is_bias_(bias),
        options_(options) {
    if (elementwise_affine_) {
      weight_ = register_parameter("weight", torch::ones({dim}, options));
      if (is_bias_) {
        bias_ = register_parameter("bias", torch::zeros({dim}, options));
      }
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto input_dtype = hidden_states.dtype();

    // Compute variance in float32 for numerical stability
    auto variance = hidden_states.to(input_dtype).pow(2).mean(-1, true);
    // RMS normalization: x / sqrt(variance + eps)
    auto output = hidden_states * torch::rsqrt(variance + eps_);
    // Apply affine transform if enabled
    if (elementwise_affine_) {
      if (weight_.dtype() != torch::kFloat32) {
        output = output.to(weight_.dtype());
      }
      output = output * weight_;
      if (is_bias_) {
        output = output + bias_;
      }
    } else {
      output = output.to(input_dtype);
    }

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (elementwise_affine_) {
      weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
      if (is_bias_) {
        weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!is_bias_ || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

 private:
  float eps_;                // Small epsilon to avoid division by zero
  bool elementwise_affine_;  // Whether to apply learnable affine parameters
  torch::Tensor weight_;     // Learnable scale parameter
  torch::Tensor bias_;       // Learnable bias parameter (optional)
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
  bool is_bias_;
  torch::TensorOptions options_;
};
TORCH_MODULE(DiTRMSNorm);

class FluxSingleAttentionImpl : public torch::nn::Module {
 public:
  explicit FluxSingleAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    auto head_dim = model_args.head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;

    fused_qkv_ = register_module(
        "fused_qkv_",
        layer::FusedAddMatmul(query_dim, 3 * out_dim, true, options_));

    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim, 1e-6f, options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& image_rotary_emb) {
    int64_t batch_size, channel, height, width;
    batch_size = hidden_states.size(0);

    auto qkv = fused_qkv_->forward(hidden_states);
    auto chunks = qkv.chunk(3, -1);

    torch::Tensor query = chunks[0];
    torch::Tensor key = chunks[1];
    torch::Tensor value = chunks[2];

    // Reshape for multi-head attention
    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = heads_;
    int64_t head_dim = inner_dim / attn_heads;
#if defined(USE_NPU)
    query = query.view({batch_size, -1, attn_heads, head_dim}).contiguous();
    key = key.view({batch_size, -1, attn_heads, head_dim}).contiguous();
    value = value.view({batch_size, -1, attn_heads, head_dim}).contiguous();

    // Apply Q/K normalization if enabled
    if (norm_q_) query = std::get<0>(norm_q_->forward(query));
    if (norm_k_) key = std::get<0>(norm_k_->forward(key));
    // Apply rotary positional embedding
    query = apply_rotary_emb(query, image_rotary_emb, false);
    key = apply_rotary_emb(key, image_rotary_emb, false);
    // Compute scaled dot-product attention (no mask, no dropout)
    // torch::Tensor attn_output = torch::scaled_dot_product_attention(
    //    query, key, value, torch::nullopt, 0.0, false);
    int64_t head_num_ = query.size(2);
    int64_t head_dim_ = query.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query,
                                                         key,
                                                         value,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);
    attn_output = attn_output.to(query.dtype());
    return attn_output.flatten(2);
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
    query = query.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, attn_heads, head_dim}).transpose(1, 2);

    // Apply Q/K normalization if enabled
    if (norm_q_) query = std::get<0>(norm_q_->forward(query));
    if (norm_k_) key = std::get<0>(norm_k_->forward(key));
    // Apply rotary positional embedding
    query = apply_rotary_emb(query, image_rotary_emb);
    key = apply_rotary_emb(key, image_rotary_emb);
    // Compute scaled dot-product attention (no mask, no dropout)
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query, key, value, torch::nullopt, 0.0, false);
    attn_output = attn_output.to(query.dtype());
    return attn_output.transpose(1, 2).flatten(2);
#else
    NOT_IMPLEMENTED();
#endif
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // fused_qkv_
    fused_qkv_->load_state_dict(state_dict, {"to_q", "to_k", "to_v"});
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fused_qkv_->verify_loaded_weights(prefix + "to_q|to_k|to_v.");
  }

 private:
  int64_t heads_;
  layer::FusedAddMatmul fused_qkv_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxSingleAttention);

class FluxAttentionImpl : public torch::nn::Module {
 public:
  explicit FluxAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    auto head_dim = model_args.head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;
    auto added_kv_proj_dim = query_dim;

    to_out_ = register_module(
        "to_out",
        layer::AddMatmul(out_dim, query_dim, /*with_bias=*/true, options_));

    to_add_out_ = register_module(
        "to_add_out",
        layer::AddMatmul(
            out_dim, added_kv_proj_dim, /*with_bias=*/true, options_));

    fused_qkv_ = register_module(
        "fused_qkv",
        layer::FusedAddMatmul(
            query_dim, 3 * out_dim, /*with_bias=*/true, options_));
    fused_add_qkv_ = register_module(
        "fused_add_qkv",
        layer::FusedAddMatmul(
            added_kv_proj_dim, 3 * out_dim, /*with_bias=*/true, options_));

    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim, 1e-6f, options_));
    norm_added_q_ = register_module("norm_added_q",
                                    layer::RMSNorm(head_dim, 1e-6f, options_));
    norm_added_k_ = register_module("norm_added_k",
                                    layer::RMSNorm(head_dim, 1e-6f, options_));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& image_rotary_emb) {
    int64_t input_ndim = hidden_states.dim();

    torch::Tensor hidden_states_reshaped = hidden_states;
    if (input_ndim == 4) {
      auto shape = hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      hidden_states_reshaped =
          hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t context_input_ndim = encoder_hidden_states.dim();
    torch::Tensor encoder_hidden_states_reshaped = encoder_hidden_states;
    if (context_input_ndim == 4) {
      auto shape = encoder_hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      encoder_hidden_states_reshaped =
          encoder_hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);

    auto qkv = fused_qkv_->forward(hidden_states_reshaped);

    auto chunks = qkv.chunk(3, -1);
    torch::Tensor query = chunks[0];
    torch::Tensor key = chunks[1];
    torch::Tensor value = chunks[2];

    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = heads_;

    int64_t head_dim = inner_dim / attn_heads;
    query = query.view({batch_size, -1, attn_heads, head_dim});
    key = key.view({batch_size, -1, attn_heads, head_dim});
    value = value.view({batch_size, -1, attn_heads, head_dim});
    if (norm_q_) query = std::get<0>(norm_q_->forward(query));
    if (norm_k_) key = std::get<0>(norm_k_->forward(key));

    auto encoder_qkv = fused_add_qkv_->forward(encoder_hidden_states_reshaped);

    auto encoder_chunks = encoder_qkv.chunk(3, -1);
    torch::Tensor encoder_hidden_states_query_proj = encoder_chunks[0];
    torch::Tensor encoder_hidden_states_key_proj = encoder_chunks[1];
    torch::Tensor encoder_hidden_states_value_proj = encoder_chunks[2];

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          std::get<0>(norm_added_q_->forward(encoder_hidden_states_query_proj));

    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          std::get<0>(norm_added_k_->forward(encoder_hidden_states_key_proj));
    // TODO some are right some are wrong query1& key1.
    // encoder_hidden_states_query_proj
    auto query1 = torch::cat({encoder_hidden_states_query_proj, query}, 1);
    auto key1 = torch::cat({encoder_hidden_states_key_proj, key}, 1);
    auto value1 = torch::cat({encoder_hidden_states_value_proj, value}, 1);
    if (image_rotary_emb.defined()) {
      query1 = apply_rotary_emb(query1, image_rotary_emb, false);
      key1 = apply_rotary_emb(key1, image_rotary_emb, false);
    }
#if defined(USE_NPU)
    // torch::Tensor attn_output = torch::scaled_dot_product_attention(
    //     query1, key1, value1, torch::nullopt, 0.0, false);
    int64_t head_num_ = query1.size(2);
    int64_t head_dim_ = query1.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query1,
                                                         key1,
                                                         value1,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);

    attn_output = attn_output.reshape({batch_size, -1, attn_heads * head_dim});
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_MUSA) || \
    defined(USE_DCU)
    // SDPA expects (B, H, S, D); our query1/key1/value1 are (B, S, H, D).
    // Transpose to match diffusers dispatch_attention_fn (permute 0,2,1,3).
    query1 = query1.transpose(1, 2);
    key1 = key1.transpose(1, 2);
    value1 = value1.transpose(1, 2);
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query1, key1, value1, torch::nullopt, 0.0, false);
    attn_output = attn_output.transpose(1, 2).reshape(
        {batch_size, -1, attn_heads * head_dim});
#else
    NOT_IMPLEMENTED();
#endif
    attn_output = attn_output.to(query.dtype());

    int64_t encoder_length = encoder_hidden_states_reshaped.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);
    hidden_output = to_out_->forward(hidden_output);
    encoder_output = to_add_out_->forward(encoder_output);
    return std::make_tuple(hidden_output, encoder_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    // to_out
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    // to_add_out
    to_add_out_->load_state_dict(
        state_dict.get_dict_with_prefix("to_add_out."));
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // norm_added_q
    norm_added_q_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_q."));
    // norm_added_k
    norm_added_k_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_k."));
    // fused_qkv_
    fused_qkv_->load_state_dict(state_dict, {"to_q", "to_k", "to_v"});
    // fused_add_qkv_
    fused_add_qkv_->load_state_dict(state_dict,
                                    {"add_q_proj", "add_k_proj", "add_v_proj"});
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_out_->verify_loaded_weights(prefix + "to_out.0.");
    to_add_out_->verify_loaded_weights(prefix + "to_add_out.");

    fused_qkv_->verify_loaded_weights(prefix + "to_q|to_k|to_v.");
    fused_add_qkv_->verify_loaded_weights(prefix +
                                          "add_q_proj|add_k_proj|add_v_proj.");
  }

 private:
  layer::FusedAddMatmul fused_qkv_{nullptr};
  layer::FusedAddMatmul fused_add_qkv_{nullptr};
  layer::AddMatmul to_out_{nullptr};
  layer::AddMatmul to_add_out_{nullptr};

  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  layer::RMSNorm norm_added_q_{nullptr};
  layer::RMSNorm norm_added_k_{nullptr};
  int64_t heads_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxAttention);

class PixArtAlphaTextProjectionImpl : public torch::nn::Module {
 public:
  explicit PixArtAlphaTextProjectionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    int64_t hidden_size = model_args.head_dim() * model_args.n_heads();
    int64_t in_features = model_args.pooled_projection_dim();
    int64_t out_dim =
        hidden_size;  //(out_features == -1) ? hidden_size : out_features;
    linear_1_ = register_module(
        "linear_1",
        layer::AddMatmul(
            in_features, hidden_size, /*with_bias=*/true, options_));

    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(hidden_size, out_dim, /*with_bias=*/true, options_));
    act_1_ = register_module("act_1", torch::nn::SiLU());
  }

  torch::Tensor forward(const torch::Tensor& caption) {
    auto hidden_states = linear_1_->forward(caption);
    hidden_states = act_1_->forward(hidden_states);
    hidden_states = linear_2_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear_1
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    // linear_2
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  layer::AddMatmul linear_1_{nullptr};
  layer::AddMatmul linear_2_{nullptr};
  torch::nn::SiLU act_1_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(PixArtAlphaTextProjection);

inline torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                            int64_t embedding_dim,
                                            bool flip_sin_to_cos = false,
                                            float downscale_freq_shift = 1.0f,
                                            float scale = 1.0f,
                                            int64_t max_period = 10000) {
  CHECK_EQ(timesteps.dim(), 1) << "Timesteps should be a 1d-array";
  int64_t half_dim = embedding_dim / 2;
  // -ln(max_period) * [0, 1, ..., half_dim-1] / (half_dim -
  // downscale_freq_shift
  torch::TensorOptions options = timesteps.options();
  auto exponent = -std::log(static_cast<float>(max_period)) *
                  torch::arange(/*start=*/0,
                                /*end=*/half_dim,
                                /*step=*/1,
                                options);
  exponent = exponent / (half_dim - downscale_freq_shift);

  // timesteps[:, None] * exp(exponent)[None, :]
  auto emb = torch::exp(exponent);                  // [half_dim]
  emb = timesteps.unsqueeze(1) * emb.unsqueeze(0);  // [N, half_dim]
  emb = scale * emb;

  // [sin(emb), cos(emb)]
  auto sin_emb = torch::sin(emb);
  auto cos_emb = torch::cos(emb);
  auto combined =
      torch::cat({sin_emb, cos_emb}, /*dim=*/-1);  // [N, 2*half_dim]

  if (flip_sin_to_cos) {
    combined = torch::cat(
        {combined.slice(
             /*dim=*/-1, /*start=*/half_dim, /*end=*/2 * half_dim),   // cos
         combined.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)},  // sin
        /*dim=*/-1);
  }

  if (embedding_dim % 2 == 1) {
    combined = torch::nn::functional::pad(
        combined, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
  }

  return combined;  // [N, embedding_dim]
}

class TimestepsImpl : public torch::nn::Module {
 public:
  explicit TimestepsImpl(ModelContext context) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  256,  // embedding_dim
                                  true,
                                  0.0f,  // flip_sin_to_cos
                                  1,
                                  10000  // max_period
    );
  }
};
TORCH_MODULE(Timesteps);

class TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  explicit TimestepEmbeddingImpl(ModelContext context)
      : options_(context.get_tensor_options()) {
    const auto& model_args = context.get_model_args();
    int64_t time_embed_dim = model_args.head_dim() * model_args.n_heads();
    linear_1_ = register_module(
        "linear_1",
        layer::AddMatmul(256, time_embed_dim, /*with_bias=*/true, options_));

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = time_embed_dim;
    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(
            time_embed_dim, time_embed_dim_out, /*with_bias=*/true, options_));
  }

  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x1 = linear_1_->forward(sample);
    x1 = act_fn_->forward(x1);
    x1 = linear_2_->forward(x1);
    return x1;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    // linear2
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  layer::AddMatmul linear_1_{nullptr};
  layer::AddMatmul linear_2_{nullptr};
  torch::nn::SiLU post_act_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  // bool has_cond_proj_;
  torch::TensorOptions options_;
};
TORCH_MODULE(TimestepEmbedding);

class CombinedTimestepTextProjEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit CombinedTimestepTextProjEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    time_proj_ = Timesteps(context);
    timestep_embedder_ = TimestepEmbedding(context);
    text_embedder_ = PixArtAlphaTextProjection(context);
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_(timestep);
    auto timesteps_emb = timestep_embedder_(timesteps_proj);

    auto pooled_projections = text_embedder_(pooled_projection);
    return timesteps_emb + pooled_projections;
  }

  void load_state_dict(const StateDict& state_dict) {
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    text_embedder_->verify_loaded_weights(prefix + "text_embedder.");
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(CombinedTimestepTextProjEmbeddings);

class CombinedTimestepGuidanceTextProjEmbeddingsImpl
    : public torch::nn::Module {
 public:
  explicit CombinedTimestepGuidanceTextProjEmbeddingsImpl(
      const ModelContext& context)
      : time_proj_(context) {
    text_embedder_ = PixArtAlphaTextProjection(context);
    timestep_embedder_ = TimestepEmbedding(
        context);  // in_channels=256, time_embed_dim=embedding_dim
    guidance_embedder_ = TimestepEmbedding(context);  // in_channels=256
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_->forward(timestep);  // [N, 256]
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj);     // [N, embedding_dim]
    auto guidance_proj = time_proj_->forward(guidance);  // [N, 256]
    auto guidance_emb =
        guidance_embedder_->forward(guidance_proj);  // [N, embedding_dim]
    auto time_guidance_emb =
        timesteps_emb + guidance_emb;  // [N, embedding_dim]
    auto pooled_projections =
        text_embedder_->forward(pooled_projection);  // [N, embedding_dim]
    return time_guidance_emb + pooled_projections;   // [N, embedding_dim]
  }

  void load_state_dict(const StateDict& state_dict) {
    // guidance_embedder
    guidance_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("guidance_embedder."));
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    guidance_embedder_->verify_loaded_weights(prefix + "guidance_embedder.");
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    text_embedder_->verify_loaded_weights(prefix + "text_embedder.");
  }

 private:
  TimestepEmbedding guidance_embedder_{nullptr};
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
};
TORCH_MODULE(CombinedTimestepGuidanceTextProjEmbeddings);

class AdaLayerNormZeroImpl : public torch::nn::Module {
 public:
  explicit AdaLayerNormZeroImpl(ModelContext context)
      : options_(context.get_tensor_options()) {
    ModelArgs model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    int64_t embedding_dim =
        num_attention_heads * attention_head_dim;  // hidden size
    silu_ = register_module("silu", torch::nn::SiLU());

    linear_ = register_module(
        "linear",
        layer::AddMatmul(
            embedding_dim, 6 * embedding_dim, /*with_bias=*/true, options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  forward(const torch::Tensor& x,
          const torch::Tensor& timestep = torch::Tensor(),
          const torch::Tensor& class_labels = torch::Tensor(),
          const torch::Tensor& emb = torch::Tensor()) {
    torch::Tensor ada_emb = emb;
    ada_emb = linear_->forward(silu_->forward(ada_emb));
    auto splits = torch::chunk(ada_emb, 6, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];
    auto shift_mlp = splits[3];
    auto scale_mlp = splits[4];
    auto gate_mlp = splits[5];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);
    return {normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  layer::AddMatmul linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZero);

class AdaLayerNormZeroSingleImpl : public torch::nn::Module {
 public:
  explicit AdaLayerNormZeroSingleImpl(ModelContext context)
      : options_(context.get_tensor_options()) {
    ModelArgs model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    int64_t embedding_dim =
        num_attention_heads * attention_head_dim;  // hidden size
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        layer::AddMatmul(
            embedding_dim, 3 * embedding_dim, /*with_bias=*/true, options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& emb = torch::Tensor()) {
    auto ada_emb = linear_->forward(silu_->forward(emb));
    auto splits = torch::chunk(ada_emb, 3, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);

    return {normalized_x, gate_msa};
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  layer::AddMatmul linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZeroSingle);

class AdaLayerNormContinuousImpl : public torch::nn::Module {
 public:
  explicit AdaLayerNormContinuousImpl(ModelContext context,
                                      bool with_bias = true)
      : options_(context.get_tensor_options()) {
    ModelArgs model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto embedding_dim = num_attention_heads * attention_head_dim;
    auto conditioning_embedding_dim = embedding_dim;
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module("linear",
                              layer::AddMatmul(conditioning_embedding_dim,
                                               2 * embedding_dim,
                                               /*with_bias=*/with_bias,
                                               options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6f)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& conditioning_embedding) {
    auto cond_emb = silu_->forward(conditioning_embedding);
    cond_emb = cond_emb.to(x.dtype());

    auto emb = linear_->forward(cond_emb);
    auto chunks = torch::chunk(emb, 2, 1);
    torch::Tensor scale, shift;

    scale = chunks[0];
    shift = chunks[1];
    auto x_norm = norm_->forward(x);
    return x_norm * (1 + scale).unsqueeze(1) + shift.unsqueeze(1);
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  layer::AddMatmul linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::string norm_type_;
  double eps_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormContinuous);

class FeedForwardImpl : public torch::nn::Module {
 public:
  explicit FeedForwardImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    auto inner_dim = dim * 4;
    auto dim_out = dim;

    // linear1
    linear1_ = register_module(
        "linear1",
        layer::AddMatmul(dim, inner_dim, /*with_bias=*/true, options_));
    // activation
    activation_ =
        register_module("activation",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    // linear2

    linear2_ = register_module(
        "linear2",
        layer::AddMatmul(inner_dim, dim_out, /*with_bias=*/true, options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor out = linear1_->forward(hidden_states);
    out = activation_(out);
    out = linear2_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    linear1_->load_state_dict(state_dict.get_dict_with_prefix("net.0.proj."));
    // linear2
    linear2_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    linear1_->verify_loaded_weights(prefix + "net.0.proj.");
    linear2_->verify_loaded_weights(prefix + "net.2.");
  }

 private:
  layer::AddMatmul linear1_{nullptr};
  torch::nn::Functional activation_{nullptr};
  layer::AddMatmul linear2_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FeedForward);

class FluxSingleTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit FluxSingleTransformerBlockImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    mlp_hidden_dim_ = dim * 4;

    norm_ = register_module("norm", AdaLayerNormZeroSingle(context));

    int64_t mlp_out_dim = mlp_hidden_dim_;
    proj_mlp_ = register_module(
        "proj_mlp",
        layer::AddMatmul(dim, mlp_out_dim, /*with_bias=*/true, options_));

    int64_t proj_in_dim = dim + mlp_hidden_dim_;
    int64_t proj_out_dim = dim;
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(
            proj_in_dim, proj_out_dim, /*with_bias=*/true, options_));
    act_mlp_ =
        register_module("gelu",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    attn_ = register_module("attn", FluxSingleAttention(context));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    auto residual = hidden_states;
    auto [norm_hidden_states, gate] = norm_(hidden_states, temb);
    auto mlp_hidden_states = act_mlp_(proj_mlp_(norm_hidden_states));
    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);
    auto hidden_states_cat = torch::cat({attn_output, mlp_hidden_states}, 2);
    auto out = proj_out_(hidden_states_cat);
    out = gate.unsqueeze(1) * out;
    out = residual + out;
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    // proj_mlp
    proj_mlp_->load_state_dict(state_dict.get_dict_with_prefix("proj_mlp."));
    // proj_out
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix + "attn.");
    norm_->verify_loaded_weights(prefix + "norm.");
    proj_mlp_->verify_loaded_weights(prefix + "proj_mlp.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

 private:
  AdaLayerNormZeroSingle norm_{nullptr};
  layer::AddMatmul proj_mlp_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  torch::nn::Functional act_mlp_{nullptr};
  FluxSingleAttention attn_{nullptr};
  int64_t mlp_hidden_dim_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxSingleTransformerBlock);

class FluxTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit FluxTransformerBlockImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();

    auto dim = num_attention_heads * attention_head_dim;
    double eps = 1e-6;

    norm1_ = register_module("norm1", AdaLayerNormZero(context));

    norm1_context_ =
        register_module("norm1_context", AdaLayerNormZero(context));

    attn_ = register_module("attn", FluxAttention(context));
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_ = register_module("ff", FeedForward(context));
    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_context_ = register_module("ff_context", FeedForward(context));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    auto [norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
        norm1_(hidden_states, torch::Tensor(), torch::Tensor(), temb);
    auto [norm_encoder_hidden_states,
          c_gate_msa,
          c_shift_mlp,
          c_scale_mlp,
          c_gate_mlp] =
        norm1_context_(
            encoder_hidden_states, torch::Tensor(), torch::Tensor(), temb);
    auto [attn_output, context_attn_output] =
        attn_(norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb);
    attn_output = gate_msa.unsqueeze(1) * attn_output;
    auto new_hidden_states = hidden_states + attn_output;
    // image latent
    auto norm_hs = norm2_(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1);
    auto ff_output = ff_->forward(norm_hs);
    new_hidden_states = new_hidden_states + gate_mlp.unsqueeze(1) * ff_output;
    // context
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states + context_attn_output;
    auto norm_enc_hs = norm2_context_(new_encoder_hidden_states);
    norm_enc_hs =
        norm_enc_hs * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1);
    auto ff_context_out = ff_context_->forward(norm_enc_hs);
    new_encoder_hidden_states =
        new_encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_context_out;
    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
      new_encoder_hidden_states =
          torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
    }
    return std::make_tuple(new_hidden_states, new_encoder_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm1
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    // norm1_context
    norm1_context_->load_state_dict(
        state_dict.get_dict_with_prefix("norm1_context."));
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // ff
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    // ff_context
    ff_context_->load_state_dict(
        state_dict.get_dict_with_prefix("ff_context."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    norm1_context_->verify_loaded_weights(prefix + "norm1_context.");
    attn_->verify_loaded_weights(prefix + "attn.");
    ff_->verify_loaded_weights(prefix + "ff.");
    ff_context_->verify_loaded_weights(prefix + "ff_context.");
  }

 private:
  AdaLayerNormZero norm1_{nullptr};
  AdaLayerNormZero norm1_context_{nullptr};
  FluxAttention attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  FeedForward ff_{nullptr};
  FeedForward ff_context_{nullptr};
  torch::nn::LayerNorm norm2_context_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxTransformerBlock);

class FluxTransformer2DModelImpl : public torch::nn::Module {
 public:
  explicit FluxTransformer2DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;
    auto pooled_projection_dim = model_args.pooled_projection_dim();
    auto joint_attention_dim = model_args.joint_attention_dim();
    auto axes_dims_rope = model_args.axes_dims_rope();
    auto num_layers = model_args.num_layers();
    auto num_single_layers = model_args.num_single_layers();
    auto patch_size = model_args.mm_patch_size();
    in_channels_ = model_args.in_channels();
    out_channels_ = model_args.out_channels();
    guidance_embeds_ = model_args.guidance_embeds();

    // Initialize the transformer model components here
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());
    if (guidance_embeds_) {
      time_text_guidance_embed_ =
          register_module("time_text_guidance_embed",
                          CombinedTimestepGuidanceTextProjEmbeddings(context));
    } else {
      time_text_embed_ = register_module(
          "time_text_embed", CombinedTimestepTextProjEmbeddings(context));
    }
    context_embedder_ = register_module(
        "context_embedder",
        layer::AddMatmul(
            joint_attention_dim, inner_dim, /*with_bias=*/true, options_));
    x_embedder_ = register_module(
        "x_embedder",
        layer::AddMatmul(
            out_channels_, inner_dim, /*with_bias=*/true, options_));
    // mm-dit block
    transformer_block_layers_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = FluxTransformerBlock(context);
      transformer_blocks_->push_back(block);
      transformer_block_layers_.push_back(block);
    }
    // single mm-dit block
    single_transformer_block_layers_.reserve(num_single_layers);
    for (int64_t i = 0; i < num_single_layers; ++i) {
      auto block = FluxSingleTransformerBlock(context);
      single_transformer_blocks_->push_back(block);
      single_transformer_block_layers_.push_back(block);
    }
    norm_out_ = register_module("norm_out", AdaLayerNormContinuous(context));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(inner_dim,
                         patch_size * patch_size * out_channels_,
                         /*with_bias=*/true,
                         options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    torch::Tensor temb;
    if (guidance.defined()) {
      auto guidance_scaled = guidance.to(hidden_states.dtype()) * 1000.0f;
      temb = time_text_guidance_embed_->forward(
          timestep_scaled, guidance_scaled, pooled_projections);
    } else {
      temb = time_text_embed_->forward(timestep_scaled, pooled_projections);
    }
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    bool use_step_cache = false;
    bool use_block_cache = false;
    torch::Tensor original_hidden_states = hidden_states;
    torch::Tensor original_encoder_hidden_states = encoder_hidden_states;

    // Step start: prepare inputs (hidden_states, original_hidden_states)
    TensorMap step_in_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_before(step_idx, step_in_map);
    use_step_cache = DiTCache::get_instance().on_before_step(stepin_before);

    if (!use_step_cache) {
      for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
        // Block start: prepare input (block_id)
        CacheBlockIn blockin_before(i);
        use_block_cache =
            DiTCache::get_instance().on_before_block(blockin_before);

        if (!use_block_cache) {
          auto block = transformer_block_layers_[i];
          auto [new_hidden, new_encoder_hidden] = block->forward(
              hidden_states, encoder_hidden_states, temb, image_rotary_emb);
          hidden_states = new_hidden;
          encoder_hidden_states = new_encoder_hidden;
        }

        // Block end: update outputs (block_id, hidden_states,
        // encoder_hidden_states, original_hidden_states,
        // original_encoder_hidden_states)
        TensorMap block_in_map = {
            {"hidden_states", hidden_states},
            {"encoder_hidden_states", encoder_hidden_states},
            {"original_hidden_states", original_hidden_states},
            {"original_encoder_hidden_states", original_encoder_hidden_states}};
        CacheBlockIn blockin_after(i, block_in_map);
        CacheBlockOut blockout_after =
            DiTCache::get_instance().on_after_block(blockin_after);

        hidden_states = blockout_after.tensors.at("hidden_states");
        encoder_hidden_states =
            blockout_after.tensors.at("encoder_hidden_states");
      }

      hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);

      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        // Block start: prepare input (block_id)
        CacheBlockIn blockin_before(i);
        use_block_cache =
            DiTCache::get_instance().on_before_block(blockin_before);

        if (!use_block_cache) {
          auto block = single_transformer_block_layers_[i];
          hidden_states = block->forward(hidden_states, temb, image_rotary_emb);
        }

        // Block end: update outputs (block_id, hidden_states,
        // original_hidden_states)
        TensorMap single_block_map = {
            {"hidden_states", hidden_states},
            {"original_hidden_states", original_hidden_states}};
        CacheBlockIn blockin_after(i, single_block_map);
        CacheBlockOut blockout_after =
            DiTCache::get_instance().on_after_block(blockin_after);

        hidden_states = blockout_after.tensors.at("hidden_states");
      }

      int64_t start = encoder_hidden_states.size(1);
      int64_t length = hidden_states.size(1) - start;
      auto output_hidden =
          hidden_states.narrow(1, start, std::max(length, int64_t(0)));
      hidden_states = output_hidden;
    }

    // Step end: update outputs (hidden_states, original_hidden_states)
    TensorMap step_after_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_after(step_idx, step_after_map);
    CacheStepOut stepout_after =
        DiTCache::get_instance().on_after_step(stepin_after);
    hidden_states = stepout_after.tensors.at("hidden_states");

    auto output_hidden = norm_out_(hidden_states, temb);
    return proj_out_(output_hidden);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    // Load model parameters from the loader
    for (const auto& state_dict : loader->get_state_dicts()) {
      // context_embedder
      context_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("context_embedder."));
      // x_embedder
      x_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("x_embedder."));
      // time_text_embed
      if (time_text_embed_) {
        time_text_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      } else {
        time_text_guidance_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      }
      // transformer_blocks
      for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
        auto block = transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      // single_transformer_blocks
      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        auto block = single_transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      // norm_out
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      // proj_out
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    // context_embedder
    context_embedder_->verify_loaded_weights(prefix + "context_embedder.");
    // x_embedder
    x_embedder_->verify_loaded_weights(prefix + "x_embedder.");
    // time_text_embed
    if (time_text_embed_) {
      time_text_embed_->verify_loaded_weights(prefix + "time_text_embed.");
    } else {
      time_text_guidance_embed_->verify_loaded_weights(prefix +
                                                       "time_text_embed.");
    }
    // transformer_blocks
    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    // single_transformer_blocks
    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "single_transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    // norm_out
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    // proj_out
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

  int64_t in_channels() { return in_channels_; }
  bool guidance_embeds() { return guidance_embeds_; }

 private:
  CombinedTimestepTextProjEmbeddings time_text_embed_{nullptr};
  CombinedTimestepGuidanceTextProjEmbeddings time_text_guidance_embed_{nullptr};
  layer::AddMatmul context_embedder_{nullptr};
  layer::AddMatmul x_embedder_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<FluxTransformerBlock> transformer_block_layers_;
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  std::vector<FluxSingleTransformerBlock> single_transformer_block_layers_;
  AdaLayerNormContinuous norm_out_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  bool guidance_embeds_;
  int64_t in_channels_;
  int64_t out_channels_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxTransformer2DModel);

class FluxDiTModelImpl : public torch::nn::Module {
 public:
  explicit FluxDiTModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    flux_transformer_2d_model_ = register_module(
        "flux_transformer_2d_model", FluxTransformer2DModel(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor output =
        flux_transformer_2d_model_->forward(hidden_states_input,
                                            encoder_hidden_states_input,
                                            pooled_projections,
                                            timestep,
                                            image_rotary_emb,
                                            guidance,
                                            step_idx);
    return output;
  }
  int64_t in_channels() { return flux_transformer_2d_model_->in_channels(); }
  bool guidance_embeds() {
    return flux_transformer_2d_model_->guidance_embeds();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux_transformer_2d_model_->load_model(std::move(loader));
    flux_transformer_2d_model_->verify_loaded_weights("");
  }

 private:
  FluxTransformer2DModel flux_transformer_2d_model_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FluxDiTModel);

REGISTER_MODEL_ARGS(FluxTransformer2DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(mm_patch_size, "patch_size", 1);
  LOAD_ARG_OR(in_channels, "in_channels", 64);
  LOAD_ARG_OR(out_channels, "out_channels", 64);
  LOAD_ARG_OR(num_layers, "num_layers", 19);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 38);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(joint_attention_dim, "joint_attention_dim", 4096);
  LOAD_ARG_OR(pooled_projection_dim, "pooled_projection_dim", 768);
  LOAD_ARG_OR(guidance_embeds, "guidance_embeds", true);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));
});
}  // namespace xllm

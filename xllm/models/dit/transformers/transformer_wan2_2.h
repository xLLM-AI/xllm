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
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "models/dit/utils/dit_parallel_linear.h"

using xllm::dit::DiTParallelLinear;
using xllm::dit::LinearType;
using xllm::dit::TpOptions;
#include "framework/model_context.h"
#include "models/dit/transformer_flux.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {

inline torch::Tensor wan_apply_rotary_emb(const torch::Tensor& hidden_states,
                                          const torch::Tensor& freqs_cos,
                                          const torch::Tensor& freqs_sin) {
  auto input_dtype = hidden_states.dtype();
  auto x = hidden_states.to(torch::kFloat32);
  auto x_flat = x.unflatten(-1, std::vector<int64_t>{-1, 2});
  auto x1 = x_flat.select(-1, 0);
  auto x2 = x_flat.select(-1, 1);

  auto cos = freqs_cos.to(torch::kFloat32).slice(-1, 0, freqs_cos.size(-1), 2);
  auto sin = freqs_sin.to(torch::kFloat32).slice(-1, 1, freqs_sin.size(-1), 2);

  auto out1 = x1 * cos - x2 * sin;
  auto out2 = x1 * sin + x2 * cos;
  auto out = torch::stack({out1, out2}, -1).flatten(-2, -1);

  return out.to(input_dtype);
}

class FP32LayerNormImpl : public torch::nn::Module {
 public:
  FP32LayerNormImpl(const ModelContext& context,
                    int64_t normalized_shape,
                    double eps = 1e-6,
                    bool elementwise_affine = true)
      : options_(context.get_tensor_options()),
        normalized_shape_(normalized_shape),
        eps_(eps),
        elementwise_affine_(elementwise_affine) {
    if (elementwise_affine) {
      weight_ = register_parameter("weight", torch::ones({normalized_shape}));
      bias_ = register_parameter("bias", torch::zeros({normalized_shape}));
    }
  }

  torch::Tensor forward(const torch::Tensor& x, bool keep_fp32 = false) {
    auto origin_dtype = x.dtype();
    auto x_fp32 = x.to(torch::kFloat32);
    torch::Tensor result;
    if (elementwise_affine_) {
      result = torch::layer_norm(x_fp32,
                                 {normalized_shape_},
                                 weight_.to(torch::kFloat32),
                                 bias_.to(torch::kFloat32),
                                 eps_);
    } else {
      result = torch::layer_norm(
          x_fp32, {normalized_shape_}, torch::nullopt, torch::nullopt, eps_);
    }
    if (keep_fp32 == true) {
      return result;
    }
    return result.to(origin_dtype);
  }

  void load_state_dict(const StateDict& state_dict) {
    if (elementwise_affine_) {
      weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
      weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (elementwise_affine_) {
      CHECK(weight_is_loaded_)
          << "weight is not loaded for " << prefix + "weight";
      CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
  torch::TensorOptions options_;
  int64_t normalized_shape_;
  double eps_;
  bool elementwise_affine_;
};
TORCH_MODULE(FP32LayerNorm);

class WanTimestepEmbeddingImpl : public torch::nn::Module {
 public:
  WanTimestepEmbeddingImpl(int64_t in_channels,
                           int64_t time_embed_dim,
                           int64_t out_dim = -1,
                           bool sample_proj_bias = true)
      : options_(torch::dtype(torch::kFloat32)) {
    linear_1_ = register_module(
        "linear_1",
        layer::AddMatmul(
            in_channels, time_embed_dim, sample_proj_bias, options_));

    act_ = register_module("act", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim > 0) ? out_dim : time_embed_dim;
    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, options_));
  }

  torch::Tensor forward(const torch::Tensor& sample) {
    torch::Tensor result = sample;

    result = linear_1_->forward(result);

    if (act_) {
      result = act_->forward(result);
    }

    result = linear_2_->forward(result);
    return result;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  torch::TensorOptions options_;
  layer::AddMatmul linear_1_{nullptr};
  torch::nn::SiLU act_{nullptr};
  layer::AddMatmul linear_2_{nullptr};
};
TORCH_MODULE(WanTimestepEmbedding);

class WanTimestepsImpl : public torch::nn::Module {
 public:
  explicit WanTimestepsImpl(int64_t num_channels,
                            bool flip_sin_to_cos = true,
                            float downscale_freq_shift = 0.0,
                            int64_t scale = 1)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  num_channels_,
                                  flip_sin_to_cos_,
                                  downscale_freq_shift_,
                                  scale_);
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
  int64_t scale_;

  torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                       int64_t embedding_dim,
                                       bool flip_sin_to_cos = false,
                                       float downscale_freq_shift = 1.0f,
                                       float scale = 1.0f,
                                       int64_t max_period = 10000) {
    int64_t half_dim = embedding_dim / 2;
    auto exponent = -std::log(static_cast<float>(max_period)) *
                    torch::arange(0,
                                  half_dim,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(timesteps.device()));
    exponent = exponent / (half_dim - downscale_freq_shift);

    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
    emb = scale * emb;
    emb = torch::cat({torch::sin(emb), torch::cos(emb)}, /*dim=*/-1);

    if (flip_sin_to_cos) {
      emb = torch::cat({emb.slice(/*dim=*/-1, /*start=*/half_dim),
                        emb.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)},
                       /*dim=*/-1);
    }

    if (embedding_dim % 2 == 1) {
      emb = torch::nn::functional::pad(
          emb, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
    }

    return emb;
  }
};
TORCH_MODULE(WanTimesteps);

class WanGELUImpl : public torch::nn::Module {
 public:
  WanGELUImpl(int64_t dim_in,
              int64_t dim_out,
              bool approximate,
              bool with_bias,
              const ModelContext& context,
              const ParallelArgs& parallel_args)
      : approximate_(approximate),
        options_(context.get_tensor_options()),
        parallel_args_(parallel_args) {
    LinearType linear_type =
        FLAGS_tp_size > 1 ? LinearType::TensorParallel : LinearType::Default;
    std::optional<TpOptions> tp_options = std::nullopt;
    if (FLAGS_tp_size > 1) {
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
          /*tp_size=*/FLAGS_tp_size,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }
    auto proj = DiTParallelLinear(dim_in,
                                  dim_out,
                                  with_bias,
                                  options_,
                                  linear_type,
                                  std::nullopt,
                                  tp_options);
    proj_ = register_module("proj", proj);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_in) {
    torch::Tensor hidden_states = proj_->forward(hidden_states_in);
    if (approximate_) {
      hidden_states = torch::gelu(hidden_states, "tanh");
    } else {
      hidden_states = torch::gelu(hidden_states);
    }
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    proj_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    proj_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "proj.");
  }

 private:
  bool approximate_;
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
  DiTParallelLinear proj_{nullptr};
};
TORCH_MODULE(WanGELU);

class WanFeedForwardImpl : public torch::nn::Module {
 public:
  WanFeedForwardImpl(const ModelContext& context,
                     const ParallelArgs& parallel_args,
                     int64_t dim,
                     int64_t dim_out = -1,
                     int64_t mult = 4,
                     float dropout = 0.0f,
                     const std::string& activation_fn = "geglu",
                     bool final_dropout = false,
                     int64_t inner_dim = -1,
                     bool with_bias = true)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    int64_t actual_inner_dim =
        (inner_dim > 0) ? inner_dim : static_cast<int64_t>(dim * mult);
    int64_t actual_dim_out = (dim_out > 0) ? dim_out : dim;

    if (activation_fn == "gelu") {
      act_fn_ = register_module("act_fn",
                                WanGELU(dim,
                                        actual_inner_dim,
                                        /*approximate*/ false,
                                        with_bias,
                                        context,
                                        parallel_args));
    } else if (activation_fn == "gelu-approximate") {
      act_fn_ = register_module("act_fn",
                                WanGELU(dim,
                                        actual_inner_dim,
                                        /*approximate*/ true,
                                        with_bias,
                                        context,
                                        parallel_args));
    } else {
      act_fn_ = register_module("act_fn",
                                WanGELU(dim,
                                        actual_inner_dim,
                                        /*approximate*/ true,
                                        with_bias,
                                        context,
                                        parallel_args));
    }

    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));

    LinearType linear_out_type =
        FLAGS_tp_size > 1 ? LinearType::TensorParallel : LinearType::Default;
    std::optional<TpOptions> tp_out_options = std::nullopt;
    if (FLAGS_tp_size > 1) {
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
          /*tp_size=*/FLAGS_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }
    auto proj_out = DiTParallelLinear(actual_inner_dim,
                                      actual_dim_out,
                                      with_bias,
                                      options_,
                                      linear_out_type,
                                      std::nullopt,
                                      tp_out_options);
    proj_out_ = register_module("proj_out", proj_out);

    if (final_dropout) {
      final_dropout_ =
          register_module("final_dropout", torch::nn::Dropout(dropout));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto output = act_fn_->forward(hidden_states);
    output = dropout_->forward(output);
    output = proj_out_->forward(output);
    if (final_dropout_) {
      output = final_dropout_->forward(output);
    }
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    act_fn_->load_state_dict(state_dict.get_dict_with_prefix("net.0."));
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    act_fn_->verify_loaded_weights(prefix + "net.0.");
    proj_out_->verify_loaded_weights(prefix + "net.2.");
  }

 private:
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
  WanGELU act_fn_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
  DiTParallelLinear proj_out_{nullptr};
  torch::nn::Dropout final_dropout_{nullptr};
};
TORCH_MODULE(WanFeedForward);

class WanPixArtAlphaTextProjectionImpl : public torch::nn::Module {
 public:
  WanPixArtAlphaTextProjectionImpl(int64_t in_features,
                                   int64_t hidden_size,
                                   int64_t out_features = -1,
                                   const std::string& act_fn = "gelu_tanh")
      : options_(torch::dtype(torch::kFloat32)) {
    int64_t actual_out_features =
        (out_features > 0) ? out_features : hidden_size;

    linear_1_ = register_module(
        "linear_1", layer::AddMatmul(in_features, hidden_size, true, options_));

    if (act_fn == "gelu_tanh") {
      act_1_ = register_module(
          "act_1",
          torch::nn::Functional(
              std::function<torch::Tensor(const torch::Tensor&)>(
                  [](const torch::Tensor& x) {
                    return torch::gelu(x, "tanh");
                  })));
    } else if (act_fn == "silu") {
      act_1_ = register_module("act_1", torch::nn::SiLU());
    } else {
      act_1_ = register_module(
          "act_1",
          torch::nn::Functional(
              std::function<torch::Tensor(const torch::Tensor&)>(
                  [](const torch::Tensor& x) {
                    return torch::gelu(x, "tanh");
                  })));
    }

    linear_2_ = register_module(
        "linear_2",
        layer::AddMatmul(hidden_size, actual_out_features, true, options_));
  }

  torch::Tensor forward(const torch::Tensor& caption) {
    auto hidden_states = linear_1_->forward(caption);
    hidden_states = act_1_.forward(hidden_states);
    hidden_states = linear_2_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  torch::TensorOptions options_;
  layer::AddMatmul linear_1_{nullptr};
  torch::nn::AnyModule act_1_;
  layer::AddMatmul linear_2_{nullptr};
};
TORCH_MODULE(WanPixArtAlphaTextProjection);

class WanAttentionImpl : public torch::nn::Module {
 public:
  explicit WanAttentionImpl(const ModelContext& context,
                            const ParallelArgs& parallel_args,
                            int64_t cross_attention_dim_head = -1)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    dim_ = model_args.head_dim() * model_args.n_heads();
    heads_ = model_args.n_heads();
    dim_head_ = model_args.head_dim();
    added_kv_proj_dim_ = model_args.added_kv_proj_dim();
    eps_ = 1e-6f;
    dropout_ = 0.0f;

    int64_t cross_dim_head = (cross_attention_dim_head > 0)
                                 ? cross_attention_dim_head
                                 : model_args.head_dim();
    is_cross_attention_ = cross_dim_head > 0;

    if (is_cross_attention_) {
      kv_inner_dim_ = cross_dim_head * heads_;
    } else {
      kv_inner_dim_ = heads_ * dim_head_;
    }
    LinearType linear_type =
        FLAGS_tp_size > 1 ? LinearType::TensorParallel : LinearType::Default;
    std::optional<TpOptions> tp_options_qk = std::nullopt;
    std::optional<TpOptions> tp_options_v = std::nullopt;
    if (FLAGS_tp_size > 1) {
      tp_options_qk = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
          /*tp_size=*/FLAGS_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
      tp_options_v = TpOptions(
          /*column_parallel=*/true,
          /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
          /*tp_size=*/FLAGS_tp_size,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }
    auto to_q = DiTParallelLinear(dim_,
                                  heads_ * dim_head_,
                                  true,
                                  options_,
                                  linear_type,
                                  std::nullopt,
                                  tp_options_qk);
    to_q_ = register_module("to_q", to_q);
    auto to_k = DiTParallelLinear(dim_,
                                  kv_inner_dim_,
                                  true,
                                  options_,
                                  linear_type,
                                  std::nullopt,
                                  tp_options_qk);
    to_k_ = register_module("to_k", to_k);
    auto to_v = DiTParallelLinear(dim_,
                                  kv_inner_dim_,
                                  true,
                                  options_,
                                  linear_type,
                                  std::nullopt,
                                  tp_options_v);
    to_v_ = register_module("to_v", to_v);
    LinearType to_out_type =
        FLAGS_tp_size > 1 ? LinearType::TensorParallel : LinearType::Default;
    std::optional<TpOptions> tp_to_out_options = std::nullopt;
    if (FLAGS_tp_size > 1) {
      tp_to_out_options = TpOptions(
          /*column_parallel=*/false,
          /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
          /*tp_size=*/FLAGS_tp_size,
          /*gather_output=*/true,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }
    auto to_out = DiTParallelLinear(heads_ * dim_head_,
                                    dim_,
                                    true,
                                    options_,
                                    to_out_type,
                                    std::nullopt,
                                    tp_to_out_options);
    to_out_ = register_module("to_out", to_out);
    norm_q_ = register_module(
        "norm_q", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    norm_k_ = register_module(
        "norm_k", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    if (added_kv_proj_dim_ > 0) {
      LinearType add_kv_type =
          FLAGS_tp_size > 1 ? LinearType::TensorParallel : LinearType::Default;
      std::optional<TpOptions> add_k_options = std::nullopt;
      std::optional<TpOptions> add_v_options = std::nullopt;
      if (FLAGS_tp_size > 1) {
        add_k_options = TpOptions(
            /*column_parallel=*/true,
            /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
            /*tp_size=*/FLAGS_tp_size,
            /*gather_output=*/false,
            /*need_scatter=*/false,
            /*process_group=*/parallel_args_.dit_tp_group_);
        add_v_options = TpOptions(
            /*column_parallel=*/true,
            /*tp_rank=*/parallel_args_.dit_tp_group_->rank(),
            /*tp_size=*/FLAGS_tp_size,
            /*gather_output=*/false,
            /*need_scatter=*/false,
            /*process_group=*/parallel_args_.dit_tp_group_);
      }
      auto add_k_proj = DiTParallelLinear(added_kv_proj_dim_,
                                          heads_ * dim_head_,
                                          true,
                                          options_,
                                          add_kv_type,
                                          std::nullopt,
                                          add_k_options);
      add_k_proj_ = register_module("add_k_proj", add_k_proj);
      auto add_v_proj = DiTParallelLinear(added_kv_proj_dim_,
                                          heads_ * dim_head_,
                                          true,
                                          options_,
                                          add_kv_type,
                                          std::nullopt,
                                          add_v_options);
      add_v_proj_ = register_module("add_v_proj", add_v_proj);
      norm_added_k_ = register_module(
          "norm_added_k", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    }
  }

  torch::Tensor at_npu_attention(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& v) {
    const auto q_t = q.transpose(1, 2);
    const auto k_t = k.transpose(1, 2);
    const auto v_t = v.transpose(1, 2);

#if defined(USE_NPU)
    const int64_t head_num = q_t.size(1);
    const int64_t head_dim = q_t.size(-1);
    const auto results = at_npu::native::custom_ops::npu_fusion_attention(
        q_t,
        k_t,
        v_t,
        head_num,
        "BNSD",
        torch::nullopt,
        torch::nullopt,
        torch::nullopt,
        std::pow(head_dim, -0.5),
        1.0,
        65535,
        65535);
    torch::Tensor out = std::get<0>(results).transpose(1, 2);
#else
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim_head_));
    auto attn_weights = torch::matmul(q_t, k_t.transpose(-2, -1)) * scale;
    attn_weights = torch::softmax(attn_weights, -1);
    torch::Tensor out = torch::matmul(attn_weights, v_t).transpose(1, 2);
#endif
    return out.flatten(2, 3).to(q.dtype());
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states_in,
      const torch::Tensor& encoder_hidden_states = torch::Tensor(),
      std::optional<std::pair<torch::Tensor, torch::Tensor>> rotary_emb =
          std::nullopt) {
    torch::Tensor hidden_states = hidden_states_in;
    bool is_self_attention =
        !encoder_hidden_states.defined() ||
        (encoder_hidden_states.size(1) == hidden_states.size(1));

    torch::Tensor encoder_hidden_states_text =
        encoder_hidden_states.defined() ? encoder_hidden_states : hidden_states;
    torch::Tensor encoder_hidden_states_img;

    if (!is_self_attention && add_k_proj_ &&
        encoder_hidden_states_text.defined() &&
        encoder_hidden_states_text.size(1) > 512) {
      int64_t image_context_length = encoder_hidden_states_text.size(1) - 512;
      encoder_hidden_states_img =
          encoder_hidden_states_text.slice(1, 0, image_context_length);
      encoder_hidden_states_text =
          encoder_hidden_states_text.slice(1, image_context_length);
    }

    torch::Tensor query = to_q_->forward(hidden_states);
    torch::Tensor key = to_k_->forward(encoder_hidden_states_text);
    torch::Tensor value = to_v_->forward(encoder_hidden_states_text);
    query = std::get<0>(norm_q_->forward(query));
    key = std::get<0>(norm_k_->forward(key));

    if (FLAGS_tp_size > 1) {
      query = parallel_state::scatter(
          query, parallel_args_.dit_tp_group_, /*dim=*/-1);
      key = parallel_state::scatter(
          key, parallel_args_.dit_tp_group_, /*dim=*/-1);
    }

    int64_t batch_size = query.size(0);
    int64_t n_heads = heads_;
    if (FLAGS_tp_size > 1) {
      n_heads = heads_ / FLAGS_tp_size;
    }
    query = query.view({batch_size, -1, n_heads, dim_head_});
    key = key.view({batch_size, -1, n_heads, dim_head_});
    value = value.view({batch_size, -1, n_heads, dim_head_});

    if (rotary_emb.has_value()) {
      torch::Tensor freqs_cos = rotary_emb->first;
      torch::Tensor freqs_sin = rotary_emb->second;
      query = wan_apply_rotary_emb(query, freqs_cos, freqs_sin);
      key = wan_apply_rotary_emb(key, freqs_cos, freqs_sin);
    }

    torch::Tensor hidden_states_img;
    if (encoder_hidden_states_img.defined()) {
      torch::Tensor key_img = add_k_proj_->forward(encoder_hidden_states_img);
      torch::Tensor value_img = add_v_proj_->forward(encoder_hidden_states_img);

      key_img = std::get<0>(norm_added_k_->forward(key_img));

      if (FLAGS_tp_size > 1) {
        key_img = parallel_state::scatter(
            key_img, parallel_args_.dit_tp_group_, /*dim=*/-1);
      }

      key_img = key_img.view({batch_size, -1, n_heads, dim_head_});
      value_img = value_img.view({batch_size, -1, n_heads, dim_head_});
      hidden_states_img = at_npu_attention(query, key_img, value_img);
    }
    hidden_states = at_npu_attention(query, key, value);
    if (hidden_states_img.defined()) {
      hidden_states = hidden_states + hidden_states_img;
    }

    hidden_states = to_out_->forward(hidden_states);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));

    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));

    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));

    if (add_k_proj_) {
      add_k_proj_->load_state_dict(
          state_dict.get_dict_with_prefix("add_k_proj."));
      add_v_proj_->load_state_dict(
          state_dict.get_dict_with_prefix("add_v_proj."));
      norm_added_k_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_k."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_q_->verify_loaded_weights(prefix + "to_q.");
    to_k_->verify_loaded_weights(prefix + "to_k.");
    to_v_->verify_loaded_weights(prefix + "to_v.");

    to_out_->verify_loaded_weights(prefix + "to_out.0.");

    if (add_k_proj_) {
      add_k_proj_->verify_loaded_weights(prefix + "add_k_proj.");
      add_v_proj_->verify_loaded_weights(prefix + "add_v_proj.");
    }
  }

 private:
  int64_t dim_;
  int64_t heads_;
  int64_t dim_head_;
  int64_t kv_inner_dim_;
  int64_t added_kv_proj_dim_;
  float eps_;
  float dropout_;
  bool is_cross_attention_;

  DiTParallelLinear to_q_{nullptr};
  DiTParallelLinear to_k_{nullptr};
  DiTParallelLinear to_v_{nullptr};
  DiTParallelLinear to_out_{nullptr};
  DiTParallelLinear add_k_proj_{nullptr};
  DiTParallelLinear add_v_proj_{nullptr};
  ParallelArgs parallel_args_;

  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  layer::RMSNorm norm_added_k_{nullptr};

  torch::TensorOptions options_;
};
TORCH_MODULE(WanAttention);

// for wan2.2 I2V, actually not used
class WanImageEmbeddingImpl : public torch::nn::Module {
 public:
  explicit WanImageEmbeddingImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    in_features_ = model_args.image_embed_dim();
    out_features_ = model_args.head_dim() * model_args.n_heads();
    pos_embed_seq_len_ = model_args.pos_embed_seq_len();

    norm1_ =
        register_module("norm1", FP32LayerNorm(context, in_features_, 1e-6));
    ff_ = register_module("ff",
                          WanFeedForward(context,
                                         parallel_args,
                                         in_features_,
                                         out_features_,
                                         1,
                                         0.0f,
                                         "gelu",
                                         false,
                                         -1,
                                         true));
    norm2_ =
        register_module("norm2", FP32LayerNorm(context, out_features_, 1e-6));

    if (pos_embed_seq_len_ > 0) {
      pos_embed_ = register_parameter(
          "pos_embed",
          torch::zeros({1, pos_embed_seq_len_, in_features_}, options_));
    }
  }

  torch::Tensor forward(const torch::Tensor& encoder_hidden_states_image) {
    torch::Tensor hidden_states = encoder_hidden_states_image;

    if (pos_embed_.defined()) {
      int64_t batch_size = hidden_states.size(0);
      int64_t seq_len = hidden_states.size(1);
      int64_t embed_dim = hidden_states.size(2);
      hidden_states = hidden_states.view({-1, 2 * seq_len, embed_dim});
      hidden_states = hidden_states + pos_embed_;
    }

    hidden_states = norm1_->forward(hidden_states);
    hidden_states = ff_->forward(hidden_states);
    hidden_states = norm2_->forward(hidden_states);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    if (pos_embed_.defined()) {
      weight::load_weight(
          state_dict, "pos_embed", pos_embed_, pos_embed_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    ff_->verify_loaded_weights(prefix + "ff.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
  }

 private:
  int64_t in_features_;
  int64_t out_features_;
  int64_t pos_embed_seq_len_;

  FP32LayerNorm norm1_{nullptr};
  WanFeedForward ff_{nullptr};
  FP32LayerNorm norm2_{nullptr};
  torch::Tensor pos_embed_;
  bool pos_embed_loaded_{false};
  torch::TensorOptions options_;
};
TORCH_MODULE(WanImageEmbedding);

class WanTimeTextImageEmbeddingImpl : public torch::nn::Module {
 public:
  explicit WanTimeTextImageEmbeddingImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    dim_ = model_args.head_dim() * model_args.n_heads();
    time_freq_dim_ = model_args.time_freq_dim();
    time_proj_dim_ = dim_ * 6;
    text_embed_dim_ = model_args.text_embed_dim();
    image_embed_dim_ = model_args.image_embed_dim();
    pos_embed_seq_len_ = model_args.pos_embed_seq_len();

    timesteps_proj_ = register_module(
        "timesteps_proj", WanTimesteps(time_freq_dim_, true, 0.0f, 1));
    time_embedder_ = register_module(
        "time_embedder", WanTimestepEmbedding(time_freq_dim_, dim_, -1, true));
    act_fn_ = register_module("act_fn", torch::nn::SiLU());
    time_proj_ = register_module(
        "time_proj", layer::AddMatmul(dim_, time_proj_dim_, true, options_));

    text_embedder_ = register_module(
        "text_embedder",
        WanPixArtAlphaTextProjection(text_embed_dim_, dim_, dim_, "gelu_tanh"));

    if (image_embed_dim_ > 0) {
      image_embedder_ =
          register_module("image_embedder", WanImageEmbedding(context));
    }
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  forward(const torch::Tensor& timestep,
          const torch::Tensor& encoder_hidden_states,
          const torch::Tensor& encoder_hidden_states_image = torch::Tensor(),
          std::optional<int64_t> timestep_seq_len = std::nullopt) {
    torch::Tensor timestep_proj = timesteps_proj_->forward(timestep);
    int64_t seq_len = timestep_seq_len.value_or(1);
    if (seq_len > 1) {
      auto ts = timestep.expand({1, seq_len}).flatten();
      timestep_proj =
          timesteps_proj_->forward(ts).view({-1, seq_len, time_freq_dim_});
    }
    timestep_proj = timestep_proj.to(torch::kFloat32);
    auto embed_dtype = encoder_hidden_states.dtype();
    torch::Tensor temb = time_embedder_->forward(timestep_proj.to(embed_dtype));
    torch::Tensor timestep_proj_out =
        time_proj_->forward(act_fn_->forward(temb));
    if (seq_len > 1) {
      timestep_proj_out = timestep_proj_out.view({-1, seq_len, 6, dim_});
    } else {
      timestep_proj_out = timestep_proj_out.view({-1, 6, dim_});
    }

    torch::Tensor text_emb = text_embedder_->forward(encoder_hidden_states);

    torch::Tensor image_emb;
    if (image_embedder_ && encoder_hidden_states_image.defined()) {
      image_emb = image_embedder_->forward(encoder_hidden_states_image);
    }

    return {temb, timestep_proj_out, text_emb, image_emb};
  }

  void load_state_dict(const StateDict& state_dict) {
    time_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder."));
    time_proj_->load_state_dict(state_dict.get_dict_with_prefix("time_proj."));
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
    if (image_embedder_) {
      image_embedder_->load_state_dict(
          state_dict.get_dict_with_prefix("image_embedder."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    time_embedder_->verify_loaded_weights(prefix + "time_embedder.");
    time_proj_->verify_loaded_weights(prefix + "time_proj.");
    text_embedder_->verify_loaded_weights(prefix + "text_embedder.");
    if (image_embedder_) {
      image_embedder_->verify_loaded_weights(prefix + "image_embedder.");
    }
  }

 private:
  int64_t dim_;
  int64_t time_freq_dim_;
  int64_t time_proj_dim_;
  int64_t text_embed_dim_;
  int64_t image_embed_dim_;
  int64_t pos_embed_seq_len_;

  WanTimesteps timesteps_proj_{nullptr};
  WanTimestepEmbedding time_embedder_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  layer::AddMatmul time_proj_{nullptr};
  WanPixArtAlphaTextProjection text_embedder_{nullptr};
  WanImageEmbedding image_embedder_{nullptr};

  torch::TensorOptions options_;
};
TORCH_MODULE(WanTimeTextImageEmbedding);

class WanRotaryPosEmbedImpl : public torch::nn::Module {
 public:
  explicit WanRotaryPosEmbedImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    attention_head_dim_ = model_args.head_dim();
    patch_size_ = model_args.wan_patch_size();
    max_seq_len_ = model_args.rope_max_seq_len();
    theta_ = 10000.0f;

    h_dim_ = w_dim_ = 2 * (attention_head_dim_ / 6);
    t_dim_ = attention_head_dim_ - h_dim_ - w_dim_;

    compute_freqs();
  }

  torch::Tensor get_freqs_cos() const { return freqs_cos_; }
  torch::Tensor get_freqs_sin() const { return freqs_sin_; }
  void set_freqs_cos(const torch::Tensor& t) { freqs_cos_ = t; }
  void set_freqs_sin(const torch::Tensor& t) { freqs_sin_ = t; }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states) {
    int64_t batch_size = hidden_states.size(0);
    int64_t num_frames = hidden_states.size(2);
    int64_t height = hidden_states.size(3);
    int64_t width = hidden_states.size(4);

    int64_t p_t = patch_size_[0];
    int64_t p_h = patch_size_[1];
    int64_t p_w = patch_size_[2];

    int64_t ppf = num_frames / p_t;
    int64_t pph = height / p_h;
    int64_t ppw = width / p_w;

    std::vector<int64_t> split_sizes = {t_dim_, h_dim_, w_dim_};

    auto freqs_cos_split = freqs_cos_.split(split_sizes, 1);
    auto freqs_sin_split = freqs_sin_.split(split_sizes, 1);

    torch::Tensor freqs_cos_f = freqs_cos_split[0]
                                    .slice(0, 0, ppf)
                                    .view({ppf, 1, 1, -1})
                                    .expand({ppf, pph, ppw, -1});
    torch::Tensor freqs_cos_h = freqs_cos_split[1]
                                    .slice(0, 0, pph)
                                    .view({1, pph, 1, -1})
                                    .expand({ppf, pph, ppw, -1});
    torch::Tensor freqs_cos_w = freqs_cos_split[2]
                                    .slice(0, 0, ppw)
                                    .view({1, 1, ppw, -1})
                                    .expand({ppf, pph, ppw, -1});

    torch::Tensor freqs_sin_f = freqs_sin_split[0]
                                    .slice(0, 0, ppf)
                                    .view({ppf, 1, 1, -1})
                                    .expand({ppf, pph, ppw, -1});
    torch::Tensor freqs_sin_h = freqs_sin_split[1]
                                    .slice(0, 0, pph)
                                    .view({1, pph, 1, -1})
                                    .expand({ppf, pph, ppw, -1});
    torch::Tensor freqs_sin_w = freqs_sin_split[2]
                                    .slice(0, 0, ppw)
                                    .view({1, 1, ppw, -1})
                                    .expand({ppf, pph, ppw, -1});

    torch::Tensor freqs_cos =
        torch::cat({freqs_cos_f, freqs_cos_h, freqs_cos_w}, -1)
            .reshape({1, ppf * pph * ppw, 1, -1});
    torch::Tensor freqs_sin =
        torch::cat({freqs_sin_f, freqs_sin_h, freqs_sin_w}, -1)
            .reshape({1, ppf * pph * ppw, 1, -1});

    return {freqs_cos, freqs_sin};
  }

 private:
  void compute_freqs() {
    std::vector<torch::Tensor> freqs_cos_list;
    std::vector<torch::Tensor> freqs_sin_list;

    for (int64_t dim : {t_dim_, h_dim_, w_dim_}) {
      torch::Tensor pos = torch::arange(
          0,
          max_seq_len_,
          torch::dtype(torch::kFloat32).device(options_.device()));
      torch::Tensor rotary_embed = get_1d_rotary_pos_embed(
          dim, pos, theta_, true, 1.0, 1.0, true, torch::kFloat64);

      torch::Tensor cos_vals = rotary_embed[0];
      torch::Tensor sin_vals = rotary_embed[1];

      freqs_cos_list.push_back(cos_vals);
      freqs_sin_list.push_back(sin_vals);
    }

    freqs_cos_ = torch::cat(freqs_cos_list, -1);
    freqs_sin_ = torch::cat(freqs_sin_list, -1);

    register_buffer("freqs_cos", freqs_cos_);
    register_buffer("freqs_sin", freqs_sin_);
  }

  int64_t attention_head_dim_;
  std::vector<int64_t> patch_size_;
  int64_t max_seq_len_;
  float theta_;
  int64_t t_dim_;
  int64_t h_dim_;
  int64_t w_dim_;

  torch::Tensor freqs_cos_;
  torch::Tensor freqs_sin_;

  torch::TensorOptions options_;
};
TORCH_MODULE(WanRotaryPosEmbed);

class WanTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit WanTransformerBlockImpl(const ModelContext& context,
                                   const ParallelArgs& parallel_args,
                                   int64_t block_idx = 0)
      : options_(context.get_tensor_options()),
        parallel_args_(parallel_args),
        block_idx_(block_idx) {
    auto model_args = context.get_model_args();
    dim_ = model_args.head_dim() * model_args.n_heads();
    ffn_dim_ = model_args.ffn_dim();
    num_heads_ = model_args.n_heads();
    eps_ = 1e-6f;
    added_kv_proj_dim_ = model_args.added_kv_proj_dim();
    cross_attn_norm_ = model_args.cross_attn_norm();
    qk_norm_ = model_args.qk_norm();

    norm1_ =
        register_module("norm1", FP32LayerNorm(context, dim_, eps_, false));
    attn1_ = register_module("attn1", WanAttention(context, parallel_args));
    attn2_ = register_module(
        "attn2", WanAttention(context, parallel_args, dim_ / num_heads_));
    if (cross_attn_norm_) {
      norm2_ =
          register_module("norm2", FP32LayerNorm(context, dim_, eps_, true));
    }
    ff_ = register_module("ff",
                          WanFeedForward(context,
                                         parallel_args,
                                         dim_,
                                         dim_,
                                         1,
                                         0.0f,
                                         "gelu-approximate",
                                         false,
                                         ffn_dim_,
                                         true));
    norm3_ =
        register_module("norm3", FP32LayerNorm(context, dim_, eps_, false));
    scale_shift_table_ =
        register_parameter("scale_shift_table",
                           torch::randn({1, 6, dim_}, options_) /
                               std::sqrt(static_cast<float>(dim_)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_in,
                        const torch::Tensor& encoder_hidden_states,
                        const torch::Tensor& timestep_proj,
                        std::optional<std::pair<torch::Tensor, torch::Tensor>>
                            rotary_emb = std::nullopt) {
    torch::Tensor hidden_states = hidden_states_in;
    torch::Tensor shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa,
        c_gate_msa;

    if (timestep_proj.dim() == 4) {
      auto scale_shift =
          scale_shift_table_.unsqueeze(0).to(hidden_states.dtype()) +
          timestep_proj.to(hidden_states.dtype());
      auto splits = scale_shift.chunk(6, 2);
      shift_msa = splits[0].squeeze(2);
      scale_msa = splits[1].squeeze(2);
      gate_msa = splits[2].squeeze(2);
      c_shift_msa = splits[3].squeeze(2);
      c_scale_msa = splits[4].squeeze(2);
      c_gate_msa = splits[5].squeeze(2);
    } else {
      auto scale_shift = scale_shift_table_.to(hidden_states.dtype()) +
                         timestep_proj.to(hidden_states.dtype());
      auto splits = scale_shift.chunk(6, 1);
      shift_msa = splits[0];
      scale_msa = splits[1];
      gate_msa = splits[2];
      c_shift_msa = splits[3];
      c_scale_msa = splits[4];
      c_gate_msa = splits[5];
    }

    torch::Tensor norm1_result = norm1_->forward(hidden_states);
    torch::Tensor norm_hidden_states =
        (norm1_result.to(hidden_states.dtype()) * (1 + scale_msa) + shift_msa);
    torch::Tensor attn_output =
        attn1_->forward(norm_hidden_states, norm_hidden_states, rotary_emb);
    hidden_states = hidden_states + attn_output * gate_msa;

    if (cross_attn_norm_) {
      norm_hidden_states = norm2_->forward(hidden_states);
    } else {
      norm_hidden_states = hidden_states;
    }

    attn_output = attn2_->forward(
        norm_hidden_states, encoder_hidden_states, std::nullopt);
    hidden_states = hidden_states + attn_output;
    torch::Tensor norm2_result = norm3_->forward(hidden_states);
    norm_hidden_states = (norm2_result * (1 + c_scale_msa) + c_shift_msa);
    torch::Tensor ff_output = ff_->forward(norm_hidden_states);
    hidden_states = hidden_states + ff_output * c_gate_msa;

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    attn1_->load_state_dict(state_dict.get_dict_with_prefix("attn1."));
    attn2_->load_state_dict(state_dict.get_dict_with_prefix("attn2."));
    if (cross_attn_norm_ && norm2_) {
      norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    }
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ffn."));
    weight::load_weight(state_dict,
                        "scale_shift_table",
                        scale_shift_table_,
                        scale_shift_table_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn1_->verify_loaded_weights(prefix + "attn1.");
    if (cross_attn_norm_) {
      norm2_->verify_loaded_weights(prefix + "norm2.");
    }
    attn2_->verify_loaded_weights(prefix + "attn2.");
    ff_->verify_loaded_weights(prefix + "ffn.");
    CHECK(scale_shift_table_loaded_) << "scale_shift_table is not loaded for "
                                     << prefix + "scale_shift_table";
  }

 private:
  int64_t dim_;
  int64_t ffn_dim_;
  int64_t num_heads_;
  float eps_;
  int64_t added_kv_proj_dim_;
  bool cross_attn_norm_;
  int64_t block_idx_ = 0;
  std::string qk_norm_;

  FP32LayerNorm norm1_{nullptr};
  WanAttention attn1_{nullptr};
  WanAttention attn2_{nullptr};
  FP32LayerNorm norm2_{nullptr};
  WanFeedForward ff_{nullptr};
  FP32LayerNorm norm3_{nullptr};
  torch::Tensor scale_shift_table_;
  bool scale_shift_table_loaded_{false};

  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(WanTransformerBlock);

class WanTransformer3DModelImpl : public torch::nn::Module {
 public:
  explicit WanTransformer3DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    patch_size_ = model_args.wan_patch_size();
    num_attention_heads_ = model_args.n_heads();
    attention_head_dim_ = model_args.head_dim();
    in_channels_ = model_args.dit_in_channels();
    out_channels_ = model_args.dit_out_channels();
    text_dim_ = model_args.text_embed_dim();
    freq_dim_ = model_args.time_freq_dim();
    ffn_dim_ = model_args.ffn_dim();
    num_layers_ = model_args.num_layers();
    image_dim_ = model_args.image_embed_dim();
    added_kv_proj_dim_ = model_args.added_kv_proj_dim();
    rope_max_seq_len_ = model_args.rope_max_seq_len();
    pos_embed_seq_len_ = model_args.pos_embed_seq_len();
    cross_attn_norm_ = model_args.cross_attn_norm();
    qk_norm_ = model_args.qk_norm();

    inner_dim_ = num_attention_heads_ * attention_head_dim_;
    if (out_channels_ <= 0) {
      out_channels_ = in_channels_;
    }
    rope_ = register_module("rope", WanRotaryPosEmbed(context));
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                in_channels_,
                inner_dim_,
                {patch_size_[0], patch_size_[1], patch_size_[2]})
                .stride({patch_size_[0], patch_size_[1], patch_size_[2]})
                .padding(0)));

    patch_embedding_->to(options_.dtype().toScalarType());
    condition_embedder_ = register_module("condition_embedder",
                                          WanTimeTextImageEmbedding(context));

    blocks_ = register_module("blocks", torch::nn::ModuleList());
    transformer_layers_.reserve(num_layers_);
    for (int64_t i = 0; i < num_layers_; ++i) {
      auto block =
          WanTransformerBlock(context, parallel_args, static_cast<int64_t>(i));
      blocks_->push_back(block);
      transformer_layers_.push_back(block);
    }

    norm_out_ = register_module(
        "norm_out", FP32LayerNorm(context, inner_dim_, 1e-6, false));
    int64_t patch_prod = patch_size_[0] * patch_size_[1] * patch_size_[2];
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(
            inner_dim_, out_channels_ * patch_prod, true, options_));
    scale_shift_table_ =
        register_parameter("scale_shift_table",
                           torch::randn({1, 2, inner_dim_}, options_) /
                               std::sqrt(static_cast<float>(inner_dim_)));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states_in,
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_image = torch::Tensor()) {
    int64_t batch_size = hidden_states_in.size(0);
    int64_t num_frames = hidden_states_in.size(2);
    int64_t height = hidden_states_in.size(3);
    int64_t width = hidden_states_in.size(4);

    int64_t p_t = patch_size_[0];
    int64_t p_h = patch_size_[1];
    int64_t p_w = patch_size_[2];
    int64_t post_patch_num_frames = num_frames / p_t;
    int64_t post_patch_height = height / p_h;
    int64_t post_patch_width = width / p_w;

    torch::Tensor hidden_states = hidden_states_in;

    auto [freqs_cos, freqs_sin] = rope_->forward(hidden_states);

    auto rotary_emb = std::make_pair(freqs_cos, freqs_sin);

    hidden_states = patch_embedding_->forward(
        hidden_states.to(patch_embedding_->weight.dtype()));
    hidden_states = hidden_states.flatten(2).transpose(1, 2);

    torch::Tensor timestep_input = timestep;
    int64_t ts_seq_len_val = hidden_states.size(1);
    std::optional<int64_t> ts_seq_len = ts_seq_len_val;
    if (timestep.dim() == 2) {
      timestep_input = timestep.flatten();
    }

    auto [temb,
          timestep_proj,
          encoder_hidden_states_embedded,
          encoder_hidden_states_image_embedded] =
        condition_embedder_->forward(timestep_input,
                                     encoder_hidden_states,
                                     encoder_hidden_states_image,
                                     ts_seq_len);

    if (timestep_proj.dim() == 4) {
    } else if (ts_seq_len.has_value() && ts_seq_len.value() > 1) {
      timestep_proj =
          timestep_proj.view({batch_size, ts_seq_len.value(), 6, -1});
    } else {
      timestep_proj = timestep_proj.view({batch_size, 6, -1});
    }

    if (encoder_hidden_states_image_embedded.defined()) {
      encoder_hidden_states_embedded =
          torch::cat({encoder_hidden_states_image_embedded,
                      encoder_hidden_states_embedded},
                     1);
    }

    for (int64_t i = 0; i < transformer_layers_.size(); ++i) {
      hidden_states =
          transformer_layers_[i]->forward(hidden_states,
                                          encoder_hidden_states_embedded,
                                          timestep_proj,
                                          rotary_emb);
    }

    torch::Tensor shift, scale;
    if (temb.dim() == 3) {
      auto scale_shift =
          scale_shift_table_.unsqueeze(0).to(temb.device()) + temb.unsqueeze(2);
      auto splits = scale_shift.chunk(2, 2);
      shift = splits[0].squeeze(2);
      scale = splits[1].squeeze(2);
    } else {
      auto scale_shift =
          scale_shift_table_.to(temb.device()) + temb.unsqueeze(1);
      auto splits = scale_shift.chunk(2, 1);
      shift = splits[0];
      scale = splits[1];
    }
    shift = shift.to(hidden_states.device());
    scale = scale.to(hidden_states.device());

    auto norm_result = norm_out_->forward(hidden_states, /*keep_fp32*/ true);
    auto one_plus_scale =
        (1 + scale.to(hidden_states.dtype())).to(torch::kFloat32);
    auto shift_fp32 = shift.to(torch::kFloat32);
    auto norm_out = norm_result * one_plus_scale + shift_fp32;
    hidden_states = norm_out.to(hidden_states.dtype());

    hidden_states = proj_out_->forward(hidden_states);
    hidden_states = hidden_states.view({batch_size,
                                        post_patch_num_frames,
                                        post_patch_height,
                                        post_patch_width,
                                        p_t,
                                        p_h,
                                        p_w,
                                        -1});
    hidden_states = hidden_states.permute({0, 7, 1, 4, 2, 5, 3, 6});
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "patch_embedding.weight",
                        patch_embedding_->weight,
                        pad_embedding_weight_loaded_);
    weight::load_weight(state_dict,
                        "patch_embedding.bias",
                        patch_embedding_->bias,
                        pad_embedding_bias_loaded_);

    condition_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("condition_embedder."));
    for (int64_t i = 0; i < transformer_layers_.size(); ++i) {
      transformer_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("blocks." + std::to_string(i) + "."));
    }
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
    weight::load_weight(state_dict,
                        "scale_shift_table",
                        scale_shift_table_,
                        scale_shift_table_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(pad_embedding_weight_loaded_) << "patch_embedding is not loaded for"
                                        << prefix << "pad_embedding.weight";
    CHECK(pad_embedding_bias_loaded_) << "patch_embedding is not loaded for"
                                      << prefix << "pad_embedding.bias";

    condition_embedder_->verify_loaded_weights(prefix + "condition_embedder.");
    for (size_t i = 0; i < transformer_layers_.size(); ++i) {
      transformer_layers_[i]->verify_loaded_weights(prefix + "blocks." +
                                                    std::to_string(i) + ".");
    }
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
    CHECK(scale_shift_table_loaded_) << "scale_shift_table is not loaded for "
                                     << prefix + "scale_shift_table";
  }

  int64_t in_channels() const { return in_channels_; }
  const std::vector<int64_t>& patch_size() const { return patch_size_; }
  bool guidance_embeds() const { return false; }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_state_dict(*state_dict);
    }
    verify_loaded_weights("");

    auto freqs_cos_fp32 = rope_->get_freqs_cos().clone();
    auto freqs_sin_fp32 = rope_->get_freqs_sin().clone();

    this->to(torch::kBFloat16);
    rope_->set_freqs_cos(freqs_cos_fp32);
    rope_->set_freqs_sin(freqs_sin_fp32);
  }

 private:
  std::vector<int64_t> patch_size_;
  int64_t num_attention_heads_;
  int64_t attention_head_dim_;
  int64_t in_channels_;
  int64_t out_channels_;
  int64_t text_dim_;
  int64_t freq_dim_;
  int64_t ffn_dim_;
  int64_t num_layers_;
  int64_t image_dim_;
  int64_t added_kv_proj_dim_;
  int64_t rope_max_seq_len_;
  int64_t pos_embed_seq_len_;
  int64_t inner_dim_;
  bool cross_attn_norm_;
  std::string qk_norm_;

  torch::nn::Conv3d patch_embedding_{nullptr};
  WanTimeTextImageEmbedding condition_embedder_{nullptr};
  WanRotaryPosEmbed rope_{nullptr};
  torch::nn::ModuleList blocks_;
  std::vector<WanTransformerBlock> transformer_layers_;
  FP32LayerNorm norm_out_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  torch::Tensor scale_shift_table_;
  bool scale_shift_table_loaded_{false};
  bool pad_embedding_weight_loaded_{false};
  bool pad_embedding_bias_loaded_{false};
  torch::TensorOptions options_;
};
TORCH_MODULE(WanTransformer3DModel);

class Wan22DiTModelImpl final : public torch::nn::Module {
 public:
  explicit Wan22DiTModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    wan2_2_transformer_ =
        register_module("wan2_2_transformer", WanTransformer3DModel(context));
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_image = torch::Tensor()) {
    return wan2_2_transformer_->forward(hidden_states,
                                        timestep,
                                        encoder_hidden_states,
                                        encoder_hidden_states_image);
  }

  int64_t in_channels() { return wan2_2_transformer_->in_channels(); }
  bool guidance_embeds() { return wan2_2_transformer_->guidance_embeds(); }
  const std::vector<int64_t>& patch_size() {
    return wan2_2_transformer_->patch_size();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    wan2_2_transformer_->load_model(std::move(loader));
    wan2_2_transformer_->verify_loaded_weights("");
  }

 private:
  WanTransformer3DModel wan2_2_transformer_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Wan22DiTModel);

REGISTER_MODEL_ARGS(WanTransformer3DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(cross_attn_norm, "cross_attn_norm", true);
  LOAD_ARG_OR(eps, "eps", 1e-6);
  LOAD_ARG_OR(ffn_dim, "ffn_dim", 13824);
  LOAD_ARG_OR(time_freq_dim, "freq_dim", 256);
  LOAD_ARG_OR(dit_in_channels, "in_channels", 36);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 40);
  LOAD_ARG_OR(num_layers, "num_layers", 40);
  LOAD_ARG_OR(dit_out_channels, "out_channels", 16);
  LOAD_ARG_OR(wan_patch_size, "patch_size", (std::vector<int64_t>{1, 2, 2}));
  LOAD_ARG_OR(qk_norm, "qk_norm", "rms_norm_across_heads");
  LOAD_ARG_OR(rope_max_seq_len, "rope_max_seq_len", 1024);
  LOAD_ARG_OR(text_embed_dim, "text_dim", 4096);
  LOAD_ARG_OR(image_embed_dim, "image_dim", -1);            // -1 for null
  LOAD_ARG_OR(added_kv_proj_dim, "added_kv_proj_dim", -1);  // -1 for null
  LOAD_ARG_OR(pos_embed_seq_len, "pos_embed_seq_len", -1);  // -1 for null
});

}  // namespace xllm

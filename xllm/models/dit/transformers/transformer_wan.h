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

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/ada_layer_norm.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "models/dit/utils/dit_parallel_linear.h"
#include "models/dit/utils/sparse_attention.h"

using xllm::dit::DiTParallelLinear;
using xllm::dit::SpOptions;
using xllm::dit::TpOptions;
#if defined(USE_NPU)
#include "core/layers/npu/loader/rolling_load_manager.h"
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#endif
#include "framework/model_context.h"
#include "models/dit/transformers/transformer_flux.h"
#if defined(USE_NPU)
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "models/dit/utils/dit_block_weight_manager.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif
#include "core/framework/quant_args.h"
#include "models/model_registry.h"

namespace xllm {

inline torch::Tensor wan_apply_rotary_emb(const torch::Tensor& hidden_states,
                                          const torch::Tensor& freqs_cos,
                                          const torch::Tensor& freqs_sin) {
#if defined(USE_NPU)
  auto x_out = at_npu::native::custom_ops::npu_rotary_mul(
      hidden_states.to(torch::kFloat), freqs_cos, freqs_sin, "interleave");
  return x_out.to(hidden_states.dtype());
#else
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
#endif
}

inline int64_t sp_pad_sequence(
    torch::Tensor& hidden_states,
    torch::Tensor& freqs_cos,
    torch::Tensor& freqs_sin,
    std::pair<torch::Tensor, torch::Tensor>& rotary_emb,
    ProcessGroup* sp_group) {
  if (!sp_group || sp_group->world_size() <= 1) {
    return hidden_states.size(1);
  }
  auto group_size = sp_group->world_size();
  int64_t seq_len = hidden_states.size(1);
  if (seq_len % group_size == 0) {
    return seq_len;
  }
  int64_t pad_seq_len = ((seq_len + group_size - 1) / group_size) * group_size;
  hidden_states = torch::nn::functional::pad(
      hidden_states,
      torch::nn::functional::PadFuncOptions({0, 0, 0, pad_seq_len - seq_len}));
  freqs_cos =
      torch::nn::functional::pad(freqs_cos,
                                 torch::nn::functional::PadFuncOptions(
                                     {0, 0, 0, 0, 0, pad_seq_len - seq_len}));
  freqs_sin =
      torch::nn::functional::pad(freqs_sin,
                                 torch::nn::functional::PadFuncOptions(
                                     {0, 0, 0, 0, 0, pad_seq_len - seq_len}));
  rotary_emb = std::make_pair(freqs_cos, freqs_sin);
  return pad_seq_len;
}

inline torch::Tensor sp_all_to_all(const torch::Tensor& input,
                                   int64_t heads,
                                   int64_t dim_head,
                                   int64_t tp_size,
                                   ProcessGroup* sp_group) {
  auto fn = parallel_state::all_to_all_4D(
      input.view({input.size(0), -1, heads / tp_size, dim_head}),
      /*scatter_dim=*/2,
      /*gather_dim=*/1,
      /*async=*/false,
      sp_group);
  return fn().view({input.size(0),
                    -1,
                    heads * dim_head / (tp_size * sp_group->world_size())});
}

inline torch::Tensor sp_slice_heads(const torch::Tensor& input,
                                    int64_t heads,
                                    int64_t dim_head,
                                    int64_t tp_size,
                                    ProcessGroup* sp_group) {
  int64_t n_heads = heads / (tp_size * sp_group->world_size());
  return input
      .view({input.size(0), -1, n_heads * sp_group->world_size(), dim_head})
      .slice(2, sp_group->rank() * n_heads, (sp_group->rank() + 1) * n_heads)
      .flatten(2, 3);
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
  WanTimestepEmbeddingImpl(const ModelContext& context,
                           int64_t in_channels,
                           int64_t time_embed_dim,
                           int64_t out_dim = -1,
                           bool sample_proj_bias = true)
      : options_(torch::dtype(torch::kFloat32)) {
    quant_args_ = context.get_quant_args();
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
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("0.", "linear_1.")));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("2.", "linear_2.")));
  }
  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("0.", "linear_1."));
    linear_2_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("2.", "linear_2."));
  }

 private:
  QuantArgs quant_args_;
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
    quant_args_ = context.get_quant_args();
    std::optional<TpOptions> tp = std::nullopt;
    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp = TpOptions::col(parallel_args_.dit_tp_group_,
                          /*gather_output=*/false);
    }
    auto proj = DiTParallelLinear(dim_in,
                                  dim_out,
                                  with_bias,
                                  options_,
                                  /*sp=*/std::nullopt,
                                  tp,
                                  quant_args_);
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
    if (quant_args_.is_quantized()) {
      // Quantized checkpoint stores weights directly under the module's
      // own prefix — no additional sub-prefix wrapping.
      proj_->as<DiTParallelLinear>()->load_state_dict(state_dict);
    } else {
      proj_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("proj."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    proj_->as<DiTParallelLinear>()->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("", "proj."));
  }

 private:
  QuantArgs quant_args_;
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
    quant_args_ = context.get_quant_args();

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

    auto tp_out = TpOptions::row(parallel_args_.dit_tp_group_,
                                 /*gather_output=*/true);
    auto proj_out = DiTParallelLinear(actual_inner_dim,
                                      actual_dim_out,
                                      with_bias,
                                      options_,
                                      /*sp=*/std::nullopt,
                                      tp_out,
                                      quant_args_);
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
    act_fn_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("0.", "net.0.")));
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("2.", "net.2.")));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    act_fn_->verify_loaded_weights(prefix +
                                   quant_args_.resolve_prefix("0.", "net.0."));
    proj_out_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("2.", "net.2."));
  }

 private:
  QuantArgs quant_args_;
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
  WanPixArtAlphaTextProjectionImpl(const ModelContext& context,
                                   int64_t in_features,
                                   int64_t hidden_size,
                                   int64_t out_features = -1,
                                   const std::string& act_fn = "gelu_tanh")
      : options_(torch::dtype(torch::kFloat32)) {
    quant_args_ = context.get_quant_args();
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
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("0.", "linear_1.")));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("2.", "linear_2.")));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("0.", "linear_1."));
    linear_2_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("2.", "linear_2."));
  }

 private:
  QuantArgs quant_args_;
  torch::TensorOptions options_;
  layer::AddMatmul linear_1_{nullptr};
  torch::nn::AnyModule act_1_;
  layer::AddMatmul linear_2_{nullptr};
};
TORCH_MODULE(WanPixArtAlphaTextProjection);

class WanAttentionImpl : public torch::nn::Module {
 public:
  explicit WanAttentionImpl(
      const ModelContext& context,
      const ParallelArgs& parallel_args,
      int64_t cross_attention_dim_head = -1,
      const xllm::dit::SparseAttnConfig& sparse_attn_config = {})
      : options_(context.get_tensor_options()),
        parallel_args_(parallel_args),
        sparse_attn_config_(sparse_attn_config) {
    auto model_args = context.get_model_args();
    quant_args_ = context.get_quant_args();
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
    // Pre-build options that get reused
    auto tp_qk = TpOptions::col(parallel_args_.dit_tp_group_,
                                /*gather_output=*/false);
    auto tp_v = TpOptions::col(parallel_args_.dit_tp_group_,
                               /*gather_output=*/false);
    auto tp_out = TpOptions::row(parallel_args_.dit_tp_group_,
                                 /*gather_output=*/true);

    auto sp_before = SpOptions(heads_,
                               dim_head_,
                               dim_,
                               /*before_attention=*/true,
                               parallel_args_.dit_sp_group_);
    auto sp_after = SpOptions(heads_,
                              dim_head_,
                              dim_,
                              /*before_attention=*/false,
                              parallel_args_.dit_sp_group_);

    // Q/K: TP only (SP handled in forward() due to norm ordering)
    to_q_ = register_module("to_q",
                            DiTParallelLinear(dim_,
                                              heads_ * dim_head_,
                                              true,
                                              options_,
                                              /*sp=*/std::nullopt,
                                              tp_qk,
                                              quant_args_));
    to_k_ = register_module("to_k",
                            DiTParallelLinear(dim_,
                                              kv_inner_dim_,
                                              true,
                                              options_,
                                              /*sp=*/std::nullopt,
                                              tp_qk,
                                              quant_args_));

    // V: TP+SP for self-attention, TP only for cross-attention
    bool is_self_attn = cross_attention_dim_head <= 0;
    to_v_ = register_module(
        "to_v",
        DiTParallelLinear(dim_,
                          kv_inner_dim_,
                          true,
                          options_,
                          is_self_attn ? sp_before : std::optional<SpOptions>{},
                          tp_v,
                          quant_args_));

    // to_out: TP+SP row parallel
    to_out_ = register_module("to_out",
                              DiTParallelLinear(heads_ * dim_head_,
                                                dim_,
                                                true,
                                                options_,
                                                sp_after,
                                                tp_out,
                                                quant_args_));
    norm_q_ = register_module(
        "norm_q", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    norm_k_ = register_module(
        "norm_k", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    if (added_kv_proj_dim_ > 0) {
      auto add_k_tp = TpOptions::col(parallel_args_.dit_tp_group_,
                                     /*gather_output=*/false);
      auto add_v_tp = TpOptions::col(parallel_args_.dit_tp_group_,
                                     /*gather_output=*/false);
      auto add_k_proj = DiTParallelLinear(added_kv_proj_dim_,
                                          heads_ * dim_head_,
                                          true,
                                          options_,
                                          /*sp=*/std::nullopt,
                                          add_k_tp);
      add_k_proj_ = register_module("add_k_proj", add_k_proj);
      auto add_v_proj = DiTParallelLinear(added_kv_proj_dim_,
                                          heads_ * dim_head_,
                                          true,
                                          options_,
                                          /*sp=*/std::nullopt,
                                          add_v_tp);
      add_v_proj_ = register_module("add_v_proj", add_v_proj);
      norm_added_k_ = register_module(
          "norm_added_k", layer::RMSNorm(dim_head_ * heads_, eps_, options_));
    }
  }

  torch::Tensor at_npu_attention(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& v,
      xllm::dit::SparseAttnState& sparse_attn_state) {
    const auto q_t = q.transpose(1, 2);
    const auto k_t = k.transpose(1, 2);
    const auto v_t = v.transpose(1, 2);

#if defined(USE_NPU)
    if (sparse_attn_config_.enabled &&
        sparse_attn_state.current_step >=
            sparse_attn_config_.sparse_start_step &&
        q_t.size(2) == k_t.size(2)) {
      // Strip SP padding: latent_shape uses the unpadded seq_len,
      // but SP may have padded the sequence to be divisible by sp_size.
      auto q_use = q_t;
      auto k_use = k_t;
      auto v_use = v_t;
      int64_t pad_len = 0;
      if (sparse_attn_state.seq_len > 0 &&
          q_t.size(2) > sparse_attn_state.seq_len) {
        pad_len = q_t.size(2) - sparse_attn_state.seq_len;
        q_use = q_t.slice(2, 0, sparse_attn_state.seq_len);
        k_use = k_t.slice(2, 0, sparse_attn_state.seq_len);
        v_use = v_t.slice(2, 0, sparse_attn_state.seq_len);
      }
      auto [out_bnsd, unused] = [&]() {
        if (sparse_attn_config_.version == "sparse_attention") {
          return xllm::dit::sparse_attention::attention(
              q_use, k_use, v_use, sparse_attn_config_, sparse_attn_state);
        }
        return xllm::dit::rain_fusion::attention(
            q_use, k_use, v_use, sparse_attn_config_, sparse_attn_state);
      }();
      // SparseAttnState cache (cached_select_idx/_num_idx) managed internally
      if (pad_len > 0) {
        out_bnsd = torch::nn::functional::pad(
            out_bnsd,
            torch::nn::functional::PadFuncOptions({0, 0, 0, pad_len}));
      }
      return out_bnsd.transpose(1, 2).flatten(2, 3).to(q.dtype());
    }

    const int64_t head_num = q_t.size(1);
    const int64_t head_dim = q_t.size(-1);
    torch::Tensor out;
    // Laser attention only supports equal-length q/k (self-attention); cross
    // attention (q/k different seq len) falls back to npu_fusion_attention.
    const bool laser_enable =
        DiTConfig::get_instance().dit_laser_attention_enabled() &&
        q_t.size(2) == k_t.size(2);
    if (laser_enable) {
      out = xllm::kernel::npu::laser_attention(
                q_t, k_t, v_t, std::pow(head_dim, -0.5), head_num)
                .transpose(1, 2);
    } else {
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
      out = std::get<0>(results).transpose(1, 2);
    }
#else
    constexpr int64_t kAttentionChunkSize = 512;
    constexpr int64_t kHeadDim = 1;
    constexpr int64_t kSequenceDim = 2;
    torch::Tensor out;
    if (q_t.size(kSequenceDim) <= kAttentionChunkSize) {
      out = torch::scaled_dot_product_attention(q_t,
                                                k_t,
                                                v_t,
                                                torch::nullopt,
                                                /*dropout_p=*/0.0,
                                                /*is_causal=*/false)
                .transpose(kHeadDim, kSequenceDim);
    } else {
      std::vector<torch::Tensor> chunks;
      const int64_t num_chunks =
          (q_t.size(kSequenceDim) + kAttentionChunkSize - 1) /
          kAttentionChunkSize;
      chunks.reserve(num_chunks);
      for (int64_t start = 0; start < q_t.size(kSequenceDim);
           start += kAttentionChunkSize) {
        int64_t chunk_size =
            std::min(kAttentionChunkSize, q_t.size(kSequenceDim) - start);
        torch::Tensor q_chunk = q_t.narrow(kSequenceDim, start, chunk_size);
        chunks.emplace_back(
            torch::scaled_dot_product_attention(q_chunk,
                                                k_t,
                                                v_t,
                                                torch::nullopt,
                                                /*dropout_p=*/0.0,
                                                /*is_causal=*/false));
      }
      out = torch::cat(chunks, kSequenceDim).transpose(kHeadDim, kSequenceDim);
    }
#endif
    return out.flatten(2, 3);
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states_in,
      const torch::Tensor& encoder_hidden_states,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> rotary_emb,
      xllm::dit::SparseAttnState& sparse_attn_state) {
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

    // ── Step 1: Linear projections ──
    torch::Tensor query = to_q_->forward(hidden_states);
    torch::Tensor key = to_k_->forward(encoder_hidden_states_text);
    torch::Tensor value = to_v_->forward(encoder_hidden_states_text);

    // ── Step 2: Norm on TP-sharded Q/K ──
    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      query = dit::tp_rms_norm(query, norm_q_, parallel_args_.dit_tp_group_);
      key = dit::tp_rms_norm(key, norm_k_, parallel_args_.dit_tp_group_);
    } else {
      query = std::get<0>(norm_q_->forward(query));
      key = std::get<0>(norm_k_->forward(key));
    }

    // ── Step 3: SP all2all for Q/K (V already done in layer) ──
    int64_t batch_size = query.size(0);
    int64_t n_heads = heads_;
    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      n_heads = heads_ / ::xllm::ParallelConfig::get_instance().tp_size();
    }
    if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
      query = sp_all_to_all(query,
                            heads_,
                            dim_head_,
                            ::xllm::ParallelConfig::get_instance().tp_size(),
                            parallel_args_.dit_sp_group_);
      if (is_self_attention) {
        key = sp_all_to_all(key,
                            heads_,
                            dim_head_,
                            ::xllm::ParallelConfig::get_instance().tp_size(),
                            parallel_args_.dit_sp_group_);
      } else {
        key = sp_slice_heads(key,
                             heads_,
                             dim_head_,
                             ::xllm::ParallelConfig::get_instance().tp_size(),
                             parallel_args_.dit_sp_group_);
        value = sp_slice_heads(value,
                               heads_,
                               dim_head_,
                               ::xllm::ParallelConfig::get_instance().tp_size(),
                               parallel_args_.dit_sp_group_);
      }
      n_heads = n_heads / ::xllm::ParallelConfig::get_instance().sp_size();
    }

    // ── Step 4: Reshape → RoPE → Attention → to_out ──
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

      if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
        key_img = dit::tp_rms_norm(
            key_img, norm_added_k_, parallel_args_.dit_tp_group_);
      } else {
        key_img = std::get<0>(norm_added_k_->forward(key_img));
      }
      if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
        key_img =
            sp_slice_heads(key_img,
                           heads_,
                           dim_head_,
                           ::xllm::ParallelConfig::get_instance().tp_size(),
                           parallel_args_.dit_sp_group_);
        value_img =
            sp_slice_heads(value_img,
                           heads_,
                           dim_head_,
                           ::xllm::ParallelConfig::get_instance().tp_size(),
                           parallel_args_.dit_sp_group_);
      }

      key_img = key_img.view({batch_size, -1, n_heads, dim_head_});
      value_img = value_img.view({batch_size, -1, n_heads, dim_head_});
      hidden_states_img =
          at_npu_attention(query, key_img, value_img, sparse_attn_state);
    }
    hidden_states = at_npu_attention(query, key, value, sparse_attn_state);
    if (hidden_states_img.defined()) {
      hidden_states = hidden_states + hidden_states_img;
    }
    hidden_states = to_out_->forward(hidden_states);

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("q.", "to_q.")));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("k.", "to_k.")));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("v.", "to_v.")));
    to_out_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("o.", "to_out.0.")));

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
    to_q_->verify_loaded_weights(prefix +
                                 quant_args_.resolve_prefix("q.", "to_q."));
    to_k_->verify_loaded_weights(prefix +
                                 quant_args_.resolve_prefix("k.", "to_k."));
    to_v_->verify_loaded_weights(prefix +
                                 quant_args_.resolve_prefix("v.", "to_v."));
    to_out_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("o.", "to_out.0."));
    if (add_k_proj_) {
      add_k_proj_->verify_loaded_weights(prefix + "add_k_proj.");
      add_v_proj_->verify_loaded_weights(prefix + "add_v_proj.");
    }
  }

 private:
  QuantArgs quant_args_;
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

  // RainFusionV3 configuration (static, same for all requests)
  xllm::dit::SparseAttnConfig sparse_attn_config_;
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

    quant_args_ = context.get_quant_args();
    timesteps_proj_ = register_module(
        "timesteps_proj", WanTimesteps(time_freq_dim_, true, 0.0f, 1));
    time_embedder_ = register_module(
        "time_embedder",
        WanTimestepEmbedding(context, time_freq_dim_, dim_, -1, true));
    act_fn_ = register_module("act_fn", torch::nn::SiLU());
    time_proj_ = register_module(
        "time_proj", layer::AddMatmul(dim_, time_proj_dim_, true, options_));

    text_embedder_ =
        register_module("text_embedder",
                        WanPixArtAlphaTextProjection(
                            context, text_embed_dim_, dim_, dim_, "gelu_tanh"));

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
    time_embedder_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("time_embedding.", "time_embedder.")));
    time_proj_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("time_projection.1.", "time_proj.")));
    text_embedder_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("text_embedding.", "text_embedder.")));
    if (image_embedder_) {
      image_embedder_->load_state_dict(
          state_dict.get_dict_with_prefix("image_embedder."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    time_embedder_->verify_loaded_weights(
        prefix +
        quant_args_.resolve_prefix("time_embedding.", "time_embedder."));
    time_proj_->verify_loaded_weights(
        prefix +
        quant_args_.resolve_prefix("time_projection.1.", "time_proj."));
    text_embedder_->verify_loaded_weights(
        prefix +
        quant_args_.resolve_prefix("text_embedding.", "text_embedder."));
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

  QuantArgs quant_args_;
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

  std::tuple<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& hidden_states) {
    int64_t num_frames = hidden_states.size(2);
    int64_t height = hidden_states.size(3);
    int64_t width = hidden_states.size(4);

    if (num_frames != cached_num_frames_ || height != cached_height_ ||
        width != cached_width_) {
      auto [cos, sin] = forward(hidden_states);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
      cached_num_frames_ = num_frames;
      cached_height_ = height;
      cached_width_ = width;
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
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

  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t cached_num_frames_ = -1;
  int64_t cached_height_ = -1;
  int64_t cached_width_ = -1;

  torch::TensorOptions options_;
};
TORCH_MODULE(WanRotaryPosEmbed);

class WanTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit WanTransformerBlockImpl(
      const ModelContext& context,
      const ParallelArgs& parallel_args,
      int64_t block_idx = 0,
      const xllm::dit::SparseAttnConfig& sparse_attn_config = {})
      : options_(context.get_tensor_options()),
        parallel_args_(parallel_args),
        block_idx_(block_idx) {
    auto model_args = context.get_model_args();
    quant_args_ = context.get_quant_args();
    dim_ = model_args.head_dim() * model_args.n_heads();
    ffn_dim_ = model_args.ffn_dim();
    num_heads_ = model_args.n_heads();
    eps_ = 1e-6f;
    added_kv_proj_dim_ = model_args.added_kv_proj_dim();
    cross_attn_norm_ = model_args.cross_attn_norm();
    qk_norm_ = model_args.qk_norm();

    ada_norm1_ = register_module(
        "ada_norm1",
        layer::AdaLayerNorm(
            dim_, eps_, /*elementwise_affine=*/false, options_));
    attn1_ = register_module(
        "attn1", WanAttention(context, parallel_args, -1, sparse_attn_config));
    attn2_ = register_module(
        "attn2",
        WanAttention(
            context, parallel_args, dim_ / num_heads_, sparse_attn_config));
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
    ada_norm3_ = register_module(
        "ada_norm3",
        layer::AdaLayerNorm(
            dim_, eps_, /*elementwise_affine=*/false, options_));
    scale_shift_table_ =
        register_parameter("scale_shift_table",
                           torch::randn({1, 6, dim_}, options_) /
                               std::sqrt(static_cast<float>(dim_)));
  }

  torch::Tensor forward(
      const torch::Tensor& hidden_states_in,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& timestep_proj,
      std::optional<std::pair<torch::Tensor, torch::Tensor>> rotary_emb,
      xllm::dit::SparseAttnState& sparse_attn_state) {
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

    auto scale_msa_2d =
        scale_msa.dim() == 3 ? scale_msa.select(1, 0) : scale_msa;
    auto shift_msa_2d =
        shift_msa.dim() == 3 ? shift_msa.select(1, 0) : shift_msa;
    torch::Tensor norm_hidden_states =
        ada_norm1_->forward(hidden_states, scale_msa_2d, shift_msa_2d);
    torch::Tensor attn_output = attn1_->forward(
        norm_hidden_states, norm_hidden_states, rotary_emb, sparse_attn_state);
    hidden_states = hidden_states + attn_output * gate_msa;

    if (cross_attn_norm_) {
      norm_hidden_states = norm2_->forward(hidden_states);
    } else {
      norm_hidden_states = hidden_states;
    }

    attn_output = attn2_->forward(norm_hidden_states,
                                  encoder_hidden_states,
                                  std::nullopt,
                                  sparse_attn_state);
    hidden_states = hidden_states + attn_output;
    auto c_scale_msa_2d =
        c_scale_msa.dim() == 3 ? c_scale_msa.select(1, 0) : c_scale_msa;
    auto c_shift_msa_2d =
        c_shift_msa.dim() == 3 ? c_shift_msa.select(1, 0) : c_shift_msa;
    norm_hidden_states =
        ada_norm3_->forward(hidden_states, c_scale_msa_2d, c_shift_msa_2d);
    torch::Tensor ff_output = ff_->forward(norm_hidden_states);
    hidden_states = hidden_states + ff_output * c_gate_msa;

    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    attn1_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("self_attn.", "attn1.")));
    attn2_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("cross_attn.", "attn2.")));
    if (cross_attn_norm_ && norm2_) {
      norm2_->load_state_dict(state_dict.get_dict_with_prefix(
          quant_args_.resolve_prefix("norm3.", "norm2.")));
    }
    ff_->load_state_dict(state_dict.get_dict_with_prefix(
        quant_args_.resolve_prefix("ffn.", "ffn.")));
    weight::load_weight(
        state_dict,
        quant_args_.resolve_prefix("modulation", "scale_shift_table"),
        scale_shift_table_,
        scale_shift_table_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn1_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("self_attn.", "attn1."));
    if (cross_attn_norm_) {
      norm2_->verify_loaded_weights(
          prefix + quant_args_.resolve_prefix("norm3.", "norm2."));
    }
    attn2_->verify_loaded_weights(
        prefix + quant_args_.resolve_prefix("cross_attn.", "attn2."));
    ff_->verify_loaded_weights(prefix +
                               quant_args_.resolve_prefix("ffn.", "ffn."));
    auto scale_key =
        quant_args_.resolve_prefix("modulation", "scale_shift_table");
    CHECK(scale_shift_table_loaded_)
        << scale_key << " is not loaded for " << prefix + scale_key;
  }

#if defined(USE_NPU)
  void build_weight_loader() { weight_loader_.build_from_module(*this); }

  dit::BlockWeightLoader& weight_loader() { return weight_loader_; }

  void set_rolling_buffer(std::shared_ptr<layer::RollingWeightBuffer> buf,
                          int32_t slot_index) {
    weight_loader_.set_rolling_buffer(std::move(buf), slot_index);
  }
#endif

 private:
  int64_t dim_;
  int64_t ffn_dim_;
  int64_t num_heads_;
  float eps_;
  int64_t added_kv_proj_dim_;
  bool cross_attn_norm_;
  int64_t block_idx_ = 0;
  std::string qk_norm_;

  WanAttention attn1_{nullptr};
  WanAttention attn2_{nullptr};
  WanFeedForward ff_{nullptr};
  layer::AdaLayerNorm ada_norm1_{nullptr};  // self-attn pre-norm (fused)
  FP32LayerNorm norm2_{nullptr};  // cross-attn pre-norm (bf16 LayerNorm)
  layer::AdaLayerNorm ada_norm3_{nullptr};  // FFN pre-norm (fused)
  torch::Tensor scale_shift_table_;
  bool scale_shift_table_loaded_{false};

  QuantArgs quant_args_;
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
#if defined(USE_NPU)
  dit::BlockWeightLoader weight_loader_;
#endif
};
TORCH_MODULE(WanTransformerBlock);

class WanTransformer3DModelImpl : public torch::nn::Module {
 public:
  explicit WanTransformer3DModelImpl(
      const ModelContext& context,
      const xllm::dit::SparseAttnConfig& sparse_attn_config = {})
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    sp_group_ = parallel_args.dit_sp_group_;
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
    quant_args_ = context.get_quant_args();
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
      auto block = WanTransformerBlock(
          context, parallel_args, static_cast<int64_t>(i), sparse_attn_config);
      blocks_->push_back(block);
      transformer_layers_.push_back(block);
    }

    ada_norm_out_ = register_module(
        "ada_norm_out",
        layer::AdaLayerNorm(
            inner_dim_, 1e-6, /*elementwise_affine=*/false, options_));
    int64_t patch_prod = patch_size_[0] * patch_size_[1] * patch_size_[2];
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(
            inner_dim_, out_channels_ * patch_prod, true, options_));
    scale_shift_table_ =
        register_parameter("scale_shift_table",
                           torch::randn({1, 2, inner_dim_}, options_) /
                               std::sqrt(static_cast<float>(inner_dim_)));

    if (LoadConfig::get_instance().enable_rolling_load() &&
        ParallelConfig::get_instance().tp_size() == 1) {
      // Free NPU memory early — weights will be streamed via rolling buffer.
      this->to(torch::kCPU);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_in,
                        const torch::Tensor& timestep,
                        const torch::Tensor& encoder_hidden_states,
                        const torch::Tensor& encoder_hidden_states_image,
                        xllm::dit::SparseAttnState& sparse_attn_state,
                        std::function<void(int32_t)> before_layer_cb = nullptr,
                        std::function<void(int32_t)> after_layer_cb = nullptr) {
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

    std::vector<int64_t> latent_shape = {
        post_patch_num_frames, post_patch_height, post_patch_width};

    torch::Tensor hidden_states = hidden_states_in;

    auto [freqs_cos, freqs_sin] = rope_->forward_cache(hidden_states);

    auto rotary_emb = std::make_pair(freqs_cos, freqs_sin);

    hidden_states = patch_embedding_->forward(
        hidden_states.to(patch_embedding_->weight.dtype()));
    hidden_states = hidden_states.flatten(2).transpose(1, 2);

    int64_t seq_len = hidden_states.size(1);
    int64_t pad_seq_len = sp_pad_sequence(
        hidden_states, freqs_cos, freqs_sin, rotary_emb, sp_group_);

    sparse_attn_state.latent_shape = latent_shape;
    sparse_attn_state.seq_len = seq_len;

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

    if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
      hidden_states =
          dit::sp_split_sequence(hidden_states, /*dim=*/1, sp_group_);
      if (timestep_proj.dim() == 4) {
        timestep_proj =
            dit::sp_split_sequence(timestep_proj, /*dim=*/1, sp_group_);
      }
    }

    for (int64_t i = 0; i < transformer_layers_.size(); ++i) {
      if (before_layer_cb) {
        before_layer_cb(static_cast<int32_t>(i));
      }
      hidden_states =
          transformer_layers_[i]->forward(hidden_states,
                                          encoder_hidden_states_embedded,
                                          timestep_proj,
                                          rotary_emb,
                                          sparse_attn_state);
      if (after_layer_cb) {
        after_layer_cb(static_cast<int32_t>(i));
      }
    }

    if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
      hidden_states =
          dit::sp_gather_sequence(hidden_states, /*dim=*/1, sp_group_);
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

    auto hidden_states_dtype = hidden_states.dtype();

    // Drop the redundant sequence dim so the fused kernel uses the fast 2D
    // [B,H] path instead of the token-wise fold.
    auto scale_2d = scale.dim() == 3 ? scale.select(1, 0) : scale;
    auto shift_2d = shift.dim() == 3 ? shift.select(1, 0) : shift;
    hidden_states = ada_norm_out_->forward(hidden_states,
                                           scale_2d.to(hidden_states_dtype),
                                           shift_2d.to(hidden_states_dtype));

    if (::xllm::ParallelConfig::get_instance().sp_size() > 1 &&
        seq_len != pad_seq_len) {
      hidden_states = hidden_states.slice(1, 0, seq_len);
    }
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

    if (quant_args_.is_quantized()) {
      // Quantized checkpoint stores condition_embedder weights directly
      // under the top-level prefix — no "condition_embedder." wrapper.
      condition_embedder_->load_state_dict(state_dict);
      proj_out_->load_state_dict(state_dict.get_dict_with_prefix("head.head."));
    } else {
      condition_embedder_->load_state_dict(
          state_dict.get_dict_with_prefix("condition_embedder."));
      proj_out_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
    }
    for (int64_t i = 0; i < transformer_layers_.size(); ++i) {
      transformer_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("blocks." + std::to_string(i) + "."));
    }
    weight::load_weight(
        state_dict,
        quant_args_.resolve_prefix("head.modulation", "scale_shift_table"),
        scale_shift_table_,
        scale_shift_table_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(pad_embedding_weight_loaded_) << "patch_embedding is not loaded for"
                                        << prefix << "pad_embedding.weight";
    CHECK(pad_embedding_bias_loaded_) << "patch_embedding is not loaded for"
                                      << prefix << "pad_embedding.bias";

    if (quant_args_.is_quantized()) {
      condition_embedder_->verify_loaded_weights(prefix);
      proj_out_->verify_loaded_weights(prefix + "head.head.");
    } else {
      condition_embedder_->verify_loaded_weights(prefix +
                                                 "condition_embedder.");
      proj_out_->verify_loaded_weights(prefix + "proj_out.");
    }
    for (size_t i = 0; i < transformer_layers_.size(); ++i) {
      transformer_layers_[i]->verify_loaded_weights(prefix + "blocks." +
                                                    std::to_string(i) + ".");
    }
    auto scale_key =
        quant_args_.resolve_prefix("head.modulation", "scale_shift_table");
    CHECK(scale_shift_table_loaded_)
        << scale_key << " is not loaded for " << prefix + scale_key;
  }

  int64_t in_channels() const { return in_channels_; }
  const std::vector<int64_t>& patch_size() const { return patch_size_; }
  bool guidance_embeds() const { return false; }

  void load_model(std::unique_ptr<DiTFolderLoader> loader,
                  bool rolling = false) {
    auto freqs_cos_fp32 = rope_->get_freqs_cos().clone();
    auto freqs_sin_fp32 = rope_->get_freqs_sin().clone();
    this->to(rolling ? torch::kCPU : options_.device(), torch::kBFloat16);
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_state_dict(*state_dict);
    }
    verify_loaded_weights("");

    // Restore fp32 RoPE frequencies that were cloned before dtype conversion.
    rope_->set_freqs_cos(freqs_cos_fp32);
    rope_->set_freqs_sin(freqs_sin_fp32);

#if defined(USE_NPU)
    if (rolling) {
      for (auto& block : transformer_layers_) {
        block->build_weight_loader();
      }

      auto device = options_.device();
      patch_embedding_->to(device);
      rope_->to(device);
      rope_->set_freqs_cos(freqs_cos_fp32.to(device));
      rope_->set_freqs_sin(freqs_sin_fp32.to(device));
      condition_embedder_->to(device);
      ada_norm_out_->to(device);
      proj_out_->to(device);
      scale_shift_table_.set_data(scale_shift_table_.to(device));

      LOG(INFO) << "WanTransformer3DModel ready for rolling load, "
                << transformer_layers_.size() << " blocks prepared";
    }
#endif
  }

#if defined(USE_NPU)
  std::vector<dit::BlockWeightLoader*> get_block_weight_loaders() {
    std::vector<dit::BlockWeightLoader*> loaders;
    loaders.reserve(transformer_layers_.size());
    for (auto& block : transformer_layers_) {
      loaders.push_back(&block->weight_loader());
    }
    return loaders;
  }
#endif

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
  QuantArgs quant_args_;
  ProcessGroup* sp_group_ = nullptr;
  torch::nn::Conv3d patch_embedding_{nullptr};
  WanTimeTextImageEmbedding condition_embedder_{nullptr};
  WanRotaryPosEmbed rope_{nullptr};
  torch::nn::ModuleList blocks_;
  std::vector<WanTransformerBlock> transformer_layers_;
  layer::AdaLayerNorm ada_norm_out_{nullptr};  // final norm (fused)
  layer::AddMatmul proj_out_{nullptr};
  torch::Tensor scale_shift_table_;
  bool scale_shift_table_loaded_{false};
  bool pad_embedding_weight_loaded_{false};
  bool pad_embedding_bias_loaded_{false};
  torch::TensorOptions options_;
};
TORCH_MODULE(WanTransformer3DModel);

REGISTER_MODEL_ARGS(WanTransformer3DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(cross_attn_norm, "cross_attn_norm", true);
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

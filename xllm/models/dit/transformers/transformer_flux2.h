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

// sequence parallel pad manager
#include "core/framework/config/dit_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/transformers/transformer_flux.h"
#include "models/dit/utils/dit_parallel_linear.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {
using dit::DiTParallelLinear;
using dit::TpOptions;

class Flux2SwiGLUImpl final : public torch::nn::Module {
 public:
  Flux2SwiGLUImpl() { gate_fn_ = torch::nn::SiLU(); }

  torch::Tensor forward(torch::Tensor x) {
    auto chunks = torch::chunk(x, 2, /*dim=*/-1);
    torch::Tensor x1 = chunks[0];
    torch::Tensor x2 = chunks[1];

    torch::Tensor x_out = gate_fn_(x1) * x2;
    return x_out;
  }

 private:
  torch::nn::SiLU gate_fn_;
};
TORCH_MODULE(Flux2SwiGLU);

class Flux2FeedForwardImpl final : public torch::nn::Module {
 public:
  explicit Flux2FeedForwardImpl(const ModelContext& context,
                                const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    float eps = model_args.mlp_ratio();
    int64_t num_attention_heads = model_args.n_heads();
    int64_t attention_head_dim = model_args.head_dim();
    int64_t inner_dim = num_attention_heads * attention_head_dim;
    std::optional<TpOptions> tp_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);

      if (!parallel_args_.dit_tp_group_ || !tp_options.has_value() ||
          !tp_options->process_group) {
      }
    }

    auto linear_in = DiTParallelLinear(
        inner_dim, inner_dim * 6, false, options_, std::nullopt, tp_options);
    linear_in_ = register_module("linear_in", linear_in);
    act_fn_ = register_module("act_fn", Flux2SwiGLU());

    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto linear_out = DiTParallelLinear(inner_dim * 3,
                                        inner_dim,
                                        false,
                                        options_,
                                        std::nullopt,
                                        tp_out_options);
    linear_out_ = register_module("linear_out", linear_out);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const bool is_save) {
    auto out = linear_in_->forward(hidden_states);
    out = act_fn_->forward(out);
    out = linear_out_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    int64_t tp_size = ::xllm::ParallelConfig::get_instance().tp_size() > 1
                          ? ::xllm::ParallelConfig::get_instance().tp_size()
                          : 1;
    if (tp_size > 1) {
      int64_t tp_rank = parallel_args_.dit_tp_group_->rank();

      // Manually reorder linear_in weight: split gate/up separately,
      // take rank-th slice of each, then recombine so each rank has
      // matching gate-up pairs for SwiGLU.
      auto linear_in_weight = state_dict.get_tensor("linear_in.weight");
      if (linear_in_weight.defined()) {
        // Full weight shape: [inner_dim * 6, inner_dim]
        // Rows [0, inner_dim*3) = gate projection
        // Rows [inner_dim*3, inner_dim*6) = up projection
        int64_t gate_up_dim = linear_in_weight.size(0) / 2;
        auto gate_weight = linear_in_weight.slice(0, 0, gate_up_dim);
        auto up_weight = linear_in_weight.slice(0, gate_up_dim);

        auto gate_chunks = gate_weight.chunk(tp_size, 0);
        auto up_chunks = up_weight.chunk(tp_size, 0);

        auto gate_rank = gate_chunks[tp_rank];
        auto up_rank = up_chunks[tp_rank];

        auto rank_weight = torch::cat({gate_rank, up_rank}, 0).to(options_);

        {
          torch::NoGradGuard no_grad;
          for (auto& p : linear_in_->named_parameters()) {
            if (p.key() == "weight") {
              p.value().copy_(rank_weight);
              break;
            }
          }
        }
      }
    } else {
      linear_in_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("linear_in."));
    }

    // linear_out uses default load_state_dict which handles
    // row split (dim 1) correctly for each rank.
    linear_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("linear_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    int64_t tp_size = ::xllm::ParallelConfig::get_instance().tp_size() > 1
                          ? ::xllm::ParallelConfig::get_instance().tp_size()
                          : 1;
    if (tp_size <= 1) {
      linear_in_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                 "linear_in.");
    }
    linear_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "linear_out.");
  }

 private:
  int64_t dim_;
  int64_t inner_dim_;
  float mult_;
  DiTParallelLinear linear_in_{nullptr};
  Flux2SwiGLU act_fn_{nullptr};
  DiTParallelLinear linear_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2FeedForward);

class Flux2AttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2AttentionImpl(const ModelContext& context,
                              const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    added_kv_proj_dim_ = query_dim_;

    // Determine parallelism strategy
    std::optional<TpOptions> tp_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_q = DiTParallelLinear(
        query_dim_, out_dim_, false, options_, std::nullopt, tp_options);
    to_q_ = register_module("to_q", to_q);

    auto to_k = DiTParallelLinear(
        query_dim_, out_dim_, false, options_, std::nullopt, tp_options);
    to_k_ = register_module("to_k", to_k);

    auto to_v = DiTParallelLinear(
        query_dim_, out_dim_, false, options_, std::nullopt, tp_options);
    to_v_ = register_module("to_v", to_v);

    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_out = DiTParallelLinear(
        out_dim_, query_dim_, false, options_, std::nullopt, tp_out_options);
    to_out_ = register_module("to_out", to_out);

    // ── Added KV projections (cross-attention for text stream) ──
    if (added_kv_proj_dim_ > 0) {
      std::optional<TpOptions> tp_add_out_options = std::nullopt;

      if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
        tp_add_out_options = TpOptions(
            /*column_parallel=*/false,
            /*gather_output=*/false,
            /*need_scatter=*/false,
            /*process_group=*/parallel_args_.dit_tp_group_);
      }

      auto to_add_out = DiTParallelLinear(out_dim_,
                                          added_kv_proj_dim_,
                                          false,
                                          options_,
                                          std::nullopt,
                                          tp_add_out_options);
      to_add_out_ = register_module("to_add_out", to_add_out);
      norm_added_q_ = register_module(
          "norm_added_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
      norm_added_k_ = register_module(
          "norm_added_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

      auto to_add_q = DiTParallelLinear(added_kv_proj_dim_,
                                        out_dim_,
                                        false,
                                        options_,
                                        std::nullopt,
                                        tp_options);
      to_add_q_ = register_module("to_add_q", to_add_q);

      auto to_add_k = DiTParallelLinear(added_kv_proj_dim_,
                                        out_dim_,
                                        false,
                                        options_,
                                        std::nullopt,
                                        tp_options);
      to_add_k_ = register_module("to_add_k", to_add_k);

      auto to_add_v = DiTParallelLinear(added_kv_proj_dim_,
                                        out_dim_,
                                        false,
                                        options_,
                                        std::nullopt,
                                        tp_options);
      to_add_v_ = register_module("to_add_v", to_add_v);
    }
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

    hidden_states_reshaped = hidden_states_reshaped.squeeze(1).squeeze(1);
    encoder_hidden_states_reshaped =
        encoder_hidden_states_reshaped.squeeze(1).squeeze(1);

    auto query = to_q_->forward(hidden_states_reshaped);
    auto key = to_k_->forward(hidden_states_reshaped);
    auto value = to_v_->forward(hidden_states_reshaped);
    int64_t inner_dim = key.size(-1);
    int64_t attn_heads = inner_dim / head_dim_;

    int64_t head_dim = head_dim_;
    query = query.view({batch_size, -1, attn_heads, head_dim});
    key = key.view({batch_size, -1, attn_heads, head_dim});
    value = value.view({batch_size, -1, attn_heads, head_dim});
    if (norm_q_) {
      query = std::get<0>(norm_q_(query));
    }
    if (norm_k_) {
      key = std::get<0>(norm_k_(key));
    }
    auto encoder_hidden_states_query_proj =
        to_add_q_->forward(encoder_hidden_states_reshaped);
    auto encoder_hidden_states_key_proj =
        to_add_k_->forward(encoder_hidden_states_reshaped);
    auto encoder_hidden_states_value_proj =
        to_add_v_->forward(encoder_hidden_states_reshaped);
    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    if (norm_added_q_) {
      encoder_hidden_states_query_proj =
          std::get<0>(norm_added_q_(encoder_hidden_states_query_proj));
    }

    if (norm_added_k_) {
      encoder_hidden_states_key_proj =
          std::get<0>(norm_added_k_(encoder_hidden_states_key_proj));
    }

    // Concatenate for joint attention: order [text, image]
    auto query1 = torch::cat({encoder_hidden_states_query_proj, query}, 1);
    auto key1 = torch::cat({encoder_hidden_states_key_proj, key}, 1);
    auto value1 = torch::cat({encoder_hidden_states_value_proj, value}, 1);
    if (image_rotary_emb.defined()) {
      query1 = apply_rotary_emb(query1, image_rotary_emb, false);
      key1 = apply_rotary_emb(key1, image_rotary_emb, false);
    }

    // After SP all2all (before_attention=true), attn_heads already equals
    // heads/sp_size. No further division needed.
    int64_t local_heads = attn_heads;

#if defined(USE_NPU)
    int64_t head_num_ = query1.size(2);
    int64_t head_dim_ = query1.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query1,
                                                         key1,
                                                         value1,
                                                         local_heads,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);

    attn_output = attn_output.reshape({batch_size, -1, local_heads * head_dim});

#elif defined(USE_CUDA)
    query1 = query1.transpose(1, 2);
    key1 = key1.transpose(1, 2);
    value1 = value1.transpose(1, 2);
    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        query1, key1, value1, torch::nullopt, 0.0, false);
    attn_output = attn_output.transpose(1, 2).reshape(
        {batch_size, -1, local_heads * head_dim});
#else
    NOT_IMPLEMENTED();
#endif

    attn_output = attn_output.to(query.dtype());
    int64_t encoder_length = encoder_hidden_states_query_proj.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);

    hidden_output = to_out_->forward(hidden_output);
    encoder_output = to_add_out_->forward(encoder_output);

    return std::make_tuple(hidden_output, encoder_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_q."));
    to_k_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_k."));
    to_v_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_v."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));

    to_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out.0."));

    if (added_kv_proj_dim_ > 0) {
      norm_added_q_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_q."));
      norm_added_k_->load_state_dict(
          state_dict.get_dict_with_prefix("norm_added_k."));
      to_add_q_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_q_proj."));
      to_add_k_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_k_proj."));
      to_add_v_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("add_v_proj."));
      to_add_out_->as<DiTParallelLinear>()->load_state_dict(
          state_dict.get_dict_with_prefix("to_add_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_q_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_q.");
    to_k_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_k.");
    to_v_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_v.");

    to_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                            "to_out.0.");
    if (added_kv_proj_dim_ > 0) {
      to_add_q_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_q_proj.");
      to_add_k_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_k_proj.");
      to_add_v_->as<DiTParallelLinear>()->verify_loaded_weights(prefix +
                                                                "add_v_proj.");
      to_add_out_->as<DiTParallelLinear>()->verify_loaded_weights(
          prefix + "to_add_out.");
    }
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  int64_t added_kv_proj_dim_;
  DiTParallelLinear to_q_{nullptr};
  DiTParallelLinear to_k_{nullptr};
  DiTParallelLinear to_v_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  DiTParallelLinear to_out_{nullptr};
  layer::RMSNorm norm_added_q_{nullptr};
  layer::RMSNorm norm_added_k_{nullptr};
  DiTParallelLinear to_add_out_{nullptr};
  DiTParallelLinear to_add_q_{nullptr};
  DiTParallelLinear to_add_k_{nullptr};
  DiTParallelLinear to_add_v_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2Attention);

class Flux2ModulationImpl : public torch::nn::Module {
 public:
  explicit Flux2ModulationImpl(const ModelContext& context,
                               int64_t dim,
                               int64_t mod_param_sets,
                               bool bias = false)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();

    linear_ = register_module(
        "linear",
        layer::AddMatmul(dim, dim * 3 * mod_param_sets, bias, options_));
    act_fn_ = torch::nn::SiLU();
  }

  torch::Tensor forward(const torch::Tensor& temb) {
    auto mod = act_fn_->forward(temb);
    mod = linear_->forward(mod);
    return mod;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

  static std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
  split(const torch::Tensor& mod, int64_t mod_param_sets) {
    torch::Tensor mod_reshaped;
    if (mod.dim() == 2) {
      mod_reshaped = mod.unsqueeze(1);
    } else {
      mod_reshaped = mod;
    }

    auto mod_params = torch::chunk(mod_reshaped, 3 * mod_param_sets, -1);

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    for (int64_t i = 0; i < mod_param_sets; ++i) {
      int64_t start_idx = 3 * i;
      auto param_tuple = std::make_tuple(mod_params[start_idx],
                                         mod_params[start_idx + 1],
                                         mod_params[start_idx + 2]);
      result.push_back(param_tuple);
    }

    return result;
  }

 private:
  layer::AddMatmul linear_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2Modulation);

class Flux2TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  Flux2TimestepEmbeddingImpl(int64_t in_channels,
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
TORCH_MODULE(Flux2TimestepEmbedding);

class Flux2TimestepsImpl : public torch::nn::Module {
 public:
  explicit Flux2TimestepsImpl(int64_t num_channels,
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
                                       int embedding_dim,
                                       bool flip_sin_to_cos = false,
                                       float downscale_freq_shift = 1.0f,
                                       float scale = 1.0f,
                                       int max_period = 10000) {
    int half_dim = embedding_dim / 2;
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
TORCH_MODULE(Flux2Timesteps);

class Flux2TimestepGuidanceEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit Flux2TimestepGuidanceEmbeddingsImpl(const ModelContext& context,
                                               int64_t embedding_dim,
                                               bool bias = false,
                                               bool guidance_embeds = true)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    in_channels_ = model_args.timestep_guidance_channels();
    embedding_dim_ = embedding_dim;
    guidance_embeds_ = guidance_embeds;

    time_proj_ =
        register_module("time_proj", Flux2Timesteps(in_channels_, true, 0.0));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        Flux2TimestepEmbedding(in_channels_, embedding_dim_, -1, bias));
    if (guidance_embeds_) {
      guidance_embedder_ = register_module(
          "guidance_embedder",
          Flux2TimestepEmbedding(in_channels_, embedding_dim_, -1, bias));
    }
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj.to(timestep.dtype()));

    if (guidance_embeds_ && guidance.defined()) {
      auto guidance_proj = time_proj_->forward(guidance);
      auto guidance_emb =
          guidance_embedder_->forward(guidance_proj.to(guidance.dtype()));
      return timesteps_emb + guidance_emb;
    } else {
      return timesteps_emb;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    if (guidance_embeds_) {
      guidance_embedder_->load_state_dict(
          state_dict.get_dict_with_prefix("guidance_embedder."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    if (guidance_embeds_) {
      guidance_embedder_->verify_loaded_weights(prefix + "guidance_embedder.");
    }
  }

 private:
  int64_t in_channels_;
  int64_t embedding_dim_;
  bool guidance_embeds_;
  Flux2Timesteps time_proj_{nullptr};
  Flux2TimestepEmbedding timestep_embedder_{nullptr};
  Flux2TimestepEmbedding guidance_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(Flux2TimestepGuidanceEmbeddings);

class Flux2TransformerBlockImpl : public torch::nn::Module {
 public:
  explicit Flux2TransformerBlockImpl(const ModelContext& context,
                                     int64_t dim,
                                     const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    double eps = model_args.eps();

    norm1_ = register_module(
        "norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    norm1_context_ = register_module(
        "norm1_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    attn_ = register_module("attn", Flux2Attention(context, parallel_args));

    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_ = register_module("ff", Flux2FeedForward(context, parallel_args));

    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_context_ =
        register_module("ff_context", Flux2FeedForward(context, parallel_args));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb_img,
      const torch::Tensor& temb_txt,
      const torch::Tensor& image_rotary_emb) {
    auto img_mod_params = Flux2ModulationImpl::split(temb_img, 2);
    auto txt_mod_params = Flux2ModulationImpl::split(temb_txt, 2);

    auto [shift_msa_img, scale_msa_img, gate_msa_img] = img_mod_params[0];
    auto [shift_mlp_img, scale_mlp_img, gate_mlp_img] = img_mod_params[1];

    auto [shift_msa_txt, scale_msa_txt, gate_msa_txt] = txt_mod_params[0];
    auto [shift_mlp_txt, scale_mlp_txt, gate_mlp_txt] = txt_mod_params[1];

    auto norm_hidden_states = norm1_->forward(hidden_states);
    norm_hidden_states =
        (1 + scale_msa_img) * norm_hidden_states + shift_msa_img;

    auto norm_encoder_hidden_states =
        norm1_context_->forward(encoder_hidden_states);
    norm_encoder_hidden_states =
        (1 + scale_msa_txt) * norm_encoder_hidden_states + shift_msa_txt;

    auto [attn_output, context_attn_output] = attn_->forward(
        norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb);

    attn_output = gate_msa_img * attn_output;
    torch::Tensor new_hidden_states = hidden_states + attn_output;

    auto norm_hs = norm2_->forward(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp_img) + shift_mlp_img;

    auto ff_output = ff_->forward(norm_hs, /*is_save*/ true);

    new_hidden_states = new_hidden_states + gate_mlp_img * ff_output;

    context_attn_output = gate_msa_txt * context_attn_output;
    torch::Tensor new_encoder_hidden_states =
        encoder_hidden_states + context_attn_output;

    auto norm_enc_hs = norm2_context_->forward(new_encoder_hidden_states);
    norm_enc_hs = norm_enc_hs * (1 + scale_mlp_txt) + shift_mlp_txt;

    auto ff_context_out = ff_context_->forward(norm_enc_hs, /*is_save*/ false);
    new_encoder_hidden_states =
        new_encoder_hidden_states + gate_mlp_txt * ff_context_out;

    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
      new_encoder_hidden_states =
          torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
    }

    return std::make_tuple(new_encoder_hidden_states, new_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    ff_context_->load_state_dict(
        state_dict.get_dict_with_prefix("ff_context."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    ff_->verify_loaded_weights(prefix + "ff.");
    ff_context_->verify_loaded_weights(prefix + "ff_context.");
  }

 private:
  torch::nn::LayerNorm norm1_{nullptr};
  torch::nn::LayerNorm norm1_context_{nullptr};
  Flux2Attention attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  Flux2FeedForward ff_{nullptr};
  torch::nn::LayerNorm norm2_context_{nullptr};
  Flux2FeedForward ff_context_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2TransformerBlock);

class Flux2ParallelSelfAttentionImpl : public torch::nn::Module {
 public:
  explicit Flux2ParallelSelfAttentionImpl(const ModelContext& context,
                                          const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    query_dim_ = heads_ * head_dim_;
    out_dim_ = query_dim_;
    mlp_ratio_ = model_args.mlp_ratio();
    mlp_hidden_dim_ = static_cast<int64_t>(query_dim_ * mlp_ratio_);
    mlp_mult_factor_ = 2;

    // ── Determine parallelism strategy for QKV ──
    std::optional<TpOptions> tp_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_options = TpOptions(
          /*column_parallel=*/true,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    // Individual Q, K, V projections (instead of fused)
    auto to_q = DiTParallelLinear(
        query_dim_, query_dim_, false, options_, std::nullopt, tp_options);
    to_q_ = register_module("to_q", to_q);

    auto to_k = DiTParallelLinear(
        query_dim_, query_dim_, false, options_, std::nullopt, tp_options);
    to_k_ = register_module("to_k", to_k);

    auto to_v = DiTParallelLinear(
        query_dim_, query_dim_, false, options_, std::nullopt, tp_options);
    to_v_ = register_module("to_v", to_v);

    // ── MLP projection (not SP-aware, local computation) ──
    std::optional<TpOptions> mlp_tp_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      mlp_tp_options = TpOptions(
          /*column_parallel=*/true,
          /*gather_output=*/false,
          /*need_scatter=*/false,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    int64_t mlp_proj_dim = mlp_hidden_dim_ * mlp_mult_factor_;
    auto mlp_proj = DiTParallelLinear(query_dim_,
                                      mlp_proj_dim,
                                      false,
                                      options_,
                                      std::nullopt,
                                      mlp_tp_options);
    mlp_proj_ = register_module("mlp_proj", mlp_proj);

    mlp_act_fn_ = register_module("mlp_act_fn", Flux2SwiGLU());
    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim_, 1e-6f, options_));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim_, 1e-6f, options_));

    // ── Output projection ──
    // In SP mode, we handle the all2all manually in forward() before this
    // layer, so to_out_ is always Default or TP (not SP).
    int64_t out_input_dim = query_dim_ + mlp_hidden_dim_;
    std::optional<TpOptions> tp_out_options = std::nullopt;

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      tp_out_options = TpOptions(
          /*column_parallel=*/false,
          /*gather_output=*/true,
          /*need_scatter=*/true,
          /*process_group=*/parallel_args_.dit_tp_group_);
    }

    auto to_out = DiTParallelLinear(
        out_input_dim, out_dim_, false, options_, std::nullopt, tp_out_options);
    to_out_ = register_module("to_out", to_out);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& image_rotary_emb) {
    int64_t batch_size = hidden_states.size(0);

    // ── Separate Q/K/V and MLP projections ──
    auto q = to_q_->forward(hidden_states);
    auto k = to_k_->forward(hidden_states);
    auto v = to_v_->forward(hidden_states);
    auto mlp_output = mlp_proj_->forward(hidden_states);

    int64_t attn_heads = q.size(-1) / head_dim_;
    int64_t head_dim = head_dim_;

    q = q.view({batch_size, -1, attn_heads, head_dim});
    k = k.view({batch_size, -1, attn_heads, head_dim});
    v = v.view({batch_size, -1, attn_heads, head_dim});

    if (norm_q_) {
      q = std::get<0>(norm_q_->forward(q));
    }
    if (norm_k_) {
      k = std::get<0>(norm_k_->forward(k));
    }

    if (image_rotary_emb.defined()) {
      q = apply_rotary_emb(q, image_rotary_emb, false);
      k = apply_rotary_emb(k, image_rotary_emb, false);
    }

    // After SP all2all (before_attention=true), attn_heads already equals
    // heads/sp_size. No further division needed.
    int64_t local_heads = attn_heads;

#if defined(USE_NPU)
    int64_t head_num_ = q.size(2);
    int64_t head_dim_ = q.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(q,
                                                         k,
                                                         v,
                                                         local_heads,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);

    attn_output = attn_output.reshape({batch_size, -1, local_heads * head_dim});

#elif defined(USE_CUDA)
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    torch::Tensor attn_output = torch::scaled_dot_product_attention(
        q, k, v, torch::nullopt, 0.0, false);
    attn_output = attn_output.transpose(1, 2).reshape(
        {batch_size, -1, local_heads * head_dim});
#else
    NOT_IMPLEMENTED();
#endif

    attn_output = attn_output.to(q.dtype());

    if (::xllm::ParallelConfig::get_instance().tp_size() > 1) {
      mlp_output = mlp_output.contiguous();
      mlp_output =
          parallel_state::gather(mlp_output, parallel_args_.dit_tp_group_, -1);

      attn_output = attn_output.contiguous();
      attn_output =
          parallel_state::gather(attn_output, parallel_args_.dit_tp_group_, -1);
    }

    mlp_output = mlp_act_fn_(mlp_output);
    auto output =
        torch::cat(std::vector<torch::Tensor>{attn_output, mlp_output}, -1);

    output = to_out_->forward(output);

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    // The original checkpoint has a fused "to_qkv_mlp_proj.weight" tensor.
    // We split it into separate Q/K/V/MLP weights and load individually.
    auto fused_weight = state_dict.get_tensor("to_qkv_mlp_proj.weight");
    if (fused_weight.defined()) {
      int64_t tp_size = ::xllm::ParallelConfig::get_instance().tp_size() > 1
                            ? ::xllm::ParallelConfig::get_instance().tp_size()
                            : 1;
      // Original fused layout: rows = [Q, K, V, MLP_gate, MLP_up]
      // each of Q/K/V has query_dim_ rows, MLP has mlp_hidden_dim_*2 rows
      auto q_weight = fused_weight.slice(0, 0, query_dim_);
      auto k_weight = fused_weight.slice(0, query_dim_, query_dim_ * 2);
      auto v_weight = fused_weight.slice(0, query_dim_ * 2, query_dim_ * 3);
      auto mlp_weight = fused_weight.slice(
          0,
          query_dim_ * 3,
          query_dim_ * 3 + mlp_hidden_dim_ * mlp_mult_factor_);

      if (tp_size > 1) {
        // TP or Combined TP+SP mode: shard each weight along dim 0 by rank.
        // SP doesn't shard weights, so combined mode uses the same TP sharding.
        int64_t tp_rank = parallel_args_.dit_tp_group_->rank();
        auto q_rank = q_weight.chunk(tp_size, 0)[tp_rank];
        auto k_rank = k_weight.chunk(tp_size, 0)[tp_rank];
        auto v_rank = v_weight.chunk(tp_size, 0)[tp_rank];
        auto mlp_rank = mlp_weight.chunk(tp_size, 0)[tp_rank];

        // Directly copy already-sharded weights to parameters.
        // Same pattern as transformer_flux2.h: bypass
        // DiTParallelLinear::load_state_dict which would double-shard.
        torch::NoGradGuard no_grad;
        for (auto& p : to_q_->named_parameters()) {
          if (p.key() == "weight") {
            p.value().copy_(q_rank.to(options_));
            break;
          }
        }
        for (auto& p : to_k_->named_parameters()) {
          if (p.key() == "weight") {
            p.value().copy_(k_rank.to(options_));
            break;
          }
        }
        for (auto& p : to_v_->named_parameters()) {
          if (p.key() == "weight") {
            p.value().copy_(v_rank.to(options_));
            break;
          }
        }
        for (auto& p : mlp_proj_->named_parameters()) {
          if (p.key() == "weight") {
            p.value().copy_(mlp_rank.to(options_));
            break;
          }
        }
      } else {
        // Default mode: full weights
        {
          std::unordered_map<std::string, torch::Tensor> temp_dict;
          temp_dict["weight"] = q_weight;
          StateDict temp_state_dict(temp_dict);
          to_q_->as<DiTParallelLinear>()->load_state_dict(temp_state_dict);
        }
        {
          std::unordered_map<std::string, torch::Tensor> temp_dict;
          temp_dict["weight"] = k_weight;
          StateDict temp_state_dict(temp_dict);
          to_k_->as<DiTParallelLinear>()->load_state_dict(temp_state_dict);
        }
        {
          std::unordered_map<std::string, torch::Tensor> temp_dict;
          temp_dict["weight"] = v_weight;
          StateDict temp_state_dict(temp_dict);
          to_v_->as<DiTParallelLinear>()->load_state_dict(temp_state_dict);
        }
        {
          std::unordered_map<std::string, torch::Tensor> temp_dict;
          temp_dict["weight"] = mlp_weight;
          StateDict temp_state_dict(temp_dict);
          mlp_proj_->as<DiTParallelLinear>()->load_state_dict(temp_state_dict);
        }
      }
    }
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    to_out_->as<DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    to_out_->as<DiTParallelLinear>()->verify_loaded_weights(prefix + "to_out.");
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  int64_t query_dim_;
  int64_t out_dim_;
  float mlp_ratio_;
  int64_t mlp_hidden_dim_;
  int64_t mlp_mult_factor_;
  DiTParallelLinear to_q_{nullptr};
  DiTParallelLinear to_k_{nullptr};
  DiTParallelLinear to_v_{nullptr};
  DiTParallelLinear mlp_proj_{nullptr};
  Flux2SwiGLU mlp_act_fn_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
  DiTParallelLinear to_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2ParallelSelfAttention);

class Flux2SingleTransformerBlockImpl : public torch::nn::Module {
 public:
  explicit Flux2SingleTransformerBlockImpl(const ModelContext& context,
                                           int64_t inner_dim,
                                           const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();

    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({inner_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));

    attn_ = register_module("attn",
                            Flux2ParallelSelfAttention(context, parallel_args));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb_mod,
                        const torch::Tensor& image_rotary_emb,
                        bool split_hidden_states,
                        int64_t text_seq_len) {
    auto mod_params = Flux2ModulationImpl::split(temb_mod, 1);
    auto [shift, scale, gate] = mod_params[0];

    auto norm_hidden_states = norm_->forward(hidden_states);
    norm_hidden_states = (1 + scale) * norm_hidden_states + shift;
    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);

    auto gate_attn = gate * attn_output;
    auto output = hidden_states + gate_attn;
    if (output.dtype() == torch::kFloat16) {
      output = torch::clamp(output, -65504.0f, 65504.0f);
    }

    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
  }

 private:
  torch::nn::LayerNorm norm_{nullptr};
  Flux2ParallelSelfAttention attn_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2SingleTransformerBlock);

class Flux2Transformer2DModelImpl : public torch::nn::Module {
 public:
  explicit Flux2Transformer2DModelImpl(const ModelContext& context,
                                       const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    auto model_args = context.get_model_args();
    int64_t num_attention_heads = model_args.n_heads();
    int64_t attention_head_dim = model_args.head_dim();
    int64_t joint_attention_dim = model_args.joint_attention_dim();
    int64_t num_layers = model_args.num_layers();
    int64_t num_single_layers = model_args.num_single_layers();
    int64_t patch_size = model_args.patch_size();
    float rope_theta = model_args.rope_theta();
    auto axes_dims_rope = model_args.axes_dims_rope();
    in_channels_ = model_args.in_channels();
    out_channels_ = model_args.out_channels();

    int64_t inner_dim = num_attention_heads * attention_head_dim;

    time_guidance_embed_ =
        Flux2TimestepGuidanceEmbeddings(context, inner_dim, false, true);
    register_module("time_guidance_embed", time_guidance_embed_);

    double_stream_modulation_img_ =
        register_module("double_stream_modulation_img",
                        Flux2Modulation(context, inner_dim, 2, false));
    double_stream_modulation_txt_ =
        register_module("double_stream_modulation_txt",
                        Flux2Modulation(context, inner_dim, 2, false));
    single_stream_modulation_ =
        register_module("single_stream_modulation",
                        Flux2Modulation(context, inner_dim, 1, false));

    x_embedder_ = register_module(
        "x_embedder",
        layer::AddMatmul(in_channels_, inner_dim, false, options_));
    context_embedder_ = register_module(
        "context_embedder",
        layer::AddMatmul(joint_attention_dim, inner_dim, false, options_));

    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());

    transformer_block_layers_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = Flux2TransformerBlock(context, inner_dim, parallel_args);
      transformer_blocks_->push_back(block);
      transformer_block_layers_.push_back(block);
    }

    single_transformer_block_layers_.reserve(num_single_layers);
    for (int64_t i = 0; i < num_single_layers; ++i) {
      auto block =
          Flux2SingleTransformerBlock(context, inner_dim, parallel_args);
      single_transformer_blocks_->push_back(block);
      single_transformer_block_layers_.push_back(block);
    }

    norm_out_ =
        register_module("norm_out", AdaLayerNormContinuous(context, false));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmul(inner_dim,
                         patch_size * patch_size * out_channels_,
                         false,
                         options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& image_rotary_emb,
                        int64_t step_idx = 0) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    auto guidance_scaled = guidance.defined()
                               ? guidance.to(hidden_states.dtype()) * 1000.0f
                               : torch::Tensor();
    auto temb = time_guidance_embed_->forward(timestep_scaled, guidance_scaled);

    auto double_stream_mod_img = double_stream_modulation_img_->forward(temb);
    auto double_stream_mod_txt = double_stream_modulation_txt_->forward(temb);
    auto single_stream_mod = single_stream_modulation_->forward(temb);

    // ── Double-stream transformer blocks ──
    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      auto [new_encoder_hidden, new_hidden] =
          block->forward(hidden_states,
                         encoder_hidden_states,
                         double_stream_mod_img,
                         double_stream_mod_txt,
                         image_rotary_emb);

      hidden_states = new_hidden;
      encoder_hidden_states = new_encoder_hidden;
    }

    // ── Merge into single stream: [txt_seq, img_seq]
    hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);

    // ── Single-stream transformer blocks (DiTCache: use_cfg=true) ──
    // NOTE: Flux2 does NOT support CFG. The "use_cfg" parameter routes to
    // the second cache instance (active_cond_cache_) to isolate single-stream.
    torch::Tensor ss_original_hidden_states = hidden_states;
    auto dummy_encoder =
        torch::zeros({hidden_states.size(0), 1, hidden_states.size(2)},
                     hidden_states.options());

    // Step start for single-stream phase
    TensorMap ss_step_in_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", ss_original_hidden_states}};
    CacheStepIn ss_stepin_before(step_idx, ss_step_in_map);
    bool ss_use_step_cache =
        DiTCache::get_instance().on_before_step(ss_stepin_before,
                                                /*use_cfg=*/true);

    if (!ss_use_step_cache) {
      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        TensorMap ss_block_in_before_map = {};
        CacheBlockIn ss_blockin_before(i, ss_block_in_before_map);
        bool ss_use_block_cache =
            DiTCache::get_instance().on_before_block(ss_blockin_before,
                                                     /*use_cfg=*/true);

        if (!ss_use_block_cache) {
          auto block = single_transformer_block_layers_[i];
          hidden_states = block->forward(
              hidden_states, single_stream_mod, image_rotary_emb, false, 0);
        }

        TensorMap ss_block_in_after_map = {
            {"hidden_states", hidden_states},
            {"encoder_hidden_states", dummy_encoder},
            {"original_hidden_states", ss_original_hidden_states}};
        CacheBlockIn ss_blockin_after(i, ss_block_in_after_map);
        CacheBlockOut ss_blockout_after =
            DiTCache::get_instance().on_after_block(ss_blockin_after,
                                                    /*use_cfg=*/true);

        hidden_states = ss_blockout_after.tensors.at("hidden_states");
      }
    }

    // Step end for single-stream phase
    TensorMap ss_step_after_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", ss_original_hidden_states}};
    CacheStepIn ss_stepin_after(step_idx, ss_step_after_map);
    CacheStepOut ss_stepout_after =
        DiTCache::get_instance().on_after_step(ss_stepin_after,
                                               /*use_cfg=*/true);
    hidden_states = ss_stepout_after.tensors.at("hidden_states");

    int64_t start = encoder_hidden_states.size(1);
    int64_t length = hidden_states.size(1) - start;
    auto output_hidden =
        hidden_states.narrow(1, start, std::max(length, int64_t(0)));

    auto output_hidden_final = norm_out_->forward(output_hidden, temb);

    auto final_output = proj_out_->forward(output_hidden_final);

    return final_output;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      context_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("context_embedder."));
      x_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("x_embedder."));
      time_guidance_embed_->load_state_dict(
          state_dict->get_dict_with_prefix("time_guidance_embed."));
      double_stream_modulation_img_->load_state_dict(
          state_dict->get_dict_with_prefix("double_stream_modulation_img."));
      double_stream_modulation_txt_->load_state_dict(
          state_dict->get_dict_with_prefix("double_stream_modulation_txt."));
      single_stream_modulation_->load_state_dict(
          state_dict->get_dict_with_prefix("single_stream_modulation."));
      for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
        auto block = transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
        auto block = single_transformer_block_layers_[i];
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    context_embedder_->verify_loaded_weights(prefix + "context_embedder.");
    x_embedder_->verify_loaded_weights(prefix + "x_embedder.");
    time_guidance_embed_->verify_loaded_weights(prefix +
                                                "time_guidance_embed.");
    double_stream_modulation_img_->verify_loaded_weights(
        prefix + "double_stream_modulation_img.");
    double_stream_modulation_txt_->verify_loaded_weights(
        prefix + "double_stream_modulation_txt.");
    single_stream_modulation_->verify_loaded_weights(
        prefix + "single_stream_modulation.");
    for (int64_t i = 0; i < transformer_block_layers_.size(); ++i) {
      auto block = transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    for (int64_t i = 0; i < single_transformer_block_layers_.size(); ++i) {
      auto block = single_transformer_block_layers_[i];
      block->verify_loaded_weights(prefix + "single_transformer_blocks." +
                                   std::to_string(i) + ".");
    }
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
  }

  int64_t in_channels() { return in_channels_; }

 private:
  int64_t in_channels_;
  int64_t out_channels_;
  layer::AddMatmul context_embedder_{nullptr};
  layer::AddMatmul x_embedder_{nullptr};
  Flux2TimestepGuidanceEmbeddings time_guidance_embed_{nullptr};
  Flux2Modulation double_stream_modulation_img_{nullptr};
  Flux2Modulation double_stream_modulation_txt_{nullptr};
  Flux2Modulation single_stream_modulation_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<Flux2TransformerBlock> transformer_block_layers_;
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  std::vector<Flux2SingleTransformerBlock> single_transformer_block_layers_;
  AdaLayerNormContinuous norm_out_{nullptr};
  layer::AddMatmul proj_out_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2Transformer2DModel);

class Flux2DiTModelImpl : public torch::nn::Module {
 public:
  explicit Flux2DiTModelImpl(const ModelContext& context,
                             const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    flux2_transformer_2d_model_ =
        register_module("flux2_transformer_2d_model_",
                        Flux2Transformer2DModel(context, parallel_args));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& image_rotary_emb,
                        int64_t step_idx = 0) {
    torch::Tensor output =
        flux2_transformer_2d_model_->forward(hidden_states_input,
                                             encoder_hidden_states_input,
                                             timestep,
                                             guidance,
                                             image_rotary_emb,
                                             step_idx);
    return output;
  }
  int64_t in_channels() { return flux2_transformer_2d_model_->in_channels(); }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux2_transformer_2d_model_->load_model(std::move(loader));
    flux2_transformer_2d_model_->verify_loaded_weights("");
  }

 private:
  Flux2Transformer2DModel flux2_transformer_2d_model_{nullptr};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
};
TORCH_MODULE(Flux2DiTModel);

REGISTER_MODEL_ARGS(Flux2Transformer2DModel, [&] {
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{32, 32, 32, 32}));
  LOAD_ARG_OR(eps, "eps", 1e-6);
  LOAD_ARG_OR(in_channels, "in_channels", 128);
  LOAD_ARG_OR(joint_attention_dim, "joint_attention_dim", 15360);
  LOAD_ARG_OR(mlp_ratio, "mlp_ratio", 3.0f);
  LOAD_ARG_OR(num_layers, "num_layers", 8);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 48);
  LOAD_ARG_OR(out_channels, "out_channels", 128);
  LOAD_ARG_OR(patch_size, "patch_size", 1);
  LOAD_ARG_OR(rope_theta, "rope_theta", 2000.0f);
  LOAD_ARG_OR(timestep_guidance_channels, "timestep_guidance_channels", 256);
});

}  // namespace xllm

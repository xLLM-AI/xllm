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
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#if defined(USE_DCU)
#include "core/layers/dcu/flash_attention.h"
#endif
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/utils/dit_parallel_linear.h"
#include "models/dit/utils/sequence_parallel_pad_manager.h"
#include "models/model_registry.h"

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>
#endif

namespace xllm {

inline bool use_dit_sp_communication_overlap() {
  return DiTConfig::get_instance().dit_sp_communication_overlap() &&
         ParallelConfig::get_instance().sp_size() > 1;
}

inline torch::Tensor qwen_image_scaled_dot_product_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
#if defined(USE_DCU)
  CHECK_EQ(query.dim(), 4) << "Qwen-Image query must be [B,H,S,D]";
  CHECK_EQ(key.dim(), 4) << "Qwen-Image key must be [B,H,S,D]";
  CHECK_EQ(value.dim(), 4) << "Qwen-Image value must be [B,H,S,D]";
  CHECK_EQ(query.size(0), key.size(0)) << "Qwen-Image q/k batch mismatch";
  CHECK_EQ(query.size(0), value.size(0)) << "Qwen-Image q/v batch mismatch";
  CHECK_EQ(query.size(1), key.size(1)) << "Qwen-Image q/k head mismatch";
  CHECK_EQ(key.size(1), value.size(1)) << "Qwen-Image k/v head mismatch";
  CHECK_EQ(query.size(3), key.size(3)) << "Qwen-Image q/k head dim mismatch";
  CHECK_EQ(query.size(3), value.size(3)) << "Qwen-Image q/v head dim mismatch";
  CHECK_EQ(key.size(2), value.size(2)) << "Qwen-Image k/v seq mismatch";

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t q_seq_len = query.size(2);
  const int64_t kv_seq_len = key.size(2);
  const int64_t head_dim = query.size(3);

  auto cu_options =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  auto cu_seqlens_q =
      torch::arange(batch_size + 1, cu_options).mul_(q_seq_len).contiguous();
  auto cu_seqlens_k =
      torch::arange(batch_size + 1, cu_options).mul_(kv_seq_len).contiguous();

  auto dense_query = query.transpose(1, 2).contiguous().view(
      {batch_size * q_seq_len, num_heads, head_dim});
  auto dense_key = key.transpose(1, 2).contiguous().view(
      {batch_size * kv_seq_len, key.size(1), head_dim});
  auto dense_value = value.transpose(1, 2).contiguous().view(
      {batch_size * kv_seq_len, value.size(1), head_dim});

  auto output = layer::dense_varlen_flash_attention(
      dense_query,
      dense_key,
      dense_value,
      cu_seqlens_q,
      cu_seqlens_k,
      std::pow(static_cast<double>(head_dim), -0.5),
      /*is_causal=*/false);
  return output.view({batch_size, q_seq_len, num_heads, head_dim})
      .transpose(1, 2)
      .contiguous();
#else
  return torch::scaled_dot_product_attention(query,
                                             key,
                                             value,
                                             torch::nullopt,
                                             /*dropout_p=*/0.0,
                                             /*is_causal=*/false);
#endif
}

inline torch::Tensor qwen_image_joint_attention(
    const torch::Tensor& joint_query,
    const torch::Tensor& joint_key,
    const torch::Tensor& joint_value) {
  constexpr int64_t kAttentionChunkSize = 512;
  auto query = joint_query.transpose(1, 2);
  auto key = joint_key.transpose(1, 2);
  auto value = joint_value.transpose(1, 2);
  auto attention_output_dtype = query.dtype();

  torch::Tensor output;
  if (query.size(2) <= kAttentionChunkSize) {
    output = qwen_image_scaled_dot_product_attention(query, key, value);
  } else {
    std::vector<torch::Tensor> attention_chunks;
    const int64_t num_chunks =
        (query.size(2) + kAttentionChunkSize - 1) / kAttentionChunkSize;
    attention_chunks.reserve(num_chunks);
    for (int64_t start = 0; start < query.size(2);
         start += kAttentionChunkSize) {
      int64_t chunk_size = std::min(kAttentionChunkSize, query.size(2) - start);
      auto query_chunk = query.narrow(2, start, chunk_size);
      attention_chunks.emplace_back(
          qwen_image_scaled_dot_product_attention(query_chunk, key, value));
    }
    output = torch::cat(attention_chunks, 2);
  }
  if (output.dtype() != attention_output_dtype) {
    output = output.to(attention_output_dtype);
  }
  return output.transpose(1, 2);
}

namespace qwenimage {
// TODO: This class should be extracted from dit class and integrated into a
// common class.
class RMSNormImpl final : public torch::nn::Module {
 public:
  // Constructor: dim (normalization dimension), eps (stabilization term)
  // elementwise_affine (enable affine transform), bias (enable bias term)
  RMSNormImpl(int64_t dim, double eps, bool elementwise_affine, bool bias)
      : eps_(eps), elementwise_affine_(elementwise_affine), is_bias_(bias) {
    if (elementwise_affine_) {
      weight_ = register_parameter("weight", torch::ones({dim}));
      if (is_bias_) {
        bias_ = register_parameter("bias", torch::zeros({dim}));
      }
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
#if defined(USE_NPU)
    auto [output, rstd] =
        at_npu::native::custom_ops::npu_rms_norm(hidden_states, weight_, eps_);
#else
    torch::Tensor output = hidden_states.to(torch::kFloat32);
    output =
        output * torch::rsqrt(output.pow(2).mean(-1, /*keepdim=*/true) + eps_);
    if (elementwise_affine_) {
      output = output * weight_.to(output.device(), output.dtype());
    }
    output = output.to(hidden_states.dtype());
#endif
    if (is_bias_ && bias_.defined()) {
      output = output + bias_;
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
  double eps_;               // Small epsilon to avoid division by zero
  bool elementwise_affine_;  // Whether to apply learnable affine parameters
  torch::Tensor weight_;     // Learnable scale parameter
  torch::Tensor bias_;       // Learnable bias parameter (optional)
  bool is_bias_;
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
};
TORCH_MODULE(RMSNorm);

// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AdaLayerNormContinuousImpl final : public torch::nn::Module {
 public:
  explicit AdaLayerNormContinuousImpl(const ModelContext& context,
                                      int64_t embedding_dim,
                                      int64_t conditioning_embedding_dim,
                                      bool elementwise_affine = true,
                                      double eps = 1e-5,
                                      bool bias = true)
      : options_(context.get_tensor_options()) {
    ModelArgs model_args = context.get_model_args();
    quant_args_ = context.get_quant_args();
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        layer::AddMatmulWeightTransposed(conditioning_embedding_dim,
                                         2 * embedding_dim,
                                         bias,
                                         options_,
                                         quant_args_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));
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
  layer::AddMatmulWeightTransposed linear_{nullptr};
  torch::nn::SiLU silu_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  QuantArgs quant_args_;
  double eps_;
  std::string norm_type_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormContinuous);

class TimestepsImpl final : public torch::nn::Module {
 public:
  TimestepsImpl(const ModelContext& context,
                int64_t num_channels,
                bool flip_sin_to_cos,
                double downscale_freq_shift,
                double scale,
                int64_t max_period = 10000)
      : embedding_dim_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale),
        max_period_(max_period) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    CHECK(timesteps.dim() == 1) << "Timesteps should be a 1d-array";

    int64_t half_dim = embedding_dim_ / 2;

    auto exponent =
        -std::log(max_period_) * torch::arange(0,
                                               half_dim,
                                               torch::TensorOptions()
                                                   .dtype(torch::kFloat32)
                                                   .device(timesteps.device()));

    exponent = exponent / (half_dim - downscale_freq_shift_);
    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat) * emb.unsqueeze(0);

    emb = scale_ * emb;

    // concat sine and cosine embeddings
    auto sin_emb = torch::sin(emb);
    auto cos_emb = torch::cos(emb);
    emb = torch::cat({sin_emb, cos_emb}, /*dim=*/-1);
    // flip sine and cosine embeddings
    if (flip_sin_to_cos_) {
      emb = torch::cat({cos_emb, sin_emb}, /*dim=*/-1);
    }
    // zero pad
    if (embedding_dim_ % 2 == 1) {
      emb = torch::nn::functional::pad(
          emb, torch::nn::functional::PadFuncOptions({0, 1}));
    }
    return emb;
  }

 private:
  int64_t embedding_dim_;
  int64_t max_period_;
  bool flip_sin_to_cos_;
  double scale_;
  double downscale_freq_shift_;
};
TORCH_MODULE(Timesteps);

// TODO: a factory function that provides activation functions based on string
// input
std::function<torch::Tensor(const torch::Tensor&)> get_activation(
    const std::string& act_fn) {
  if (act_fn == "silu") {
    return [](const torch::Tensor& x) { return torch::silu(x); };
  } else if (act_fn == "relu") {
    return [](const torch::Tensor& x) { return torch::relu(x); };
  } else if (act_fn == "gelu") {
    return [](const torch::Tensor& x) { return torch::gelu(x); };
  } else if (act_fn == "tanh") {
    return [](const torch::Tensor& x) { return torch::tanh(x); };
  } else if (act_fn == "sigmoid") {
    return [](const torch::Tensor& x) { return torch::sigmoid(x); };
  } else if (act_fn == "none" || act_fn.empty()) {
    return [](const torch::Tensor& x) { return x; };
  } else {
    LOG(FATAL) << "Unsupported activation function: " << act_fn;
    return [](const torch::Tensor& x) { return x; };
  }
}

class TimestepEmbeddingImpl final : public torch::nn::Module {
 public:
  TimestepEmbeddingImpl(const ModelContext& context,
                        int64_t in_channels,
                        int64_t time_embed_dim,
                        const std::string& act_fn = "silu",
                        int64_t out_dim = -1,
                        const std::string& post_act_fn = "",
                        int64_t cond_proj_dim = -1,
                        bool sample_proj_bias = true)
      : options_(context.get_tensor_options()) {
    quant_args_ = context.get_quant_args();
    linear_1_ =
        register_module("linear_1",
                        layer::AddMatmulWeightTransposed(in_channels,
                                                         time_embed_dim,
                                                         sample_proj_bias,
                                                         options_,
                                                         quant_args_));

    if (cond_proj_dim > 0) {
      cond_proj_ = register_module(
          "cond_proj",
          layer::AddMatmulWeightTransposed(
              cond_proj_dim, in_channels, false, options_, quant_args_));
    }

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim > 0) ? out_dim : time_embed_dim;

    linear_2_ =
        register_module("linear_2",
                        layer::AddMatmulWeightTransposed(time_embed_dim,
                                                         time_embed_dim_out,
                                                         sample_proj_bias,
                                                         options_,
                                                         quant_args_));
  }

  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x = sample;

    if (cond_proj_) {
      x = x + cond_proj_->forward(condition);
    }
    x = linear_1_->forward(x);
    x = act_fn_(x);
    x = linear_2_->forward(x);

    return x;
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
  torch::TensorOptions options_;
  QuantArgs quant_args_;
  torch::nn::SiLU act_fn_{nullptr};
  layer::AddMatmulWeightTransposed linear_1_{nullptr};
  layer::AddMatmulWeightTransposed linear_2_{nullptr};
  layer::AddMatmulWeightTransposed cond_proj_{nullptr};
};
TORCH_MODULE(TimestepEmbedding);

// A internel class that only register necessary modules for attention
// implementation The attention forward shouldn't be implemented here, but in
// processor classes
// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AttentionImpl final : public torch::nn::Module {
 public:
  AttentionImpl(const ModelContext& context,
                int64_t query_dim,
                std::optional<int64_t> cross_attention_dim = std::nullopt,
                int64_t heads = 8,
                std::optional<int64_t> kv_heads = std::nullopt,
                int64_t dim_head = 64,
                double dropout = 0.0,
                bool bias = false,
                const std::string& qk_norm = "",
                const std::string& cross_attention_norm = "",
                std::optional<int64_t> added_kv_proj_dim = std::nullopt,
                bool added_proj_bias = true,
                bool out_bias = true,
                bool scale_qk = true,
                bool only_cross_attention = false,
                double eps = 1e-5,
                double rescale_output_factor = 1.0,
                bool residual_connection = false,
                std::optional<int64_t> out_dim = std::nullopt,
                std::optional<int64_t> out_context_dim = std::nullopt,
                std::optional<bool> context_pre_only = std::nullopt,
                bool pre_only = false,
                bool elementwise_affine = true,
                bool is_causal = false,
                ProcessGroup* sp_group = nullptr)
      : options_(context.get_tensor_options()),
        quant_args_(context.get_quant_args()),
        heads_(heads),
        kv_heads_(kv_heads.has_value() ? kv_heads.value() : heads),
        dim_head_(dim_head),
        bias_(bias),
        out_bias_(out_bias),
        added_proj_bias_(added_proj_bias),
        sp_group_(sp_group) {
    if (qk_norm == "layer_norm") {
      layer_norm_q_ = register_module(
          "norm_q",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_head})
                                   .eps(eps)
                                   .elementwise_affine(elementwise_affine)));
      layer_norm_k_ = register_module(
          "norm_k",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_head})
                                   .eps(eps)
                                   .elementwise_affine(elementwise_affine)));
    } else if (qk_norm == "layer_norm_across_heads") {
      // Lumina applies qk norm across all heads
      CHECK(kv_heads.has_value())
          << "qk_norm is set to: " + qk_norm + ", but get no kv_heads ";
      layer_norm_q_ = register_module(
          "norm_q",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * heads}).eps(eps)));
      layer_norm_k_ = register_module(
          "norm_k",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * kv_heads.value()})
                  .eps(eps)));
    } else if (qk_norm == "rms_norm") {
      // Assuming you have an RMSNorm implementation
      norm_q_ = register_module(
          "norm_q",
          RMSNorm(dim_head, eps, /*elementwise_affine=*/true, /*bias=*/false));
      norm_k_ = register_module(
          "norm_k",
          RMSNorm(dim_head, eps, /*elementwise_affine=*/true, /*bias=*/false));
    } else if (qk_norm == "rms_norm_across_heads") {
      // LTX applies qk norm across all heads
      CHECK(kv_heads.has_value())
          << "qk_norm is set to: " + qk_norm + ", but get no kv_heads ";

      norm_q_ = register_module(
          "norm_q",
          RMSNorm(dim_head, eps, /*elementwise_affine=*/true, /*bias=*/false));
      norm_k_ = register_module("norm_k",
                                RMSNorm(dim_head * kv_heads.value(),
                                        eps,
                                        /*elementwise_affine=*/true,
                                        /*bias=*/false));
    } else {
      CHECK(qk_norm.empty()) << "unknown qk_norm: " + qk_norm +
                                    ". Should be "
                                    "'','layer_norm','rms_norm','layer_norm_"
                                    "across_heads', 'rms_norm_across_heads'";
    }

    if (cross_attention_norm == "layer_norm") {
      norm_cross_ = register_module(
          "norm_cross",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({cross_attention_dim.value()})));
    } else {
      CHECK(cross_attention_norm.empty())
          << "unknown cross_attention_norm: " + cross_attention_norm +
                 ". Should be '', 'layer_norm'";
    }

    int64_t q_dim = out_dim.has_value() ? out_dim.value() : dim_head * heads;
    int64_t kv_dim =
        !kv_heads.has_value() ? q_dim : dim_head * kv_heads.value();
    cross_attention_dim = cross_attention_dim.has_value()
                              ? cross_attention_dim.value()
                              : query_dim;
    out_context_dim =
        out_context_dim.has_value() ? out_context_dim.value() : query_dim;

    xllm::dit::SpOptions q_sp_option;
    xllm::dit::SpOptions kv_sp_option;
    xllm::dit::LinearType linear_type = xllm::dit::LinearType::Default;
    if (::xllm::ParallelConfig::get_instance().sp_size() > 1 &&
        !use_dit_sp_communication_overlap()) {
      q_sp_option = xllm::dit::SpOptions(/*head_num=*/heads,
                                         /*head_dim=*/dim_head,
                                         /*hidden_size=*/q_dim,
                                         /*before_attention=*/true,
                                         /*process_group=*/sp_group_);

      kv_sp_option = xllm::dit::SpOptions(
          /*head_num=*/kv_heads.has_value() ? kv_heads.value() : heads,
          /*head_dim=*/dim_head,
          /*hidden_size=*/kv_dim,
          /*before_attention=*/true,
          /*process_group=*/sp_group_);
      linear_type = xllm::dit::LinearType::SequenceParallel;
    }

    to_q_ = register_module("q_linear",
                            xllm::dit::DiTParallelLinear(query_dim,
                                                         q_dim,
                                                         bias,
                                                         options_,
                                                         q_sp_option,
                                                         /*tp=*/std::nullopt,
                                                         quant_args_));

    // Key-Value projections (if not only cross attention)
    if (!only_cross_attention) {
      to_k_ = register_module(
          "k_linear",
          xllm::dit::DiTParallelLinear(cross_attention_dim.value(),
                                       kv_dim,
                                       bias,
                                       options_,
                                       kv_sp_option,
                                       /*tp=*/std::nullopt,
                                       quant_args_));

      to_v_ = register_module(
          "v_linear",
          xllm::dit::DiTParallelLinear(cross_attention_dim.value(),
                                       kv_dim,
                                       bias,
                                       options_,
                                       kv_sp_option,
                                       /*tp=*/std::nullopt,
                                       quant_args_));
    }

    if (added_kv_proj_dim.has_value()) {
      add_k_proj_ = register_module(
          "add_k_linear",
          xllm::dit::DiTParallelLinear(added_kv_proj_dim.value(),
                                       kv_dim,
                                       added_proj_bias,
                                       options_,
                                       kv_sp_option,
                                       /*tp=*/std::nullopt,
                                       quant_args_));

      add_v_proj_ = register_module(
          "add_v_linear",
          xllm::dit::DiTParallelLinear(added_kv_proj_dim.value(),
                                       kv_dim,
                                       added_proj_bias,
                                       options_,
                                       kv_sp_option,
                                       /*tp=*/std::nullopt,
                                       quant_args_));
      if (context_pre_only.has_value()) {
        add_q_proj_ = register_module(
            "add_q_linear",
            xllm::dit::DiTParallelLinear(added_kv_proj_dim.value(),
                                         q_dim,
                                         added_proj_bias,
                                         options_,
                                         q_sp_option,
                                         /*tp=*/std::nullopt,
                                         quant_args_));
      }
    }

    xllm::dit::SpOptions out_sp_option;
    if (::xllm::ParallelConfig::get_instance().sp_size() > 1 &&
        !use_dit_sp_communication_overlap()) {
      out_sp_option = xllm::dit::SpOptions(/*head_num=*/heads,
                                           /*head_dim=*/dim_head,
                                           /*hidden_size=*/q_dim,
                                           /*before_attention=*/false,
                                           /*process_group=*/sp_group_);
    }

    // Output projections
    if (!pre_only) {
      to_out_ = register_module("to_out", torch::nn::Sequential());

      to_out_->push_back(xllm::dit::DiTParallelLinear(q_dim,
                                                      out_dim.value(),
                                                      out_bias,
                                                      options_,
                                                      out_sp_option,
                                                      /*tp=*/std::nullopt,
                                                      quant_args_));
      to_out_->push_back(
          torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
    }

    // Additional output for context
    if (context_pre_only.has_value() && context_pre_only) {
      to_add_out_ =
          register_module("to_add_out_linear",
                          xllm::dit::DiTParallelLinear(q_dim,
                                                       out_context_dim.value(),
                                                       out_bias,
                                                       options_,
                                                       out_sp_option,
                                                       /*tp=*/std::nullopt,
                                                       quant_args_));
    }

    // Added QK normalization for added KV projections
    if (!qk_norm.empty() && added_kv_proj_dim.has_value()) {
      if (qk_norm == "rms_norm") {
        norm_added_q_ = register_module("norm_added_q",
                                        RMSNorm(dim_head, eps, true, false));
        norm_added_k_ = register_module("norm_added_k",
                                        RMSNorm(dim_head, eps, true, false));
      } else {
        CHECK(qk_norm.empty()) << "unknown qk_norm: " + qk_norm +
                                      ". Should be one of '','rms_norm'";
        // For layer_norm, we would register similar layers here
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    // to_out
    to_out_[0]->as<xllm::dit::DiTParallelLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out.0."));
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

    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));

    add_q_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_q_proj."));
    add_k_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_k_proj."));
    add_v_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_v_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    // to_out
    to_out_[0]->as<xllm::dit::DiTParallelLinear>()->verify_loaded_weights(
        prefix + "to_out.0.");
    // to_add_out
    to_add_out_->verify_loaded_weights(prefix + "to_add_out.");
    // norm_q
    norm_q_->verify_loaded_weights(prefix + "norm_q.");
    // norm_k
    norm_k_->verify_loaded_weights(prefix + "norm_k.");
    // norm_added_q
    norm_added_q_->verify_loaded_weights(prefix + "norm_added_q.");
    // norm_added_k
    norm_added_k_->verify_loaded_weights(prefix + "norm_added_k.");

    to_q_->verify_loaded_weights(prefix + "to_q.");
    to_k_->verify_loaded_weights(prefix + "to_k.");
    to_v_->verify_loaded_weights(prefix + "to_v.");

    add_q_proj_->verify_loaded_weights(prefix + "add_q_proj.");
    add_k_proj_->verify_loaded_weights(prefix + "add_k_proj.");
    add_v_proj_->verify_loaded_weights(prefix + "add_v_proj.");
  }

 public:
  int64_t heads_;
  int64_t kv_heads_;
  int64_t dim_head_;
  bool bias_;
  bool out_bias_;
  bool added_proj_bias_;
  ProcessGroup* sp_group_;

  torch::TensorOptions options_;
  QuantArgs quant_args_;
  torch::nn::LayerNorm layer_norm_q_{nullptr}, layer_norm_k_{nullptr},
      norm_cross_{nullptr};
  xllm::dit::DiTParallelLinear to_q_{nullptr}, to_k_{nullptr}, to_v_{nullptr};
  xllm::dit::DiTParallelLinear add_k_proj_{nullptr}, add_v_proj_{nullptr},
      add_q_proj_{nullptr};
  torch::nn::Sequential to_out_{nullptr};
  xllm::dit::DiTParallelLinear to_add_out_{nullptr};

  // Assuming you have RMSNorm implemented
  RMSNorm norm_q_{nullptr}, norm_k_{nullptr}, norm_added_q_{nullptr},
      norm_added_k_{nullptr};
};
TORCH_MODULE(Attention);

class FeedForwardImpl final : public torch::nn::Module {
 public:
  explicit FeedForwardImpl(const ModelContext& context,
                           int64_t dim,
                           int64_t dim_out,
                           int64_t mult = 4,
                           double dropout = 0.0)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    quant_args_ = context.get_quant_args();
    int64_t inner_dim = dim * 4;

    // linear1
    linear1_ =
        register_module("linear1",
                        layer::AddMatmulWeightTransposed(
                            dim, inner_dim, true, options_, quant_args_));

    // activation
    activation_ =
        register_module("activation",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    // linear2
    linear2_ =
        register_module("linear2",
                        layer::AddMatmulWeightTransposed(
                            inner_dim, dim_out, true, options_, quant_args_));
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
  layer::AddMatmulWeightTransposed linear1_{nullptr};
  layer::AddMatmulWeightTransposed linear2_{nullptr};
  torch::nn::Functional activation_{nullptr};
  QuantArgs quant_args_;
  torch::TensorOptions options_;
};
TORCH_MODULE(FeedForward);

}  // namespace qwenimage

// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AdaLayerNormImpl final : public torch::nn::Module {
 public:
  AdaLayerNormImpl(const ModelContext& contex,
                   int64_t hidden_size,
                   double eps = 1e-6)
      : hidden_size_(hidden_size), eps_(eps) {
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})
                                 .elementwise_affine(false)
                                 .eps(eps)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& mod_params,
      const torch::Tensor& index = torch::Tensor()) {
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];
    torch::Tensor shift_result, scale_result, gate_result;

    if (index.defined()) {
      // Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
      // So shift, scale, gate have shape [2*actual_batch, d]
      int64_t actual_batch = shift.size(0) / 2;

      // Split into two parts
      auto shift_0 = shift.slice(0, 0, actual_batch);
      auto shift_1 = shift.slice(0, actual_batch, shift.size(0));

      auto scale_0 = scale.slice(0, 0, actual_batch);
      auto scale_1 = scale.slice(0, actual_batch, scale.size(0));

      auto gate_0 = gate.slice(0, 0, actual_batch);
      auto gate_1 = gate.slice(0, actual_batch, gate.size(0));

      // index: [b, l] where b is actual batch size
      // Expand to [b, l, 1] to match feature dimension
      auto index_expanded = index.unsqueeze(-1);  // [b, l, 1]

      // Expand chunks to [b, 1, d] then broadcast to [b, l, d]
      auto shift_0_exp = shift_0.unsqueeze(1);  // [b, 1, d]
      auto shift_1_exp = shift_1.unsqueeze(1);  // [b, 1, d]
      auto scale_0_exp = scale_0.unsqueeze(1);
      auto scale_1_exp = scale_1.unsqueeze(1);
      auto gate_0_exp = gate_0.unsqueeze(1);
      auto gate_1_exp = gate_1.unsqueeze(1);

      // Use torch::where to select based on index
      shift_result =
          torch::where(index_expanded == 0, shift_0_exp, shift_1_exp);
      scale_result =
          torch::where(index_expanded == 0, scale_0_exp, scale_1_exp);
      gate_result = torch::where(index_expanded == 0, gate_0_exp, gate_1_exp);
    } else {
      shift_result = shift.unsqueeze(1);
      scale_result = scale.unsqueeze(1);
      gate_result = gate.unsqueeze(1);
    }

    scale_result = 1 + scale_result;

    // auto result = at_npu::native::custom_ops::npu_layer_norm_eval(
    //     x, {hidden_size_}, scale_result, shift_result, eps_);
    auto x_norm = norm_->forward(x);
    auto result = x_norm * scale_result + shift_result;
    return std::make_tuple(result, gate_result);
  }

 private:
  double eps_;
  int64_t hidden_size_;
  torch::nn::LayerNorm norm_{nullptr};
};
TORCH_MODULE(AdaLayerNorm);

torch::Tensor apply_rotary_emb_qwen(const torch::Tensor& x,
                                    const torch::Tensor& freqs_cis,
                                    bool use_real = true,
                                    int64_t use_real_unbind_dim = -1) {
  auto cos = torch::real(freqs_cis);
  auto sin = torch::imag(freqs_cis);

  int64_t seqlen = cos.size(0);

  auto cos_expanded = cos.unsqueeze(0)
                          .unsqueeze(2)
                          .unsqueeze(-1)
                          .expand({-1, -1, -1, -1, 2})
                          .reshape({1, seqlen, 1, -1});
  auto sin_expanded = sin.unsqueeze(0)
                          .unsqueeze(2)
                          .unsqueeze(-1)
                          .expand({-1, -1, -1, -1, 2})
                          .reshape({1, seqlen, 1, -1});
#if defined(USE_NPU)
  auto x_out = at_npu::native::custom_ops::npu_rotary_mul(
      x.to(torch::kFloat), cos_expanded, sin_expanded, "interleave");
  return x_out.to(x.dtype());
#else
  auto input_dtype = x.dtype();
  auto x_float = x.to(torch::kFloat32);
  auto x_flat = x_float.unflatten(-1, std::vector<int64_t>{-1, 2});
  auto x1 = x_flat.select(-1, 0);
  auto x2 = x_flat.select(-1, 1);
  auto cos_half = cos.unsqueeze(0).unsqueeze(2);
  auto sin_half = sin.unsqueeze(0).unsqueeze(2);
  auto out1 = x1 * cos_half - x2 * sin_half;
  auto out2 = x1 * sin_half + x2 * cos_half;
  auto out = torch::stack({out1, out2}, -1).flatten(-2, -1);
  return out.to(input_dtype);
#endif
}

std::tuple<int64_t, std::optional<torch::Tensor>, std::optional<torch::Tensor>>
compute_text_seq_len_from_mask(
    const torch::Tensor& encoder_hidden_states,
    const std::optional<torch::Tensor>& encoder_hidden_states_mask) {
  int64_t batch_size = encoder_hidden_states.size(0);
  int64_t text_seq_len = encoder_hidden_states.size(1);

  if (!encoder_hidden_states_mask.has_value()) {
    return std::make_tuple(text_seq_len, std::nullopt, std::nullopt);
  }

  auto mask =
      encoder_hidden_states_mask.value().to(encoder_hidden_states.device());

  if (mask.size(0) != batch_size || mask.size(1) != text_seq_len) {
    LOG(ERROR) << "`encoder_hidden_states_mask` shape " << mask.sizes()
               << " must match (batch_size, text_seq_len)=(" << batch_size
               << ", " << text_seq_len << ").";
  }

  if (mask.dtype() != torch::kBool) {
    mask = mask.to(torch::kBool);
  }

  auto device = encoder_hidden_states.device();
  auto position_ids = torch::arange(
      text_seq_len, torch::TensorOptions().device(device).dtype(torch::kLong));

  // Compute active positions (use position ID where mask is True, else 0)
  auto zero_tensor = torch::zeros(
      {}, torch::TensorOptions().device(device).dtype(torch::kLong));

  auto active_positions = torch::where(mask, position_ids, zero_tensor);

  // Check which samples have active positions
  auto has_active = mask.any(/*dim=*/1);

  // Compute per-sample length: max position + 1 if active, else use full length
  auto max_positions = std::get<0>(active_positions.max(/*dim=*/1));
  auto per_sample_len = torch::where(
      has_active,
      max_positions + 1,
      torch::tensor(text_seq_len,
                    torch::TensorOptions().device(device).dtype(torch::kLong)));

  return std::make_tuple(text_seq_len, per_sample_len, mask);
}

class QwenTimestepProjEmbeddingsImpl final : public torch::nn::Module {
 public:
  QwenTimestepProjEmbeddingsImpl(const ModelContext& context,
                                 int64_t embedding_dim,
                                 bool use_additional_t_cond = false)
      : use_additional_t_cond_(use_additional_t_cond) {
    time_proj_ =
        register_module("time_proj",
                        qwenimage::Timesteps(context,
                                             /*num_channels=*/256,
                                             /*flip_sin_to_cos=*/true,
                                             /*downscale_freq_shift=*/0.0,
                                             /*scale=*/1000));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        qwenimage::TimestepEmbedding(context,
                                     /*in_channels=*/256,
                                     /*time_embed_dim*/ embedding_dim));
    if (use_additional_t_cond) {
      addition_t_embedding_ =
          register_module("addition_t_embedding",
                          torch::nn::Embedding(torch::nn::EmbeddingOptions(
                              /*num=*/2, embedding_dim)));
    }
  }

  torch::Tensor forward(
      const torch::Tensor& timestep,
      const torch::Tensor& hidden_states,
      const torch::Tensor& addition_t_cond = torch::Tensor()) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj.to(hidden_states.dtype()));

    torch::Tensor conditioning = timesteps_emb;
    if (use_additional_t_cond_) {
      CHECK(addition_t_cond.defined())
          << "expected to pass addition_t_cond when"
          << " use_additional_t_cond_ is setup to true";
      auto addition_t_emb = addition_t_embedding_->forward(addition_t_cond);
      addition_t_emb = addition_t_emb.to(hidden_states.dtype());
      conditioning = conditioning + addition_t_emb;
    }

    return conditioning;
  }
  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    if (use_additional_t_cond_) {
      weight::load_weight(state_dict,
                          "addition_t_embedding.weight",
                          addition_t_embedding_->weight,
                          weight_is_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    if (use_additional_t_cond_) {
      CHECK(weight_is_loaded_)
          << "weight is not loaded for " << prefix + "weight";
    }
  }

 private:
  qwenimage::Timesteps time_proj_{nullptr};
  qwenimage::TimestepEmbedding timestep_embedder_{nullptr};
  torch::nn::Embedding addition_t_embedding_{nullptr};
  bool use_additional_t_cond_;
  bool weight_is_loaded_{false};
};
TORCH_MODULE(QwenTimestepProjEmbeddings);

class QwenEmbedRopeImpl : public torch::nn::Module {
 public:
  QwenEmbedRopeImpl(const ModelContext& context,
                    int64_t theta,
                    std::vector<int64_t> axes_dim,
                    bool scale_rope = false)
      : theta_(theta), axes_dim_(axes_dim), scale_rope_(scale_rope) {
    auto pos_index = torch::arange(4096);
    auto neg_index = torch::arange(4096).flip(0) * -1 - 1;

    pos_freqs_ = torch::cat({rope_params(pos_index, axes_dim[0], theta),
                             rope_params(pos_index, axes_dim[1], theta),
                             rope_params(pos_index, axes_dim[2], theta)},
                            1);

    neg_freqs_ = torch::cat({rope_params(neg_index, axes_dim[0], theta),
                             rope_params(neg_index, axes_dim[1], theta),
                             rope_params(neg_index, axes_dim[2], theta)},
                            1);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      const std::optional<int64_t>& txt_seq_lens,
      torch::Device device,
      const std::optional<int64_t>& max_txt_seq_len) {
    if (pos_freqs_.device() != device) {
      pos_freqs_ = pos_freqs_.to(device);
      neg_freqs_ = neg_freqs_.to(device);
    }

    std::vector<torch::Tensor> vid_freqs;
    vid_freqs.reserve(video_fhw.size());
    int64_t max_vid_index = 0;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const auto& fhw = video_fhw[idx];
      int64_t frame = fhw[0], height = fhw[1], width = fhw[2];

      std::string rope_key = std::to_string(idx) + "_" +
                             std::to_string(height) + "_" +
                             std::to_string(width);

      auto video_freq = _compute_video_freqs(frame, height, width, idx, device);
      vid_freqs.emplace_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_len;
    if (txt_seq_lens.has_value() && !max_txt_seq_len.has_value()) {
      max_len = txt_seq_lens.value();
    } else if (max_txt_seq_len.has_value()) {
      max_len = max_txt_seq_len.value();
    } else {
      LOG(FATAL) << "need to pass txt_seq_lens or max_txt_seq_len "
                 << "to calculate the mrope";
    }

    auto txt_freqs =
        pos_freqs_.slice(0, max_vid_index, max_vid_index + max_len);
    auto vid_freqs_cat = torch::cat(vid_freqs, 0);
    return std::make_tuple(vid_freqs_cat, txt_freqs);
  }

 protected:
  torch::Tensor rope_params(const torch::Tensor& index,
                            int64_t dim,
                            int64_t theta) {
    CHECK(dim % 2 == 0) << "dim must be even";

    auto exponents =
        torch::arange(
            0, dim, 2, torch::TensorOptions().dtype(torch::kFloat32)) /
        static_cast<float>(dim);
    auto freqs = 1.0 / torch::pow(theta, exponents);

    auto outer_result = torch::outer(index.to(torch::kFloat32), freqs);

    auto complex_freqs =
        torch::polar(torch::ones_like(outer_result), outer_result);

    return complex_freqs;
  }

  torch::Tensor _compute_video_freqs(int64_t frame,
                                     int64_t height,
                                     int64_t width,
                                     int64_t idx,
                                     torch::Device device) {
    int64_t seq_lens = frame * height * width;

    auto pos_freqs = pos_freqs_.to(device);
    auto neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (auto dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos_chunks = pos_freqs_.split_with_sizes(split_sizes, 1);
    auto freqs_neg_chunks = neg_freqs_.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_pos_chunks[0]
                           .slice(0, idx, idx + frame)
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height, freqs_width;
    if (scale_rope_) {
      auto height_neg_part = freqs_neg_chunks[1].slice(
          0, -(height - height / 2), torch::indexing::None);
      auto height_pos_part = freqs_pos_chunks[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({height_neg_part, height_pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});

      auto width_neg_part = freqs_neg_chunks[2].slice(
          0, -(width - width / 2), torch::indexing::None);
      auto width_pos_part = freqs_pos_chunks[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({width_neg_part, width_pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos_chunks[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});

      freqs_width = freqs_pos_chunks[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs = torch::cat({freqs_frame, freqs_height, freqs_width}, -1)
                     .reshape({seq_lens, -1});
    return freqs.contiguous();
  }

  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  bool scale_rope_;
  torch::Tensor pos_freqs_;
  torch::Tensor neg_freqs_;
  std::unordered_map<std::string, torch::Tensor> rope_cache_;
};

TORCH_MODULE(QwenEmbedRope);

class QwenEmbedRopeWithCacheImpl : public QwenEmbedRopeImpl {
 public:
  QwenEmbedRopeWithCacheImpl(const ModelContext& context,
                             int64_t theta,
                             std::vector<int64_t> axes_dim,
                             bool scale_rope = false)
      : QwenEmbedRopeImpl(context, theta, axes_dim, scale_rope) {}

 private:
  torch::Tensor _compute_video_freqs_cached(int64_t frame,
                                            int64_t height,
                                            int64_t width,
                                            int64_t idx,
                                            torch::Device device) {
    std::string key = std::to_string(idx) + "_" + std::to_string(height) + "_" +
                      std::to_string(width);

    auto it = rope_cache_.find(key);
    if (it != rope_cache_.end()) {
      return it->second;
    } else {
      auto result = _compute_video_freqs(frame, height, width, idx, device);
      rope_cache_[key] = result;
      return result;
    }
  }

  std::unordered_map<std::string, torch::Tensor> rope_cache_;
};
TORCH_MODULE(QwenEmbedRopeWithCache);

class QwenEmbedLayer3DRopeImpl : public torch::nn::Module {
 public:
  QwenEmbedLayer3DRopeImpl(const ModelContext& context,
                           int64_t theta,
                           std::vector<int64_t>& axes_dim,
                           bool scale_rope = false)
      : theta_(theta), axes_dim_(axes_dim), scale_rope_(scale_rope) {
    auto pos_index = torch::arange(4096);
    auto neg_index = torch::arange(4096).flip(0) * -1 - 1;

    std::vector<torch::Tensor> pos_freqs_parts;
    pos_freqs_ = torch::cat({rope_params(pos_index, axes_dim[0], theta),
                             rope_params(pos_index, axes_dim[1], theta),
                             rope_params(pos_index, axes_dim[2], theta)},
                            1);

    neg_freqs_ = torch::cat({rope_params(neg_index, axes_dim[0], theta),
                             rope_params(neg_index, axes_dim[1], theta),
                             rope_params(neg_index, axes_dim[2], theta)},
                            1);
  }

  virtual std::pair<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      int64_t max_txt_seq_len,
      torch::Device device = torch::Device(torch::kCPU)) {
    std::vector<torch::Tensor> vid_freqs_list;
    int64_t max_vid_index = 0;
    int64_t layer_num = video_fhw.size() - 1;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const std::vector<int64_t>& fhw = video_fhw[idx];

      int64_t frame = fhw[0];
      int64_t height = fhw[1];
      int64_t width = fhw[2];

      torch::Tensor video_freq;

      if (idx != layer_num) {
        video_freq = _compute_video_freqs(frame, height, width, idx, device);
      } else {
        video_freq = _compute_condition_freqs(frame, height, width, device);
      }
      vid_freqs_list.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_txt_seq_len_int = std::max(max_vid_index, layer_num);

    torch::Tensor txt_freqs = pos_freqs_.to(device).slice(
        0, max_vid_index, max_vid_index + max_txt_seq_len_int);

    torch::Tensor vid_freqs = torch::cat(vid_freqs_list, 0);

    return {vid_freqs, txt_freqs};
  }

 protected:
  torch::Tensor rope_params(torch::Tensor index, int64_t dim, int64_t theta) {
    CHECK(dim % 2 == 0) << "dim must be even";

    auto exponents =
        torch::arange(
            0, dim, 2, torch::TensorOptions().dtype(torch::kFloat32)) /
        static_cast<float>(dim);
    auto freqs = 1.0 / torch::pow(theta, exponents);

    auto outer_result = torch::outer(index.to(torch::kFloat32), freqs);

    auto complex_freqs =
        torch::polar(torch::ones_like(outer_result), outer_result);

    return complex_freqs;
  }

  torch::Tensor _compute_video_freqs(int64_t frame,
                                     int64_t height,
                                     int64_t width,
                                     int64_t idx,
                                     torch::Device device) {
    int64_t seq_lens = frame * height * width;

    torch::Tensor pos_freqs = pos_freqs_.to(device);
    torch::Tensor neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (int64_t dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos = pos_freqs.split_with_sizes(split_sizes, 1);
    auto freqs_neg = neg_freqs.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_pos[0]
                           .slice(0, idx, idx + frame)
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height;
    if (scale_rope_) {
      auto height_neg_part =
          freqs_neg[1].slice(0, -(height / 2), freqs_neg[1].size(0));
      auto height_pos_part = freqs_pos[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({height_neg_part, height_pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    }

    torch::Tensor freqs_width;
    if (scale_rope_) {
      auto neg_part = freqs_neg[2].slice(0, -(width / 2), freqs_neg[2].size(0));
      auto pos_part = freqs_pos[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({neg_part, pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_width = freqs_pos[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs =
        torch::cat({freqs_frame, freqs_height, freqs_width}, /*dim=*/-1)
            .reshape({seq_lens, -1})
            .clone()
            .contiguous();

    return freqs;
  }

  torch::Tensor _compute_condition_freqs(int64_t frame,
                                         int64_t height,
                                         int64_t width,
                                         torch::Device device) {
    int64_t seq_lens = frame * height * width;

    torch::Tensor pos_freqs = pos_freqs_.to(device);
    torch::Tensor neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (int64_t dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos = pos_freqs.split_with_sizes(split_sizes, 1);
    auto freqs_neg = neg_freqs.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_neg[0]
                           .slice(0, -1, freqs_neg[0].size(0))
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height;
    if (scale_rope_) {
      auto neg_part =
          freqs_neg[1].slice(0, -(height / 2), freqs_neg[1].size(0));
      auto pos_part = freqs_pos[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({neg_part, pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    }
    torch::Tensor freqs_width;
    if (scale_rope_) {
      auto neg_part = freqs_neg[2].slice(0, -(width / 2), freqs_neg[2].size(0));
      auto pos_part = freqs_pos[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({neg_part, pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_width = freqs_pos[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs = torch::cat({freqs_frame, freqs_height, freqs_width}, -1)
                     .reshape({seq_lens, -1})
                     .clone()
                     .contiguous();

    return freqs;
  }

  int64_t theta_;
  std::vector<int64_t>& axes_dim_;
  bool scale_rope_;
  torch::Tensor pos_freqs_;
  torch::Tensor neg_freqs_;
};

TORCH_MODULE(QwenEmbedLayer3DRope);

class QwenEmbedLayer3DRopeWithCacheImpl : public QwenEmbedLayer3DRopeImpl {
 public:
  QwenEmbedLayer3DRopeWithCacheImpl(const ModelContext& context,
                                    int64_t theta,
                                    std::vector<int64_t>& axes_dim,
                                    bool scale_rope = false)
      : QwenEmbedLayer3DRopeImpl(context, theta, axes_dim, scale_rope) {}

  std::pair<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      int64_t max_txt_seq_len,
      torch::Device device = torch::Device(torch::kCPU)) override {
    std::vector<torch::Tensor> vid_freqs_list;
    int64_t max_vid_index = 0;
    int64_t layer_num = video_fhw.size() - 1;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const std::vector<int64_t>& fhw = video_fhw[idx];

      int64_t frame = fhw[0];
      int64_t height = fhw[1];
      int64_t width = fhw[2];

      torch::Tensor video_freq;

      if (idx != layer_num) {
        video_freq =
            _compute_video_freqs_with_cache(frame, height, width, idx, device);
      } else {
        video_freq =
            _compute_condition_freqs_with_cache(frame, height, width, device);
      }
      vid_freqs_list.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_txt_seq_len_int = std::max(max_vid_index, layer_num);

    torch::Tensor txt_freqs = pos_freqs_.to(device).slice(
        0, max_vid_index, max_vid_index + max_txt_seq_len_int);

    torch::Tensor vid_freqs = torch::cat(vid_freqs_list, 0);

    return {vid_freqs, txt_freqs};
  }

 private:
  torch::Tensor _compute_video_freqs_with_cache(int64_t frame,
                                                int64_t height,
                                                int64_t width,
                                                int64_t idx,
                                                torch::Device device) {
    std::string key = std::to_string(frame) + "_" + std::to_string(idx) + "_" +
                      std::to_string(height) + "_" + std::to_string(width);

    // TODO: currently the freqs tensors are cached on device
    // need to check whether to swap them to cpu to save device memory
    auto it = video_freqs_cache_.find(key);
    if (it != video_freqs_cache_.end()) {
      return it->second.clone().contiguous();
    } else {
      auto result = _compute_video_freqs(frame, height, width, idx, device);
      video_freqs_cache_[key] = result.clone();
      return result;
    }
  }

  torch::Tensor _compute_condition_freqs_with_cache(int64_t frame,
                                                    int64_t height,
                                                    int64_t width,
                                                    torch::Device device) {
    std::string key = std::to_string(frame) + "_" + std::to_string(height) +
                      "_" + std::to_string(width);

    // TODO: currently the freqs tensors are cached on device
    // need to check whether to swap them to cpu to save device memory
    auto it = condition_cache_.find(key);
    if (it != condition_cache_.end()) {
      return it->second.clone().contiguous();
    } else {
      auto result = _compute_condition_freqs(frame, height, width, device);
      condition_cache_[key] = result.clone();
      return result;
    }
  }

  std::unordered_map<std::string, torch::Tensor> video_freqs_cache_;
  std::unordered_map<std::string, torch::Tensor> condition_cache_;
};

TORCH_MODULE(QwenEmbedLayer3DRopeWithCache);

// Implementation of attention forward
class QwenDoubleStreamAttnProcessor2_0Impl : public torch::nn::Module {
 public:
  QwenDoubleStreamAttnProcessor2_0Impl(qwenimage::Attention&& attn_module,
                                       const ParallelArgs& parallel_args)
      : parallel_args_(parallel_args) {
    attn_ = register_module("attn", std::move(attn_module));
  }

  virtual std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,          // Image stream
      const torch::Tensor& encoder_hidden_states,  // Text stream
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      const torch::Tensor& attention_mask = torch::Tensor(),
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {}) {
    // int64_t seq_txt = encoder_hidden_states.size(1);
    // int64_t seq_img = hidden_states.size(1);
    //  Compute QKV for image stream (sample projections)
    auto img_query = attn_->to_q_->forward(hidden_states);
    auto img_key = attn_->to_k_->forward(hidden_states);
    auto img_value = attn_->to_v_->forward(hidden_states);

    // Compute QKV for text stream (context projections)
    auto txt_query = attn_->add_q_proj_->forward(encoder_hidden_states);
    auto txt_key = attn_->add_k_proj_->forward(encoder_hidden_states);
    auto txt_value = attn_->add_v_proj_->forward(encoder_hidden_states);

    // Reshape for multi-head attention
    int64_t heads = attn_->heads_;
    auto reshape_dims = std::vector<int64_t>{
        heads / ::xllm::ParallelConfig::get_instance().sp_size(), -1};

    img_query = img_query.unflatten(-1, reshape_dims);
    img_key = img_key.unflatten(-1, reshape_dims);
    img_value = img_value.unflatten(-1, reshape_dims);
    txt_query = txt_query.unflatten(-1, reshape_dims);
    txt_key = txt_key.unflatten(-1, reshape_dims);
    txt_value = txt_value.unflatten(-1, reshape_dims);
    // Apply QK normalization
    if (attn_->norm_q_) {
      img_query = attn_->norm_q_->forward(img_query);
    }
    if (attn_->norm_k_) {
      img_key = attn_->norm_k_->forward(img_key);
    }
    if (attn_->norm_added_q_) {
      txt_query = attn_->norm_added_q_->forward(txt_query);
    }
    if (attn_->norm_added_k_) {
      txt_key = attn_->norm_added_k_->forward(txt_key);
    }

    // Apply RoPE if provided
    auto img_freqs = std::get<0>(image_rotary_emb);
    auto txt_freqs = std::get<1>(image_rotary_emb);

    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_query, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_key, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_value, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);

    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_query, /*tensor_name=*/"hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_key, /*tensor_name=*/"hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_value, /*tensor_name=*/"hidden_states", /*dim=*/1);

    img_query = apply_rotary_emb_qwen(img_query, img_freqs, false);
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, false);
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, false);
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, false);

    // Concatenate for joint attention - Order: [text, image]
    auto joint_query = torch::cat({txt_query, img_query}, 1);
    auto joint_key = torch::cat({txt_key, img_key}, 1);
    auto joint_value = torch::cat({txt_value, img_value}, 1);

#if defined(USE_NPU)
    auto results = at_npu::native::custom_ops::npu_fusion_attention(
        joint_query,
        joint_key,
        joint_value,
        heads / ::xllm::ParallelConfig::get_instance().sp_size(),
        /*input_layout=*/"BSND",
        /*pse=*/torch::nullopt,
        /*padding_mask=*/torch::nullopt,
        /*atten_mask*/ torch::nullopt,
        /*scale=*/pow(joint_query.size(3), -0.5),
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);

    auto joint_hidden_states = std::get<0>(results);
#else
    auto joint_hidden_states =
        qwen_image_joint_attention(joint_query, joint_key, joint_value);
#endif
    // Reshape back
    joint_hidden_states = joint_hidden_states.flatten(2, 3);
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype());

    int64_t seq_txt = txt_query.size(1);
    int64_t seq_img = img_query.size(1);
    // Split attention outputs back
    auto chunks = torch::split(joint_hidden_states, {seq_txt, seq_img}, 1);
    auto txt_attn_output = chunks[0];
    auto img_attn_output = chunks[1];

    txt_attn_output =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            txt_attn_output,
            /*tensor_name=*/"encoder_hidden_states",
            /*dim=*/1);

    img_attn_output =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            img_attn_output, /*tensor_name=*/"hidden_states", /*dim=*/1);

    // Apply output projections
    img_attn_output = attn_->to_out_->forward(img_attn_output);

    txt_attn_output = attn_->to_add_out_->forward(txt_attn_output);
    return std::make_tuple(img_attn_output, txt_attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix);
  }

 protected:
  qwenimage::Attention attn_{nullptr};
  ParallelArgs parallel_args_;
};
TORCH_MODULE(QwenDoubleStreamAttnProcessor2_0);

class QwenDoubleStreamAttnProcessorCMO2_0Impl : public torch::nn::Module {
 public:
  QwenDoubleStreamAttnProcessorCMO2_0Impl(qwenimage::Attention&& attn_module,
                                          const ParallelArgs& parallel_args)
      : parallel_args_(parallel_args) {
    attn_ = register_module("attn", std::move(attn_module));
    q_heads_ = attn_->heads_;
    kv_heads_ = attn_->kv_heads_;
    dim_head_ = attn_->dim_head_;
    q_hidden_size_ = q_heads_ * dim_head_;
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      const torch::Tensor& attention_mask = torch::Tensor(),
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {}) {
    const auto sp_size = ::xllm::ParallelConfig::get_instance().sp_size();
    CHECK_GT(sp_size, 1) << "CMO attention requires sequence parallelism";

    auto img_query = attn_->to_q_->forward(hidden_states);
    auto img_query_handler = parallel_state::all_to_all_4D(
        img_query.view({hidden_states.size(0), -1, q_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto img_key = attn_->to_k_->forward(hidden_states);
    auto img_key_handler = parallel_state::all_to_all_4D(
        img_key.view({hidden_states.size(0), -1, kv_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto img_value = attn_->to_v_->forward(hidden_states);
    auto img_value_handler = parallel_state::all_to_all_4D(
        img_value.view({hidden_states.size(0), -1, kv_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto txt_query = attn_->add_q_proj_->forward(encoder_hidden_states);
    auto txt_query_handler = parallel_state::all_to_all_4D(
        txt_query.view(
            {encoder_hidden_states.size(0), -1, q_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto txt_key = attn_->add_k_proj_->forward(encoder_hidden_states);
    auto txt_key_handler = parallel_state::all_to_all_4D(
        txt_key.view({encoder_hidden_states.size(0), -1, kv_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto txt_value = attn_->add_v_proj_->forward(encoder_hidden_states);
    auto txt_value_handler = parallel_state::all_to_all_4D(
        txt_value.view(
            {encoder_hidden_states.size(0), -1, kv_heads_, dim_head_}),
        /*scatter_idx=*/2,
        /*gather_idx=*/1,
        /*async_ops=*/true,
        attn_->sp_group_);

    img_query = img_query_handler();
    img_key = img_key_handler();
    txt_query = txt_query_handler();
    txt_key = txt_key_handler();

    if (attn_->norm_q_) {
      img_query = attn_->norm_q_->forward(img_query);
    }
    if (attn_->norm_k_) {
      img_key = attn_->norm_k_->forward(img_key);
    }
    if (attn_->norm_added_q_) {
      txt_query = attn_->norm_added_q_->forward(txt_query);
    }
    if (attn_->norm_added_k_) {
      txt_key = attn_->norm_added_k_->forward(txt_key);
    }

    auto img_freqs = std::get<0>(image_rotary_emb);
    auto txt_freqs = std::get<1>(image_rotary_emb);

    img_value = img_value_handler();
    txt_value = txt_value_handler();

    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_query, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_key, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        txt_value, /*tensor_name=*/"encoder_hidden_states", /*dim=*/1);

    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_query, /*tensor_name=*/"hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_key, /*tensor_name=*/"hidden_states", /*dim=*/1);
    xllm::dit::SequenceParallelPadManager::get_instance().unpad_tensor(
        img_value, /*tensor_name=*/"hidden_states", /*dim=*/1);

    img_query = apply_rotary_emb_qwen(img_query, img_freqs, false);
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, false);
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, false);
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, false);

    auto joint_query = torch::cat({txt_query, img_query}, 1);
    auto joint_key = torch::cat({txt_key, img_key}, 1);
    auto joint_value = torch::cat({txt_value, img_value}, 1);

#if defined(USE_NPU)
    auto results = at_npu::native::custom_ops::npu_fusion_attention(
        joint_query,
        joint_key,
        joint_value,
        q_heads_ / sp_size,
        /*input_layout=*/"BSND",
        /*pse=*/torch::nullopt,
        /*padding_mask=*/torch::nullopt,
        /*atten_mask*/ torch::nullopt,
        /*scale=*/pow(joint_query.size(3), -0.5),
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);

    auto joint_hidden_states = std::get<0>(results);
#else
    auto joint_hidden_states =
        qwen_image_joint_attention(joint_query, joint_key, joint_value);
#endif
    joint_hidden_states = joint_hidden_states.flatten(2, 3);
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype());

    int64_t seq_txt = txt_query.size(1);
    int64_t seq_img = img_query.size(1);
    auto chunks = torch::split(joint_hidden_states, {seq_txt, seq_img}, 1);
    auto txt_attn_output = chunks[0];
    auto img_attn_output = chunks[1];

    txt_attn_output =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            txt_attn_output,
            /*tensor_name=*/"encoder_hidden_states",
            /*dim=*/1);

    img_attn_output =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            img_attn_output, /*tensor_name=*/"hidden_states", /*dim=*/1);

    auto img_out_handler = parallel_state::all_to_all_4D(
        img_attn_output.view(
            {hidden_states.size(0), -1, q_heads_ / sp_size, dim_head_}),
        /*scatter_idx=*/1,
        /*gather_idx=*/2,
        /*async_ops=*/true,
        attn_->sp_group_);

    auto txt_out_handler = parallel_state::all_to_all_4D(
        txt_attn_output.view(
            {encoder_hidden_states.size(0), -1, q_heads_ / sp_size, dim_head_}),
        /*scatter_idx=*/1,
        /*gather_idx=*/2,
        /*async_ops=*/true,
        attn_->sp_group_);

    img_attn_output = img_out_handler();
    img_attn_output =
        img_attn_output.view({hidden_states.size(0), -1, q_hidden_size_});
    img_attn_output = attn_->to_out_->forward(img_attn_output);

    txt_attn_output = txt_out_handler();
    txt_attn_output = txt_attn_output.view(
        {encoder_hidden_states.size(0), -1, q_hidden_size_});
    txt_attn_output = attn_->to_add_out_->forward(txt_attn_output);

    return std::make_tuple(img_attn_output, txt_attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix);
  }

 private:
  qwenimage::Attention attn_{nullptr};
  const ParallelArgs parallel_args_;
  int64_t q_heads_ = 0;
  int64_t kv_heads_ = 0;
  int64_t dim_head_ = 0;
  int64_t q_hidden_size_ = 0;
};
TORCH_MODULE(QwenDoubleStreamAttnProcessorCMO2_0);

bool ADALN_FUSE = true;

class QwenImageTransformerBlockImpl : public torch::nn::Module {
 public:
  QwenImageTransformerBlockImpl(const ModelContext& context,
                                int64_t dim,
                                int64_t num_attention_heads,
                                int64_t attention_head_dim,
                                const ParallelArgs& parallel_args,
                                bool zero_cond_t = false,
                                const std::string& qk_norm = "rms_norm",
                                double eps = 1e-6)
      : options_(context.get_tensor_options()),
        zero_cond_t_(zero_cond_t),
        parallel_args_(parallel_args) {
    quant_args_ = context.get_quant_args();
    // Image processing modules
    img_mod_ = register_module(
        "img_mod",
        torch::nn::Sequential(torch::nn::SiLU(),
                              layer::AddMatmulWeightTransposed(
                                  dim, 6 * dim, true, options_, quant_args_)));

    // Image normalization
    img_norm1_ = register_module("img_norm1", AdaLayerNorm(context, dim, eps));
    // Attention module
    auto attn_ =
        qwenimage::Attention(context,
                             /*query_dim=*/dim,
                             /*cross_attention_dim=*/std::nullopt,
                             /*heads=*/num_attention_heads,
                             /*kv_heads=*/std::nullopt,
                             /*dim_head=*/attention_head_dim,
                             /*drop_out=*/0.0,
                             /*bias=*/true,
                             /*qk_norm=*/qk_norm,
                             /*cross_attention_norm=*/"",
                             /*added_kv_proj_dim=*/dim,
                             /*added_proj_bias*/ true,
                             /*out_bias*/ true,
                             /*scale_qk*/ true,
                             /*only_cross_attention=*/false,
                             eps,
                             /*rescale_output_factor=*/1.0,
                             /*residual_connection=*/false,
                             /*out_dim=*/dim,
                             /*out_context_dim=*/std::nullopt,
                             /*context_pre_only=*/true,
                             /*pre_only=*/false,
                             /*elementwise_affine=*/true,
                             /*is_causal=*/false,
                             /*sp_group=*/parallel_args_.dit_sp_group_);
    if (use_dit_sp_communication_overlap()) {
      attn_cmo_processor_ =
          register_module("attn_processor_",
                          QwenDoubleStreamAttnProcessorCMO2_0(std::move(attn_),
                                                              parallel_args_));
    } else {
      attn_processor_ = register_module(
          "attn_processor_",
          QwenDoubleStreamAttnProcessor2_0(std::move(attn_), parallel_args_));
    }
    // Image normalization 2
    img_norm2_ = register_module("img_norm2", AdaLayerNorm(context, dim, eps));

    // Image MLP
    img_mlp_ =
        register_module("img_mlp", qwenimage::FeedForward(context, dim, dim));

    // Text processing modules
    txt_mod_ = register_module(
        "txt_mod",
        torch::nn::Sequential(torch::nn::SiLU(),
                              layer::AddMatmulWeightTransposed(
                                  dim, 6 * dim, true, options_, quant_args_)));

    // Text normalization 1
    txt_norm1_ = register_module("txt_norm1", AdaLayerNorm(context, dim, eps));

    // Text normalization 2
    txt_norm2_ = register_module("txt_norm2", AdaLayerNorm(context, dim, eps));

    // Text MLP
    txt_mlp_ =
        register_module("txt_mlp", qwenimage::FeedForward(context, dim, dim));
  }

  std::pair<torch::Tensor, torch::Tensor> _modulate(
      const torch::Tensor& x,
      const torch::Tensor& mod_params,
      const torch::Tensor& index = torch::Tensor()) {
    // x: b l d, shift: b d, scale: b d, gate: b d
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];

    torch::Tensor shift_result, scale_result, gate_result;

    if (index.defined()) {
      // Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
      // So shift, scale, gate have shape [2*actual_batch, d]
      int64_t actual_batch = shift.size(0) / 2;

      // Split into two parts
      auto shift_0 = shift.slice(0, 0, actual_batch);
      auto shift_1 = shift.slice(0, actual_batch, shift.size(0));

      auto scale_0 = scale.slice(0, 0, actual_batch);
      auto scale_1 = scale.slice(0, actual_batch, scale.size(0));

      auto gate_0 = gate.slice(0, 0, actual_batch);
      auto gate_1 = gate.slice(0, actual_batch, gate.size(0));

      // index: [b, l] where b is actual batch size
      // Expand to [b, l, 1] to match feature dimension
      auto index_expanded = index.unsqueeze(-1);  // [b, l, 1]

      // Expand chunks to [b, 1, d] then broadcast to [b, l, d]
      auto shift_0_exp = shift_0.unsqueeze(1);  // [b, 1, d]
      auto shift_1_exp = shift_1.unsqueeze(1);  // [b, 1, d]
      auto scale_0_exp = scale_0.unsqueeze(1);
      auto scale_1_exp = scale_1.unsqueeze(1);
      auto gate_0_exp = gate_0.unsqueeze(1);
      auto gate_1_exp = gate_1.unsqueeze(1);

      // Use torch::where to select based on index
      shift_result =
          torch::where(index_expanded == 0, shift_0_exp, shift_1_exp);
      scale_result =
          torch::where(index_expanded == 0, scale_0_exp, scale_1_exp);
      gate_result = torch::where(index_expanded == 0, gate_0_exp, gate_1_exp);
    } else {
      shift_result = shift.unsqueeze(1);
      scale_result = scale.unsqueeze(1);
      gate_result = gate.unsqueeze(1);
    }

    // Apply modulation: x * (1 + scale_result) + shift_result
    auto modulated_x = x * (1 + scale_result) + shift_result;

    return {modulated_x, gate_result};
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_mask,
      const torch::Tensor& temb,
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {},
      const std::unordered_map<std::string, torch::Tensor>&
          joint_attention_kwargs = {},
      const torch::Tensor& modulate_index = torch::Tensor()) {
    // Get modulation parameters for both streams
    auto img_mod_params = img_mod_->forward(temb);  // [B, 6*dim]
    torch::Tensor new_temb;
    if (zero_cond_t_) {
      new_temb = temb.chunk(2, 0)[0];
    } else {
      new_temb = temb;
    }
    auto txt_mod_params = txt_mod_->forward(new_temb);  // [B, 6*dim]
    //  Split modulation parameters for norm1 and norm2
    auto img_mod_chunks = img_mod_params.chunk(2, -1);
    auto img_mod1 = img_mod_chunks[0];  // [B, 3*dim]
    auto img_mod2 = img_mod_chunks[1];  // [B, 3*dim]

    auto txt_mod_chunks = txt_mod_params.chunk(2, -1);
    auto txt_mod1 = txt_mod_chunks[0];  // [B, 3*dim]
    auto txt_mod2 = txt_mod_chunks[1];  // [B, 3*dim]

    // Process image stream - norm1 + modulation
    torch::Tensor img_modulated, img_gate1;
    std::tie(img_modulated, img_gate1) =
        img_norm1_->forward(hidden_states, img_mod1, modulate_index);
    //  Process text stream - norm1 + modulation
    torch::Tensor txt_modulated, txt_gate1;
    std::tie(txt_modulated, txt_gate1) =
        txt_norm1_->forward(encoder_hidden_states, txt_mod1);

    std::tuple<torch::Tensor, torch::Tensor> attn_output;
    // Use QwenAttnProcessor2_0 for joint attention computation
    if (use_dit_sp_communication_overlap()) {
      attn_output = attn_cmo_processor_->forward(img_modulated,  // Image stream
                                                 txt_modulated,  // Text stream
                                                 encoder_hidden_states_mask,
                                                 torch::Tensor(),  // timestep
                                                 image_rotary_emb);
    } else {
      attn_output = attn_processor_->forward(img_modulated,  // Image stream
                                             txt_modulated,  // Text stream
                                             encoder_hidden_states_mask,
                                             torch::Tensor(),  // timestep
                                             image_rotary_emb);
    }

    // QwenAttnProcessor2_0 returns (img_output, txt_output)
    auto img_attn_output = std::get<0>(attn_output);
    auto txt_attn_output = std::get<1>(attn_output);

    //  Apply attention gates and add residual
    auto new_hidden_states = hidden_states + img_gate1 * img_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states + txt_gate1 * txt_attn_output;

    // Process image stream - norm2 + MLP
    torch::Tensor img_modulated2, img_gate2;
    std::tie(img_modulated2, img_gate2) =
        img_norm2_->forward(new_hidden_states, img_mod2, modulate_index);

    auto img_mlp_output = img_mlp_->forward(img_modulated2);
    new_hidden_states = new_hidden_states + img_gate2 * img_mlp_output;

    // Process text stream - norm2 + MLP
    torch::Tensor txt_modulated2, txt_gate2;
    std::tie(txt_modulated2, txt_gate2) =
        txt_norm2_->forward(new_encoder_hidden_states, txt_mod2);

    auto txt_mlp_output = txt_mlp_->forward(txt_modulated2);
    new_encoder_hidden_states =
        new_encoder_hidden_states + txt_gate2 * txt_mlp_output;

    //  Clip to prevent overflow for fp16
    if (new_encoder_hidden_states.dtype() == torch::kFloat16) {
      new_encoder_hidden_states =
          new_encoder_hidden_states.clamp(-65504, 65504);
    }
    if (new_hidden_states.dtype() == torch::kFloat16) {
      new_hidden_states = new_hidden_states.clamp(-65504, 65504);
    }

    return std::make_tuple(new_hidden_states, new_encoder_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    img_mod_[1]->as<layer::AddMatmulWeightTransposed>()->load_state_dict(
        state_dict.get_dict_with_prefix("img_mod.1."));
    img_mlp_->load_state_dict(state_dict.get_dict_with_prefix("img_mlp."));
    txt_mod_[1]->as<layer::AddMatmulWeightTransposed>()->load_state_dict(
        state_dict.get_dict_with_prefix("txt_mod.1."));
    txt_mlp_->load_state_dict(state_dict.get_dict_with_prefix("txt_mlp."));
    if (use_dit_sp_communication_overlap()) {
      attn_cmo_processor_->load_state_dict(
          state_dict.get_dict_with_prefix("attn."));
    } else {
      attn_processor_->load_state_dict(
          state_dict.get_dict_with_prefix("attn."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    img_mod_[1]->as<layer::AddMatmulWeightTransposed>()->verify_loaded_weights(
        prefix + "img_mod.1.");
    img_mlp_->verify_loaded_weights(prefix + "img_mlp.");
    txt_mod_[1]->as<layer::AddMatmulWeightTransposed>()->verify_loaded_weights(
        prefix + "txt_mod.1.");
    txt_mlp_->verify_loaded_weights(prefix + "txt_mlp.");
    if (use_dit_sp_communication_overlap()) {
      attn_cmo_processor_->verify_loaded_weights(prefix + "attn.");
    } else {
      attn_processor_->verify_loaded_weights(prefix + "attn.");
    }
  }

 private:
  torch::TensorOptions options_;
  QuantArgs quant_args_;
  torch::nn::Sequential img_mod_{nullptr};
  AdaLayerNorm img_norm1_{nullptr};
  AdaLayerNorm img_norm2_{nullptr};
  std::shared_ptr<qwenimage::Attention> attn_{nullptr};
  QwenDoubleStreamAttnProcessor2_0 attn_processor_{nullptr};
  QwenDoubleStreamAttnProcessorCMO2_0 attn_cmo_processor_{nullptr};
  qwenimage::FeedForward img_mlp_{nullptr};

  torch::nn::Sequential txt_mod_{nullptr};
  AdaLayerNorm txt_norm1_{nullptr};
  AdaLayerNorm txt_norm2_{nullptr};
  qwenimage::FeedForward txt_mlp_{nullptr};
  bool zero_cond_t_;
  ParallelArgs parallel_args_;
};

TORCH_MODULE(QwenImageTransformerBlock);

class QwenImageTransformer2DModelImpl : public torch::nn::Module {
 public:
  QwenImageTransformer2DModelImpl(const ModelContext& context,
                                  const ParallelArgs& parallel_args)
      : options_(context.get_tensor_options()), parallel_args_(parallel_args) {
    quant_args_ = context.get_quant_args();
    auto model_args = context.get_model_args();
    int64_t num_attention_heads = model_args.n_heads();
    int64_t attention_head_dim = model_args.head_dim();
    int64_t joint_attention_dim = model_args.joint_attention_dim();
    int64_t num_layers = model_args.num_layers();
    int64_t patch_size = model_args.mm_patch_size();
    int64_t in_channels = model_args.in_channels();
    int64_t out_channels = model_args.out_channels();
    bool zero_cond_t = model_args.zero_cond_t();
    bool use_additional_t_cond = model_args.use_additional_t_cond();
    use_layer3d_rope_ = model_args.use_layer3d_rope();

    out_channels = (out_channels > 0) ? out_channels : in_channels;
    auto inner_dim = num_attention_heads * attention_head_dim;

    // Time-text embedding
    time_text_embed_ = register_module(
        "time_text_embed",
        QwenTimestepProjEmbeddings(context, inner_dim, use_additional_t_cond));

    // Text normalization
    txt_norm_ = register_module(
        "txt_norm", qwenimage::RMSNorm(joint_attention_dim, 1e-6, true, false));

    // Input projections
    img_in_ = register_module(
        "img_in",
        layer::AddMatmulWeightTransposed(
            in_channels, inner_dim, true, options_, quant_args_));
    txt_in_ = register_module(
        "txt_in",
        layer::AddMatmulWeightTransposed(
            joint_attention_dim, inner_dim, true, options_, quant_args_));
    // Transformer blocks
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
      transformer_blocks_->push_back(
          QwenImageTransformerBlock(context,
                                    inner_dim,
                                    num_attention_heads,
                                    attention_head_dim,
                                    parallel_args_,
                                    zero_cond_t));
    }

    // Output layers
    norm_out_ =
        register_module("norm_out",
                        qwenimage::AdaLayerNormContinuous(
                            context, inner_dim, inner_dim, false, 1e-6));
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmulWeightTransposed(inner_dim,
                                         patch_size * patch_size * out_channels,
                                         true,
                                         options_,
                                         quant_args_));

    // Cache for conditional and unconditional
    cache_cond_ = false;
    cache_uncond_ = false;

    zero_cond_t_ = zero_cond_t;
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states = torch::Tensor(),
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      torch::Tensor timestep = torch::Tensor(),
      std::vector<std::vector<int64_t>> img_shapes = {},
      torch::Tensor txt_seq_lens = torch::Tensor(),
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {},
      bool use_cfg = false,
      int64_t step_idx = 0,
      torch::Tensor addition_t_cond = torch::Tensor(),
      torch::Tensor guidance = torch::Tensor(),
      const std::unordered_map<std::string, torch::Tensor>& attention_kwargs =
          {},
      const std::vector<torch::Tensor>& controlnet_block_samples = {}) {
    auto new_hidden_states = img_in_->forward(hidden_states);
    auto new_timestep = timestep.to(new_hidden_states.dtype());
    torch::Tensor modulate_index;
    if (zero_cond_t_) {
      new_timestep = torch::cat({new_timestep, new_timestep * 0}, /*dim=*/0);
      std::vector<torch::Tensor> modulate_index_list;
      for (size_t sample_index = 0; sample_index < 1; sample_index++) {
        auto zero_prods = torch::zeros({img_shapes[0][1] * img_shapes[0][2]},
                                       torch::TensorOptions()
                                           .device(new_timestep.device())
                                           .dtype(torch::kInt64));
        int64_t one_prods_size = 0;
        for (size_t index = 1; index < img_shapes.size(); index++) {
          one_prods_size += img_shapes[index][1] * img_shapes[index][2];
        }
        auto ones_prods = torch::ones({one_prods_size},
                                      torch::TensorOptions()
                                          .device(new_timestep.device())
                                          .dtype(torch::kInt64));
        modulate_index_list.emplace_back(
            torch::cat({zero_prods, ones_prods}, /*dim=*/0));
      }
      modulate_index = torch::stack(modulate_index_list, /*dim=*/0);
    } else {
      modulate_index = torch::Tensor();
    }

    // padding mask for sequence parallel scene
    auto padded_encoder_hidden_states_mask =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            encoder_hidden_states_mask,
            /*tensor_name=*/"encoder_hidden_states_mask",
            /*dim=*/1);

    auto new_encoder_hidden_states =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            encoder_hidden_states,
            /*tensor_name=*/"encoder_hidden_states",
            /*dim=*/1);

    new_hidden_states =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            new_hidden_states, /*tensor_name=*/"hidden_states", /*dim=*/1);

    modulate_index =
        xllm::dit::SequenceParallelPadManager::get_instance().pad_tensor(
            modulate_index, /*tensor_name=*/"modulate_index", /*dim=*/1);

    new_encoder_hidden_states = txt_norm_->forward(new_encoder_hidden_states);
    new_encoder_hidden_states = txt_in_->forward(new_encoder_hidden_states);

    // Use the encoder_hidden_states sequence length for RoPE computation and
    // normalize mask
    auto [text_seq_len, per_sample_len, new_encoder_hidden_states_mask] =
        compute_text_seq_len_from_mask(new_encoder_hidden_states,
                                       padded_encoder_hidden_states_mask);
    auto temb = time_text_embed_->forward(
        new_timestep, new_hidden_states, addition_t_cond);

    std::unordered_map<std::string, torch::Tensor> block_attention_kwargs;
    if (new_encoder_hidden_states_mask.has_value() &&
        new_encoder_hidden_states_mask.value().defined()) {
      int64_t batch_size = new_hidden_states.size(0);
      int64_t image_seq_len = new_hidden_states.size(1);
      auto image_mask = torch::ones({batch_size, image_seq_len},
                                    torch::TensorOptions()
                                        .device(new_hidden_states.device())
                                        .dtype(torch::kBool));
      auto joint_attention_mask = torch::cat(
          {new_encoder_hidden_states_mask.value(), image_mask}, /*dim=*/1);
      block_attention_kwargs["attention_mask"] = joint_attention_mask;
    }

    if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
      new_hidden_states = dit::sp_split_sequence(new_hidden_states,
                                                 /*dim=*/1,
                                                 parallel_args_.dit_sp_group_);
      new_encoder_hidden_states =
          dit::sp_split_sequence(new_encoder_hidden_states,
                                 /*dim=*/1,
                                 parallel_args_.dit_sp_group_);
      if (modulate_index.defined()) {
        modulate_index = dit::sp_split_sequence(modulate_index,
                                                /*dim=*/1,
                                                parallel_args_.dit_sp_group_);
      }
    }

    auto image_rot = std::get<0>(image_rotary_emb);
    auto txt_rot = std::get<1>(image_rotary_emb);

    bool use_step_cache = false;
    bool use_block_cache = false;

    torch::Tensor original_hidden_states = new_hidden_states;
    torch::Tensor original_encoder_hidden_states = new_encoder_hidden_states;
    // Step start: prepare inputs (hidden_states, original_hidden_states)
    TensorMap step_in_map = {
        {"hidden_states", new_hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_before(step_idx, step_in_map);
    use_step_cache =
        DiTCache::get_instance().on_before_step(stepin_before, use_cfg);

    if (!use_step_cache) {
      for (int64_t index_block = 0; index_block < transformer_blocks_->size();
           ++index_block) {
        TensorMap block_in_before_map = {};
        CacheBlockIn blockin_before(index_block, block_in_before_map);
        use_block_cache =
            DiTCache::get_instance().on_before_block(blockin_before, use_cfg);

        if (!use_block_cache) {
          std::tie(new_hidden_states, new_encoder_hidden_states) =
              transformer_blocks_[index_block]
                  ->as<QwenImageTransformerBlock>()
                  ->forward(new_hidden_states,
                            new_encoder_hidden_states,
                            /*encoder_hidden_states_mask=*/torch::Tensor(),
                            temb,
                            image_rotary_emb,
                            block_attention_kwargs,
                            modulate_index);
        }

        TensorMap block_in_after_map = {
            {"hidden_states", new_hidden_states},
            {"encoder_hidden_states", new_encoder_hidden_states},
            {"original_hidden_states", original_hidden_states},
            {"original_encoder_hidden_states", original_encoder_hidden_states}};
        CacheBlockIn blockin_after(index_block, block_in_after_map);
        CacheBlockOut blockout_after =
            DiTCache::get_instance().on_after_block(blockin_after, use_cfg);

        new_hidden_states = blockout_after.tensors.at("hidden_states");
        new_encoder_hidden_states =
            blockout_after.tensors.at("encoder_hidden_states");
      }
    }

    // Step end: update outputs (hidden_states, original_hidden_states)
    TensorMap step_after_map = {
        {"hidden_states", new_hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_after(step_idx, step_after_map);
    CacheStepOut stepout_after =
        DiTCache::get_instance().on_after_step(stepin_after, use_cfg);
    new_hidden_states = stepout_after.tensors.at("hidden_states");

    if (zero_cond_t_) {
      temb = temb.chunk(2, 0)[0];
    }

    new_hidden_states = norm_out_->forward(new_hidden_states, temb);
    new_hidden_states = proj_out_->forward(new_hidden_states);
    if (::xllm::ParallelConfig::get_instance().sp_size() > 1) {
      new_hidden_states = dit::sp_gather_sequence(
          new_hidden_states, /*dim=*/1, parallel_args_.dit_sp_group_);
    }
    return new_hidden_states;
  }

  void verify_loaded_weights(const std::string& prefix) {
    time_text_embed_->verify_loaded_weights(prefix + "time_text_embed.");
    txt_norm_->verify_loaded_weights(prefix + "txt_norm.");
    img_in_->verify_loaded_weights(prefix + "img_in.");
    txt_in_->verify_loaded_weights(prefix + "txt_in.");
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
    for (size_t i = 0; i < transformer_blocks_->size(); i++) {
      auto block_prefix = "transformer_blocks." + std::to_string(i) + ".";
      transformer_blocks_[i]
          ->as<QwenImageTransformerBlock>()
          ->verify_loaded_weights(prefix + block_prefix);
    }
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      time_text_embed_->load_state_dict(
          state_dict->get_dict_with_prefix("time_text_embed."));
      txt_norm_->load_state_dict(state_dict->get_dict_with_prefix("txt_norm."));

      img_in_->load_state_dict(state_dict->get_dict_with_prefix("img_in."));
      txt_in_->load_state_dict(state_dict->get_dict_with_prefix("txt_in."));

      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));

      for (size_t i = 0; i < transformer_blocks_->size(); i++) {
        auto prefix = "transformer_blocks." + std::to_string(i) + ".";
        transformer_blocks_[i]
            ->as<QwenImageTransformerBlock>()
            ->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      }
    }
    verify_loaded_weights("");
    LOG(INFO) << "qwen image vae model loaded successfully.";
  }

 private:
  torch::TensorOptions options_;
  QuantArgs quant_args_;
  QwenTimestepProjEmbeddings time_text_embed_{nullptr};
  qwenimage::RMSNorm txt_norm_{nullptr};
  layer::AddMatmulWeightTransposed img_in_{nullptr};
  layer::AddMatmulWeightTransposed txt_in_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  qwenimage::AdaLayerNormContinuous norm_out_{nullptr};
  layer::AddMatmulWeightTransposed proj_out_{nullptr};

  ParallelArgs parallel_args_;

  // Cache objects
  bool cache_cond_;
  bool cache_uncond_;

  bool zero_cond_t_;
  bool use_layer3d_rope_;
};

TORCH_MODULE(QwenImageTransformer2DModel);

REGISTER_MODEL_ARGS(QwenImageTransformer2DModel, [&] {
  // qwen-image 2509 params
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(in_channels, "in_channels", 64);
  LOAD_ARG_OR(out_channels, "out_channels", 16);
  LOAD_ARG_OR(num_layers, "num_layers", 60);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 24);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(joint_attention_dim, "joint_attention_dim", 3584);
  LOAD_ARG_OR(mm_patch_size, "patch_size", 2);
  LOAD_ARG_OR(guidance_embeds, "guidance_embeds", false);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));

  // qwen-image 2511 params
  LOAD_ARG_OR(zero_cond_t, "zero_cond_t", false);
  LOAD_ARG_OR(use_additional_t_cond, "use_additional_t_cond", false);
  LOAD_ARG_OR(use_layer3d_rope, "use_layer3d_rope", false);
});

}  // namespace xllm

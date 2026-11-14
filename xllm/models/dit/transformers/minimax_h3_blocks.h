/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "models/dit/utils/minimax_h3_packing.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif

namespace xllm {

struct MiniMaxH3C4Config {
  int64_t hidden_size = 5376;
  int64_t num_attention_heads = 56;
  int64_t attention_head_dim = 128;
  int64_t ffn_hidden_size = 14336;
  int64_t video_patch_dim = 96;
  int64_t audio_dim = 32;
  int64_t text_dim = 5120;
  int64_t timestep_input_dim = 256;
  int64_t time_embed_hidden_size = 5376;
  int64_t time_embed_dim = 2688;
  int64_t rope_inv_freq_len = 16;
  int64_t num_refiner_layers = 2;
  double norm_eps = 1e-5;
  double qk_norm_eps = 1e-5;
  double final_norm_eps = 1e-5;
};

struct MiniMaxH3ResidualBranchTrace {
  torch::Tensor adaln_parameters;
  torch::Tensor shift_msa;
  torch::Tensor scale_msa;
  torch::Tensor gate_msa;
  torch::Tensor shift_mlp;
  torch::Tensor scale_mlp;
  torch::Tensor gate_mlp;
  torch::Tensor norm1_output;
  torch::Tensor attention_input;
  torch::Tensor attention_output;
  torch::Tensor attention_delta;
  torch::Tensor norm2_output;
  torch::Tensor mlp_input;
  torch::Tensor mlp_output;
  torch::Tensor mlp_delta;
  torch::Tensor output;
};

struct MiniMaxH3FinalOutput {
  torch::Tensor activation;
  torch::Tensor all_video_logits;
  torch::Tensor all_audio_logits;
  torch::Tensor raw_selected_video_logits;
  torch::Tensor raw_selected_audio_logits;
  torch::Tensor selected_video_logits;
  torch::Tensor selected_audio_logits;
};

inline torch::Tensor minimax_h3_reorder_grouped_qkv(const torch::Tensor& weight,
                                                    int64_t num_heads,
                                                    int64_t head_dim) {
  if (!weight.defined() || weight.dim() < 1 || num_heads <= 0 ||
      head_dim <= 0 || weight.size(0) != 3 * num_heads * head_dim) {
    throw std::invalid_argument(
        "MiniMax-H3 grouped QKV weight has incompatible shape");
  }
  std::vector<int64_t> grouped_shape = weight.sizes().vec();
  grouped_shape[0] = num_heads;
  grouped_shape.insert(grouped_shape.begin() + 1, 3 * head_dim);
  torch::Tensor grouped = weight.reshape(grouped_shape);

  std::vector<int64_t> output_shape = weight.sizes().vec();
  output_shape[0] = num_heads * head_dim;
  torch::Tensor query = grouped.narrow(1, 0, head_dim).reshape(output_shape);
  torch::Tensor key =
      grouped.narrow(1, head_dim, head_dim).reshape(output_shape);
  torch::Tensor value =
      grouped.narrow(1, 2 * head_dim, head_dim).reshape(output_shape);
  return torch::cat({query, key, value}, 0).contiguous();
}

inline torch::Tensor minimax_h3_reorder_gate_up_to_up_gate(
    const torch::Tensor& weight) {
  if (!weight.defined() || weight.dim() < 1 || weight.size(0) % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 gate/up weight has incompatible shape");
  }
  const std::vector<torch::Tensor> gate_up = weight.chunk(2, 0);
  return torch::cat({gate_up[1], gate_up[0]}, 0).contiguous();
}

inline torch::Tensor minimax_h3_rms_norm(const torch::Tensor& input,
                                         const torch::Tensor& weight,
                                         double eps) {
  if (!input.defined() || !weight.defined() || input.dim() < 1 ||
      weight.dim() != 1 || input.size(-1) != weight.size(0)) {
    throw std::invalid_argument("MiniMax-H3 RMSNorm shape mismatch");
  }
  const torch::ScalarType input_dtype = input.scalar_type();
  torch::Tensor normalized = input.to(torch::kFloat32);
  // The frozen Ascend reference scales each FP32 square before reduction.
  // mean() can change this order across CANN versions and flip BF16 rounding.
  torch::Tensor variance =
      normalized.pow(2)
          .mul(1.0 / static_cast<double>(normalized.size(-1)))
          .sum(-1, /*keepdim=*/true);
  normalized = normalized * torch::rsqrt(variance + eps);
  normalized = normalized * weight.to(torch::kFloat32);
  return normalized.to(input_dtype);
}

inline torch::Tensor minimax_h3_rope_frequencies(
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq) {
  if (!position_ids.defined() || position_ids.dim() != 2 ||
      position_ids.size(1) != 3 || !inv_freq.defined() || inv_freq.dim() != 1) {
    throw std::invalid_argument(
        "MiniMax-H3 RoPE expects position IDs [S,3] and inv_freq [F]");
  }
  torch::Tensor per_axis = position_ids.to(torch::kFloat32).unsqueeze(-1) *
                           inv_freq.to(torch::kFloat32).view({1, 1, -1});
  torch::Tensor half = per_axis.reshape({position_ids.size(0), -1});
  return torch::cat({half, half}, -1);
}

inline torch::Tensor minimax_h3_apply_rope(const torch::Tensor& input,
                                           const torch::Tensor& frequencies) {
  if (!input.defined() || input.dim() != 3 || !frequencies.defined() ||
      frequencies.dim() != 2 || input.size(0) != frequencies.size(0) ||
      frequencies.size(1) <= 0 || frequencies.size(1) % 2 != 0 ||
      frequencies.size(1) > input.size(2)) {
    throw std::invalid_argument("MiniMax-H3 RoPE tensor shape mismatch");
  }
  const int64_t rotary_dim = frequencies.size(1);
  const int64_t half = rotary_dim / 2;
  torch::Tensor rotary = input.narrow(-1, 0, rotary_dim);
  torch::Tensor rotated = torch::cat(
      {-rotary.narrow(-1, half, half), rotary.narrow(-1, 0, half)}, -1);
  torch::Tensor cosine =
      torch::cos(frequencies).to(input.scalar_type()).unsqueeze(1);
  torch::Tensor sine =
      torch::sin(frequencies).to(input.scalar_type()).unsqueeze(1);
  torch::Tensor output_rotary = rotary * cosine + rotated * sine;
  torch::Tensor output_pass =
      input.narrow(-1, rotary_dim, input.size(2) - rotary_dim);
  return torch::cat({output_rotary, output_pass}, -1).contiguous();
}

inline torch::Tensor minimax_h3_combined_adaln_indices(
    const torch::Tensor& inverse_timestep_indices,
    const torch::Tensor& token_tags) {
  if (!inverse_timestep_indices.defined() || !token_tags.defined() ||
      inverse_timestep_indices.dim() != 1 || token_tags.dim() != 1 ||
      inverse_timestep_indices.size(0) != token_tags.size(0) ||
      inverse_timestep_indices.scalar_type() != torch::kInt64 ||
      token_tags.scalar_type() != torch::kInt64) {
    throw std::invalid_argument(
        "MiniMax-H3 AdaLN indices require matching int64 vectors");
  }
  return inverse_timestep_indices * 3 + token_tags.clamp_min(0);
}

inline torch::Tensor minimax_h3_timestep_embedding(
    const torch::Tensor& timestep,
    int64_t embedding_size) {
  if (!timestep.defined() || timestep.dim() != 1 ||
      !timestep.is_floating_point() || embedding_size <= 0 ||
      embedding_size % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 timestep embedding requires [M] and an even dimension");
  }
  const int64_t half = embedding_size / 2;
  torch::Tensor frequency =
      torch::exp(-std::log(10000.0) *
                 torch::arange(half,
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(timestep.device())) /
                 static_cast<double>(half));
  torch::Tensor arguments =
      timestep.to(torch::kFloat32).view({-1, 1}) * frequency.view({1, -1});
  return torch::cat({torch::cos(arguments), torch::sin(arguments)}, -1);
}

inline torch::Tensor minimax_h3_select_adaln_parameter(
    const torch::Tensor& parameters,
    int64_t logical_parameter,
    const torch::Tensor& combined_indices) {
  if (!parameters.defined() || parameters.dim() != 4 ||
      parameters.size(1) != 3 || logical_parameter < 0 ||
      logical_parameter >= parameters.size(2) ||
      combined_indices.scalar_type() != torch::kInt64 ||
      combined_indices.dim() != 1) {
    throw std::invalid_argument(
        "MiniMax-H3 AdaLN parameter selection mismatch");
  }
  return parameters.flatten(0, 1)
      .select(1, logical_parameter)
      .index_select(0, combined_indices);
}

inline torch::Tensor minimax_h3_segmented_sdpa(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens) {
  if (!query.defined() || query.dim() != 3 || key.sizes() != query.sizes() ||
      value.sizes() != query.sizes() || !cu_seqlens.defined() ||
      cu_seqlens.dim() != 1 || cu_seqlens.numel() < 2 ||
      (cu_seqlens.scalar_type() != torch::kInt32 &&
       cu_seqlens.scalar_type() != torch::kInt64)) {
    throw std::invalid_argument("MiniMax-H3 segmented SDPA shape mismatch");
  }
  torch::Tensor bounds =
      cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
  const int64_t* values = bounds.data_ptr<int64_t>();
  if (values[0] != 0 || values[bounds.numel() - 1] != query.size(0)) {
    throw std::invalid_argument(
        "MiniMax-H3 cu_seqlens must cover every packed row");
  }

  torch::Tensor output = torch::empty_like(query);
  for (int64_t index = 0; index + 1 < bounds.numel(); ++index) {
    const int64_t start = values[index];
    const int64_t stop = values[index + 1];
    if (start < 0 || stop < start || stop > query.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 cu_seqlens must be monotonic and in range");
    }
    if (start == stop) {
      continue;
    }
    const int64_t length = stop - start;
    torch::Tensor segment_query =
        query.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_key =
        key.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_value =
        value.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_output;
#if defined(USE_NPU)
    if (query.device().type() == c10::DeviceType::PrivateUse1) {
      // Match the native H3Attention BNSD kernel and parameters. Keep used
      // rows and alignment-only rows in separate segments.
      const auto results = at_npu::native::custom_ops::npu_fusion_attention(
          segment_query.contiguous(),
          segment_key.contiguous(),
          segment_value.contiguous(),
          segment_query.size(1),
          "BNSD",
          torch::nullopt,
          torch::nullopt,
          torch::nullopt,
          std::pow(query.size(2), -0.5),
          1.0,
          65535,
          65535);
      segment_output = std::get<0>(results);
    } else {
      CHECK(query.device().is_cpu())
          << "Qualified attention supports CPU or Ascend NPU tensors";
#else
    {
#endif
      segment_output = torch::scaled_dot_product_attention(segment_query,
                                                           segment_key,
                                                           segment_value,
                                                           torch::nullopt,
                                                           /*dropout_p=*/0.0,
                                                           /*is_causal=*/false);
    }
    output.narrow(0, start, length)
        .copy_(segment_output.squeeze(0).transpose(0, 1));
  }
  return output;
}

class MiniMaxH3DenseImpl final : public torch::nn::Module {
 public:
  MiniMaxH3DenseImpl(int64_t input_size,
                     int64_t output_size,
                     bool with_bias,
                     const torch::TensorOptions& options) {
    weight_ = register_parameter(
        "weight", torch::empty({output_size, input_size}, options));
    if (with_bias) {
      bias_ = register_parameter("bias", torch::empty({output_size}, options));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    if (bias_.defined()) {
      return torch::nn::functional::linear(input, weight_, bias_);
    }
    return torch::nn::functional::linear(input, weight_);
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3Dense);

class MiniMaxH3RMSNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3RMSNormImpl(int64_t size,
                       double eps,
                       const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({size}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return minimax_h3_rms_norm(input, weight_, eps_);
  }

 private:
  torch::Tensor weight_;
  double eps_;
};
TORCH_MODULE(MiniMaxH3RMSNorm);

class MiniMaxH3RopeImpl final : public torch::nn::Module {
 public:
  MiniMaxH3RopeImpl(int64_t inv_freq_len, const torch::TensorOptions& options) {
    inv_freq_ = register_buffer(
        "inv_freq",
        torch::empty({inv_freq_len}, options.dtype(torch::kFloat32)));
  }

  torch::Tensor forward(const torch::Tensor& position_ids) const {
    return minimax_h3_rope_frequencies(position_ids, inv_freq_);
  }

 private:
  torch::Tensor inv_freq_;
};
TORCH_MODULE(MiniMaxH3Rope);

class MiniMaxH3AttentionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AttentionImpl(const MiniMaxH3C4Config& config,
                         const torch::TensorOptions& options)
      : num_heads_(config.num_attention_heads),
        head_dim_(config.attention_head_dim) {
    const int64_t inner_size = num_heads_ * head_dim_;
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    q_norm_ = register_module(
        "q_norm",
        MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16_options));
    k_norm_ = register_module(
        "k_norm",
        MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16_options));
    qkv_proj_ = register_module("qkv_proj",
                                MiniMaxH3Dense(config.hidden_size,
                                               3 * inner_size,
                                               /*with_bias=*/false,
                                               bf16_options));
    out_proj_ = register_module("out_proj",
                                MiniMaxH3Dense(inner_size,
                                               config.hidden_size,
                                               /*with_bias=*/false,
                                               bf16_options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const std::optional<torch::Tensor>& rope_frequencies,
                        const torch::Tensor& cu_seqlens) const {
    if (!input.defined() || input.dim() != 2 ||
        input.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 attention input must be BF16 [T,H]");
    }
    const int64_t row_count = input.size(0);
    const int64_t inner_size = num_heads_ * head_dim_;
    const std::vector<torch::Tensor> qkv_weights =
        qkv_proj_->weight().split({inner_size, inner_size, inner_size}, 0);
    std::vector<torch::Tensor> qkv;
    qkv.reserve(3);
    for (const torch::Tensor& weight : qkv_weights) {
      qkv.emplace_back(torch::nn::functional::linear(input, weight));
    }
    torch::Tensor query =
        q_norm_->forward(qkv[0].view({row_count, num_heads_, head_dim_}));
    torch::Tensor key =
        k_norm_->forward(qkv[1].view({row_count, num_heads_, head_dim_}));
    torch::Tensor value = qkv[2].view({row_count, num_heads_, head_dim_});
    if (rope_frequencies.has_value()) {
      query = minimax_h3_apply_rope(query, *rope_frequencies);
      key = minimax_h3_apply_rope(key, *rope_frequencies);
    }
    torch::Tensor output =
        minimax_h3_segmented_sdpa(query, key, value, cu_seqlens);
    return out_proj_->forward(output.reshape({row_count, inner_size}));
  }

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  MiniMaxH3RMSNorm q_norm_{nullptr};
  MiniMaxH3RMSNorm k_norm_{nullptr};
  MiniMaxH3Dense qkv_proj_{nullptr};
  MiniMaxH3Dense out_proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3Attention);

class MiniMaxH3MLPImpl final : public torch::nn::Module {
 public:
  MiniMaxH3MLPImpl(const MiniMaxH3C4Config& config,
                   const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    fc1_ = register_module("fc1",
                           MiniMaxH3Dense(config.hidden_size,
                                          2 * config.ffn_hidden_size,
                                          /*with_bias=*/false,
                                          bf16_options));
    fc2_ = register_module("fc2",
                           MiniMaxH3Dense(config.ffn_hidden_size,
                                          config.hidden_size,
                                          /*with_bias=*/false,
                                          bf16_options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    std::vector<torch::Tensor> up_gate = fc1_->forward(input).chunk(2, -1);
    return fc2_->forward(up_gate[0] * torch::silu(up_gate[1]));
  }

 private:
  MiniMaxH3Dense fc1_{nullptr};
  MiniMaxH3Dense fc2_{nullptr};
};
TORCH_MODULE(MiniMaxH3MLP);

class MiniMaxH3TokenRefinerBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TokenRefinerBlockImpl(const MiniMaxH3C4Config& config,
                                 const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    norm2_ = register_module(
        "norm2",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    attn_ = register_module("attn", MiniMaxH3Attention(config, options));
    mlp_ = register_module("mlp", MiniMaxH3MLP(config, options));
  }

  MiniMaxH3ResidualBranchTrace forward(const torch::Tensor& input,
                                       const torch::Tensor& cu_seqlens) const {
    MiniMaxH3ResidualBranchTrace trace;
    trace.attention_delta =
        attn_->forward(norm1_->forward(input), std::nullopt, cu_seqlens);
    torch::Tensor after_attention = input + trace.attention_delta;
    trace.mlp_delta = mlp_->forward(norm2_->forward(after_attention));
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor after_attention;
    {
      torch::Tensor attention_delta =
          attn_->forward(norm1_->forward(input), std::nullopt, cu_seqlens);
      after_attention = input + attention_delta;
    }
    return after_attention + mlp_->forward(norm2_->forward(after_attention));
  }

 private:
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3Attention attn_{nullptr};
  MiniMaxH3MLP mlp_{nullptr};
};
TORCH_MODULE(MiniMaxH3TokenRefinerBlock);

class MiniMaxH3TokenRefinerImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TokenRefinerImpl(const MiniMaxH3C4Config& config,
                            const torch::TensorOptions& options) {
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    block_layers_.reserve(static_cast<size_t>(config.num_refiner_layers));
    for (int64_t index = 0; index < config.num_refiner_layers; ++index) {
      MiniMaxH3TokenRefinerBlock block(config, options);
      blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
    final_norm_ =
        register_module("final_norm",
                        MiniMaxH3RMSNorm(config.hidden_size,
                                         config.final_norm_eps,
                                         options.dtype(torch::kBFloat16)));
  }

  std::pair<torch::Tensor, std::vector<MiniMaxH3ResidualBranchTrace>> forward(
      const torch::Tensor& input,
      const torch::Tensor& cu_seqlens) const {
    torch::Tensor hidden = input;
    std::vector<MiniMaxH3ResidualBranchTrace> traces;
    traces.reserve(block_layers_.size());
    for (const MiniMaxH3TokenRefinerBlock& block : block_layers_) {
      MiniMaxH3ResidualBranchTrace trace = block->forward(hidden, cu_seqlens);
      hidden = trace.output;
      traces.emplace_back(std::move(trace));
    }
    return {final_norm_->forward(hidden), std::move(traces)};
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor hidden = input;
    for (const MiniMaxH3TokenRefinerBlock& block : block_layers_) {
      hidden = block->forward_output_only(hidden, cu_seqlens);
    }
    return final_norm_->forward(hidden);
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MiniMaxH3TokenRefinerBlock> block_layers_;
  MiniMaxH3RMSNorm final_norm_{nullptr};
};
TORCH_MODULE(MiniMaxH3TokenRefiner);

class MiniMaxH3AdaLNProjectionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AdaLNProjectionImpl(int64_t time_embed_dim,
                               int64_t hidden_size,
                               int64_t parameter_count,
                               int64_t modality_count,
                               const torch::TensorOptions& options)
      : hidden_size_(hidden_size),
        parameter_count_(parameter_count),
        modality_count_(modality_count) {
    linear_ = register_module(
        "linear",
        MiniMaxH3Dense(time_embed_dim,
                       hidden_size * parameter_count * modality_count,
                       /*with_bias=*/true,
                       options.dtype(torch::kBFloat16)));
  }

  torch::Tensor forward(const torch::Tensor& time_embedding) const {
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 AdaLN time embedding must be FP32 [M,D]");
    }
    torch::Tensor projected =
        linear_->forward(torch::silu(time_embedding).to(torch::kBFloat16));
    return projected.view({time_embedding.size(0),
                           modality_count_,
                           parameter_count_,
                           hidden_size_});
  }

 private:
  int64_t hidden_size_;
  int64_t parameter_count_;
  int64_t modality_count_;
  MiniMaxH3Dense linear_{nullptr};
};
TORCH_MODULE(MiniMaxH3AdaLNProjection);

class MiniMaxH3DiTBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3DiTBlockImpl(const MiniMaxH3C4Config& config,
                        const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    norm2_ = register_module(
        "norm2",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    attn_ = register_module("attn", MiniMaxH3Attention(config, options));
    mlp_ = register_module("mlp", MiniMaxH3MLP(config, options));
    adaln_proj_ =
        register_module("adaln_proj",
                        MiniMaxH3AdaLNProjection(config.time_embed_dim,
                                                 config.hidden_size,
                                                 /*parameter_count=*/6,
                                                 /*modality_count=*/3,
                                                 options));
  }

  MiniMaxH3ResidualBranchTrace forward(const torch::Tensor& input,
                                       const torch::Tensor& time_embedding,
                                       const torch::Tensor& combined_indices,
                                       const torch::Tensor& rope_frequencies,
                                       const torch::Tensor& cu_seqlens) const {
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor shift_msa =
        minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
    torch::Tensor scale_msa =
        minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
    torch::Tensor gate_msa =
        minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
    torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);

    MiniMaxH3ResidualBranchTrace trace;
    trace.adaln_parameters = parameters;
    trace.shift_msa = shift_msa;
    trace.scale_msa = scale_msa;
    trace.gate_msa = gate_msa;
    trace.shift_mlp = shift_mlp;
    trace.scale_mlp = scale_mlp;
    trace.gate_mlp = gate_mlp;
    trace.norm1_output = norm1_->forward(input);
    trace.attention_input = trace.norm1_output * (scale_msa + 1.0) + shift_msa;
    trace.attention_output =
        attn_->forward(trace.attention_input, rope_frequencies, cu_seqlens);
    trace.attention_delta = gate_msa * trace.attention_output;
    torch::Tensor after_attention = input + trace.attention_delta;

    trace.norm2_output = norm2_->forward(after_attention);
    trace.mlp_input = trace.norm2_output * (scale_mlp + 1.0) + shift_mlp;
    trace.mlp_output = mlp_->forward(trace.mlp_input);
    trace.mlp_delta = gate_mlp * trace.mlp_output;
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& time_embedding,
                                    const torch::Tensor& combined_indices,
                                    const torch::Tensor& rope_frequencies,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor after_attention;
    {
      torch::Tensor shift_msa =
          minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
      torch::Tensor scale_msa =
          minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
      torch::Tensor gate_msa =
          minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
      torch::Tensor attention_input =
          norm1_->forward(input) * (scale_msa + 1.0) + shift_msa;
      torch::Tensor attention_output =
          attn_->forward(attention_input, rope_frequencies, cu_seqlens);
      after_attention = input + gate_msa * attention_output;
    }
    torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);
    torch::Tensor mlp_input =
        norm2_->forward(after_attention) * (scale_mlp + 1.0) + shift_mlp;
    return after_attention + gate_mlp * mlp_->forward(mlp_input);
  }

 private:
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3Attention attn_{nullptr};
  MiniMaxH3MLP mlp_{nullptr};
  MiniMaxH3AdaLNProjection adaln_proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3DiTBlock);

class MiniMaxH3TimeEmbedderImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TimeEmbedderImpl(const MiniMaxH3C4Config& config,
                            const torch::TensorOptions& options)
      : frequency_embedding_size_(config.timestep_input_dim) {
    const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
    proj_in_ = register_module("proj_in",
                               MiniMaxH3Dense(config.timestep_input_dim,
                                              config.time_embed_hidden_size,
                                              /*with_bias=*/true,
                                              fp32_options));
    proj_out_ = register_module("proj_out",
                                MiniMaxH3Dense(config.time_embed_hidden_size,
                                               config.time_embed_dim,
                                               /*with_bias=*/true,
                                               fp32_options));
  }

  torch::Tensor forward(const torch::Tensor& timestep) const {
    if (!timestep.defined() || timestep.dim() != 1 ||
        !timestep.is_floating_point()) {
      throw std::invalid_argument(
          "MiniMax-H3 timestep must be a floating-point vector");
    }
    if (!torch::logical_and(timestep >= 0, timestep <= 1).all().item<bool>()) {
      throw std::invalid_argument("MiniMax-H3 timestep must be in [0,1]");
    }
    torch::Tensor sinusoidal =
        minimax_h3_timestep_embedding(timestep, frequency_embedding_size_);
    return proj_out_->forward(torch::silu(proj_in_->forward(sinusoidal)));
  }

 private:
  int64_t frequency_embedding_size_;
  MiniMaxH3Dense proj_in_{nullptr};
  MiniMaxH3Dense proj_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3TimeEmbedder);

class MiniMaxH3FinalLayerImpl final : public torch::nn::Module {
 public:
  MiniMaxH3FinalLayerImpl(const MiniMaxH3C4Config& config,
                          const torch::TensorOptions& options) {
    norm_ = register_module("norm",
                            MiniMaxH3RMSNorm(config.hidden_size,
                                             config.final_norm_eps,
                                             options.dtype(torch::kBFloat16)));
    adaln_proj_ =
        register_module("adaln_proj",
                        MiniMaxH3AdaLNProjection(config.time_embed_dim,
                                                 config.hidden_size,
                                                 /*parameter_count=*/2,
                                                 /*modality_count=*/1,
                                                 options));
    video_out_ =
        register_module("video_out",
                        MiniMaxH3Dense(config.hidden_size,
                                       config.video_patch_dim,
                                       /*with_bias=*/true,
                                       options.dtype(torch::kFloat32)));
    audio_out_ =
        register_module("audio_out",
                        MiniMaxH3Dense(config.hidden_size,
                                       config.audio_dim,
                                       /*with_bias=*/true,
                                       options.dtype(torch::kFloat32)));
  }

  MiniMaxH3FinalOutput forward(const torch::Tensor& input,
                               const torch::Tensor& time_embedding,
                               const torch::Tensor& inverse_indices,
                               const H3PackedLayout& layout) const {
    if (layout.used_length <= 0 || layout.used_length > input.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 final layer used-row geometry mismatch");
    }
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor shift =
        parameters.select(1, 0).select(1, 0).index_select(0, inverse_indices);
    torch::Tensor scale =
        parameters.select(1, 0).select(1, 1).index_select(0, inverse_indices);
    MiniMaxH3FinalOutput output;
    output.activation = norm_->forward(input) * (scale + 1.0) + shift;
    const torch::Tensor fp32_activation =
        output.activation.narrow(0, 0, layout.used_length).to(torch::kFloat32);
    const torch::Tensor used_video_logits =
        video_out_->forward(fp32_activation);
    const torch::Tensor used_audio_logits =
        audio_out_->forward(fp32_activation);
    output.all_video_logits =
        torch::zeros({input.size(0), used_video_logits.size(1)},
                     used_video_logits.options());
    output.all_audio_logits =
        torch::zeros({input.size(0), used_audio_logits.size(1)},
                     used_audio_logits.options());
    output.all_video_logits.narrow(0, 0, layout.used_length)
        .copy_(used_video_logits);
    output.all_audio_logits.narrow(0, 0, layout.used_length)
        .copy_(used_audio_logits);

    torch::Tensor image_positions = layout.img_pos.to(input.device());
    torch::Tensor audio_positions = layout.audio_pos.to(input.device());
    output.raw_selected_video_logits =
        output.all_video_logits.index_select(0, image_positions);
    output.raw_selected_audio_logits =
        output.all_audio_logits.index_select(0, audio_positions);
    torch::Tensor video_mask =
        layout.update_mask.to(input.device()).to(torch::kFloat32).unsqueeze(-1);
    torch::Tensor audio_mask = layout.audio_update_mask.to(input.device())
                                   .to(torch::kFloat32)
                                   .unsqueeze(-1);
    output.selected_video_logits =
        output.raw_selected_video_logits * video_mask;
    output.selected_audio_logits =
        output.raw_selected_audio_logits * audio_mask;
    return output;
  }

 private:
  MiniMaxH3RMSNorm norm_{nullptr};
  MiniMaxH3AdaLNProjection adaln_proj_{nullptr};
  MiniMaxH3Dense video_out_{nullptr};
  MiniMaxH3Dense audio_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3FinalLayer);

}  // namespace xllm

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
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/npu/npu_qwen3_audio_encoder_layer_impl.h"
#include "processors/qwen3_audio_common.h"

namespace xllm::npu::model {

class Qwen3AudioSinusoidalPositionEmbeddingImpl final
    : public torch::nn::Module {
 public:
  Qwen3AudioSinusoidalPositionEmbeddingImpl(int64_t length,
                                            int64_t channels,
                                            double max_timescale = 10000.0) {
    CHECK_EQ(channels % 2, 0)
        << "Qwen3 audio sinusoidal position embedding requires an even "
           "channel count.";

    const double log_timescale_increment =
        std::log(max_timescale) / (channels / 2 - 1);
    const torch::Tensor inverse_timescales =
        torch::exp(-log_timescale_increment * torch::arange(channels / 2))
            .to(torch::kFloat32);
    const torch::Tensor scaled_time =
        torch::arange(length).unsqueeze(1) * inverse_timescales.unsqueeze(0);
    position_embedding_ =
        torch::cat({torch::sin(scaled_time), torch::cos(scaled_time)}, 1);
  }

  torch::Tensor forward(int64_t sequence_length) {
    return position_embedding_.slice(0, 0, sequence_length);
  }

 private:
  torch::Tensor position_embedding_;
};
TORCH_MODULE(Qwen3AudioSinusoidalPositionEmbedding);

class Qwen3AudioEncoderBlockImpl final : public torch::nn::Module {
 public:
  explicit Qwen3AudioEncoderBlockImpl(const ModelContext& context) {
    encoder_layer_ = register_module("encoder_layer",
                                     layer::NpuQwen3AudioEncoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& cumulative_sequence_lengths,
                        std::vector<int32_t>& cumulative_sequence_lengths_vec,
                        int32_t node_id) {
    return encoder_layer_(hidden_states,
                          cumulative_sequence_lengths,
                          cumulative_sequence_lengths_vec,
                          node_id);
  }

  void load_state_dict(const StateDict& state_dict) {
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights() const {
    encoder_layer_->verify_loaded_weights();
  }

  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuQwen3AudioEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3AudioEncoderBlock);

class Qwen3AudioEncoderImpl final : public torch::nn::Module {
 public:
  explicit Qwen3AudioEncoderImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    options_ = context.get_tensor_options();
    const int64_t downsample_hidden_size =
        model_args.mm_audio_downsample_hidden_size();
    const int64_t num_mel_bins = model_args.mm_audio_num_mel_bins();
    const int64_t max_source_positions =
        model_args.mm_audio_max_source_positions();
    embed_dim_ = model_args.mm_audio_hidden_size();
    n_window_ = model_args.mm_audio_n_window();
    n_window_infer_ = model_args.mm_audio_n_window_infer();
    conv_chunk_size_ = model_args.mm_audio_conv_chunksize();
    CHECK_GT(embed_dim_, 0);
    CHECK_GT(n_window_, 0);
    CHECK_GE(n_window_infer_, n_window_ * 2);
    CHECK_EQ(n_window_infer_ % (n_window_ * 2), 0);
    CHECK_GT(conv_chunk_size_, 0);

    positional_embedding_ =
        register_module("positional_embedding",
                        Qwen3AudioSinusoidalPositionEmbedding(
                            max_source_positions, embed_dim_));

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t index = 0; index < model_args.mm_audio_encoder_layers();
         ++index) {
      Qwen3AudioEncoderBlock layer(context);
      layers_->push_back(layer);
    }

    ln_post_ = register_module(
        "ln_post",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})
                                 .elementwise_affine(true)));

    conv2d1_ = register_module(
        "conv2d1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));
    conv2d2_ = register_module(
        "conv2d2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              downsample_hidden_size, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));
    conv2d3_ = register_module(
        "conv2d3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              downsample_hidden_size, downsample_hidden_size, 3)
                              .stride(2)
                              .padding(1)
                              .bias(true)));

    const int64_t conv_output_dim = (((num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(downsample_hidden_size * conv_output_dim,
                                     embed_dim_)
                .bias(false)));
    proj1_ = register_module(
        "proj1",
        torch::nn::Linear(
            torch::nn::LinearOptions(embed_dim_, embed_dim_).bias(true)));
    proj2_ = register_module(
        "proj2",
        torch::nn::Linear(torch::nn::LinearOptions(
                              embed_dim_, model_args.mm_audio_output_dim())
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& input_features,
                        const torch::Tensor& feature_lengths) {
    const torch::Tensor output_lengths =
        qwen3_audio::get_feature_output_lengths(feature_lengths, n_window_ * 2);
    const torch::Tensor chunk_counts =
        torch::ceil(feature_lengths / (n_window_ * 2)).to(torch::kLong);
    const int64_t total_chunks = chunk_counts.sum().item<int64_t>();

    torch::Tensor chunk_lengths =
        torch::full({total_chunks},
                    n_window_ * 2,
                    torch::TensorOptions()
                        .dtype(torch::kLong)
                        .device(feature_lengths.device()));
    const torch::Tensor padded_chunk_counts = torch::nn::functional::pad(
        chunk_counts, torch::nn::functional::PadFuncOptions({1, 0}).value(-1));
    const torch::Tensor tail_chunk_indices =
        padded_chunk_counts.cumsum(0).slice(0, 1);
    const torch::Tensor remainder = feature_lengths % (n_window_ * 2);
    chunk_lengths.index_put_({torch::indexing::TensorIndex(tail_chunk_indices)},
                             remainder);
    chunk_lengths.index_put_({torch::indexing::TensorIndex(chunk_lengths == 0)},
                             n_window_ * 2);

    const torch::Tensor transposed_features = input_features.t();
    const torch::Tensor chunk_lengths_cpu =
        chunk_lengths.to(torch::kCPU).contiguous();
    const c10::IntArrayRef split_sizes(
        chunk_lengths_cpu.data_ptr<int64_t>(),
        static_cast<size_t>(chunk_lengths_cpu.size(0)));
    const std::vector<torch::Tensor> chunks =
        transposed_features.split_with_sizes(split_sizes, 0);
    torch::Tensor padded_features =
        torch::nn::utils::rnn::pad_sequence(chunks, true).transpose(1, 2);
    const torch::Tensor chunk_output_lengths =
        qwen3_audio::get_feature_output_lengths(chunk_lengths.to(torch::kLong),
                                                n_window_ * 2);

    std::vector<torch::Tensor> mask_tensors;
    mask_tensors.reserve(static_cast<size_t>(chunk_output_lengths.size(0)));
    for (int64_t index = 0; index < chunk_output_lengths.size(0); ++index) {
      const int64_t length = chunk_output_lengths[index].item<int64_t>();
      mask_tensors.emplace_back(
          torch::full({length},
                      1,
                      torch::TensorOptions()
                          .dtype(torch::kBool)
                          .device(padded_features.device())));
    }
    const torch::Tensor padded_mask_after_cnn =
        torch::nn::utils::rnn::pad_sequence(mask_tensors, true);

    padded_features = padded_features.unsqueeze(1);
    std::vector<torch::Tensor> padded_embeddings;
    const int64_t batch_size = padded_features.size(0);
    padded_embeddings.reserve(static_cast<size_t>(
        (batch_size + conv_chunk_size_ - 1) / conv_chunk_size_));
    for (int64_t start = 0; start < batch_size; start += conv_chunk_size_) {
      const int64_t end = std::min(start + conv_chunk_size_, batch_size);
      const torch::Tensor chunk = padded_features.slice(0, start, end);
      torch::Tensor embedding = torch::gelu(conv2d1_(chunk));
      embedding = torch::gelu(conv2d2_(embedding));
      embedding = torch::gelu(conv2d3_(embedding));
      padded_embeddings.emplace_back(embedding);
    }

    torch::Tensor padded_embedding = torch::cat(padded_embeddings, 0);
    const int64_t batch = padded_embedding.size(0);
    const int64_t channels = padded_embedding.size(1);
    const int64_t frequency = padded_embedding.size(2);
    const int64_t time = padded_embedding.size(3);
    const torch::Tensor reshaped =
        padded_embedding.permute({0, 3, 1, 2})
            .contiguous()
            .view({batch, time, channels * frequency});
    padded_embedding = conv_out_(reshaped);
    const torch::Tensor position_embedding =
        positional_embedding_->forward(padded_embedding.size(1))
            .unsqueeze(0)
            .to(options_.device(), padded_embedding.dtype());
    padded_embedding = padded_embedding + position_embedding;

    torch::Tensor hidden_states =
        padded_embedding
            .masked_select(
                padded_mask_after_cnn.unsqueeze(-1).expand_as(padded_embedding))
            .view({-1, padded_embedding.size(-1)});
    const int64_t window_after_cnn =
        padded_mask_after_cnn.size(-1) * (n_window_infer_ / (n_window_ * 2));

    std::vector<int32_t> cumulative_chunk_lengths;
    for (int64_t index = 0; index < output_lengths.size(0); ++index) {
      const int64_t cnn_length = output_lengths[index].item<int64_t>();
      const int64_t full_windows = cnn_length / window_after_cnn;
      for (int64_t window = 0; window < full_windows; ++window) {
        cumulative_chunk_lengths.emplace_back(
            static_cast<int32_t>(window_after_cnn));
      }
      const int64_t tail_length = cnn_length % window_after_cnn;
      if (tail_length != 0) {
        cumulative_chunk_lengths.emplace_back(
            static_cast<int32_t>(tail_length));
      }
    }

    torch::Tensor cumulative_sequence_lengths =
        torch::tensor(cumulative_chunk_lengths,
                      torch::TensorOptions()
                          .device(output_lengths.device())
                          .dtype(torch::kInt32));
    const torch::Tensor cumulative_sequence_lengths_cpu =
        cumulative_sequence_lengths.cpu().contiguous();
    std::vector<int32_t> cumulative_sequence_lengths_vec(
        cumulative_sequence_lengths_cpu.data_ptr<int32_t>(),
        cumulative_sequence_lengths_cpu.data_ptr<int32_t>() +
            cumulative_sequence_lengths_cpu.numel());

    const int32_t layer_count = static_cast<int32_t>(layers_->size());
    for (int32_t index = 0; index < layer_count; ++index) {
      hidden_states = layers_[index]->as<Qwen3AudioEncoderBlock>()->forward(
          hidden_states,
          cumulative_sequence_lengths,
          cumulative_sequence_lengths_vec,
          index);
    }

    hidden_states = ln_post_(hidden_states);
    hidden_states = proj1_(hidden_states);
    hidden_states = torch::gelu(hidden_states);
    return proj2_(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "conv_out.weight",
                        conv_out_->weight,
                        is_conv_out_weight_loaded_);
    weight::load_weight(
        state_dict, "proj1.weight", proj1_->weight, is_proj1_weight_loaded_);
    weight::load_weight(
        state_dict, "proj1.bias", proj1_->bias, is_proj1_bias_loaded_);
    weight::load_weight(
        state_dict, "proj2.weight", proj2_->weight, is_proj2_weight_loaded_);
    weight::load_weight(
        state_dict, "proj2.bias", proj2_->bias, is_proj2_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv2d1.weight",
                        conv2d1_->weight,
                        is_conv2d1_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d1.bias", conv2d1_->bias, is_conv2d1_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv2d2.weight",
                        conv2d2_->weight,
                        is_conv2d2_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d2.bias", conv2d2_->bias, is_conv2d2_bias_loaded_);
    weight::load_weight(state_dict,
                        "conv2d3.weight",
                        conv2d3_->weight,
                        is_conv2d3_weight_loaded_);
    weight::load_weight(
        state_dict, "conv2d3.bias", conv2d3_->bias, is_conv2d3_bias_loaded_);
    weight::load_weight(state_dict,
                        "ln_post.weight",
                        ln_post_->weight,
                        is_ln_post_weight_loaded_);
    weight::load_weight(
        state_dict, "ln_post.bias", ln_post_->bias, is_ln_post_bias_loaded_);

    const size_t layer_count = layers_->size();
    for (size_t index = 0; index < layer_count; ++index) {
      const std::string prefix = "layers." + std::to_string(index) + ".";
      layers_[index]->as<Qwen3AudioEncoderBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_conv_out_weight_loaded_)
        << "weight is not loaded for " << prefix << "conv_out.weight";
    CHECK(is_proj1_weight_loaded_)
        << "weight is not loaded for " << prefix << "proj1.weight";
    CHECK(is_proj1_bias_loaded_)
        << "weight is not loaded for " << prefix << "proj1.bias";
    CHECK(is_proj2_weight_loaded_)
        << "weight is not loaded for " << prefix << "proj2.weight";
    CHECK(is_proj2_bias_loaded_)
        << "weight is not loaded for " << prefix << "proj2.bias";
    CHECK(is_conv2d1_weight_loaded_)
        << "weight is not loaded for " << prefix << "conv2d1.weight";
    CHECK(is_conv2d1_bias_loaded_)
        << "weight is not loaded for " << prefix << "conv2d1.bias";
    CHECK(is_conv2d2_weight_loaded_)
        << "weight is not loaded for " << prefix << "conv2d2.weight";
    CHECK(is_conv2d2_bias_loaded_)
        << "weight is not loaded for " << prefix << "conv2d2.bias";
    CHECK(is_conv2d3_weight_loaded_)
        << "weight is not loaded for " << prefix << "conv2d3.weight";
    CHECK(is_conv2d3_bias_loaded_)
        << "weight is not loaded for " << prefix << "conv2d3.bias";
    CHECK(is_ln_post_weight_loaded_)
        << "weight is not loaded for " << prefix << "ln_post.weight";
    CHECK(is_ln_post_bias_loaded_)
        << "weight is not loaded for " << prefix << "ln_post.bias";

    const size_t layer_count = layers_->size();
    for (size_t index = 0; index < layer_count; ++index) {
      layers_[index]->as<Qwen3AudioEncoderBlock>()->verify_loaded_weights();
    }
  }

  void merge_loaded_weights() {
    const size_t layer_count = layers_->size();
    for (size_t index = 0; index < layer_count; ++index) {
      layers_[index]->as<Qwen3AudioEncoderBlock>()->merge_loaded_weights();
    }
  }

 private:
  int64_t embed_dim_ = 0;
  int64_t n_window_ = 0;
  int64_t n_window_infer_ = 0;
  int64_t conv_chunk_size_ = 0;
  torch::TensorOptions options_;

  Qwen3AudioSinusoidalPositionEmbedding positional_embedding_{nullptr};
  torch::nn::ModuleList layers_{nullptr};
  torch::nn::LayerNorm ln_post_{nullptr};
  torch::nn::Conv2d conv2d1_{nullptr};
  torch::nn::Conv2d conv2d2_{nullptr};
  torch::nn::Conv2d conv2d3_{nullptr};
  torch::nn::Linear conv_out_{nullptr};
  torch::nn::Linear proj1_{nullptr};
  torch::nn::Linear proj2_{nullptr};

  bool is_conv_out_weight_loaded_ = false;
  bool is_conv2d1_weight_loaded_ = false;
  bool is_conv2d1_bias_loaded_ = false;
  bool is_conv2d2_weight_loaded_ = false;
  bool is_conv2d2_bias_loaded_ = false;
  bool is_conv2d3_weight_loaded_ = false;
  bool is_conv2d3_bias_loaded_ = false;
  bool is_ln_post_weight_loaded_ = false;
  bool is_ln_post_bias_loaded_ = false;
  bool is_proj1_weight_loaded_ = false;
  bool is_proj1_bias_loaded_ = false;
  bool is_proj2_weight_loaded_ = false;
  bool is_proj2_bias_loaded_ = false;
};
TORCH_MODULE(Qwen3AudioEncoder);

struct Qwen3AudioInputs {
  torch::Tensor input_features;
  torch::Tensor feature_lengths;
  torch::Tensor feature_origin_lengths;
};

}  // namespace xllm::npu::model

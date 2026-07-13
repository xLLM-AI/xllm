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

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "models/llm/npu/minimax_m3.h"
#include "models/model_registry.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "processors/minimax_m3_vl_image_processor.h"
#include "processors/minimax_m3_vl_prompt_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm::npu::model {
namespace {

void load_linear_weight(const StateDict& state_dict,
                        const std::string& weight_name,
                        torch::nn::Linear& linear,
                        bool& weight_loaded) {
  torch::Tensor weight = state_dict.get_tensor(weight_name);
  if (!weight.defined()) {
    return;
  }
  weight = weight.reshape(linear->weight.sizes());
  CHECK_EQ(linear->weight.sizes(), weight.sizes())
      << "weight size mismatch for " << weight_name;
  linear->weight.data().copy_(weight);
  weight_loaded = true;
}

void load_linear_bias(const StateDict& state_dict,
                      const std::string& bias_name,
                      torch::nn::Linear& linear,
                      bool& bias_loaded) {
  torch::Tensor bias = state_dict.get_tensor(bias_name);
  if (!bias.defined()) {
    return;
  }
  CHECK(linear->bias.defined()) << "linear layer has no bias: " << bias_name;
  CHECK_EQ(linear->bias.sizes(), bias.sizes())
      << "bias size mismatch for " << bias_name;
  linear->bias.data().copy_(bias);
  bias_loaded = true;
}

void load_layer_norm_weight(const StateDict& state_dict,
                            torch::nn::LayerNorm& layer_norm,
                            bool& weight_loaded,
                            bool& bias_loaded) {
  torch::Tensor weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(layer_norm->weight.sizes(), weight.sizes())
        << "layer norm weight size mismatch.";
    layer_norm->weight.data().copy_(weight);
    weight_loaded = true;
  }

  torch::Tensor bias = state_dict.get_tensor("bias");
  if (bias.defined()) {
    CHECK_EQ(layer_norm->bias.sizes(), bias.sizes())
        << "layer norm bias size mismatch.";
    layer_norm->bias.data().copy_(bias);
    bias_loaded = true;
  }
}

torch::Tensor build_axis_freq(const torch::Tensor& pos_ids,
                              int64_t section_size,
                              double theta) {
  if (section_size == 0) {
    return torch::empty(
        {pos_ids.size(0), 0},
        torch::TensorOptions().dtype(torch::kFloat32).device(pos_ids.device()));
  }
  torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(pos_ids.device());
  torch::Tensor inv_freq =
      1.0 / torch::pow(theta,
                       torch::arange(0, section_size * 2, 2, options) /
                           static_cast<double>(section_size * 2));
  return pos_ids.to(torch::kFloat32).unsqueeze(1) * inv_freq.unsqueeze(0);
}

torch::Tensor apply_rotary_pos_emb(torch::Tensor input,
                                   const torch::Tensor& cos,
                                   const torch::Tensor& sin) {
  torch::Tensor input_float = input.to(torch::kFloat32);
  torch::Tensor cos_broadcast = cos.unsqueeze(1).to(torch::kFloat32);
  torch::Tensor sin_broadcast = sin.unsqueeze(1).to(torch::kFloat32);
  const int64_t rotary_dim = cos.size(-1);
  torch::Tensor input_first = input_float.slice(/*dim=*/-1,
                                                /*start=*/0,
                                                /*end=*/rotary_dim);
  torch::Tensor input_second = input_float.slice(/*dim=*/-1,
                                                 /*start=*/rotary_dim,
                                                 /*end=*/rotary_dim * 2);
  torch::Tensor output_first =
      input_first * cos_broadcast - input_second * sin_broadcast;
  torch::Tensor output_second =
      input_second * cos_broadcast + input_first * sin_broadcast;
  torch::Tensor output = torch::cat({output_first, output_second}, -1);
  return output.to(input.dtype());
}

}  // namespace

class MiniMaxM3VisionPatchEmbedImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3VisionPatchEmbedImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions& options = context.get_tensor_options();

    const int64_t in_features =
        model_args.mm_num_channels() * model_args.mm_temporal_patch_size() *
        model_args.mm_patch_size() * model_args.mm_patch_size();
    const int64_t out_features = model_args.mm_hidden_size();
    proj_ = register_module(
        "patch_embedding",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(false)));
    proj_->weight.set_data(proj_->weight.to(options));
  }

  torch::Tensor forward(torch::Tensor input) { return proj_(input); }

  void load_state_dict(const StateDict& state_dict) {
    load_linear_weight(
        state_dict, "patch_embedding.weight", proj_, weight_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_loaded_) << "weight is not loaded for " << prefix
                          << "patch_embedding.weight";
  }

 private:
  torch::nn::Linear proj_{nullptr};
  bool weight_loaded_ = false;
};
TORCH_MODULE(MiniMaxM3VisionPatchEmbed);

class MiniMaxM3VisionEncoderLayerImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3VisionEncoderLayerImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions& options = context.get_tensor_options();

    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    head_dim_ = hidden_size_ / num_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    layer_norm1_ = register_module(
        "layer_norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                 .eps(model_args.mm_layer_norm_eps())));
    layer_norm2_ = register_module(
        "layer_norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                 .eps(model_args.mm_layer_norm_eps())));
    q_proj_ = register_module(
        "q_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    k_proj_ = register_module(
        "k_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    v_proj_ = register_module(
        "v_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    out_proj_ = register_module(
        "out_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    fc1_ = register_module(
        "fc1",
        torch::nn::Linear(torch::nn::LinearOptions(
                              hidden_size_, model_args.mm_intermediate_size())
                              .bias(true)));
    fc2_ = register_module(
        "fc2",
        torch::nn::Linear(torch::nn::LinearOptions(
                              model_args.mm_intermediate_size(), hidden_size_)
                              .bias(true)));

    layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
    layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
    layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
    layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
    q_proj_->weight.set_data(q_proj_->weight.to(options));
    q_proj_->bias.set_data(q_proj_->bias.to(options));
    k_proj_->weight.set_data(k_proj_->weight.to(options));
    k_proj_->bias.set_data(k_proj_->bias.to(options));
    v_proj_->weight.set_data(v_proj_->weight.to(options));
    v_proj_->bias.set_data(v_proj_->bias.to(options));
    out_proj_->weight.set_data(out_proj_->weight.to(options));
    out_proj_->bias.set_data(out_proj_->bias.to(options));
    fc1_->weight.set_data(fc1_->weight.to(options));
    fc1_->bias.set_data(fc1_->bias.to(options));
    fc2_->weight.set_data(fc2_->weight.to(options));
    fc2_->bias.set_data(fc2_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin) {
    torch::Tensor residual = hidden_states;
    hidden_states = layer_norm1_(hidden_states);
    hidden_states = residual + attention_forward(hidden_states, cos, sin);

    residual = hidden_states;
    hidden_states = layer_norm2_(hidden_states);
    hidden_states = fc2_(torch::gelu(fc1_(hidden_states)));
    return residual + hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    load_layer_norm_weight(state_dict.get_dict_with_prefix("layer_norm1."),
                           layer_norm1_,
                           layer_norm1_weight_loaded_,
                           layer_norm1_bias_loaded_);
    load_layer_norm_weight(state_dict.get_dict_with_prefix("layer_norm2."),
                           layer_norm2_,
                           layer_norm2_weight_loaded_,
                           layer_norm2_bias_loaded_);
    const StateDict attn_dict = state_dict.get_dict_with_prefix("self_attn.");
    load_linear_weight(attn_dict, "q_proj.weight", q_proj_, q_weight_loaded_);
    load_linear_bias(attn_dict, "q_proj.bias", q_proj_, q_bias_loaded_);
    load_linear_weight(attn_dict, "k_proj.weight", k_proj_, k_weight_loaded_);
    load_linear_bias(attn_dict, "k_proj.bias", k_proj_, k_bias_loaded_);
    load_linear_weight(attn_dict, "v_proj.weight", v_proj_, v_weight_loaded_);
    load_linear_bias(attn_dict, "v_proj.bias", v_proj_, v_bias_loaded_);
    load_linear_weight(
        attn_dict, "out_proj.weight", out_proj_, out_weight_loaded_);
    load_linear_bias(attn_dict, "out_proj.bias", out_proj_, out_bias_loaded_);
    const StateDict mlp_dict = state_dict.get_dict_with_prefix("mlp.");
    load_linear_weight(mlp_dict, "fc1.weight", fc1_, fc1_weight_loaded_);
    load_linear_bias(mlp_dict, "fc1.bias", fc1_, fc1_bias_loaded_);
    load_linear_weight(mlp_dict, "fc2.weight", fc2_, fc2_weight_loaded_);
    load_linear_bias(mlp_dict, "fc2.bias", fc2_, fc2_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(layer_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix << "layer_norm1.weight";
    CHECK(layer_norm1_bias_loaded_)
        << "bias is not loaded for " << prefix << "layer_norm1.bias";
    CHECK(layer_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix << "layer_norm2.weight";
    CHECK(layer_norm2_bias_loaded_)
        << "bias is not loaded for " << prefix << "layer_norm2.bias";
    CHECK(q_weight_loaded_)
        << "weight is not loaded for " << prefix << "self_attn.q_proj.weight";
    CHECK(q_bias_loaded_) << "bias is not loaded for " << prefix
                          << "self_attn.q_proj.bias";
    CHECK(k_weight_loaded_)
        << "weight is not loaded for " << prefix << "self_attn.k_proj.weight";
    CHECK(k_bias_loaded_) << "bias is not loaded for " << prefix
                          << "self_attn.k_proj.bias";
    CHECK(v_weight_loaded_)
        << "weight is not loaded for " << prefix << "self_attn.v_proj.weight";
    CHECK(v_bias_loaded_) << "bias is not loaded for " << prefix
                          << "self_attn.v_proj.bias";
    CHECK(out_weight_loaded_)
        << "weight is not loaded for " << prefix << "self_attn.out_proj.weight";
    CHECK(out_bias_loaded_)
        << "bias is not loaded for " << prefix << "self_attn.out_proj.bias";
    CHECK(fc1_weight_loaded_)
        << "weight is not loaded for " << prefix << "mlp.fc1.weight";
    CHECK(fc1_bias_loaded_)
        << "bias is not loaded for " << prefix << "mlp.fc1.bias";
    CHECK(fc2_weight_loaded_)
        << "weight is not loaded for " << prefix << "mlp.fc2.weight";
    CHECK(fc2_bias_loaded_)
        << "bias is not loaded for " << prefix << "mlp.fc2.bias";
  }

 private:
  torch::Tensor attention_forward(torch::Tensor hidden_states,
                                  const torch::Tensor& cos,
                                  const torch::Tensor& sin) {
    const int64_t seq_len = hidden_states.size(0);
    torch::Tensor query =
        q_proj_(hidden_states).view({seq_len, num_heads_, head_dim_});
    torch::Tensor key =
        k_proj_(hidden_states).view({seq_len, num_heads_, head_dim_});
    torch::Tensor value =
        v_proj_(hidden_states).view({seq_len, num_heads_, head_dim_});

    query = apply_rotary_pos_emb(query, cos, sin);
    key = apply_rotary_pos_emb(key, cos, sin);

    torch::Tensor query_heads = query.transpose(0, 1).to(torch::kFloat32);
    torch::Tensor key_heads = key.transpose(0, 1).to(torch::kFloat32);
    torch::Tensor value_heads = value.transpose(0, 1).to(torch::kFloat32);
    torch::Tensor scores =
        torch::matmul(query_heads, key_heads.transpose(-2, -1)) * scale_;
    scores = torch::softmax(scores, -1);
    torch::Tensor context = torch::matmul(scores, value_heads)
                                .to(hidden_states.dtype())
                                .transpose(0, 1)
                                .contiguous()
                                .view({seq_len, hidden_size_});
    return out_proj_(context);
  }

  int64_t hidden_size_ = 0;
  int64_t num_heads_ = 0;
  int64_t head_dim_ = 0;
  double scale_ = 1.0;

  torch::nn::LayerNorm layer_norm1_{nullptr};
  torch::nn::LayerNorm layer_norm2_{nullptr};
  torch::nn::Linear q_proj_{nullptr};
  torch::nn::Linear k_proj_{nullptr};
  torch::nn::Linear v_proj_{nullptr};
  torch::nn::Linear out_proj_{nullptr};
  torch::nn::Linear fc1_{nullptr};
  torch::nn::Linear fc2_{nullptr};

  bool layer_norm1_weight_loaded_ = false;
  bool layer_norm1_bias_loaded_ = false;
  bool layer_norm2_weight_loaded_ = false;
  bool layer_norm2_bias_loaded_ = false;
  bool q_weight_loaded_ = false;
  bool q_bias_loaded_ = false;
  bool k_weight_loaded_ = false;
  bool k_bias_loaded_ = false;
  bool v_weight_loaded_ = false;
  bool v_bias_loaded_ = false;
  bool out_weight_loaded_ = false;
  bool out_bias_loaded_ = false;
  bool fc1_weight_loaded_ = false;
  bool fc1_bias_loaded_ = false;
  bool fc2_weight_loaded_ = false;
  bool fc2_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxM3VisionEncoderLayer);

class MiniMaxM3VisionTransformerImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3VisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    const ModelArgs& model_args = context.get_model_args();
    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    head_dim_ = hidden_size_ / num_heads_;
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    theta_ = 10000.0;

    patch_embed_ =
        register_module("embeddings", MiniMaxM3VisionPatchEmbed(context));
    pre_layrnorm_ = register_module(
        "pre_layrnorm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                 .eps(model_args.mm_layer_norm_eps())));
    pre_layrnorm_->weight.set_data(pre_layrnorm_->weight.to(options_));
    pre_layrnorm_->bias.set_data(pre_layrnorm_->bias.to(options_));

    layers_.reserve(model_args.mm_num_hidden_layers());
    for (int32_t index = 0; index < model_args.mm_num_hidden_layers();
         ++index) {
      layers_.push_back(register_module(std::to_string(index),
                                        MiniMaxM3VisionEncoderLayer(context)));
    }
  }

  torch::Tensor forward(torch::Tensor pixel_values, torch::Tensor grid_thw) {
    torch::Tensor hidden_states = patch_embed_(pixel_values);
    hidden_states = pre_layrnorm_(hidden_states);

    std::vector<torch::Tensor> image_outputs;
    image_outputs.reserve(grid_thw.size(0));
    torch::Tensor grid_cpu = grid_thw.cpu();
    int64_t offset = 0;
    const int64_t image_count = grid_cpu.size(0);
    for (int64_t image_index = 0; image_index < image_count; ++image_index) {
      const int64_t grid_t = grid_cpu[image_index][0].item<int64_t>();
      const int64_t grid_h = grid_cpu[image_index][1].item<int64_t>();
      const int64_t grid_w = grid_cpu[image_index][2].item<int64_t>();
      const int64_t patch_count = grid_t * grid_h * grid_w;
      torch::Tensor image_states = hidden_states.index(
          {torch::indexing::Slice(offset, offset + patch_count)});
      torch::Tensor rotary_emb = rotary_pos_emb(grid_t, grid_h, grid_w);
      torch::Tensor cos = rotary_emb.cos().to(hidden_states.dtype());
      torch::Tensor sin = rotary_emb.sin().to(hidden_states.dtype());
      for (size_t layer_index = 0; layer_index < layers_.size();
           ++layer_index) {
        image_states = layers_[layer_index](image_states, cos, sin);
      }
      image_outputs.push_back(image_states);
      offset += patch_count;
    }

    return torch::cat(image_outputs, 0);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    load_layer_norm_weight(state_dict.get_dict_with_prefix("pre_layrnorm."),
                           pre_layrnorm_,
                           pre_layrnorm_weight_loaded_,
                           pre_layrnorm_bias_loaded_);
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->load_state_dict(state_dict.get_dict_with_prefix(
          "encoder.layers." + std::to_string(index) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "embeddings.");
    CHECK(pre_layrnorm_weight_loaded_)
        << "weight is not loaded for " << prefix << "pre_layrnorm.weight";
    CHECK(pre_layrnorm_bias_loaded_)
        << "bias is not loaded for " << prefix << "pre_layrnorm.bias";
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->verify_loaded_weights(prefix + "encoder.layers." +
                                            std::to_string(index) + ".");
    }
  }

 private:
  torch::Tensor rotary_pos_emb(int64_t grid_t,
                               int64_t grid_h,
                               int64_t grid_w) const {
    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kLong).device(options_.device());
    torch::Tensor hpos_ids =
        torch::arange(grid_h, options).unsqueeze(1).expand({-1, grid_w});
    hpos_ids = hpos_ids
                   .reshape({grid_h / spatial_merge_size_,
                             spatial_merge_size_,
                             grid_w / spatial_merge_size_,
                             spatial_merge_size_})
                   .permute({0, 2, 1, 3})
                   .flatten();
    torch::Tensor wpos_ids =
        torch::arange(grid_w, options).unsqueeze(0).expand({grid_h, -1});
    wpos_ids = wpos_ids
                   .reshape({grid_h / spatial_merge_size_,
                             spatial_merge_size_,
                             grid_w / spatial_merge_size_,
                             spatial_merge_size_})
                   .permute({0, 2, 1, 3})
                   .flatten();
    hpos_ids = hpos_ids.repeat({grid_t});
    wpos_ids = wpos_ids.repeat({grid_t});
    torch::Tensor tpos_ids =
        torch::arange(grid_t, options).repeat_interleave(grid_h * grid_w);

    const int64_t half_dim = head_dim_ / 2;
    const int64_t temporal_section = half_dim / 3;
    const int64_t height_section = (half_dim - temporal_section) / 2;
    const int64_t width_section = half_dim - temporal_section - height_section;
    return torch::cat({build_axis_freq(tpos_ids, temporal_section, theta_),
                       build_axis_freq(hpos_ids, height_section, theta_),
                       build_axis_freq(wpos_ids, width_section, theta_)},
                      -1);
  }

  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  int64_t num_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t spatial_merge_size_ = 2;
  double theta_ = 10000.0;
  MiniMaxM3VisionPatchEmbed patch_embed_{nullptr};
  torch::nn::LayerNorm pre_layrnorm_{nullptr};
  std::vector<MiniMaxM3VisionEncoderLayer> layers_;
  bool pre_layrnorm_weight_loaded_ = false;
  bool pre_layrnorm_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxM3VisionTransformer);

class MiniMaxM3PatchMergeMLPImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3PatchMergeMLPImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    hidden_size_ = model_args.hidden_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    const int64_t merged_hidden_size =
        hidden_size_ * spatial_merge_size_ * spatial_merge_size_;

    linear_1_ =
        register_module("linear_1",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              merged_hidden_size, hidden_size_)
                                              .bias(true)));
    linear_2_ = register_module(
        "linear_2",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    linear_1_->weight.set_data(linear_1_->weight.to(options));
    linear_1_->bias.set_data(linear_1_->bias.to(options));
    linear_2_->weight.set_data(linear_2_->weight.to(options));
    linear_2_->bias.set_data(linear_2_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor image_embeds,
                        const torch::Tensor& grid_thw) {
    torch::Tensor grid_cpu = grid_thw.cpu();
    std::vector<torch::Tensor> outputs;
    outputs.reserve(grid_cpu.size(0));
    int64_t offset = 0;
    const int64_t merge_length = spatial_merge_size_ * spatial_merge_size_;
    const int64_t image_count = grid_cpu.size(0);
    for (int64_t image_index = 0; image_index < image_count; ++image_index) {
      const int64_t patch_count = grid_cpu[image_index].prod().item<int64_t>();
      const int64_t token_count = patch_count / merge_length;
      torch::Tensor image_slice =
          image_embeds.slice(/*dim=*/0, offset, offset + patch_count)
              .contiguous()
              .view({token_count, hidden_size_ * merge_length});
      outputs.push_back(linear_2_(torch::gelu(linear_1_(image_slice))));
      offset += patch_count;
    }
    return torch::cat(outputs, 0);
  }

  void load_state_dict(const StateDict& state_dict) {
    load_linear_weight(
        state_dict, "linear_1.weight", linear_1_, linear_1_weight_loaded_);
    load_linear_bias(
        state_dict, "linear_1.bias", linear_1_, linear_1_bias_loaded_);
    load_linear_weight(
        state_dict, "linear_2.weight", linear_2_, linear_2_weight_loaded_);
    load_linear_bias(
        state_dict, "linear_2.bias", linear_2_, linear_2_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(linear_1_weight_loaded_)
        << "weight is not loaded for " << prefix << "linear_1.weight";
    CHECK(linear_1_bias_loaded_)
        << "bias is not loaded for " << prefix << "linear_1.bias";
    CHECK(linear_2_weight_loaded_)
        << "weight is not loaded for " << prefix << "linear_2.weight";
    CHECK(linear_2_bias_loaded_)
        << "bias is not loaded for " << prefix << "linear_2.bias";
  }

 private:
  int64_t hidden_size_ = 0;
  int64_t spatial_merge_size_ = 2;
  torch::nn::Linear linear_1_{nullptr};
  torch::nn::Linear linear_2_{nullptr};
  bool linear_1_weight_loaded_ = false;
  bool linear_1_bias_loaded_ = false;
  bool linear_2_weight_loaded_ = false;
  bool linear_2_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxM3PatchMergeMLP);

class MiniMaxM3MultiModalProjectorImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3MultiModalProjectorImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    linear_1_ = register_module(
        "linear_1",
        torch::nn::Linear(
            torch::nn::LinearOptions(model_args.mm_hidden_size(),
                                     model_args.mm_projector_hidden_size())
                .bias(true)));
    linear_2_ = register_module(
        "linear_2",
        torch::nn::Linear(
            torch::nn::LinearOptions(model_args.mm_projector_hidden_size(),
                                     model_args.hidden_size())
                .bias(true)));
    linear_1_->weight.set_data(linear_1_->weight.to(options));
    linear_1_->bias.set_data(linear_1_->bias.to(options));
    linear_2_->weight.set_data(linear_2_->weight.to(options));
    linear_2_->bias.set_data(linear_2_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor image_features) {
    return linear_2_(torch::gelu(linear_1_(image_features)));
  }

  void load_state_dict(const StateDict& state_dict) {
    load_linear_weight(
        state_dict, "linear_1.weight", linear_1_, linear_1_weight_loaded_);
    load_linear_bias(
        state_dict, "linear_1.bias", linear_1_, linear_1_bias_loaded_);
    load_linear_weight(
        state_dict, "linear_2.weight", linear_2_, linear_2_weight_loaded_);
    load_linear_bias(
        state_dict, "linear_2.bias", linear_2_, linear_2_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(linear_1_weight_loaded_)
        << "weight is not loaded for " << prefix << "linear_1.weight";
    CHECK(linear_1_bias_loaded_)
        << "bias is not loaded for " << prefix << "linear_1.bias";
    CHECK(linear_2_weight_loaded_)
        << "weight is not loaded for " << prefix << "linear_2.weight";
    CHECK(linear_2_bias_loaded_)
        << "bias is not loaded for " << prefix << "linear_2.bias";
  }

 private:
  torch::nn::Linear linear_1_{nullptr};
  torch::nn::Linear linear_2_{nullptr};
  bool linear_1_weight_loaded_ = false;
  bool linear_1_bias_loaded_ = false;
  bool linear_2_weight_loaded_ = false;
  bool linear_2_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxM3MultiModalProjector);

struct MiniMaxM3VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

class MiniMaxM3ForConditionalGenerationImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxM3ForConditionalGenerationImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    visual_ =
        register_module("vision_tower", MiniMaxM3VisionTransformer(context));
    multi_modal_projector_ = register_module(
        "multi_modal_projector", MiniMaxM3MultiModalProjector(context));
    patch_merge_mlp_ =
        register_module("patch_merge_mlp", MiniMaxM3PatchMergeMLP(context));
    language_model_ =
        register_module("language_model", MiniMaxM3ForCausalLM(context));
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<MiniMaxM3VLImageInputs> image_input =
        prepare_encoder_input(input_params);
    MMDict multimodal_embeds;
    if (!image_input) {
      return multimodal_embeds;
    }

    torch::Tensor image_pixels = image_input->pixel_values.to(options_);
    torch::Tensor image_grid =
        image_input->image_grid_thw.to(options_.device()).reshape({-1, 3});
    torch::Tensor image_features = visual_(image_pixels, image_grid);
    torch::Tensor image_embeds = multi_modal_projector_(image_features);
    image_embeds = patch_merge_mlp_(image_embeds, image_grid);
    std::vector<int32_t> image_token_nums =
        get_mm_token_nums(input_params.multimodal.mm_data, MMType::IMAGE);
    multimodal_embeds["image|embedding"] =
        split_by_token_nums(image_embeds, image_token_nums);
    return multimodal_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const MMBatchData& mm_data = input_params.multimodal.mm_data;
    torch::Tensor inputs_embeds =
        language_model_->get_word_embedding()(input_ids);
    const std::optional<torch::Tensor> image_embeds =
        mm_data.get<torch::Tensor>("image|embedding");
    const std::optional<torch::Tensor> image_mask =
        mm_data.get<torch::Tensor>("image|mask");
    if (image_embeds && image_mask) {
      inputs_embeds.index_put_({image_mask.value()}, image_embeds.value());
    }
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    return language_model_->logits(hidden_states, selected_idxes);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    return language_model_->pooler(hidden_states, selected_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("vision_tower.vision_model."));
      multi_modal_projector_->load_state_dict(
          state_dict->get_dict_with_prefix("multi_modal_projector."));
      patch_merge_mlp_->load_state_dict(
          state_dict->get_dict_with_prefix("patch_merge_mlp."));
    }
    visual_->verify_loaded_weights("vision_tower.vision_model.");
    multi_modal_projector_->verify_loaded_weights("multi_modal_projector.");
    patch_merge_mlp_->verify_loaded_weights("patch_merge_mlp.");
    language_model_->load_model(std::move(loader));
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }

  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  std::optional<MiniMaxM3VLImageInputs> prepare_encoder_input(
      const ModelInputParams& input_params) const {
    const MMBatchData& mm_data = input_params.multimodal.mm_data;
    torch::Tensor pixel_values;
    if (const std::optional<torch::Tensor>& res =
            mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }
    torch::Tensor image_grid_thw;
    if (const std::optional<torch::Tensor>& res =
            mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }
    if (pixel_values.defined() && image_grid_thw.defined()) {
      return MiniMaxM3VLImageInputs{pixel_values, image_grid_thw};
    }
    return std::nullopt;
  }

  torch::TensorOptions options_;
  MiniMaxM3VisionTransformer visual_{nullptr};
  MiniMaxM3MultiModalProjector multi_modal_projector_{nullptr};
  MiniMaxM3PatchMergeMLP patch_merge_mlp_{nullptr};
  MiniMaxM3ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(MiniMaxM3ForConditionalGeneration);

using MiniMaxM3VLMultimodalProcessor =
    MultimodalProcessor<MiniMaxM3VLPromptProcessor, MiniMaxM3VLImageProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(minimax_m3_vl, MiniMaxM3VLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(minimax_m3_vl_vlm,
                                       minimax_m3_vl,
                                       MiniMaxM3ForConditionalGeneration);

}  // namespace xllm::npu::model

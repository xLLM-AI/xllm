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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/nn/options/vision.h>
#include <torch/torch.h>

#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "models/llm/npu/glm4.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "processors/glm4v_image_processor.h"
#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "xllm/core/layers/npu/npu_glm4_vision_encoder_layer_impl.h"

namespace xllm::npu::model {

class Glm4VisionRmsNormImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  Glm4VisionRmsNormImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    weight = torch::empty({model_args.mm_hidden_size()}, options);
    epsilon_ = 1e-5;
  }

  torch::Tensor forward(torch::Tensor& x) {
    auto results =
        at_npu::native::custom_ops::npu_rms_norm(x, weight, epsilon_);
    return std::get<0>(results);
  }

 private:
  double epsilon_;
};
TORCH_MODULE(Glm4VisionRmsNorm);

class Glm4VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Glm4VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_temporal_patch_size() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();

    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(true)));

    proj_->weight.set_data(proj_->weight.to(options));
    proj_->bias.set_data(proj_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor x) { return proj_(x); }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }
    auto bias = state_dict.get_tensor("proj.bias");
    if (bias.defined()) {
      bias = bias.reshape({bias.size(0)});
      DCHECK_EQ(proj_->bias.sizes(), bias.sizes())
          << "proj bias size mismatch for " << name();
      proj_->bias.data().copy_(bias);
      proj_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(proj_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  bool proj_weight_loaded_ = false;
  bool proj_bias_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Glm4VisionPatchEmbed);

class Glm4_VisionBlockImpl : public torch::nn::Module {
 public:
  Glm4_VisionBlockImpl(const ModelContext& context) {
    // register submodules
    encoder_layer_ = register_module("encoder_layer",
                                     layer::NpuGlm4VisionEncoderLayer(context));
  }
  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int>& cu_seq_len_vec,
                        int node_id) {
    return encoder_layer_(
        x, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec, node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuGlm4VisionEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Glm4_VisionBlock);

class Glm4VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  Glm4VisionRotaryEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    dim_ = model_args.mm_head_dim() / 2;
    theta_ = 10000.0;

    auto opts = options.dtype(torch::kFloat32);
    auto inv_freq =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, opts) / dim_);
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) return;

    seqlen *= 2;
    seq_len_cached_ = seqlen;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(inv_freq_.device());
    inv_freq_ =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, options) / dim_);
    auto seq = torch::arange(seqlen, options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(0, 0, seqlen);
  }

 private:
  int dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Glm4VisionRotaryEmbedding);

class Glm4vVisionEmbeddingsImpl : public torch::nn::Module {
 public:
  Glm4vVisionEmbeddingsImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_dim_ = model_args.mm_hidden_size();
    image_size_ = model_args.mm_image_size();
    patch_size_ = model_args.mm_patch_size();
    num_positions_ = image_size_ / patch_size_;
    num_positions_ = num_positions_ * num_positions_;
    position_embedding_ = register_module(
        "position_embedding", torch::nn::Embedding(num_positions_, embed_dim_));
    position_embedding_->weight.set_data(
        position_embedding_->weight.to(options));
  }
  torch::Tensor forward(torch::Tensor x,
                        std::vector<int> lengths,
                        torch::Tensor image_shapes,
                        torch::Tensor h_coords,
                        torch::Tensor w_coords) {
    const auto& pos_embed_weight = position_embedding_->weight;
    const int64_t hidden_size = pos_embed_weight.size(1);
    const int64_t total_seq = x.size(0);
    const auto device = pos_embed_weight.device();
    const auto dtype = pos_embed_weight.dtype();

    image_shapes = image_shapes.to(device);
    h_coords = h_coords.to(device);
    w_coords = w_coords.to(device);
    x = x.to(device, dtype);

    torch::Tensor adapted_pos_embed;
    if (total_seq == 0) {
      adapted_pos_embed = torch::empty(
          {0, hidden_size}, torch::TensorOptions().device(device).dtype(dtype));
    } else {
      const int64_t batch_size = static_cast<int64_t>(lengths.size());
      const int64_t orig_size_sq = pos_embed_weight.size(0);
      const int64_t orig_size = static_cast<int64_t>(std::sqrt(orig_size_sq));
      auto pos_embed_2d =
          pos_embed_weight.view({orig_size, orig_size, hidden_size})
              .permute({2, 0, 1})
              .unsqueeze(0)
              .to(torch::kFloat32);

      std::vector<torch::Tensor> target_h_list;
      std::vector<torch::Tensor> target_w_list;
      target_h_list.reserve(batch_size);
      target_w_list.reserve(batch_size);
      for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t seq_len = lengths[i];
        const auto img_h = image_shapes.index({i, 1}).to(torch::kFloat32);
        const auto img_w = image_shapes.index({i, 2}).to(torch::kFloat32);

        target_h_list.push_back(img_h.repeat({seq_len}));
        target_w_list.push_back(img_w.repeat({seq_len}));
      }

      auto target_h = torch::cat(target_h_list, 0);
      auto target_w = torch::cat(target_w_list, 0);

      auto h_coords_fp32 = h_coords.to(torch::kFloat32);
      auto w_coords_fp32 = w_coords.to(torch::kFloat32);

      const auto norm_w = ((w_coords_fp32 + 0.5f) / target_w) * 2.0f - 1.0f;
      const auto norm_h = ((h_coords_fp32 + 0.5f) / target_h) * 2.0f - 1.0f;
      auto grid = torch::stack({norm_w, norm_h}, -1).unsqueeze(0).unsqueeze(2);
      namespace F = torch::nn::functional;
      auto interpolated_embed = F::grid_sample(pos_embed_2d,
                                               grid,
                                               F::GridSampleFuncOptions()
                                                   .mode(torch::kBicubic)
                                                   .padding_mode(torch::kBorder)
                                                   .align_corners(false));
      adapted_pos_embed =
          interpolated_embed.squeeze(0).squeeze(-1).permute({1, 0}).to(dtype);
    }

    return x + adapted_pos_embed;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("position_embedding.weight");
    if (weight.defined()) {
      position_embedding_->weight.data().copy_(weight);
      position_embedding_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(position_embedding_weight_loaded_)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
  }

 private:
  int64_t embed_dim_ = 0;
  int64_t image_size_ = 0;
  int64_t patch_size_ = 0;
  int64_t num_positions_ = 0;
  bool position_embedding_weight_loaded_ = false;
  torch::nn::Embedding position_embedding_{nullptr};
};
TORCH_MODULE(Glm4vVisionEmbeddings);

class Glm4_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Glm4_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    options_ = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    int64_t dim = model_args.mm_projection_dim();
    int64_t context_dim = model_args.mm_intermediate_size();
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm_->weight.set_data(norm_->weight.to(options_));
    norm_->bias.set_data(norm_->bias.to(options_));
    proj_ = register_module(
        "proj",
        torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
    proj_->weight.set_data(proj_->weight.to(options_));
    act_ = register_module("act", torch::nn::GELU());
    silu_ = register_module("silu", torch::nn::SiLU());

    gate_ = register_module(
        "gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, context_dim).bias(false)));
    gate_->weight.set_data(gate_->weight.to(options_));
    up_ = register_module(
        "up",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, context_dim).bias(false)));
    up_->weight.set_data(up_->weight.to(options_));
    down_ = register_module(
        "down",
        torch::nn::Linear(
            torch::nn::LinearOptions(context_dim, dim).bias(false)));
    down_->weight.set_data(down_->weight.to(options_));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = proj_(x);
    x = act_(norm_(x));
    x = down_(torch::mul(silu_((gate_(x))), up_(x)));
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm
    const auto& norm_dict =
        state_dict.get_dict_with_prefix("post_projection_norm.");
    const auto& norm_weight = norm_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(norm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      norm_->weight.data().copy_(norm_weight);
      is_norm_weight_loaded = true;
    }
    const auto norm_bias = norm_dict.get_tensor("bias");
    if (norm_bias.defined()) {
      CHECK_EQ(norm_->bias.sizes(), norm_bias.sizes())
          << "bias size mismatch for " << name();
      norm_->bias.data().copy_(norm_bias);
      is_norm_bias_loaded = true;
    }

    const auto& proj_dict = state_dict.get_dict_with_prefix("proj.");
    const auto& proj_weight = proj_dict.get_tensor("weight");
    if (proj_weight.defined()) {
      proj_->weight.data().copy_(proj_weight);
      is_proj_weight_loaded = true;
    }

    const auto& up_dict = state_dict.get_dict_with_prefix("up_proj.");
    const auto& up_weight = up_dict.get_tensor("weight");
    if (up_weight.defined()) {
      up_->weight.data().copy_(up_weight);
      is_up_weight_loaded = true;
    }

    const auto& down_dict = state_dict.get_dict_with_prefix("down_proj.");
    const auto& down_weight = down_dict.get_tensor("weight");
    if (down_weight.defined()) {
      down_->weight.data().copy_(down_weight);
      is_down_weight_loaded = true;
    }

    const auto& gate_dict = state_dict.get_dict_with_prefix("gate_proj.");
    const auto& gate_weight = gate_dict.get_tensor("weight");
    if (gate_weight.defined()) {
      gate_->weight.data().copy_(gate_weight);
      is_gate_weight_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_proj_weight_loaded)
        << "weight is not loaded for " << prefix + "proj_weight" + ".weight";
    CHECK(is_up_weight_loaded)
        << "weight is not loaded for " << prefix + "up_weight" + ".weight";
    CHECK(is_down_weight_loaded)
        << "weight is not loaded for " << prefix + "down_weight" + ".weight";
    CHECK(is_gate_weight_loaded)
        << "weight is not loaded for " << prefix + "gate_weight" + ".weight";
    CHECK(is_norm_weight_loaded)
        << "weight is not loaded for " << prefix + "norm" + ".weight";
    CHECK(is_norm_bias_loaded)
        << "bias is not loaded for " << prefix + "norm" + ".bias";
  }

 private:
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear proj_{nullptr};
  torch::nn::Linear up_{nullptr};
  torch::nn::Linear gate_{nullptr};
  torch::nn::Linear down_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::nn::SiLU silu_{nullptr};
  torch::TensorOptions options_;

  bool is_proj_weight_loaded = false;
  bool is_up_weight_loaded = false;
  bool is_down_weight_loaded = false;
  bool is_gate_weight_loaded = false;
  bool is_norm_weight_loaded = false;
  bool is_norm_bias_loaded = false;
};
TORCH_MODULE(Glm4_VisionPatchMerger);

class Glm4VisionTransformerImpl : public torch::nn::Module {
 public:
  Glm4VisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    hidden_size_ = model_args.mm_hidden_size();
    out_hidden_size_ = model_args.mm_projection_dim();

    patch_embed_ =
        register_module("patch_embed", Glm4VisionPatchEmbed(context));
    rotary_pos_emb_ =
        register_module("rotary_pos_emb", Glm4VisionRotaryEmbedding(context));
    post_conv_layernorm_ =
        register_module("post_conv_layernorm", Glm4VisionRmsNorm(context));

    embeddings_ = register_module("embeddings", Glm4vVisionEmbeddings(context));

    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = Glm4_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    post_layernorm_ =
        register_module("post_layernorm", Glm4VisionRmsNorm(context));

    downsample_ = register_module(
        "downsample",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_size_,
                                                   out_hidden_size_,
                                                   spatial_merge_size_)
                              .stride(spatial_merge_size_)
                              .bias(true)
                              .padding(0)));
    downsample_->weight.set_data(downsample_->weight.to(options_));
    downsample_->bias.set_data(downsample_->bias.to(options_));
    merger_ = register_module("merger", Glm4_VisionPatchMerger(context));
  }
  std::tuple<torch::Tensor, torch::Tensor> rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    auto count = grid_thw.sizes()[0];
    pos_ids_vec.reserve(count);
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    auto grid_thw_cpu = grid_thw.cpu();
    for (int idx = 0; idx < count; ++idx) {
      auto t = grid_thw_cpu[idx][0].item<int64_t>();
      auto h = grid_thw_cpu[idx][1].item<int64_t>();
      auto w = grid_thw_cpu[idx][2].item<int64_t>();
      auto hpos_ids = torch::arange(h, options).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();
      auto wpos_ids = torch::arange(w, options).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();
      pos_ids_vec.push_back(
          torch::stack({hpos_ids, wpos_ids}, -1).repeat({t, 1}));
    }
    auto pos_ids = torch::cat(pos_ids_vec, 0);
    auto max_grid_size =
        grid_thw
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None)})
            .max();
    auto rotary_pos_emb_full = rotary_pos_emb_(max_grid_size.item<int64_t>());
    auto rotary_pos_emb = rotary_pos_emb_full.index({pos_ids}).flatten(1);

    return std::make_tuple(rotary_pos_emb, pos_ids);
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor grid_thw) {
    hidden_states = patch_embed_(hidden_states);
    hidden_states = post_conv_layernorm_(hidden_states);

    auto [rotary_pos_emb, image_type_ids] = rot_pos_emb(grid_thw);
    auto emb = torch::cat({rotary_pos_emb, rotary_pos_emb}, -1);
    auto m_cos = emb.cos().type_as(hidden_states);
    auto m_sin = emb.sin().type_as(hidden_states);

    auto device = grid_thw.device();
    auto grid_t = grid_thw.index_select(
        1,
        torch::tensor(
            {0}, torch::TensorOptions().dtype(torch::kInt).device(device)));
    auto grid_h = grid_thw.index_select(
        1,
        torch::tensor(
            {1}, torch::TensorOptions().dtype(torch::kInt).device(device)));
    auto grid_w = grid_thw.index_select(
        1,
        torch::tensor(
            {2}, torch::TensorOptions().dtype(torch::kInt).device(device)));
    auto h_times_w = (grid_h * grid_w).squeeze(1);
    auto repeats = grid_t.squeeze(1);
    auto repeated = torch::repeat_interleave(h_times_w, repeats, 0);
    c10::optional<torch::ScalarType> cumsum_dtype;

    cumsum_dtype = torch::kInt32;
    auto cu_seqlens = torch::cumsum(repeated, 0, cumsum_dtype);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));
    cu_seqlens = torch::diff(cu_seqlens).cpu().to(torch::kInt);
    std::vector<int> seqlens;
    seqlens.assign(cu_seqlens.data_ptr<int>(),
                   cu_seqlens.data_ptr<int>() + cu_seqlens.numel());

    hidden_states = embeddings_(hidden_states,
                                seqlens,
                                grid_thw,
                                image_type_ids.select(1, 0),
                                image_type_ids.select(1, 1));
    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int>(),
        cu_seqlens_cpu.data_ptr<int>() + cu_seqlens_cpu.numel());
    cu_seqlens = cu_seqlens.to(hidden_states.device());
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      hidden_states = layers_[idx](
          hidden_states, m_cos, m_sin, cu_seqlens, cu_seqlens_vec, idx);
    }
    hidden_states = post_layernorm_(hidden_states);
    hidden_states = hidden_states.view(
        {-1, spatial_merge_size_, spatial_merge_size_, hidden_states.size(-1)});
    hidden_states = hidden_states.permute({0, 3, 1, 2});
    hidden_states = downsample_(hidden_states).view({-1, out_hidden_size_});
    hidden_states = merger_(hidden_states);
    return hidden_states;
  };

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    const auto& norm_weight =
        state_dict.get_dict_with_prefix("post_conv_layernorm.")
            .get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(post_conv_layernorm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      post_conv_layernorm_->weight.data().copy_(norm_weight);
      is_post_conv_layernorm_weight_loaded = true;
    }
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    const auto& post_norm_weight =
        state_dict.get_dict_with_prefix("post_layernorm.").get_tensor("weight");
    if (post_norm_weight.defined()) {
      CHECK_EQ(post_layernorm_->weight.sizes(), post_norm_weight.sizes())
          << "weight size mismatch for " << name();
      post_layernorm_->weight.data().copy_(post_norm_weight);
      is_post_layernorm_weight_loaded = true;
    }
    const auto& downsample_dict =
        state_dict.get_dict_with_prefix("downsample.");
    const auto& downsample_weight = downsample_dict.get_tensor("weight");
    const auto& downsample_bias = downsample_dict.get_tensor("bias");
    if (downsample_weight.defined()) {
      downsample_->weight.data().copy_(downsample_weight);
      is_downsample_weight_loaded_ = true;
    }
    if (downsample_bias.defined()) {
      downsample_->bias.data().copy_(downsample_bias);
      is_downsample_bias_loaded_ = true;
    }
    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    CHECK(is_post_conv_layernorm_weight_loaded)
        << "weight is not loaded for " << prefix + "post_conv_layernorm.weight";
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    CHECK(is_post_layernorm_weight_loaded)
        << "weight is not loaded for " << prefix + "post_layernorm.weight";
    merger_->verify_loaded_weights(prefix + "merger.");

    CHECK(is_downsample_weight_loaded_)
        << "weight is not loaded for " << prefix + "downsample.weight";
    CHECK(is_downsample_bias_loaded_)
        << "bias is not loaded for " << prefix + "downsample.bias";
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
  }

 private:
  int hidden_size_ = 0;
  int out_hidden_size_ = 0;
  int spatial_merge_size_ = 0;

  Glm4VisionPatchEmbed patch_embed_{nullptr};
  Glm4VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  Glm4vVisionEmbeddings embeddings_{nullptr};
  Glm4VisionRmsNorm post_conv_layernorm_{nullptr};
  Glm4VisionRmsNorm post_layernorm_{nullptr};
  torch::nn::Conv2d downsample_{nullptr};
  std::vector<Glm4_VisionBlock> layers_;
  Glm4_VisionPatchMerger merger_{nullptr};
  torch::TensorOptions options_;
  bool is_post_conv_layernorm_weight_loaded = false;
  bool is_post_layernorm_weight_loaded = false;
  bool is_downsample_weight_loaded_ = false;
  bool is_downsample_bias_loaded_ = false;
  torch::Tensor m_cos;
  torch::Tensor m_sin;
};
TORCH_MODULE(Glm4VisionTransformer);

struct Glm4VImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Glm4VVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
};

class Glm4vForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Glm4vForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    encoder_dp_group_ = context.get_parallel_args().encoder_dp_group_;
    use_encoder_dp_ =
        encoder_dp_group_ != nullptr && encoder_dp_group_->world_size() > 1;
    if (use_encoder_dp_) {
      ModelContext visual_context = context.with_parallel_args(ParallelArgs(
          /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr));
      visual_ =
          register_module("visual", Glm4VisionTransformer(visual_context));
    } else {
      visual_ = register_module("visual", Glm4VisionTransformer(context));
    }

    language_model_ =
        register_module("language_model", Glm4ForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Glm4VImageInputs>& image_inputs,
                             std::optional<Glm4VVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Glm4VImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined())
      video_inputs = Glm4VVideoInputs{pixel_values_videos, video_grid_thw};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Glm4VImageInputs> image_input;
    std::optional<Glm4VVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    if (image_input) {
      torch::Tensor image_pixels = image_input->pixel_values.to(options_);
      torch::Tensor image_grid =
          image_input->image_grid_thw.to(options_.device());
      std::vector<int32_t> image_token_nums =
          get_mm_token_nums(input_params.multimodal.mm_data, MMType::IMAGE);
      if (!use_encoder_dp_) {
        auto image_embeds = visual_(image_pixels, image_grid);
        multimodal_embeds["image|embedding"] =
            split_by_token_nums(image_embeds, image_token_nums);
      } else {
        multimodal_embeds["image|embedding"] =
            run_dp_encoder(visual_,
                           image_pixels,
                           image_grid,
                           image_token_nums,
                           model_args_.mm_projection_dim(),
                           encoder_dp_group_);
      }
    }
    if (video_input) {
      std::vector<torch::Tensor> temp_frames_hw;
      torch::Tensor video_grid =
          video_input->video_grid_thw.to(options_.device());
      for (int i = 0; i < video_input->video_grid_thw.size(0); ++i) {
        auto t = video_grid[i][0].item<int32_t>();
        auto h = video_grid[i][1].item<int32_t>();
        auto w = video_grid[i][2].item<int32_t>();
        auto repeated_row =
            torch::tensor({1, h, w}).unsqueeze(0).repeat({t, 1});
        temp_frames_hw.push_back(repeated_row);
      }
      auto flatten_video_grid_thw = torch::cat(temp_frames_hw, 0);
      torch::Tensor video_pixels =
          video_input->pixel_values_videos.to(options_);
      std::vector<int32_t> video_token_nums =
          get_mm_token_nums(input_params.multimodal.mm_data, MMType::VIDEO);
      auto video_embeds = visual_(video_pixels, flatten_video_grid_thw);
      multimodal_embeds["video|embedding"] =
          split_by_token_nums(video_embeds, video_token_nums);
    }
    return multimodal_embeds;
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.multimodal.mm_data;
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    auto merge_modality = [&](const std::string& embed_key,
                              const std::string& mask_key) {
      auto emb = mm_data.get<torch::Tensor>(embed_key);
      if (!emb.has_value()) return;
      auto mask = mm_data.get<torch::Tensor>(mask_key);
      if (!mask.has_value()) return;
      inputs_embeds =
          merge_multimodal_embeddings(inputs_embeds, emb.value(), mask.value());
    };

    merge_modality("image|embedding", "image|mask");
    merge_modality("video|embedding", "video|mask");

    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

  layer::NpuLmHead get_npu_lm_head() {
    return language_model_->get_npu_lm_head();
  }
  void set_npu_lm_head(layer::NpuLmHead& head) {
    language_model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  ProcessGroup* encoder_dp_group_ = nullptr;
  bool use_encoder_dp_ = false;
  Glm4VisionTransformer visual_{nullptr};
  Glm4ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Glm4vForConditionalGeneration);

using Glm4vMultimodalProcessor = MultimodalProcessor<GLM4VPromptProcessor,
                                                     Glm4VImageProcessor,
                                                     Glm4VVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(glm4v, Glm4vMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(glm4v, Glm4vForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(glm4v, xllm::Glm4VMPositionGenerator);
// register the model args
REGISTER_MODEL_ARGS(glm4v, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4v");
  LOAD_ARG_OR(image_start_token_id, "image_start_token_id", 151339);
  LOAD_ARG_OR(image_end_token_id, "image_end_token_id", 151340);
  LOAD_ARG_OR(video_start_token_id, "video_start_token_id", 151341);
  LOAD_ARG_OR(video_end_token_id, "video_end_token_id", 151342);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151363);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151364);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151552);
  // LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 151329);
  LOAD_ARG_OR(
      eos_token_id_vec, "text_config.eos_token_id", std::vector<int>{151329});
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 4096);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 10944);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 131072);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 46);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  // LOAD_ARG_OR(partial_rotary_factor, "text_config.partial_rotary_factor",
  // 0.5);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_parameters.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_parameters.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_parameters.rope_theta", 500000.0f);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.0);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  // LOAD_ARG_OR(use_cache, "text_config.use_cache", true);
  LOAD_ARG_OR(use_qk_norm, "text_config.use_qk_norm", false);

  // vision config
  // LOAD_ARG_OR(mm_attention_bias, "vision_config.attention_bias", false);
  // LOAD_ARG_OR(mm_attention_dropout, "vision_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 10944);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // LOAD_ARG_OR(mm_rms_norm_eps, "vision_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
  SET_ARG(rope_scaling_rope_type, "mrope");
});
}  // namespace xllm::npu::model

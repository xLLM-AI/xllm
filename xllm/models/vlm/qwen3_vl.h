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

#include "core/framework/model/model_output.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/qwen3_vision_layer.h"
#include "models/llm/qwen3.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/qwen3_vl_base.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"
#include "processors/qwen3_vl_video_processor.h"
#include "qwen2_5_vl.h"

namespace xllm {

class Qwen3_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Qwen3_VisionPatchEmbedImpl(const ModelContext& context) {
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
TORCH_MODULE(Qwen3_VisionPatchEmbed);

class Qwen3_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen3_VisionRotaryEmbeddingImpl(const ModelContext& context) {
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
TORCH_MODULE(Qwen3_VisionRotaryEmbedding);

class Qwen3_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen3_VisionPatchMergerImpl(const ModelContext& context,
                              bool use_postshuffle_norm = false) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();
    int64_t d_model = model_args.mm_projection_dim();
    int context_dim = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();
    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
    use_postshuffle_norm_ = use_postshuffle_norm;
    if (use_postshuffle_norm)
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    else
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    norm_->weight.set_data(norm_->weight.to(options));
    norm_->bias.set_data(norm_->bias.to(options));

    auto fc1 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true));
    fc1->weight.set_data(fc1->weight.to(options));
    fc1->bias.set_data(fc1->bias.to(options));
    auto act = torch::nn::GELU();
    auto fc2 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, d_model).bias(true));
    fc2->weight.set_data(fc2->weight.to(options));
    fc2->bias.set_data(fc2->bias.to(options));
    mlp_ = register_module("mlp", torch::nn::Sequential(fc1, act, fc2));
    layers_ = std::make_tuple(fc1, act, fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    if (use_postshuffle_norm_)
      x = norm_(x.view({-1, hidden_size_}));
    else
      x = norm_(x).view({-1, hidden_size_});
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm
    const auto& norm_dict = state_dict.get_dict_with_prefix("norm.");
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

    const auto& fc1_dict = state_dict.get_dict_with_prefix("linear_fc1.");
    const auto& fc1_weight = fc1_dict.get_tensor("weight");
    if (fc1_weight.defined()) {
      CHECK_EQ(std::get<0>(layers_)->weight.sizes(), fc1_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<0>(layers_)->weight.data().copy_(fc1_weight);
      is_fc1_weight_loaded = true;
    }
    const auto fc1_bias = fc1_dict.get_tensor("bias");
    if (fc1_bias.defined()) {
      CHECK_EQ(std::get<0>(layers_)->bias.sizes(), fc1_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<0>(layers_)->bias.data().copy_(fc1_bias);
      is_fc1_bias_loaded = true;
    }

    const auto& fc2_dict = state_dict.get_dict_with_prefix("linear_fc2.");
    const auto& fc2_weight = fc2_dict.get_tensor("weight");
    if (fc2_weight.defined()) {
      CHECK_EQ(std::get<2>(layers_)->weight.sizes(), fc2_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<2>(layers_)->weight.data().copy_(fc2_weight);
      is_fc2_weight_loaded = true;
    }
    const auto fc2_bias = fc2_dict.get_tensor("bias");
    if (fc2_bias.defined()) {
      CHECK_EQ(std::get<2>(layers_)->bias.sizes(), fc2_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<2>(layers_)->bias.data().copy_(fc2_bias);
      is_fc2_bias_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_fc1_weight_loaded)
        << "weight is not loaded for " << prefix + "linear_fc1" + ".weight";
    CHECK(is_fc1_bias_loaded)
        << "bias is not loaded for " << prefix + "linear_fc1" + ".bias";
    CHECK(is_fc2_weight_loaded)
        << "weight is not loaded for " << prefix + "linear_fc2" + ".weight";
    CHECK(is_fc2_bias_loaded)
        << "bias is not loaded for " << prefix + "linear_fc2" + ".bias";
    CHECK(is_norm_weight_loaded)
        << "weight is not loaded for " << prefix + "norm" + ".weight";
    CHECK(is_norm_bias_loaded)
        << "bias is not loaded for " << prefix + "norm" + ".bias";
  }

 private:
  int hidden_size_;
  bool use_postshuffle_norm_;
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<torch::nn::Linear, torch::nn::GELU, torch::nn::Linear> layers_ = {
      nullptr,
      nullptr,
      nullptr};
  bool is_fc1_weight_loaded = false;
  bool is_fc1_bias_loaded = false;
  bool is_fc2_weight_loaded = false;
  bool is_fc2_bias_loaded = false;
  bool is_norm_weight_loaded = false;
  bool is_norm_bias_loaded = false;
};
TORCH_MODULE(Qwen3_VisionPatchMerger);

class Qwen3_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen3_VisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    auto& visual_indexes = model_args.mm_deepstack_visual_indexes();
    deepstack_visual_indexes_.insert(deepstack_visual_indexes_.end(),
                                     visual_indexes.begin(),
                                     visual_indexes.end());
    num_position_embeddings_ = model_args.mm_num_position_embeddings();
    spatial_merge_unit_ =
        static_cast<int>(spatial_merge_size_ * spatial_merge_size_);
    num_grid_per_side_ = static_cast<int>(std::sqrt(num_position_embeddings_));

    patch_embed_ =
        register_module("patch_embed", Qwen3_VisionPatchEmbed(context));
    rotary_pos_emb_ =
        register_module("rotary_pos_emb", Qwen3_VisionRotaryEmbedding(context));

    deepstack_mergers_ =
        register_module("deepstack_mergers", torch::nn::ModuleList());

    emb_ = register_module(
        "embedding",
        torch::nn::Embedding(num_position_embeddings_, hidden_size_));
    emb_->weight.set_data(emb_->weight.to(options_));

    merger_ = register_module("merger", Qwen3_VisionPatchMerger(context));

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto layer = layer::Qwen3_VisionLayer(context);
      layers_.push_back(layer);
    }
    for (int32_t idx = 0; idx < deepstack_visual_indexes_.size(); idx++) {
      auto merger = Qwen3_VisionPatchMerger(context, true);
      deepstack_mergers_->push_back(merger);
      deepstack_merger_layers_.push_back(merger);
    }
  }

  torch::Tensor rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    auto count = grid_thw.sizes()[0];
    pos_ids_vec.reserve(count);

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

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

    return rotary_pos_emb;
  }

  torch::Tensor fast_pos_embed_interpolate(const torch::Tensor& grid_thw) {
    auto device = grid_thw.device();
    int64_t hidden_dim = hidden_size_;
    int64_t m_size = spatial_merge_size_;

    auto grid_cpu = grid_thw.to(torch::kCPU);
    int64_t count = grid_thw.size(0);

    std::vector<torch::Tensor> outputs;
    outputs.reserve(count);

    for (int64_t idx = 0; idx < count; ++idx) {
      int64_t t = grid_cpu[idx][0].item<int64_t>();
      int64_t h = grid_cpu[idx][1].item<int64_t>();
      int64_t w = grid_cpu[idx][2].item<int64_t>();

      auto h_idxs =
          torch::linspace(
              0, static_cast<float>(num_grid_per_side_ - 1), h, torch::kFloat32)
              .to(device);
      auto w_idxs =
          torch::linspace(
              0, static_cast<float>(num_grid_per_side_ - 1), w, torch::kFloat32)
              .to(device);

      auto h_floor = h_idxs.to(torch::kLong);
      auto w_floor = w_idxs.to(torch::kLong);
      auto h_ceil = torch::clamp(h_floor + 1, 0, num_grid_per_side_ - 1);
      auto w_ceil = torch::clamp(w_floor + 1, 0, num_grid_per_side_ - 1);

      auto dh = h_idxs - h_floor;
      auto dw = w_idxs - w_floor;

      auto mesh_d = torch::meshgrid({dh, dw}, "ij");
      auto dh_grid = mesh_d[0], dw_grid = mesh_d[1];

      auto mesh_floor = torch::meshgrid({h_floor, w_floor}, "ij");
      auto h_floor_grid = mesh_floor[0];
      auto w_floor_grid = mesh_floor[1];

      auto mesh_ceil = torch::meshgrid({h_ceil, w_ceil}, "ij");
      auto h_ceil_grid = mesh_ceil[0];
      auto w_ceil_grid = mesh_ceil[1];

      auto h_floor_grid_idx = h_floor_grid * num_grid_per_side_;
      auto h_ceil_grid_idx = h_ceil_grid * num_grid_per_side_;

      auto w11 = dh_grid * dw_grid;
      auto w10 = dh_grid - w11;
      auto w01 = dw_grid - w11;
      auto w00 = 1.0f - dh_grid - dw_grid + w11;

      auto idx00 = h_floor_grid_idx + w_floor_grid;
      auto idx01 = h_floor_grid_idx + w_ceil_grid;
      auto idx10 = h_ceil_grid_idx + w_floor_grid;
      auto idx11 = h_ceil_grid_idx + w_ceil_grid;

      auto indices = torch::stack({idx00, idx01, idx10, idx11}, 0)
                         .reshape({4, -1})
                         .to(torch::kLong);
      auto weights = torch::stack({w00, w01, w10, w11}, 0)
                         .reshape({4, -1, 1})
                         .to(options_);

      auto embeds = emb_(indices);

      auto combined = (embeds * weights).sum(0);  // [h*w, hidden_dim]

      auto repeated = combined.unsqueeze(0).expand({t, -1, -1}).contiguous();
      repeated = repeated.view(
          {t, h / m_size, m_size, w / m_size, m_size, hidden_dim});
      repeated = repeated.permute({0, 1, 3, 2, 4, 5}).reshape({-1, hidden_dim});

      outputs.push_back(repeated);
    }

    return torch::cat(outputs, 0);
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    hidden_states = patch_embed_(hidden_states);
    auto pos_embeds = fast_pos_embed_interpolate(grid_thw);
    hidden_states = hidden_states + pos_embeds;
    //   compute position embedding
    auto rotary_pos_emb = rot_pos_emb(grid_thw);
    // compute cu_seqlens
    auto cu_seqlens = torch::repeat_interleave(
                          grid_thw.index({torch::indexing::Slice(), 1}) *
                              grid_thw.index({torch::indexing::Slice(), 2}),
                          grid_thw.index({torch::indexing::Slice(), 0}))
                          .cumsum(0, torch::kInt32);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));

    m_cos = rotary_pos_emb.cos().type_as(hidden_states);
    m_cos = m_cos.repeat({1, 2});
    m_sin = rotary_pos_emb.sin().type_as(hidden_states);
    m_sin = m_sin.repeat({1, 2});

    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int>(),  // full seqlen vec
        cu_seqlens_cpu.data_ptr<int>() + cu_seqlens_cpu.numel());
    std::vector<torch::Tensor> deepstack_feature_lists;
    deepstack_feature_lists.reserve(deepstack_visual_indexes_.size());
    for (int idx = 0; idx < layers_.size(); ++idx) {
      hidden_states = layers_[idx](
          hidden_states, m_cos, m_sin, cu_seqlens, cu_seqlens_vec, idx);
      auto it = std::find(deepstack_visual_indexes_.begin(),
                          deepstack_visual_indexes_.end(),
                          idx);

      if (it != deepstack_visual_indexes_.end()) {
        int index = std::distance(deepstack_visual_indexes_.begin(), it);
        deepstack_feature_lists.push_back(
            deepstack_merger_layers_[index](hidden_states));
      }
    }
    // adapter
    hidden_states = merger_(hidden_states);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(deepstack_feature_lists.size() + 1);
    tensors.push_back(hidden_states);
    tensors.insert(tensors.end(),
                   deepstack_feature_lists.begin(),
                   deepstack_feature_lists.end());
    return torch::cat(tensors, /*dim=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));

    for (int idx = 0; idx < deepstack_merger_layers_.size(); ++idx) {
      deepstack_merger_layers_[idx]->load_state_dict(
          state_dict.get_dict_with_prefix("deepstack_merger_list." +
                                          std::to_string(idx) + "."));
    }

    const auto& emb_dict = state_dict.get_dict_with_prefix("pos_embed.");
    const auto& emb_weight = emb_dict.get_tensor("weight");
    if (emb_weight.defined()) {
      CHECK_EQ(emb_->weight.sizes(), emb_weight.sizes())
          << "weight size mismatch for " << name();
      emb_->weight.data().copy_(emb_weight);
      is_emb_weight_loaded = true;
    }
  }

 private:
  int hidden_size_ = 0;
  int num_heads_ = 0;
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::vector<int64_t> deepstack_visual_indexes_;
  int spatial_merge_unit_ = 0;
  int64_t num_position_embeddings_ = 0;
  int num_grid_per_side_ = 0;

  Qwen3_VisionPatchEmbed patch_embed_{nullptr};
  Qwen3_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::Embedding emb_{nullptr};

  std::vector<layer::Qwen3_VisionLayer> layers_;

  torch::nn::ModuleList deepstack_mergers_{nullptr};
  std::vector<Qwen3_VisionPatchMerger> deepstack_merger_layers_;
  Qwen3_VisionPatchMerger merger_{nullptr};

  torch::Tensor m_cos;
  torch::Tensor m_sin;
  int device_id = 0;
  bool is_emb_weight_loaded = false;
  torch::TensorOptions options_;
};
TORCH_MODULE(Qwen3_VisionTransformer);

class Qwen3_VLForConditionalGenerationImpl
    : public Qwen3VLForConditionalGenerationBase<Qwen3_VisionTransformer,
                                                 QWen3ForCausalLM> {
 public:
  Qwen3_VLForConditionalGenerationImpl(const ModelContext& context)
      : Qwen3VLForConditionalGenerationBase(context) {}
};
TORCH_MODULE(Qwen3_VLForConditionalGeneration);

using Qwen3VLMultimodalProcessor = MultimodalProcessor<Qwen3VLPromptProcessor,
                                                       Qwen2VLImageProcessor,
                                                       Qwen3VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen3_vl, Qwen3VLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_vl, Qwen3_VLForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_vl, Qwen3VLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_vl, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl");
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR(
      vision_start_token_id, "text_config.vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "text_config.vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "text_config.vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "text_config.image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "text_config.video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 3584);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 18944);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  LOAD_ARG_OR(max_window_layers, "text_config.max_window_layers", 28);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(sliding_window, "text_config.sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "text_config.tie_word_embeddings", false);
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(
      rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope");

  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);
});
}  // namespace xllm

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
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_qwen3_vision_encoder_layer_impl.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/npu/qwen3_audio_encoder.h"
#include "models/vlm/npu/qwen3_vl.h"
#include "processors/qwen3_audio_common.h"
#include "processors/qwen3_omni_moe_processor.h"

namespace xllm::npu::model {

inline void load_qwen3_omni_moe_model_args(const JsonReader& json,
                                           ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe");
  LOAD_ARG_OR(has_feature_extractor, "has_feature_extractor", true);
  LOAD_ARG_OR(mm_audio_feature_size, "feature_size", 128);
  LOAD_ARG_OR(mm_audio_sampling_rate, "sampling_rate", 16000);
  LOAD_ARG_OR(mm_audio_hop_length, "hop_length", 160);
  LOAD_ARG_OR(mm_audio_chunk_length, "chunk_length", 30);
  LOAD_ARG_OR(mm_audio_n_fft, "n_fft", 400);
  LOAD_ARG_OR(mm_audio_dither, "dither", 0.0);
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);
  SET_ARG(mm_use_audio_in_video,
          ModelConfig::get_instance().use_audio_in_video());
  LOAD_ARG_OR(mm_position_id_per_seconds, "position_id_per_seconds", 13);
  LOAD_ARG_OR(mm_fps, "fps", 1.0);

  LOAD_ARG_OR(
      vision_start_token_id, "thinker_config.vision_start_token_id", 151652);
  LOAD_ARG_OR(
      vision_end_token_id, "thinker_config.vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "thinker_config.vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "thinker_config.image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "thinker_config.video_token_id", 151656);
  LOAD_ARG_OR(audio_token_id, "thinker_config.audio_token_id", 151675);
  LOAD_ARG_OR(
      audio_start_token_id, "thinker_config.audio_start_token_id", 151669);
  LOAD_ARG_OR(audio_end_token_id, "thinker_config.audio_end_token_id", 151670);
  LOAD_ARG_OR(dtype, "thinker_config.dtype", "bfloat16");

  LOAD_ARG_OR(
      attention_bias, "thinker_config.text_config.attention_bias", false);
  LOAD_ARG_OR(
      attention_dropout, "thinker_config.text_config.attention_dropout", 0.0);
  LOAD_ARG_OR(
      decoder_sparse_step, "thinker_config.text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(bos_token_id, "thinker_config.text_config.bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "thinker_config.text_config.eos_token_id", 151645);
  LOAD_ARG_OR(hidden_act, "thinker_config.text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "thinker_config.text_config.hidden_size", 2048);
  LOAD_ARG_OR(
      intermediate_size, "thinker_config.text_config.intermediate_size", 768);
  LOAD_ARG_OR(max_position_embeddings,
              "thinker_config.text_config.max_position_embeddings",
              65536);
  LOAD_ARG_OR(
      max_window_layers, "thinker_config.text_config.max_window_layers", 28);
  LOAD_ARG_OR(n_heads, "thinker_config.text_config.num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "thinker_config.text_config.num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "thinker_config.text_config.num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "thinker_config.text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(
      sliding_window, "thinker_config.text_config.sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings,
              "thinker_config.text_config.tie_word_embeddings",
              false);
  LOAD_ARG_OR(
      initializer_range, "thinker_config.text_config.initializer_range", 0.02);
  LOAD_ARG_OR(use_sliding_window,
              "thinker_config.text_config.use_sliding_window",
              false);
  LOAD_ARG_OR(moe_intermediate_size,
              "thinker_config.text_config.moe_intermediate_size",
              768);
  LOAD_ARG_OR(
      norm_topk_prob, "thinker_config.text_config.norm_topk_prob", true);
  LOAD_ARG_OR(num_experts, "thinker_config.text_config.num_experts", 128);
  LOAD_ARG_OR(
      num_experts_per_tok, "thinker_config.text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR_FUNC(head_dim, "thinker_config.text_config.head_dim", [args] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(output_router_logits,
              "thinker_config.text_config.output_router_logits",
              false);
  LOAD_ARG_OR(router_aux_loss_coef,
              "thinker_config.text_config.router_aux_loss_coef",
              0.001f);
  LOAD_ARG_OR(mlp_only_layers,
              "thinker_config.text_config.mlp_only_layers",
              std::vector<int32_t>());
  LOAD_ARG_OR(rope_scaling_rope_type,
              "thinker_config.text_config.rope_scaling.type",
              "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "thinker_config.text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(rope_theta, "thinker_config.text_config.rope_theta", 1000000.0f);
  LOAD_ARG_OR(vocab_size, "thinker_config.text_config.vocab_size", 152064);

  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>{args->eos_token_id()});

  LOAD_ARG_OR(mm_num_hidden_layers, "thinker_config.vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act,
              "thinker_config.vision_config.hidden_act",
              "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "thinker_config.vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size,
              "thinker_config.vision_config.intermediate_size",
              4304);
  LOAD_ARG_OR(
      mm_num_attention_heads, "thinker_config.vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "thinker_config.vision_config.in_channels", 3);
  LOAD_ARG_OR(
      mm_projection_dim, "thinker_config.vision_config.out_hidden_size", 2048);
  LOAD_ARG_OR(mm_patch_size, "thinker_config.vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "thinker_config.vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_spatial_merge_size,
              "thinker_config.vision_config.spatial_merge_size",
              2);
  LOAD_ARG(mm_deepstack_visual_indexes,
           "thinker_config.vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_temporal_patch_size,
              "thinker_config.vision_config.temporal_patch_size",
              2);
  LOAD_ARG_OR(mm_image_size, "thinker_config.vision_config.image_size", 768);
  LOAD_ARG_OR_FUNC(
      mm_head_dim, "thinker_config.vision_config.head_dim", [args] {
        return args->mm_hidden_size() / args->mm_num_attention_heads();
      });

  LOAD_ARG_OR(mm_audio_num_attention_heads,
              "thinker_config.audio_config.encoder_attention_heads",
              20);
  LOAD_ARG_OR(
      mm_audio_hidden_size, "thinker_config.audio_config.d_model", 1280);
  LOAD_ARG_OR(mm_audio_layer_norm_eps,
              "thinker_config.audio_config.layer_norm_eps",
              1e-5);
  LOAD_ARG_OR(mm_audio_downsample_hidden_size,
              "thinker_config.audio_config.downsample_hidden_size",
              480);
  LOAD_ARG_OR(
      mm_audio_num_mel_bins, "thinker_config.audio_config.num_mel_bins", 128);
  LOAD_ARG_OR(mm_audio_max_source_positions,
              "thinker_config.audio_config.max_source_positions",
              1500);
  LOAD_ARG_OR(mm_audio_n_window, "thinker_config.audio_config.n_window", 50);
  LOAD_ARG_OR(mm_audio_n_window_infer,
              "thinker_config.audio_config.n_window_infer",
              800);
  LOAD_ARG_OR(mm_audio_conv_chunksize,
              "thinker_config.audio_config.conv_chunksize",
              500);
  LOAD_ARG_OR(mm_audio_encoder_layers,
              "thinker_config.audio_config.encoder_layers",
              32);
  LOAD_ARG_OR(
      mm_audio_output_dim, "thinker_config.audio_config.output_dim", 2048);
}

class Qwen3OmniMoeThinkerVisionPatchEmbedImpl final : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerVisionPatchEmbedImpl(
      const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();

    const int64_t in_features =
        model_args.mm_num_channels() * model_args.mm_temporal_patch_size() *
        model_args.mm_patch_size() * model_args.mm_patch_size();

    const int64_t out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(true)));

    proj_->weight.set_data(proj_->weight.to(options));
    proj_->bias.set_data(proj_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    return proj_(hidden_states);
  }

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
TORCH_MODULE(Qwen3OmniMoeThinkerVisionPatchEmbed);

class Qwen3OmniMoeThinkerVisionBlockImpl final : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerVisionBlockImpl(const ModelContext& context) {
    encoder_layer_ = register_module(
        "encoder_layer", layer::NpuQwen3VisionEncoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec,
                        int32_t node_id) {
    return encoder_layer_(hidden_states,
                          m_cos_pos,
                          m_sin_pos,
                          cu_seq_len,
                          cu_seq_len_vec,
                          node_id);
  }

  void load_state_dict(const StateDict& state_dict) {
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuQwen3VisionEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoeThinkerVisionBlock);

class Qwen3OmniMoeThinkerVisionRotaryEmbeddingImpl final
    : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerVisionRotaryEmbeddingImpl(
      const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();

    dim_ = model_args.mm_head_dim() / 2;
    theta_ = 10000.0;

    const torch::TensorOptions float_options = options.dtype(torch::kFloat32);
    const torch::Tensor inv_freq =
        1.0 /
        torch::pow(theta_, torch::arange(0, dim_, 2, float_options) / dim_);
    inv_freq_ = register_buffer("inv_freq", inv_freq);
  }

  void update_freqs_cache(int64_t seqlen) {
    if (seqlen <= seq_len_cached_) {
      return;
    }

    seqlen *= 2;
    seq_len_cached_ = seqlen;

    const torch::TensorOptions options = torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(inv_freq_.device());
    inv_freq_ =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, options) / dim_);
    auto seq = torch::arange(seqlen, options);
    freqs_cached_ = torch::outer(seq, inv_freq_);
  }

  torch::Tensor forward(int64_t seqlen) {
    update_freqs_cache(seqlen);
    return freqs_cached_.slice(0, 0, seqlen);
  }

 private:
  int64_t dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen3OmniMoeThinkerVisionRotaryEmbedding);

class Qwen3OmniMoeThinkerVisionPatchMergerImpl final
    : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerVisionPatchMergerImpl(
      const ModelContext& context,
      bool use_postshuffle_norm = false) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();
    const int64_t output_size = model_args.mm_projection_dim();
    const int64_t context_size = model_args.mm_hidden_size();
    const int64_t spatial_merge_size = model_args.mm_spatial_merge_size();
    hidden_size_ = context_size * spatial_merge_size * spatial_merge_size;
    use_postshuffle_norm_ = use_postshuffle_norm;
    if (use_postshuffle_norm) {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    } else {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_size})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    }

    norm_->weight.set_data(norm_->weight.to(options));
    norm_->bias.set_data(norm_->bias.to(options));

    auto fc1 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true));
    fc1->weight.set_data(fc1->weight.to(options));
    fc1->bias.set_data(fc1->bias.to(options));
    auto act = torch::nn::GELU();
    auto fc2 = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, output_size).bias(true));
    fc2->weight.set_data(fc2->weight.to(options));
    fc2->bias.set_data(fc2->bias.to(options));
    mlp_ = register_module("mlp", torch::nn::Sequential(fc1, act, fc2));
    layers_ = std::make_tuple(fc1, act, fc2);
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    if (use_postshuffle_norm_) {
      hidden_states = norm_(hidden_states.view({-1, hidden_size_}));
    } else {
      hidden_states = norm_(hidden_states).view({-1, hidden_size_});
    }
    return mlp_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm
    const auto& norm_dict = state_dict.get_dict_with_prefix("ln_q.");
    const auto& norm_weight = norm_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(norm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      norm_->weight.data().copy_(norm_weight);
      is_norm_weight_loaded_ = true;
    }
    const auto norm_bias = norm_dict.get_tensor("bias");
    if (norm_bias.defined()) {
      CHECK_EQ(norm_->bias.sizes(), norm_bias.sizes())
          << "bias size mismatch for " << name();
      norm_->bias.data().copy_(norm_bias);
      is_norm_bias_loaded_ = true;
    }

    const auto& fc1_dict = state_dict.get_dict_with_prefix("mlp.0.");
    const auto& fc1_weight = fc1_dict.get_tensor("weight");
    if (fc1_weight.defined()) {
      CHECK_EQ(std::get<0>(layers_)->weight.sizes(), fc1_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<0>(layers_)->weight.data().copy_(fc1_weight);
      is_fc1_weight_loaded_ = true;
    }
    const auto fc1_bias = fc1_dict.get_tensor("bias");
    if (fc1_bias.defined()) {
      CHECK_EQ(std::get<0>(layers_)->bias.sizes(), fc1_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<0>(layers_)->bias.data().copy_(fc1_bias);
      is_fc1_bias_loaded_ = true;
    }

    const auto& fc2_dict = state_dict.get_dict_with_prefix("mlp.2.");
    const auto& fc2_weight = fc2_dict.get_tensor("weight");
    if (fc2_weight.defined()) {
      CHECK_EQ(std::get<2>(layers_)->weight.sizes(), fc2_weight.sizes())
          << "weight size mismatch for " << name();
      std::get<2>(layers_)->weight.data().copy_(fc2_weight);
      is_fc2_weight_loaded_ = true;
    }
    const auto fc2_bias = fc2_dict.get_tensor("bias");
    if (fc2_bias.defined()) {
      CHECK_EQ(std::get<2>(layers_)->bias.sizes(), fc2_bias.sizes())
          << "bias size mismatch for " << name();
      std::get<2>(layers_)->bias.data().copy_(fc2_bias);
      is_fc2_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_fc1_weight_loaded_)
        << "weight is not loaded for " << prefix + "mlp.0.weight";
    CHECK(is_fc1_bias_loaded_)
        << "bias is not loaded for " << prefix + "mlp.0.bias";
    CHECK(is_fc2_weight_loaded_)
        << "weight is not loaded for " << prefix + "mlp.2.weight";
    CHECK(is_fc2_bias_loaded_)
        << "bias is not loaded for " << prefix + "mlp.2.bias";
    CHECK(is_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "ln_q.weight";
    CHECK(is_norm_bias_loaded_)
        << "bias is not loaded for " << prefix + "ln_q.bias";
  }

 private:
  int64_t hidden_size_ = 0;
  bool use_postshuffle_norm_ = false;
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<torch::nn::Linear, torch::nn::GELU, torch::nn::Linear> layers_ = {
      nullptr,
      nullptr,
      nullptr};
  bool is_fc1_weight_loaded_ = false;
  bool is_fc1_bias_loaded_ = false;
  bool is_fc2_weight_loaded_ = false;
  bool is_fc2_bias_loaded_ = false;
  bool is_norm_weight_loaded_ = false;
  bool is_norm_bias_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniMoeThinkerVisionPatchMerger);

class Qwen3OmniMoeThinkerVisionTransformerImpl final
    : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerVisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    const ModelArgs& model_args = context.get_model_args();
    hidden_size_ = model_args.mm_hidden_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    deepstack_visual_indexes_ = model_args.mm_deepstack_visual_indexes();
    image_size_ = model_args.mm_image_size();
    num_grid_per_side_ = image_size_ / patch_size_;

    patch_embed_ = register_module(
        "patch_embed", Qwen3OmniMoeThinkerVisionPatchEmbed(context));
    rotary_pos_emb_ = register_module(
        "rotary_pos_emb", Qwen3OmniMoeThinkerVisionRotaryEmbedding(context));

    blocks_ = register_module("blocks", torch::nn::ModuleList());
    deepstack_mergers_ =
        register_module("deepstack_mergers", torch::nn::ModuleList());

    emb_ = register_module(
        "embedding",
        torch::nn::Embedding(num_grid_per_side_ * num_grid_per_side_,
                             hidden_size_));
    emb_->weight.set_data(emb_->weight.to(options_));

    merger_ = register_module("merger",
                              Qwen3OmniMoeThinkerVisionPatchMerger(context));

    layers_.reserve(static_cast<size_t>(model_args.mm_num_hidden_layers()));
    for (int32_t index = 0; index < model_args.mm_num_hidden_layers();
         ++index) {
      Qwen3OmniMoeThinkerVisionBlock block(context);
      blocks_->push_back(block);
      layers_.emplace_back(block);
    }
    const size_t deepstack_count = deepstack_visual_indexes_.size();
    deepstack_merger_layers_.reserve(deepstack_count);
    for (size_t index = 0; index < deepstack_count; ++index) {
      Qwen3OmniMoeThinkerVisionPatchMerger merger(
          context, /*use_postshuffle_norm=*/true);
      deepstack_mergers_->push_back(merger);
      deepstack_merger_layers_.emplace_back(merger);
    }
  }

  torch::Tensor rot_pos_emb(torch::Tensor grid_thw) {
    std::vector<torch::Tensor> pos_ids_vec;
    const int64_t count = grid_thw.size(0);
    pos_ids_vec.reserve(static_cast<size_t>(count));

    const torch::Tensor grid_thw_cpu = grid_thw.cpu();
    const torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int64_t index = 0; index < count; ++index) {
      const int64_t temporal = grid_thw_cpu[index][0].item<int64_t>();
      const int64_t height = grid_thw_cpu[index][1].item<int64_t>();
      const int64_t width = grid_thw_cpu[index][2].item<int64_t>();

      torch::Tensor height_position_ids =
          torch::arange(height, options).unsqueeze(1).expand({-1, width});
      height_position_ids = height_position_ids
                                .reshape({height / spatial_merge_size_,
                                          spatial_merge_size_,
                                          width / spatial_merge_size_,
                                          spatial_merge_size_})
                                .permute({0, 2, 1, 3})
                                .flatten();

      torch::Tensor width_position_ids =
          torch::arange(width, options).unsqueeze(0).expand({height, -1});
      width_position_ids = width_position_ids
                               .reshape({height / spatial_merge_size_,
                                         spatial_merge_size_,
                                         width / spatial_merge_size_,
                                         spatial_merge_size_})
                               .permute({0, 2, 1, 3})
                               .flatten();

      pos_ids_vec.emplace_back(
          torch::stack({height_position_ids, width_position_ids}, /*dim=*/-1)
              .repeat({temporal, 1}));
    }

    const torch::Tensor position_ids = torch::cat(pos_ids_vec, /*dim=*/0);
    const torch::Tensor max_grid_size =
        grid_thw
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None)})
            .max();

    const torch::Tensor rotary_position_embedding =
        rotary_pos_emb_(max_grid_size.item<int64_t>())
            .index({position_ids})
            .flatten(1);

    return rotary_position_embedding;
  }

  torch::Tensor fast_pos_embed_interpolate(const torch::Tensor& grid_thw) {
    const torch::Device device = grid_thw.device();
    const int64_t hidden_dim = hidden_size_;
    const int64_t merge_size = spatial_merge_size_;

    const torch::Tensor grid_cpu = grid_thw.to(torch::kCPU);
    const int64_t count = grid_thw.size(0);

    std::vector<torch::Tensor> outputs;
    outputs.reserve(static_cast<size_t>(count));

    for (int64_t index = 0; index < count; ++index) {
      const int64_t temporal = grid_cpu[index][0].item<int64_t>();
      const int64_t height = grid_cpu[index][1].item<int64_t>();
      const int64_t width = grid_cpu[index][2].item<int64_t>();

      auto h_idxs = torch::linspace(0,
                                    static_cast<float>(num_grid_per_side_ - 1),
                                    height,
                                    torch::kFloat32)
                        .to(device);
      auto w_idxs = torch::linspace(0,
                                    static_cast<float>(num_grid_per_side_ - 1),
                                    width,
                                    torch::kFloat32)
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

      const torch::Tensor combined = (embeds * weights).sum(/*dim=*/0);

      auto repeated =
          combined.unsqueeze(0).expand({temporal, -1, -1}).contiguous();
      repeated = repeated.view({temporal,
                                height / merge_size,
                                merge_size,
                                width / merge_size,
                                merge_size,
                                hidden_dim});
      repeated = repeated.permute({0, 1, 3, 2, 4, 5}).reshape({-1, hidden_dim});

      outputs.emplace_back(repeated);
    }

    return torch::cat(outputs, 0);
  }

  std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(
      torch::Tensor hidden_states,
      torch::Tensor grid_thw) {
    hidden_states = patch_embed_(hidden_states);
    const torch::Tensor pos_embeds = fast_pos_embed_interpolate(grid_thw);
    hidden_states = hidden_states + pos_embeds;
    const torch::Tensor rotary_pos_emb = rot_pos_emb(grid_thw);
    torch::Tensor cu_seqlens =
        torch::repeat_interleave(
            grid_thw.index({torch::indexing::Slice(), 1}) *
                grid_thw.index({torch::indexing::Slice(), 2}),
            grid_thw.index({torch::indexing::Slice(), 0}))
            .cumsum(/*dim=*/0, torch::kInt32);
    cu_seqlens =
        torch::nn::functional::pad(cu_seqlens,
                                   torch::nn::functional::PadFuncOptions({1, 0})
                                       .mode(torch::kConstant)
                                       .value(0));
    cu_seqlens = torch::diff(cu_seqlens);

    m_cos_ = rotary_pos_emb.cos().type_as(hidden_states);
    m_cos_ = m_cos_.repeat({1, 2});
    m_sin_ = rotary_pos_emb.sin().type_as(hidden_states);
    m_sin_ = m_sin_.repeat({1, 2});

    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int32_t> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int32_t>(),
        cu_seqlens_cpu.data_ptr<int32_t>() + cu_seqlens_cpu.numel());
    std::vector<torch::Tensor> deepstack_feature_lists;
    deepstack_feature_lists.reserve(deepstack_visual_indexes_.size());
    const int32_t layer_count = static_cast<int32_t>(blocks_->size());
    for (int32_t index = 0; index < layer_count; ++index) {
      hidden_states = layers_[index](
          hidden_states, m_cos_, m_sin_, cu_seqlens, cu_seqlens_vec, index);
      auto it = std::find(deepstack_visual_indexes_.begin(),
                          deepstack_visual_indexes_.end(),
                          index);

      if (it != deepstack_visual_indexes_.end()) {
        const size_t merger_index = static_cast<size_t>(
            std::distance(deepstack_visual_indexes_.begin(), it));
        deepstack_feature_lists.emplace_back(
            deepstack_merger_layers_[merger_index](hidden_states));
      }
    }
    hidden_states = merger_(hidden_states);
    return std::make_tuple(hidden_states, deepstack_feature_lists);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    const size_t layer_count = layers_.size();
    for (size_t index = 0; index < layer_count; ++index) {
      layers_[index]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(index) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));

    const size_t merger_count = deepstack_merger_layers_.size();
    for (size_t index = 0; index < merger_count; ++index) {
      deepstack_merger_layers_[index]->load_state_dict(
          state_dict.get_dict_with_prefix("merger_list." +
                                          std::to_string(index) + "."));
    }

    const auto& emb_dict = state_dict.get_dict_with_prefix("pos_embed.");
    const auto& emb_weight = emb_dict.get_tensor("weight");
    if (emb_weight.defined()) {
      CHECK_EQ(emb_->weight.sizes(), emb_weight.sizes())
          << "weight size mismatch for " << name();
      emb_->weight.data().copy_(emb_weight);
      is_emb_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    const size_t layer_count = layers_.size();
    for (size_t index = 0; index < layer_count; ++index) {
      layers_[index]->verify_loaded_weights(prefix + "blocks." +
                                            std::to_string(index) + ".");
    }
    merger_->verify_loaded_weights(prefix + "merger.");

    const size_t merger_count = deepstack_merger_layers_.size();
    for (size_t index = 0; index < merger_count; ++index) {
      deepstack_merger_layers_[index]->verify_loaded_weights(
          prefix + "merger_list." + std::to_string(index) + ".");
    }
    CHECK(is_emb_weight_loaded_)
        << "weight is not loaded for " << prefix + "pos_embed.weight";
  }

  void merge_loaded_weights() {
    for (Qwen3OmniMoeThinkerVisionBlock& layer : layers_) {
      layer->merge_loaded_weights();
    }
  }

 private:
  int64_t hidden_size_ = 0;
  int64_t patch_size_ = 0;
  int64_t spatial_merge_size_ = 0;
  std::vector<int64_t> deepstack_visual_indexes_;
  int64_t image_size_ = 0;
  int64_t num_grid_per_side_ = 0;

  Qwen3OmniMoeThinkerVisionPatchEmbed patch_embed_{nullptr};
  Qwen3OmniMoeThinkerVisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::Embedding emb_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3OmniMoeThinkerVisionBlock> layers_;

  torch::nn::ModuleList deepstack_mergers_{nullptr};
  std::vector<Qwen3OmniMoeThinkerVisionPatchMerger> deepstack_merger_layers_;
  Qwen3OmniMoeThinkerVisionPatchMerger merger_{nullptr};

  torch::Tensor m_cos_;
  torch::Tensor m_sin_;
  bool is_emb_weight_loaded_ = false;
  torch::TensorOptions options_;
};
TORCH_MODULE(Qwen3OmniMoeThinkerVisionTransformer);

class Qwen3OmniMoeThinkerForConditionalGenerationImpl final
    : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeThinkerForConditionalGenerationImpl(
      const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual",
                              Qwen3OmniMoeThinkerVisionTransformer(context));
    audio_tower_ = register_module("audio_tower", Qwen3AudioEncoder(context));
    language_model_ =
        register_module("language_model", Qwen3MoeForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs,
                             std::optional<Qwen3AudioInputs>& audio_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor pixel_values;
    if (std::optional<torch::Tensor> value =
            mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = value.value();
    }

    torch::Tensor image_grid_thw;
    if (std::optional<torch::Tensor> value =
            mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = value.value();
    }

    torch::Tensor pixel_values_videos;
    if (std::optional<torch::Tensor> value =
            mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = value.value();
    }

    torch::Tensor video_grid_thw;
    if (std::optional<torch::Tensor> value =
            mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = value.value();
    }

    torch::Tensor input_features;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kInputFeaturesKey)) {
      input_features = res.value();
    }

    torch::Tensor feature_lengths;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kFeatureLengthKey)) {
      feature_lengths = res.value();
    }

    torch::Tensor feature_origin_lengths;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kFeatureOriginLengthsKey)) {
      feature_origin_lengths = res.value();
    }

    if (pixel_values.defined() && image_grid_thw.defined()) {
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};
    }

    if (pixel_values_videos.defined() && video_grid_thw.defined()) {
      video_inputs = Qwen3_VLVideoInputs{pixel_values_videos, video_grid_thw};
    }

    if (input_features.defined() && feature_lengths.defined() &&
        feature_origin_lengths.defined()) {
      audio_inputs = Qwen3AudioInputs{
          input_features, feature_lengths, feature_origin_lengths};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    std::optional<Qwen3AudioInputs> audio_input;
    prepare_encoder_input(input_params, image_input, video_input, audio_input);
    MMDict multimodal_embeds;
    const int64_t merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      auto [image_embeds, deep_stacks] =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()));

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);
      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      std::vector<torch::Tensor> image_embedding_parts{image_embeds};
      image_embedding_parts.insert(
          image_embedding_parts.end(), deep_stacks.begin(), deep_stacks.end());
      multimodal_embeds[get_embedding_key(MMType::IMAGE)] =
          torch::cat(image_embedding_parts, /*dim=*/1)
              .split(image_tokens_vec, /*dim=*/0);
    }
    if (video_input) {
      auto [video_embeds, deep_stacks] =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()));
      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);
      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      std::vector<torch::Tensor> video_embedding_parts{video_embeds};
      video_embedding_parts.insert(
          video_embedding_parts.end(), deep_stacks.begin(), deep_stacks.end());
      multimodal_embeds[get_embedding_key(MMType::VIDEO)] =
          torch::cat(video_embedding_parts, /*dim=*/1)
              .split(video_tokens_vec, /*dim=*/0);
    }
    if (audio_input) {
      const torch::Tensor feature_origin_lengths =
          audio_input->feature_origin_lengths.to(options_.device(),
                                                 torch::kLong);

      const torch::Tensor input_features =
          audio_input->input_features.permute({1, 0}).to(options_);

      const torch::Tensor audio_embeds =
          audio_tower_->forward(input_features, feature_origin_lengths);

      const torch::Tensor audio_tokens =
          audio_input->feature_lengths.cpu().contiguous().to(torch::kLong);

      std::vector<int64_t> feature_lens_vec(
          audio_tokens.data_ptr<int64_t>(),
          audio_tokens.data_ptr<int64_t>() + audio_tokens.numel());

      multimodal_embeds[get_embedding_key(MMType::AUDIO)] =
          audio_embeds.split(feature_lens_vec, /*dim=*/0);
    }
    if (model_args_.mm_use_audio_in_video() && video_input && audio_input) {
      const std::vector<torch::Tensor> origin_audio_embeds =
          std::get<std::vector<torch::Tensor>>(
              multimodal_embeds[get_embedding_key(MMType::AUDIO)]);
      const std::vector<torch::Tensor> origin_video_embeds =
          std::get<std::vector<torch::Tensor>>(
              multimodal_embeds[get_embedding_key(MMType::VIDEO)]);
      CHECK_GE(origin_audio_embeds.size(), origin_video_embeds.size());
      CHECK(!origin_video_embeds.empty());
      std::vector<torch::Tensor> audio_in_video_embeds;
      audio_in_video_embeds.reserve(origin_video_embeds.size());
      std::vector<torch::Tensor> scattered_audio_embeds;
      scattered_audio_embeds.reserve(origin_audio_embeds.size());
      size_t audio_index = 0;
      size_t video_index = 0;

      const auto& mm_data = input_params.multimodal.mm_data;
      const std::vector<MMData>& mm_data_vec = mm_data.mm_data_vec();
      for (const MMData& sequence_mm_data : mm_data_vec) {
        const MMItemVec& mm_items = sequence_mm_data.items<MMItemVec>();
        for (size_t item_index = 0; item_index < mm_items.size();
             ++item_index) {
          const MMDataItem& item = mm_items[item_index];
          if (item.is_type(MMType::AUDIO)) {
            CHECK_LT(audio_index, origin_audio_embeds.size());
            scattered_audio_embeds.emplace_back(
                origin_audio_embeds[audio_index++]);
          } else if (item.is_type(MMType::VIDEO)) {
            CHECK(item_index + 1 < mm_items.size());
            CHECK(mm_items[item_index + 1].is_type(MMType::AUDIO));
            CHECK_LT(video_index, origin_video_embeds.size());
            CHECK_LT(audio_index, origin_audio_embeds.size());
            const torch::Tensor video_embedding =
                origin_video_embeds[video_index++];
            const torch::Tensor audio_embedding =
                origin_audio_embeds[audio_index++];
            std::optional<torch::Tensor> audio_in_video_token_ids =
                item.get<torch::Tensor>(
                    qwen3_omni_moe::kAudioInVideoTokenIdsKey);
            CHECK(audio_in_video_token_ids.has_value());
            torch::Tensor audio_in_video_embedding =
                torch::full({audio_in_video_token_ids->size(0),
                             origin_video_embeds[0].size(1)},
                            1,
                            options_);
            const torch::Tensor video_mask = torch::isin(
                audio_in_video_token_ids.value(), model_args_.video_token_id());
            const torch::Tensor audio_mask = torch::isin(
                audio_in_video_token_ids.value(), model_args_.audio_token_id());
            audio_in_video_embedding.index_put_({video_mask}, video_embedding);
            audio_in_video_embedding.index_put_({audio_mask}, audio_embedding);
            audio_in_video_embeds.emplace_back(audio_in_video_embedding);
            scattered_audio_embeds.emplace_back(
                audio_embedding.slice(/*dim=*/0, /*start=*/0, /*end=*/0));
            ++item_index;
          }
        }
      }
      CHECK_EQ(audio_index, origin_audio_embeds.size());
      CHECK_EQ(video_index, origin_video_embeds.size());
      CHECK_EQ(scattered_audio_embeds.size(), origin_audio_embeds.size());
      multimodal_embeds[get_embedding_key(MMType::AUDIO)] =
          scattered_audio_embeds;
      multimodal_embeds[get_embedding_key(MMType::VIDEO)] =
          audio_in_video_embeds;
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
    torch::Tensor inputs_embeds =
        language_model_->get_input_embeddings(input_ids);
    const size_t num_deepstacks =
        model_args_.mm_deepstack_visual_indexes().size();
    std::vector<torch::Tensor> deepstack_input_embeds(
        num_deepstacks, torch::zeros_like(inputs_embeds));
    auto merge_visual_modality = [&](const std::string& embed_key,
                                     const std::string& mask_key) {
      std::optional<torch::Tensor> embedding =
          mm_data.get<torch::Tensor>(embed_key);
      std::optional<torch::Tensor> mask = mm_data.get<torch::Tensor>(mask_key);
      if (!embedding.has_value() || !mask.has_value()) {
        return;
      }
      const std::vector<torch::Tensor> chunks = embedding.value().chunk(
          static_cast<int64_t>(num_deepstacks + 1), /*dim=*/1);
      inputs_embeds =
          merge_multimodal_embeddings(inputs_embeds, chunks[0], mask.value());
      for (size_t index = 0; index < num_deepstacks; ++index) {
        deepstack_input_embeds[index] = merge_multimodal_embeddings(
            deepstack_input_embeds[index], chunks[index + 1], mask.value());
      }
    };
    merge_visual_modality(get_embedding_key(MMType::IMAGE), "image|mask");
    merge_visual_modality(get_embedding_key(MMType::VIDEO), "video|mask");
    std::optional<torch::Tensor> audio_embeds =
        mm_data.get<torch::Tensor>(get_embedding_key(MMType::AUDIO));
    std::optional<torch::Tensor> audio_mask =
        mm_data.get<torch::Tensor>(qwen3_audio::kMaskKey);
    if (audio_embeds.has_value() && audio_mask.has_value()) {
      inputs_embeds = merge_multimodal_embeddings(
          inputs_embeds, audio_embeds.value(), audio_mask.value());
    }
    input_params.multimodal.deep_stacks = std::move(deepstack_input_embeds);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) {
    return language_model_->logits(hidden_states, selected_indices);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.visual."));
      audio_tower_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.audio_tower."));
    }
    // verify
    visual_->verify_loaded_weights("thinker.visual.");
    visual_->merge_loaded_weights();
    audio_tower_->verify_loaded_weights("thinker.audio_tower.");
    audio_tower_->merge_loaded_weights();
    audio_tower_->to(options_.device(),
                     torch::typeMetaToScalarType(options_.dtype()));

    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(
          std::move(loader), "thinker.model.", "thinker.lm_head.");
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
  Qwen3OmniMoeThinkerVisionTransformer visual_{nullptr};
  Qwen3AudioEncoder audio_tower_{nullptr};
  Qwen3MoeForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoeThinkerForConditionalGeneration);

REGISTER_MULTIMODAL_PROCESSOR(qwen3_omni_moe_thinker,
                              Qwen3OmniMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe_thinker,
                          Qwen3OmniMoeThinkerForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_omni_moe_thinker,
                             xllm::Qwen3VLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_omni_moe_thinker,
                    [&] { load_qwen3_omni_moe_model_args(json, args); });

}  // namespace xllm::npu::model

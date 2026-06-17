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
#include "core/layers/qwen2_decoder_layer.h"
#include "core/layers/qwen2_vision_layer.h"
#include "models/llm/qwen2.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_video_processor.h"
#include "qwen2_5_vl.h"

namespace xllm {

class Qwen2_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  Qwen2_VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_temporal_patch_size() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();

    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(false)));

    proj_->weight.set_data(proj_->weight.to(options));
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
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
  }

 private:
  bool proj_weight_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Qwen2_VisionPatchEmbed);

class Qwen2_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_VisionRotaryEmbeddingImpl(const ModelContext& context) {
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
  int64_t dim_ = 0;
  double theta_ = 0.0;

  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen2_VisionRotaryEmbedding);

class Qwen2_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen2_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto quant_args = context.get_quant_args();
    auto parallel_args = context.get_parallel_args();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    int64_t context_dim = model_args.mm_hidden_size();
    int64_t spatial_merge_size = model_args.mm_spatial_merge_size();

    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
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
    x = norm_(x).view({-1, hidden_size_});
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto& norm_dict = state_dict.get_dict_with_prefix("ln_q.");
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
    const auto& fc1_dict = state_dict.get_dict_with_prefix("mlp.0.");
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

    const auto& fc2_dict = state_dict.get_dict_with_prefix("mlp.2.");
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
        << "weight is not loaded for " << prefix + "mlp.0" + ".weight";
    CHECK(is_fc1_bias_loaded)
        << "bias is not loaded for " << prefix + "mlp.0" + ".bias";
    CHECK(is_fc2_weight_loaded)
        << "weight is not loaded for " << prefix + "mlp.2" + ".weight";
    CHECK(is_fc2_bias_loaded)
        << "bias is not loaded for " << prefix + "mlp.2" + ".bias";
    CHECK(is_norm_weight_loaded)
        << "weight is not loaded for " << prefix + "ln_q" + ".weight";
    CHECK(is_norm_bias_loaded)
        << "bias is not loaded for " << prefix + "ln_q" + ".bias";
  }

 private:
  int hidden_size_;
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
TORCH_MODULE(Qwen2_VisionPatchMerger);

class Qwen2_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen2_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();

    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    spatial_merge_unit_ = static_cast<int>(std::pow(spatial_merge_size_, 2));
    // mlp_ratio_ = model_args.mm_mlp_ratio();

    patch_embed_ =
        register_module("patch_embed", Qwen2_VisionPatchEmbed(context));
    rotary_pos_emb_ =
        register_module("rotary_pos_emb", Qwen2_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = layer::Qwen2_VisionLayer(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Qwen2_VisionPatchMerger(context));
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

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    // patchify
    // hidden_states = x.to(device=self.device, dtype=self.dtype);
    hidden_states = patch_embed_(hidden_states);
    //  compute position embedding
    auto rotary_pos_emb = rot_pos_emb(grid_thw);
    auto seq_len = hidden_states.sizes()[0];

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
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      hidden_states = layers_[idx](
          hidden_states, m_cos, m_sin, cu_seqlens, cu_seqlens_vec, idx);
    }
    // adapter
    hidden_states = merger_(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

 private:
  int hidden_size_ = 0;
  int num_heads_ = 0;
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::unordered_set<int> fullatt_block_indexes_;
  int spatial_merge_unit_ = 0;
  // int mlp_ratio_ = 4;

  Qwen2_VisionPatchEmbed patch_embed_{nullptr};
  Qwen2_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::Qwen2_VisionLayer> layers_;
  Qwen2_VisionPatchMerger merger_{nullptr};

  torch::Tensor m_cos;
  torch::Tensor m_sin;
  int device_id = 0;
};
TORCH_MODULE(Qwen2_VisionTransformer);

struct Qwen2_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen2_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class Qwen2_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen2_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_VisionTransformer(context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen2_VLImageInputs>& image_inputs,
                             std::optional<Qwen2_VLVideoInputs>& video_inputs) {
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

    torch::Tensor second_per_grid_ts;
    if (const auto& res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined() &&
        second_per_grid_ts.defined())
      video_inputs = Qwen2_VLVideoInputs{
          pixel_values_videos, video_grid_thw, second_per_grid_ts};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen2_VLImageInputs> image_input;
    std::optional<Qwen2_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);
    auto merge_size = model_args_.mm_image_merge_size();
    MMDict multimodal_embeds;
    if (image_input) {
      // visual
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw);
      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, /*dim=*/0);
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

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
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
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_VLForConditionalGeneration);

using Qwen2VLMultimodalProcessor = MultimodalProcessor<Qwen2VLPromptProcessor,
                                                       Qwen2VLImageProcessor,
                                                       Qwen2VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen2_vl, Qwen2VLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_vl, Qwen2_VLForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen2_vl, QwenVLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen2_vl, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "qwen2_vl");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  // LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.embed_dim", 1280);
  // LOAD_ARG_OR(mm_mlp_ratio, "vision_config.mlp_ratio", 4);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});
}  // namespace xllm

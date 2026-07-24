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
#include <torch/torch.h>

#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/model_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_mistral3_vision_encoder_layer_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/util/timer.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/llm_model_base.h"
#include "models/llm/npu/mistral.h"
#include "models/model_registry.h"
#include "processors/mistral3_image_processor.h"
#include "processors/mistral3_prompt_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm {

// ==================== Mistral3 Vision Encoder Block ====================
class Mistral3_VisionBlockImpl : public torch::nn::Module {
 public:
  explicit Mistral3_VisionBlockImpl(const ModelContext& context) {
    encoder_layer_ = register_module(
        "encoder_layer", layer::NpuMistral3VisionEncoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int>& cu_seq_len_vec,
                        ModelInputParams& input_params,
                        int node_id) {
    return encoder_layer_(x,
                          m_cos_pos,
                          m_sin_pos,
                          cu_seq_len,
                          cu_seq_len_vec,
                          input_params,
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
  layer::NpuMistral3VisionEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(Mistral3_VisionBlock);

// ==================== Mistral3 Vision Patch Embedding ====================
class Mistral3_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  explicit Mistral3_VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int64_t in_channels = model_args.mm_num_channels();
    int64_t hidden_size = model_args.mm_hidden_size();
    int64_t patch_size = model_args.mm_patch_size();
    in_features_ = in_channels * patch_size * patch_size;

    auto proj = torch::nn::Linear(
        torch::nn::LinearOptions(in_features_, hidden_size).bias(false));
    proj->weight.set_data(proj->weight.to(options));
    proj_ = register_module("proj", proj);
  }

  torch::Tensor forward(torch::Tensor x) { return proj_(x); }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("patch_conv.weight");
    if (weight.defined()) {
      // Conv2d weight [hidden_size, C, P, P] → Linear weight [hidden_size,
      // C*P*P]
      weight = weight.reshape({weight.size(0), -1});
      CHECK(proj_->weight.sizes() == weight.sizes())
          << "patch_conv weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "patch_conv.weight";
  }

 private:
  int64_t in_features_ = 0;
  bool proj_weight_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Mistral3_VisionPatchEmbed);

// ==================== Mistral3 Vision Rotary Embedding (2D RoPE)
// ====================
class Mistral3_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  explicit Mistral3_VisionRotaryEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    dim_ = model_args.mm_hidden_size() / model_args.mm_num_attention_heads();
    theta_ = 10000.0;

    auto opts = options.dtype(torch::kFloat32);
    auto freqs =
        1.0 / torch::pow(theta_, torch::arange(0, dim_, 2, opts) / dim_);
    inv_freq_h_ = register_buffer(
        "inv_freq_h",
        freqs.index({torch::indexing::Slice(0, torch::indexing::None, 2)}));
    inv_freq_w_ = register_buffer(
        "inv_freq_w",
        freqs.index({torch::indexing::Slice(1, torch::indexing::None, 2)}));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(int64_t height,
                                                   int64_t width) {
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(inv_freq_h_.device());

    auto h_positions = torch::arange(height, options);
    auto freqs_h = torch::outer(h_positions, inv_freq_h_);  // [height, 16]

    auto w_positions = torch::arange(width, options);
    auto freqs_w = torch::outer(w_positions, inv_freq_w_);  // [width, 16]

    auto h_grid = h_positions.unsqueeze(1).expand({-1, width});
    auto w_grid = w_positions.unsqueeze(0).expand({height, -1});

    auto h_idx = h_grid.flatten().to(torch::kInt64);
    auto w_idx = w_grid.flatten().to(torch::kInt64);
    auto h_emb = freqs_h.index({h_idx});  // [flatten, 16]
    auto w_emb = freqs_w.index({w_idx});  // [flatten, 16]

    // cat h+w → [flatten, 32], then cat with self → [flatten, 64]
    // Matches HF: inv_freq = cat((inv_freq, inv_freq), dim=-1)
    auto inv_freq_patch = torch::cat({h_emb, w_emb}, /*dim=*/-1);
    auto emb = torch::cat({inv_freq_patch, inv_freq_patch}, /*dim=*/-1);
    // auto emb = inv_freq_patch.repeat_interleave(2, /*dim=*/-1);
    auto cos_emb = torch::cos(emb);
    auto sin_emb = torch::sin(emb);

    return std::make_tuple(cos_emb, sin_emb);
  }

 private:
  int64_t dim_ = 0;
  double theta_ = 0.0;
  torch::Tensor inv_freq_h_;  // even freqs for height (16 values)
  torch::Tensor inv_freq_w_;  // odd freqs for width (16 values)
};
TORCH_MODULE(Mistral3_VisionRotaryEmbedding);

// ==================== Mistral3 Vision Patch Merger ====================

class Mistral3_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  explicit Mistral3_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int64_t vision_hidden_size = model_args.mm_hidden_size();
    int64_t projection_dim = model_args.mm_projection_dim();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();

    hidden_size_ = vision_hidden_size *
                   static_cast<int64_t>(std::pow(spatial_merge_size_, 2));

    ln_q_ = register_module("norm", layer::NpuRMSNorm(context));

    auto merging_linear = torch::nn::Linear(
        torch::nn::LinearOptions(hidden_size_, vision_hidden_size).bias(false));
    merging_linear->weight.set_data(merging_linear->weight.to(options));
    merging_layer_ = register_module("merging_layer", merging_linear);

    auto linear_1 = torch::nn::Linear(
        torch::nn::LinearOptions(vision_hidden_size, projection_dim)
            .bias(false));
    linear_1->weight.set_data(linear_1->weight.to(options));
    auto act = torch::nn::GELU();
    auto linear_2 = torch::nn::Linear(
        torch::nn::LinearOptions(projection_dim, projection_dim).bias(false));
    linear_2->weight.set_data(linear_2->weight.to(options));

    mlp_ =
        register_module("mlp", torch::nn::Sequential(linear_1, act, linear_2));
    layers_ = std::make_tuple(linear_1, act, linear_2);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor image_grid_thw) {
    x = ln_q_(x, 0);

    auto grid_thw_cpu = image_grid_thw.cpu();
    int64_t num_images = grid_thw_cpu.size(0);
    int64_t d = x.size(-1);
    int64_t merge_size = spatial_merge_size_;

    std::vector<torch::Tensor> merged_patches;
    merged_patches.reserve(num_images);

    int64_t offset = 0;
    for (int64_t i = 0; i < num_images; ++i) {
      int64_t t = grid_thw_cpu[i][0].item<int64_t>();
      int64_t h = grid_thw_cpu[i][1].item<int64_t>();
      int64_t w = grid_thw_cpu[i][2].item<int64_t>();
      int64_t tokens_per_frame = h * w;

      for (int64_t frame = 0; frame < t; ++frame) {
        auto frame_tokens = x.slice(0, offset, offset + tokens_per_frame)
                                .contiguous()
                                .reshape({h, w, d});

        auto merged =
            frame_tokens
                .reshape(
                    {h / merge_size, merge_size, w / merge_size, merge_size, d})
                .permute({0, 2, 4, 1, 3})
                .contiguous()
                .reshape({(h / merge_size) * (w / merge_size),
                          d * merge_size * merge_size});
        merged_patches.push_back(merged);
        offset += tokens_per_frame;
      }
    }

    x = torch::cat(merged_patches, /*dim=*/0);

    x = merging_layer_(x);
    x = mlp_->forward(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    ln_q_->load_state_dict(state_dict.get_dict_with_prefix("norm."));

    auto merging_weight =
        state_dict.get_tensor("patch_merger.merging_layer.weight");
    if (!merging_weight.defined()) {
      merging_weight = state_dict.get_tensor("merging_layer.weight");
    }
    if (merging_weight.defined()) {
      CHECK(merging_layer_->weight.sizes() == merging_weight.sizes())
          << "merging_layer weight size mismatch";
      merging_layer_->weight.data().copy_(merging_weight);
      is_merging_weight_loaded_ = true;
    }

    auto linear_1_weight = state_dict.get_tensor("linear_1.weight");
    if (linear_1_weight.defined()) {
      CHECK(std::get<0>(layers_)->weight.sizes() == linear_1_weight.sizes())
          << "linear_1 weight size mismatch";
      std::get<0>(layers_)->weight.data().copy_(linear_1_weight);
      is_linear_1_weight_loaded_ = true;
    }

    auto linear_2_weight = state_dict.get_tensor("linear_2.weight");
    if (linear_2_weight.defined()) {
      CHECK(std::get<2>(layers_)->weight.sizes() == linear_2_weight.sizes())
          << "linear_2 weight size mismatch";
      std::get<2>(layers_)->weight.data().copy_(linear_2_weight);
      is_linear_2_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    ln_q_->verify_loaded_weights(prefix + "norm.");
    CHECK(is_merging_weight_loaded_)
        << "weight is not loaded for "
        << prefix + "patch_merger.merging_layer.weight";
    CHECK(is_linear_1_weight_loaded_)
        << "weight is not loaded for " << prefix + "linear_1.weight";
    CHECK(is_linear_2_weight_loaded_)
        << "weight is not loaded for " << prefix + "linear_2.weight";
  }

  void merge_loaded_weights() { ln_q_->merge_loaded_weights(); }

 private:
  int64_t hidden_size_;
  int64_t spatial_merge_size_ = 0;
  layer::NpuRMSNorm ln_q_{nullptr};
  torch::nn::Linear merging_layer_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
  std::tuple<torch::nn::Linear, torch::nn::GELU, torch::nn::Linear> layers_ = {
      nullptr,
      nullptr,
      nullptr};
  bool is_merging_weight_loaded_ = false;
  bool is_linear_1_weight_loaded_ = false;
  bool is_linear_2_weight_loaded_ = false;
};
TORCH_MODULE(Mistral3_VisionPatchMerger);

// ==================== Mistral3 Vision Transformer ====================
class Mistral3_VisionTransformerImpl : public torch::nn::Module {
 public:
  explicit Mistral3_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    hidden_size_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();

    patch_embed_ =
        register_module("patch_embed", Mistral3_VisionPatchEmbed(context));
    ln_pre_ = register_module("ln_pre", layer::NpuRMSNorm(context));
    rotary_pos_emb_ = register_module("rotary_pos_emb",
                                      Mistral3_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); ++idx) {
      auto block = Mistral3_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Mistral3_VisionPatchMerger(context));
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor image_grid_thw,
                        const ModelInputParams& input_params) {
    auto grid_h = hidden_states.size(2) / patch_size_;
    auto grid_w = hidden_states.size(3) / patch_size_;
    int64_t C = hidden_states.size(1);

    hidden_states =
        hidden_states.view({1, C, grid_h, patch_size_, grid_w, patch_size_});
    hidden_states = hidden_states.permute({0, 1, 3, 5, 2, 4}).contiguous();
    hidden_states =
        hidden_states.view({1, C * patch_size_ * patch_size_, grid_h * grid_w});

    hidden_states = hidden_states.squeeze(0).transpose(0, 1).contiguous();
    // Linear projection: [num_patches, hidden_size]
    hidden_states = patch_embed_(hidden_states);

    hidden_states = hidden_states.to(torch::kBFloat16);
    hidden_states = ln_pre_(hidden_states, 0);
    // compute sequence lengths from image_grid_thw
    auto grid_thw_cpu = image_grid_thw.cpu();
    int64_t num_images = grid_thw_cpu.size(0);
    int64_t total_frames = 0;
    for (int64_t idx = 0; idx < num_images; ++idx) {
      total_frames += grid_thw_cpu[idx][0].item<int64_t>();
    }

    std::vector<int> seq_lens_vec;
    seq_lens_vec.reserve(static_cast<size_t>(total_frames));

    int64_t total_tokens = 0;
    for (int64_t idx = 0; idx < num_images; ++idx) {
      int64_t t = grid_thw_cpu[idx][0].item<int64_t>();
      int64_t h = grid_thw_cpu[idx][1].item<int64_t>();
      int64_t w = grid_thw_cpu[idx][2].item<int64_t>();
      for (int64_t frame = 0; frame < t; ++frame) {
        seq_lens_vec.push_back(static_cast<int>(h * w));
      }
      total_tokens += t * h * w;
    }
    CHECK_EQ(total_tokens, hidden_states.size(0))
        << "image_grid_thw token count mismatch with vision hidden states";

    // compute 2D RoPE cos/sin per image
    std::vector<torch::Tensor> cos_vec;
    std::vector<torch::Tensor> sin_vec;
    cos_vec.reserve(num_images);
    sin_vec.reserve(num_images);

    for (int64_t idx = 0; idx < num_images; ++idx) {
      int64_t t = grid_thw_cpu[idx][0].item<int64_t>();
      int64_t h = grid_thw_cpu[idx][1].item<int64_t>();
      int64_t w = grid_thw_cpu[idx][2].item<int64_t>();
      auto [cos_img, sin_img] = rotary_pos_emb_(h, w);
      cos_vec.push_back(cos_img.repeat({t, 1}));
      sin_vec.push_back(sin_img.repeat({t, 1}));
    }

    torch::Tensor m_cos =
        torch::cat(cos_vec, /*dim=*/0).to(hidden_states.dtype());
    torch::Tensor m_sin =
        torch::cat(sin_vec, /*dim=*/0).to(hidden_states.dtype());

    // pad cos/sin to head_dim if needed
    int64_t head_dim = hidden_size_ / num_heads_;
    int64_t current_dim = m_cos.size(1);
    if (current_dim < head_dim) {
      int64_t pad_size = head_dim - current_dim;
      m_cos = torch::nn::functional::pad(
          m_cos, torch::nn::functional::PadFuncOptions({0, pad_size}));
      m_sin = torch::nn::functional::pad(
          m_sin, torch::nn::functional::PadFuncOptions({0, pad_size}));
    }

    torch::TensorOptions opts = torch::TensorOptions()
                                    .dtype(torch::kInt32)
                                    .device(hidden_states.device());
    torch::Tensor seq_lens = torch::tensor(seq_lens_vec, opts);

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);

    // 24 vision encoder layers
    for (int32_t idx = 0; idx < static_cast<int32_t>(layers_.size()); ++idx) {
      hidden_states = layers_[idx](hidden_states,
                                   m_cos,
                                   m_sin,
                                   seq_lens,
                                   seq_lens_vec,
                                   input_params_new,
                                   idx);
    }

    // PatchMerger: RMSNorm + unfold + Linear + MLP
    hidden_states = merger_(hidden_states, image_grid_thw);
    return hidden_states;
  }

  // Load vision_tower.* weights (patch_conv, ln_pre, transformer layers).
  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(state_dict);
    ln_pre_->load_state_dict(state_dict.get_dict_with_prefix("ln_pre."));

    for (int32_t idx = 0; idx < static_cast<int32_t>(layers_.size()); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "transformer.layers." + std::to_string(idx) + "."));
    }
  }

  // Load multi_modal_projector.* weights into the merger.
  void load_merger_state_dict(const StateDict& state_dict) {
    merger_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix);
    ln_pre_->verify_loaded_weights(prefix + "ln_pre.");

    for (int32_t idx = 0; idx < static_cast<int32_t>(layers_.size()); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "transformer.layers." +
                                          std::to_string(idx) + ".");
    }
  }

  void verify_merger_loaded_weights(const std::string& prefix) const {
    merger_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() {
    ln_pre_->merge_loaded_weights();
    for (int32_t idx = 0; idx < static_cast<int32_t>(layers_.size()); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
    merger_->merge_loaded_weights();
  }

 private:
  int64_t hidden_size_ = 0;
  int64_t num_heads_ = 0;
  int64_t patch_size_ = 0;
  int64_t spatial_merge_size_ = 0;

  Mistral3_VisionPatchEmbed patch_embed_{nullptr};
  layer::NpuRMSNorm ln_pre_{nullptr};
  Mistral3_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Mistral3_VisionBlock> layers_;
  Mistral3_VisionPatchMerger merger_{nullptr};
};
TORCH_MODULE(Mistral3_VisionTransformer);

// ==================== Mistral3 For Conditional Generation (VLM)
// ====================

struct Mistral3_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

class Mistral3ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  explicit Mistral3ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Mistral3_VisionTransformer(context));
    language_model_ = register_module("language_model", MistralModel(context));
    lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  // ==================== Vision Encoder ====================
  void prepare_encoder_input(
      const ModelInputParams& input_params,
      std::optional<Mistral3_VLImageInputs>& image_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;

    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    if (pixel_values.defined() && image_grid_thw.defined()) {
      image_inputs = Mistral3_VLImageInputs{pixel_values, image_grid_thw};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    const auto& mm_data = input_params.multimodal.mm_data;
    bool has_pixel = mm_data.get<torch::Tensor>("pixel_values").has_value();
    bool has_thw = mm_data.get<torch::Tensor>("image_grid_thw").has_value();

    std::optional<Mistral3_VLImageInputs> image_input;
    prepare_encoder_input(input_params, image_input);

    MMDict multimodal_embeds;
    if (image_input) {
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw,
                                  input_params);

      int64_t spatial_merge_size = model_args_.mm_spatial_merge_size();
      auto image_tokens = (image_input->image_grid_thw.prod(-1) /
                           spatial_merge_size / spatial_merge_size)
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

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    auto special_token_ids =
        torch::tensor({model_args_.image_token_id()},
                      input_ids.options().dtype(torch::kInt64));
    return torch::isin(input_ids, special_token_ids);
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  // ==================== Embeddings ====================
  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor multimodal_embeds;
    // Pipeline stores image embeddings under "image|embedding" key
    // (via EncoderEmbeddingGatherVisitor). Also check "embedding" for
    // pre-computed embedding passthrough.
    if (const auto& emb = mm_data.get<torch::Tensor>("image|embedding")) {
      multimodal_embeds = emb.value();
    } else if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }
    auto is_multimodal = generate_multimodal_mask(input_ids);

    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
    return inputs_embeds;
  }

  // ==================== Forward ====================
  // Remove bos token (line 4 in Flux2 chat_template.jinja) to adapt xLLM
  // default add_special_tokens behavior
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto model_output =
        language_model_->forward(tokens, positions, kv_caches, input_params);
    // Mistral3 is used for Flux2 text_encoder:
    // select 3 intermediate layers (9, 19, 29) as embeddings
    bool is_embedding_task =
        ::xllm::ModelConfig::get_instance().task() == "embed";
    bool return_full_embeddings =
        ::xllm::ModelConfig::get_instance().enable_return_mm_full_embeddings();
    bool return_flux2_embeddings =
        model_args_.encoder_embedding_mode() ||
        (is_embedding_task && return_full_embeddings);
    if (return_flux2_embeddings) {
      CHECK_GE(model_output.aux_hidden_states.size(0), 30)
          << "Model must have at least 30 layers for embedding mode";
      auto indices = torch::tensor({9, 19, 29}, torch::kLong)
                         .to(model_output.aux_hidden_states.device());
      auto selected =
          model_output.aux_hidden_states.index_select(/*dim=*/0, indices);

      return ModelOutput(selected);
    }
    return model_output;
  }

  // ==================== Pooler / Logits ====================

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    // Flux2 text encoder: return raw hidden states without L2 normalization
    // hidden_states shape: [3, seq_len, hidden_size] (layers 9, 19, 29)
    auto h = hidden_states;
    if (::xllm::ModelConfig::get_instance()
            .enable_return_mm_full_embeddings()) {
      return h;
    }
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  // ==================== Accessors ====================

  layer::NpuLmHead get_npu_lm_head() { return lm_head_; }
  void set_npu_lm_head(layer::NpuLmHead& head) { lm_head_ = head; }
  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }
  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  // ==================== Weight Loading ====================

  void load_state_dict(const StateDict& state_dict) {
    language_model_->load_state_dict(state_dict.get_dict_with_prefix("model."));
    lm_head_->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    language_model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights(prefix + "lm_head.");
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "Loading Mistral3ForConditionalGeneration from ModelLoader...";

    // 1. Load vision encoder weights (prefix: vision_tower.*)
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix(
          std::vector<std::string>{"vision_tower.", "model.vision_tower."}));
    }
    visual_->verify_loaded_weights("vision_tower.");

    // 2. Load multi_modal_projector weights
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_merger_state_dict(
          state_dict->get_dict_with_prefix("multi_modal_projector."));
    }
    visual_->verify_merger_loaded_weights("multi_modal_projector.");
    visual_->merge_loaded_weights();

    // 3. Load language model weights
    if (!model_args_.encoder_embedding_mode()) {
      for (const auto& state_dict : loader->get_state_dicts()) {
        language_model_->load_state_dict(
            state_dict->get_dict_with_prefix("language_model.model."));
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("language_model.lm_head."));
      }
      language_model_->merge_loaded_weights();
      lm_head_->merge_loaded_weights();

      language_model_->verify_loaded_weights("language_model.model.");
      lm_head_->verify_loaded_weights("language_model.lm_head.");
    }
  }

  // ==================== Lifecycle ====================

  void merge_loaded_weights() {
    if (visual_) {
      visual_->merge_loaded_weights();
    }
    if (language_model_) {
      language_model_->merge_loaded_weights();
    }
    if (lm_head_) {
      lm_head_->merge_loaded_weights();
    }
  }

  void merge_and_move_pinned_host() {
    if (language_model_) {
      language_model_->merge_and_move_pinned_host();
    }
    if (lm_head_) {
      lm_head_->merge_and_move_pinned_host();
    }
  }

  void free_weights() {
    if (language_model_) {
      language_model_->free_weights();
    }
    if (lm_head_) {
      lm_head_->free_weights();
    }
  }

  void reload_weights() {
    if (language_model_) {
      language_model_->reload_weights();
    }
    if (lm_head_) {
      lm_head_->reload_weights();
    }
  }

  void reload_weights_from_device() {
    if (language_model_) {
      language_model_->reload_weights_from_device();
    }
    if (lm_head_) {
      lm_head_->reload_weights_from_device();
    }
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Mistral3_VisionTransformer visual_{nullptr};
  MistralModel language_model_{nullptr};
  layer::NpuLmHead lm_head_{nullptr};
};
TORCH_MODULE(Mistral3ForConditionalGeneration);

// ==================== Registration ====================

// REGISTER_CAUSAL_VLM_MODEL(mistral3, Mistral3ForConditionalGeneration);
// REGISTER_CAUSAL_MODEL(mistral3, Mistral3ForConditionalGeneration);
REGISTER_CAUSAL_MODEL_WITH_VARNAME(mistral3_llm,
                                   mistral3,
                                   Mistral3ForConditionalGeneration);
REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(mistral3_vlm,
                                       mistral3,
                                       Mistral3ForConditionalGeneration);

REGISTER_MODEL_ARGS(mistral3, [&] {
  // Text config (from top-level JSON)
  LOAD_ARG_OR(model_type, "model_type", "mistral3");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 131072);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5120);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 32768);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1e9);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);

  // Vision config (Pixtral ViT, from vision_config.* in JSON)
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1024);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4096);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 24);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 1540);
  LOAD_ARG_OR(mm_projection_dim, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(mm_spatial_merge_size, "spatial_merge_size", 2);

  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  // Multimodal token IDs
  LOAD_ARG_OR(image_token_id, "image_token_index", 10);

  // Image processor config (from preprocessor_config.json)
  LOAD_ARG_OR(mm_image_do_normalize, "do_normalize", true);
  LOAD_ARG_OR(mm_image_do_rescale, "do_rescale", true);
  LOAD_ARG_OR(mm_image_do_resize, "do_resize", true);
});

using Mistral3MultimodalProcessor =
    MultimodalProcessor<Mistral3PromptProcessor, Mistral3ImageProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(mistral3, Mistral3MultimodalProcessor);

}  // namespace xllm

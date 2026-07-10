/* Copyright 2025-2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/pipelines/pipeline_longcat_image.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_longcat_image.h"
#include "models/model_registry.h"
#include "models/vlm/qwen2_5_vl.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/transforms.h"

namespace xllm {

// Forward declarations
class LongCatImageEditPipelineImpl;

// Prompt template constants for LongCat-Image-Edit
// Ref:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/longcat_image/pipeline_longcat_image_edit.py
// Prefix includes <|vision_start|><|image_pad|><|vision_end|> where
// <|image_pad|> is replaced with num_image_tokens copies at runtime
constexpr const char* PROMPT_TEMPLATE_ENCODE_PREFIX_EDIT =
    "<|im_start|>system\nAs an image editing expert, first analyze the content "
    "and attributes of the input image(s). Then, based on the user's editing "
    "instructions, clearly and precisely determine how to modify the given "
    "image(s), ensuring that only the specified parts are altered and all "
    "other aspects remain consistent with the original(s).<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>";
constexpr const char* PROMPT_TEMPLATE_ENCODE_SUFFIX_EDIT =
    "<|im_end|>\n<|im_start|>assistant\n";

// Helper copied from diffusers LongCatImageEditPipeline.calculate_dimensions
inline std::pair<int64_t, int64_t> calculate_dimensions_edit(
    int64_t target_area,
    float ratio) {
  double width = std::sqrt(static_cast<double>(target_area) * ratio);
  double height = width / ratio;

  auto round_to_16 = [](double x) -> int64_t {
    int64_t v = static_cast<int64_t>(x);
    if (v % 16 != 0) v = (v / 16 + 1) * 16;
    return v;
  };

  int64_t w = round_to_16(width);
  int64_t h = round_to_16(height);
  return {w, h};
}

// LongCat-Image-Edit pipeline for CUDA backend.
class LongCatImageEditPipelineImpl : public torch::nn::Module {
 public:
  explicit LongCatImageEditPipelineImpl(const DiTModelContext& context)
      : context_(context),
        vl_image_processor_(context.get_model_args("text_encoder")) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();

    // Initialize FlashinferWorkspace for attention operations
    layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
        options_.device());

    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();

    LOG(INFO) << "Initializing LongCat-Image-Edit pipeline...";

    vae_image_processor_ =
        VAEImageProcessor(ModelContext(context.get_parallel_args(),
                                       context.get_model_args("vae"),
                                       context.get_quant_args("vae"),
                                       context.get_tensor_options()),
                          /*do_resize=*/true,
                          /*do_normalize=*/true,
                          /*do_binarize=*/false,
                          /*do_convert_rgb=*/false,
                          /*do_convert_grayscale=*/false,
                          model_args.latent_channels());
    vae_ = VAE(ModelContext(context.get_parallel_args(),
                            context.get_model_args("vae"),
                            context.get_quant_args("vae"),
                            context.get_tensor_options()));

    pos_embed_ = register_module(
        "pos_embed",
        LongCatImagePosEmbed(
            kLongCatRopeScaleBase,
            context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = LongCatImageTransformer2DModel(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("transformer"),
                     context.get_quant_args("transformer"),
                     context.get_tensor_options()));

    const auto& original_parallel_args = context.get_parallel_args();
    ParallelArgs vlm_parallel_args = original_parallel_args;
    if (original_parallel_args.tp_group_ == nullptr) {
      LOG(INFO)
          << "Creating real ProcessGroup for single-device VLM initialization.";
      vlm_tp_group_ = create_process_group(0,
                                           1,
                                           1,
                                           29501,  // Different port from T2I
                                           false,
                                           "127.0.0.1",
                                           "vlm_tp_group_longcat_edit",
                                           options_.device());
      vlm_parallel_args.tp_group_ = vlm_tp_group_.get();
    }

    LOG(INFO)
        << "LongCat-Image-Edit uses Qwen2_5_VL as text encoder (edit mode)";
    text_encoder_ = Qwen2_5_VLForConditionalGeneration(
        ModelContext(vlm_parallel_args,
                     context.get_model_args("text_encoder"),
                     context.get_quant_args("text_encoder"),
                     context.get_tensor_options()));

    scheduler_ = FlowMatchEulerDiscreteScheduler(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("scheduler"),
                     context.get_quant_args("scheduler"),
                     context.get_tensor_options()));

    prompt_template_encode_prefix_ = PROMPT_TEMPLATE_ENCODE_PREFIX_EDIT;
    prompt_template_encode_suffix_ = PROMPT_TEMPLATE_ENCODE_SUFFIX_EDIT;

    register_module("vae", vae_);
    register_module("vae_image_processor", vae_image_processor_);
    register_module("transformer", transformer_);
    register_module("text_encoder", text_encoder_);
    register_module("scheduler", scheduler_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);
    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);
    auto negative_prompts_2 =
        input.negative_prompts_2.empty()
            ? std::nullopt
            : std::make_optional(input.negative_prompts_2);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;
    auto negative_pooled_prompt_embeds =
        input.negative_pooled_prompt_embeds.defined()
            ? std::make_optional(input.negative_pooled_prompt_embeds)
            : std::nullopt;

    auto image = input.images.defined() ? std::make_optional(input.images)
                                        : std::nullopt;

    CHECK(image.has_value()) << "LongCat-Image-Edit requires an input image.";

    std::vector<torch::Tensor> output = forward_(
        image,                             // input image
        prompts,                           // prompt
        prompts_2,                         // prompt_2 (unused)
        negative_prompts,                  // negative_prompt
        negative_prompts_2,                // negative_prompt_2 (unused)
        generation_params.true_cfg_scale,  // cfg scale (edit uses guidance)
        generation_params.num_inference_steps,    // num_inference_steps
        generation_params.guidance_scale,         // guidance_scale
        generation_params.num_images_per_prompt,  // num_images_per_prompt
        seed,                                     // seed
        latents,                                  // latents
        prompt_embeds,                            // prompt_embeds
        negative_prompt_embeds,                   // negative_prompt_embeds
        pooled_prompt_embeds,                     // pooled_prompt_embeds
        negative_pooled_prompt_embeds,          // negative_pooled_prompt_embeds
        generation_params.max_sequence_length,  // max_sequence_length
        generation_params.enable_cfg_renorm,    // enable_cfg_renorm
        generation_params.cfg_renorm_min        // cfg_renorm_min
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output[0], output[0].size(0), 0);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "LongCat-Image-Edit pipeline loading model from "
              << loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto text_encoder_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");

    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    text_encoder_->load_model(std::move(text_encoder_loader));
    text_encoder_->to(options_.device());
    tokenizer_ = tokenizer_loader->tokenizer();
  }

 private:
  // Prepare text position IDs for text tokens
  torch::Tensor prepare_text_ids(int64_t num_tokens,
                                 int64_t start_height = 0,
                                 int64_t start_width = 0) {
    // Use int64 first, then convert to float32 for RoPE to avoid precision
    // issues with large indices under bfloat16.
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor text_ids_int = torch::zeros({num_tokens, 3}, int_options);

    // modality_id = 0 for text
    text_ids_int.select(1, 0).fill_(0);

    // position indices: [0, 1, 2, ..., num_tokens-1]
    torch::Tensor token_range = torch::arange(num_tokens, int_options);
    text_ids_int.select(1, 1) = token_range + start_height;
    text_ids_int.select(1, 2) = token_range + start_width;

    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor text_ids = text_ids_int.to(float32_options);

    return text_ids;
  }

  // Generic position IDs for image tokens, following
  // diffusers.prepare_pos_ids(type="image").
  torch::Tensor prepare_image_pos_ids(int64_t modality_id,
                                      int64_t start_row,
                                      int64_t start_col,
                                      int64_t height,
                                      int64_t width) {
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor pos_ids =
        torch::zeros({height, width, 3}, int_options);  // [H, W, 3]

    // modality id
    pos_ids.select(2, 0).fill_(modality_id);

    auto height_range = torch::arange(height, int_options).unsqueeze(1);
    auto width_range = torch::arange(width, int_options).unsqueeze(0);

    pos_ids.select(2, 1) += height_range + start_row;
    pos_ids.select(2, 2) += width_range + start_col;

    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor pos_ids_f32 = pos_ids.to(float32_options);
    pos_ids_f32 = pos_ids_f32.view({height * width, 3});

    return pos_ids_f32;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents,
                             int64_t batch_size,
                             int64_t num_channels_latents,
                             int64_t height,
                             int64_t width) {
    torch::Tensor latents_packed = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents_packed = latents_packed.permute({0, 2, 4, 1, 3, 5});
    latents_packed = latents_packed.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});
    return latents_packed;
  }

  torch::Tensor unpack_latents(const torch::Tensor& latents,
                               int64_t height,
                               int64_t width,
                               int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t channels = latents.size(2);
    height = 2 * (height / (vae_scale_factor * 2));
    width = 2 * (width / (vae_scale_factor * 2));

    torch::Tensor latents_unpacked =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
    latents_unpacked = latents_unpacked.reshape(
        {batch_size, channels / (2 * 2), height, width});
    return latents_unpacked;
  }

  // Helper function: split quotation marks (simplified version, shared with
  // T2I pipeline)
  static std::vector<std::pair<std::string, bool>> split_quotation(
      const std::string& prompt_text) {
    std::vector<std::pair<std::string, bool>> result;
    std::string current;
    bool in_quotes = false;
    char quote_char = '\0';

    for (size_t i = 0; i < prompt_text.length(); ++i) {
      char c = prompt_text[i];
      if ((c == '\'' || c == '\"') && !in_quotes) {
        if (!current.empty()) {
          result.emplace_back(current, false);
          current.clear();
        }
        in_quotes = true;
        quote_char = c;
        current += c;
      } else if (in_quotes && c == quote_char) {
        current += c;
        result.emplace_back(current, true);
        current.clear();
        in_quotes = false;
        quote_char = '\0';
      } else {
        current += c;
      }
    }
    if (!current.empty()) {
      result.emplace_back(current, in_quotes);
    }
    return result;
  }

  // Encode prompt using Qwen2_5_VL text encoder. When image is provided,
  // processes it with VL image processor and encodes vision + text (diffusers
  // _encode_prompt behavior).
  torch::Tensor encode_prompt_qwen(
      std::vector<std::string>& prompt,
      std::optional<torch::Tensor> image = std::nullopt,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";
    CHECK(!text_encoder_.is_empty()) << "Text encoder not loaded";
    CHECK(image.has_value())
        << "LongCat-Image-Edit encode_prompt requires an input image.";

    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    int32_t merge_size = text_encoder_args.mm_image_merge_size();
    int64_t merge_length = static_cast<int64_t>(merge_size) * merge_size;

    // 1. Process input image with VL image processor -> pixel_values,
    //    image_grid_thw. Pass float [0,255] to match diffusers Fast processor
    //    (resize is applied to float there; no uint8 /256 path).
    torch::Tensor img = image.value();
    if (img.dim() == 4) {
      img = img.index({0});
    }
    img = img.to(torch::kFloat32);
    float max_val = img.max().item<float>();
    if (max_val <= 1.1f) {
      img = img.clamp(0.0f, 1.0f) * 255.0f;
    } else {
      img = img.clamp(0.0f, 255.0f);
    }
    std::vector<torch::Tensor> pixel_values_list;
    std::vector<torch::Tensor> image_grid_thw_list;

    std::vector<MMDataItem> mm_items;

    CHECK(vl_image_processor_.process({img}, mm_items))
        << "VL image processor failed";

    torch::Tensor pixel_values =
        mm_items[0].get<torch::Tensor>("pixel_values").value();
    torch::Tensor image_grid_thw =
        mm_items[0].get<torch::Tensor>("image_grid_thw").value();

    int64_t num_image_tokens =
        image_grid_thw.prod().item<int64_t>() / merge_length;

    // 2. Build prefix string: replace <|image_pad|> with num_image_tokens
    //    copies (diffusers replacement logic).
    static const std::string kImagePad("<|image_pad|>");
    std::string prefix_str = prompt_template_encode_prefix_;
    size_t pos = 0;
    while ((pos = prefix_str.find(kImagePad, pos)) != std::string::npos) {
      std::string replacement;
      for (int64_t i = 0; i < num_image_tokens; ++i) {
        replacement += kImagePad;
      }
      prefix_str.replace(pos, kImagePad.size(), replacement);
      pos += replacement.size();
    }

    // 3. Tokenize prefix (now includes vision tokens) and suffix.
    std::vector<int32_t> prefix_tokens;
    CHECK(tokenizer_->encode(prefix_str, &prefix_tokens, false));
    std::vector<int32_t> suffix_tokens;
    CHECK(tokenizer_->encode(
        prompt_template_encode_suffix_, &suffix_tokens, false));
    int64_t suffix_len = suffix_tokens.size();

    int32_t vision_start_token_id = text_encoder_args.vision_start_token_id();
    int64_t prefix_len = 0;
    for (size_t i = 0; i < prefix_tokens.size(); ++i) {
      if (prefix_tokens[i] == vision_start_token_id) {
        prefix_len = static_cast<int64_t>(i);
        break;
      }
    }

    // 4. Tokenize prompts with split_quotation handling.
    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> batch_all_tokens;
    batch_all_tokens.reserve(batch_size);

    for (const auto& each_prompt : prompt) {
      std::vector<int32_t> all_tokens;
      auto parts = split_quotation(each_prompt);
      for (const auto& [clean_prompt_sub, _] : parts) {
        std::vector<int32_t> tokens;
        if (tokenizer_->encode(clean_prompt_sub, &tokens, false)) {
          all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
        }
      }
      if (static_cast<int64_t>(all_tokens.size()) > max_sequence_length) {
        LOG(WARNING) << "Input truncated from " << all_tokens.size() << " to "
                     << max_sequence_length << " tokens";
        all_tokens.resize(max_sequence_length);
      }
      batch_all_tokens.push_back(all_tokens);
    }

    int32_t pad_token_id = text_encoder_args.pad_token_id();
    if (pad_token_id == 0) {
      auto pad_id = tokenizer_->token_to_id("<|endoftext|>");
      if (pad_id.has_value()) {
        pad_token_id = pad_id.value();
      }
      if (pad_token_id == 0) {
        pad_token_id = 151643;
      }
    }

    for (auto& tokens : batch_all_tokens) {
      int64_t pad_len =
          max_sequence_length - static_cast<int64_t>(tokens.size());
      if (pad_len > 0) {
        tokens.insert(tokens.end(), pad_len, pad_token_id);
      }
    }

    int64_t prefix_full_len = prefix_tokens.size();
    int64_t total_seq_len = prefix_full_len + max_sequence_length + suffix_len;
    std::vector<int32_t> input_ids_flat;
    std::vector<int32_t> attention_mask_flat;
    input_ids_flat.reserve(batch_size * total_seq_len);
    attention_mask_flat.reserve(batch_size * total_seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
      input_ids_flat.insert(
          input_ids_flat.end(), prefix_tokens.begin(), prefix_tokens.end());
      attention_mask_flat.insert(
          attention_mask_flat.end(), static_cast<size_t>(prefix_full_len), 1);

      input_ids_flat.insert(input_ids_flat.end(),
                            batch_all_tokens[i].begin(),
                            batch_all_tokens[i].end());

      int64_t non_pad_count = 0;
      for (const auto& token : batch_all_tokens[i]) {
        if (token != pad_token_id) {
          non_pad_count++;
        } else {
          break;
        }
      }
      for (int64_t j = 0; j < max_sequence_length; ++j) {
        attention_mask_flat.push_back(j < non_pad_count ? 1 : 0);
      }

      input_ids_flat.insert(
          input_ids_flat.end(), suffix_tokens.begin(), suffix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), suffix_len, 1);
    }

    torch::Tensor input_ids =
        torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    torch::Tensor attention_mask =
        torch::tensor(attention_mask_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    torch::Tensor tokens_flat = input_ids.view({-1});
    torch::Tensor positions_2d = build_qwen2_5_vl_mrope_positions(
        input_ids, attention_mask, image_grid_thw);
    torch::Tensor positions_3d =
        positions_2d.view({3, batch_size, total_seq_len}).to(torch::kLong);

    std::vector<KVCache> kv_caches(text_encoder_args.n_layers());
    MMData mm_data;
    mm_data.set(MMType::IMAGE, std::move(mm_items));
    std::vector<MMData> mm_data_list(static_cast<size_t>(batch_size), mm_data);
    MMBatchData mm_batch(std::move(mm_data_list));
    ModelInputParams input_params = build_longcat_input_params(
        tokens_flat, positions_2d, attention_mask, mm_batch);
    auto model_output = text_encoder_->forward(
        tokens_flat, positions_2d, kv_caches, input_params);
    torch::Tensor hidden_states_flat = model_output.hidden_states;

    int64_t hidden_size = hidden_states_flat.size(-1);
    torch::Tensor hidden_states_last =
        hidden_states_flat.view({batch_size, total_seq_len, hidden_size});

    int64_t extracted_len = total_seq_len - prefix_len - suffix_len;
    torch::Tensor prompt_embeds =
        hidden_states_last.slice(1, prefix_len, total_seq_len - suffix_len);

    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds = prompt_embeds.view(
        {batch_size * num_images_per_prompt, extracted_len, -1});

    return prompt_embeds.to(options_);
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> image = std::nullopt,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }

    if (!prompt_embeds.has_value()) {
      prompt_embeds = encode_prompt_qwen(
          prompt_list, image, num_images_per_prompt, max_sequence_length);
    }

    torch::Tensor text_ids = prepare_text_ids(prompt_embeds.value().size(1));
    return {prompt_embeds.value(), text_ids};
  }

  torch::Tensor encode_vae_image(const torch::Tensor& image, int64_t seed) {
    torch::Tensor latents = vae_->encode(image, seed);
    latents = (latents - vae_shift_factor_) * vae_scaling_factor_;
    return latents;
  }

  // Prepare latents and image latents for LongCat-Image-Edit.
  std::tuple<torch::Tensor,  // noise latents (packed)
             torch::Tensor,  // image latents (packed)
             torch::Tensor,  // latents_ids
             torch::Tensor>  // image_latents_ids
  prepare_latents_with_image(
      const torch::Tensor& image,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      int64_t prompt_length,
      std::optional<torch::Tensor> latents_opt = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor image_latents;
    if (image.size(1) != num_channels_latents) {
      image_latents = encode_vae_image(image, seed);
    } else {
      image_latents = image;
    }

    if (batch_size > image_latents.size(0) &&
        batch_size % image_latents.size(0) == 0) {
      int64_t additional_image_per_prompt = batch_size / image_latents.size(0);
      image_latents =
          image_latents.repeat({additional_image_per_prompt, 1, 1, 1});
    } else if (batch_size > image_latents.size(0) &&
               batch_size % image_latents.size(0) != 0) {
      LOG(FATAL) << "Cannot duplicate `image` of batch size "
                 << image_latents.size(0) << " to " << batch_size
                 << " text prompts.";
    } else {
      image_latents = torch::cat({image_latents}, 0);
    }

    image_latents = pack_latents(image_latents,
                                 batch_size,
                                 num_channels_latents,
                                 adjusted_height,
                                 adjusted_width);

    torch::Tensor latents;
    if (latents_opt.has_value()) {
      latents = latents_opt.value().to(options_);
    } else {
      std::vector<int64_t> shape = {
          batch_size, num_channels_latents, adjusted_height, adjusted_width};
      torch::TensorOptions latents_options = options_.dtype(torch::kFloat32);
      torch::Tensor latents_tensor =
          xllm::dit::randn_tensor(shape, seed, latents_options).to(options_);
      latents = pack_latents(latents_tensor,
                             batch_size,
                             num_channels_latents,
                             adjusted_height,
                             adjusted_width);
    }

    // Position IDs: start at (prompt_length, prompt_length) for both
    // modalities, following diffusers LongCat-Image-Edit.
    int64_t start = prompt_length;
    torch::Tensor latents_ids = prepare_image_pos_ids(
        /*modality_id=*/1,
        /*start_row=*/start,
        /*start_col=*/start,
        /*height=*/adjusted_height / 2,
        /*width=*/adjusted_width / 2);
    torch::Tensor image_latents_ids = prepare_image_pos_ids(
        /*modality_id=*/2,
        /*start_row=*/start,
        /*start_col=*/start,
        /*height=*/adjusted_height / 2,
        /*width=*/adjusted_width / 2);

    return {latents, image_latents, latents_ids, image_latents_ids};
  }

  // Build ModelInputParams for LongCat-Image text encoding (reuse from
  // T2I pipeline). When mm_data_opt is provided (e.g. pixel_values +
  // image_grid_thw), the text encoder will use vision embeddings.
  torch::Tensor build_qwen2_5_vl_mrope_positions(
      const torch::Tensor& input_ids,         // [B, S]
      const torch::Tensor& attention_mask,    // [B, S]
      const torch::Tensor& image_grid_thw) {  // [num_images, 3]
    CHECK(input_ids.dim() == 2) << "input_ids must be [B, S]";
    CHECK(attention_mask.dim() == 2) << "attention_mask must be [B, S]";

    auto long_opts =
        torch::TensorOptions().dtype(torch::kLong).device(input_ids.device());
    torch::Tensor position_ids =
        torch::ones({3, input_ids.size(0), input_ids.size(1)}, long_opts);

    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    int64_t spatial_merge_size = text_encoder_args.mm_image_merge_size();
    int64_t image_token_id = text_encoder_args.image_token_id();
    int64_t vision_start_token_id = text_encoder_args.vision_start_token_id();

    int64_t global_image_index = 0;
    int64_t num_image_grids =
        image_grid_thw.defined() ? image_grid_thw.size(0) : 0;

    for (int64_t b = 0; b < input_ids.size(0); ++b) {
      auto mask_b = attention_mask[b].to(torch::kBool);
      auto ids_b = input_ids[b].index({mask_b}).to(torch::kLong).cpu();

      std::vector<int64_t> tokens(ids_b.data_ptr<int64_t>(),
                                  ids_b.data_ptr<int64_t>() + ids_b.numel());

      int64_t image_nums = 0;
      for (int64_t i = 0; i + 1 < static_cast<int64_t>(tokens.size()); ++i) {
        if (tokens[i] == vision_start_token_id &&
            tokens[i + 1] == image_token_id) {
          image_nums++;
        }
      }

      std::vector<torch::Tensor> llm_pos_ids_list;
      llm_pos_ids_list.reserve(static_cast<size_t>(image_nums) * 2 + 1);

      int64_t st = 0;
      int64_t remain_images = image_nums;
      for (int64_t n = 0; n < image_nums; ++n) {
        int64_t ed = static_cast<int64_t>(tokens.size()) + 1;
        if (remain_images > 0) {
          for (int64_t p = st; p < static_cast<int64_t>(tokens.size()); ++p) {
            if (tokens[p] == image_token_id) {
              ed = p;
              break;
            }
          }
        }

        CHECK(num_image_grids > 0)
            << "image_grid_thw is required when image tokens exist";
        CHECK(global_image_index < num_image_grids)
            << "image_grid_thw count is smaller than required image tokens";
        auto grid_idx = global_image_index;
        int64_t t = image_grid_thw[grid_idx][0].item<int64_t>();
        int64_t h = image_grid_thw[grid_idx][1].item<int64_t>();
        int64_t w = image_grid_thw[grid_idx][2].item<int64_t>();

        int64_t llm_grid_t = t;
        int64_t llm_grid_h = h / spatial_merge_size;
        int64_t llm_grid_w = w / spatial_merge_size;
        int64_t text_len = ed - st;

        int64_t st_idx = 0;
        if (!llm_pos_ids_list.empty()) {
          st_idx = llm_pos_ids_list.back().max().item<int64_t>() + 1;
        }

        auto text_pos =
            torch::arange(text_len, long_opts).view({1, -1}).expand({3, -1}) +
            st_idx;
        llm_pos_ids_list.push_back(text_pos);

        auto t_index = torch::arange(llm_grid_t, long_opts)
                           .view({-1, 1})
                           .expand({-1, llm_grid_h * llm_grid_w})
                           .flatten();
        auto h_index = torch::arange(llm_grid_h, long_opts)
                           .view({1, -1, 1})
                           .expand({llm_grid_t, -1, llm_grid_w})
                           .flatten();
        auto w_index = torch::arange(llm_grid_w, long_opts)
                           .view({1, 1, -1})
                           .expand({llm_grid_t, llm_grid_h, -1})
                           .flatten();
        auto vision_pos =
            torch::stack({t_index, h_index, w_index}, 0) + text_len + st_idx;
        llm_pos_ids_list.push_back(vision_pos);

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
        global_image_index++;
        remain_images--;
      }

      if (st < static_cast<int64_t>(tokens.size())) {
        int64_t st_idx = 0;
        if (!llm_pos_ids_list.empty()) {
          st_idx = llm_pos_ids_list.back().max().item<int64_t>() + 1;
        }
        int64_t text_len = static_cast<int64_t>(tokens.size()) - st;
        auto text_pos =
            torch::arange(text_len, long_opts).view({1, -1}).expand({3, -1}) +
            st_idx;
        llm_pos_ids_list.push_back(text_pos);
      }

      if (!llm_pos_ids_list.empty()) {
        auto llm_positions = torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
        position_ids.index_put_({torch::indexing::Slice(), b, mask_b},
                                llm_positions.to(position_ids.device()));
      }
    }

    return position_ids.reshape({3, -1}).contiguous();
  }

  ModelInputParams build_longcat_input_params(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      const torch::Tensor& attention_mask,
      std::optional<MMBatchData> mm_data_opt = std::nullopt) {
    ModelInputParams params;

    int64_t actual_seq_len;
    if (positions.dim() == 2) {
      actual_seq_len = positions.size(1);
    } else if (positions.dim() == 1) {
      actual_seq_len = positions.size(0);
    } else {
      actual_seq_len = positions.numel();
    }

    if (actual_seq_len > 0) {
      params.meta.q_max_seq_len = actual_seq_len;
      params.meta.kv_max_seq_len = actual_seq_len;
      auto cu_seqlens =
          torch::tensor({0, static_cast<int>(actual_seq_len)}, torch::kInt)
              .to(tokens.device());
      params.attention.device.q_seq_lens = cu_seqlens;
      params.attention.device.kv_seq_lens = cu_seqlens;
      params.meta.batch_forward_type = BatchForwardType::PREFILL;
    }

    // Set mm_data before get_input_embeddings
    if (mm_data_opt.has_value()) {
      params.multimodal.mm_data =
          MMBatchData::to(mm_data_opt.value(), options_.device());
    }

    // Build attention metadata before get_input_embeddings
    // (may be needed by multimodal processing)
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, context_.get_model_args("text_encoder").enable_mla()));
    params.attn_metadata->is_causal = true;
    params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, params);
    if (attention_mask.defined() && attention_mask.size(0) > 0) {
      params.graph.attn_mask = attention_mask.view({-1}).to(torch::kFloat32);
    }

    return params;
  }

  std::vector<torch::Tensor> forward_(
      std::optional<torch::Tensor> image = std::nullopt,
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      int64_t num_inference_steps = 50,
      float guidance_scale = 4.5f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512,
      bool enable_cfg_renorm = true,
      float cfg_renorm_min = 0.0f) {
    torch::NoGradGuard no_grad;

    CHECK(image.has_value())
        << "LongCat-Image-Edit forward_ called without image.";

    // Determine image dimensions & resize to ~1M pixels, preserving aspect.
    torch::Tensor input_image = image.value();
    int64_t in_height = input_image.size(-2);
    int64_t in_width = input_image.size(-1);
    float ratio = static_cast<float>(in_width) /
                  static_cast<float>(std::max<int64_t>(in_height, 1));

    auto [calculated_width, calculated_height] =
        calculate_dimensions_edit(1024 * 1024, ratio);

    // Preprocess image for VAE.
    torch::Tensor processed_image = vae_image_processor_->preprocess(
        input_image, calculated_height, calculated_width);

    // Create prompt_image: half-resolution for text encoder (matches diffusers
    // LongCatImageEditPipeline: image = resize(., calc_h, calc_w); prompt_image
    // = resize(image, calc_h//2, calc_w//2)). Use VL processor's PIL-compatible
    // resize instead of torch interpolate to align pixel_values with diffusers.
    torch::Tensor prompt_image = input_image;
    if (prompt_image.dim() == 3) {
      prompt_image = prompt_image.unsqueeze(0);
    }
    if (prompt_image.dtype() != torch::kFloat32) {
      prompt_image = prompt_image.to(torch::kFloat32);
    }
    // Resize expects [0,255] range
    if (prompt_image.max().item<float>() <= 1.1f) {
      prompt_image = prompt_image * 255.0f;
    }
    prompt_image = prompt_image.squeeze(0);  // [C,H,W]
    // Use BICUBIC resize to match diffusers VaeImageProcessor.resize behavior
    // for PIL.Image.Image (uses PIL LANCZOS). BICUBIC is the closest match
    // to LANCZOS in PyTorch's interpolate function.
    // Step 1: resize to full target resolution (matches diffusers:
    //   image = self.image_processor.resize(image, calc_h, calc_w))
    prompt_image =
        transforms::resize(prompt_image,
                           {calculated_height, calculated_width},
                           /*resample=*/3,  // BICUBIC (approximate LANCZOS)
                           /*antialias=*/true);
    // Step 2: resize to half resolution for VL text encoder (matches diffusers:
    //   prompt_image = self.image_processor.resize(image, calc_h//2,
    //   calc_w//2))
    prompt_image =
        transforms::resize(prompt_image,
                           {calculated_height / 2, calculated_width / 2},
                           /*resample=*/3,  // BICUBIC (approximate LANCZOS)
                           /*antialias=*/true);
    prompt_image = prompt_image.unsqueeze(0);  // [1,C,H,W] for encode_prompt

    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0) / num_images_per_prompt;
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;

    bool do_classifier_free_guidance = (guidance_scale > 1.0f);

    if (do_classifier_free_guidance && !negative_prompt.has_value() &&
        !negative_prompt_embeds.has_value()) {
      negative_prompt = std::vector<std::string>{""};
    }

    // Encode prompt (use prompt_image = half-res for VL text encoder, matches
    // diffusers).
    auto [encoded_prompt_embeds, text_ids] =
        encode_prompt(prompt,
                      prompt_embeds,
                      prompt_image,
                      num_images_per_prompt,
                      max_sequence_length);

    torch::Tensor negative_encoded_embeds;
    torch::Tensor negative_text_ids;
    if (do_classifier_free_guidance) {
      auto [neg_emb, neg_ids] = encode_prompt(negative_prompt,
                                              negative_prompt_embeds,
                                              prompt_image,
                                              num_images_per_prompt,
                                              max_sequence_length);
      negative_encoded_embeds = neg_emb;
      negative_text_ids = neg_ids;
    }

    // Prepare latents (noise) and image latents.
    int64_t num_channels_latents = 16;  // LongCat-Image uses 16 channels
    auto [prepared_latents, image_latents, latents_ids, image_latents_ids] =
        prepare_latents_with_image(processed_image,
                                   total_batch_size,
                                   num_channels_latents,
                                   calculated_height,
                                   calculated_width,
                                   seed.value_or(0),
                                   encoded_prompt_embeds.size(1),
                                   latents);

    // Prepare timesteps: sigmas = linspace(1.0, 1.0 / n, n)
    std::vector<float> sigmas;
    double start = 1.0;
    double end = 1.0 / static_cast<double>(num_inference_steps);
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      double v = start + (end - start) * static_cast<double>(i) /
                             (num_inference_steps - 1);
      sigmas.push_back(static_cast<float>(v));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_longcat_shift(image_seq_len,
                                       scheduler_->base_image_seq_len(),
                                       scheduler_->max_image_seq_len(),
                                       scheduler_->base_shift(),
                                       scheduler_->max_shift());

    auto [timesteps, _] = retrieve_longcat_timesteps(
        scheduler_, num_inference_steps, options_.device(), sigmas, mu);

    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    // Build combined image position IDs (latents + image_latents).
    torch::Tensor all_image_ids =
        torch::cat({latents_ids, image_latents_ids}, 0);

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  all_image_ids,
                                  calculated_height / (vae_scale_factor_ * 2),
                                  calculated_width / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb = torch::stack({rot_emb1, rot_emb2}, 0);

    torch::Tensor negative_image_rotary_emb;
    if (do_classifier_free_guidance) {
      auto [neg_rot1, neg_rot2] =
          pos_embed_->forward_cache(negative_text_ids,
                                    all_image_ids,
                                    calculated_height / (vae_scale_factor_ * 2),
                                    calculated_width / (vae_scale_factor_ * 2));
      negative_image_rotary_emb = torch::stack({neg_rot1, neg_rot2}, 0);
    }

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>());
      timestep = timestep.to(prepared_latents.dtype());
      timestep.div_(kLongCatTimestepScale);

      // Concatenate noise latents and encoded image latents along sequence
      // dimension, matching diffusers.
      torch::Tensor latent_model_input =
          torch::cat({prepared_latents, image_latents}, 1);

      // Match diffusers LongCatImageEditPipeline behavior:
      // only the first `image_seq_len` tokens (corresponding to noisy latents)
      // are used for the scheduler step; the image latents part is ignored.
      torch::Tensor noise_pred_text_full =
          transformer_->forward(latent_model_input,
                                encoded_prompt_embeds,
                                timestep,
                                image_rotary_emb);
      torch::Tensor noise_pred_text = noise_pred_text_full.narrow(
          /*dim=*/1, /*start=*/0, image_seq_len);

      torch::Tensor noise_pred = noise_pred_text;

      if (do_classifier_free_guidance) {
        torch::Tensor negative_noise_pred_full = transformer_->forward(
            latent_model_input,
            negative_encoded_embeds,
            timestep,
            negative_image_rotary_emb,
            i + 10000);  // step_idx for cache separation (future use)
        torch::Tensor negative_noise_pred = negative_noise_pred_full.narrow(
            /*dim=*/1, /*start=*/0, image_seq_len);

        noise_pred = negative_noise_pred +
                     guidance_scale * (noise_pred_text - negative_noise_pred);

        if (enable_cfg_renorm) {
          torch::Tensor cond_norm = torch::norm(noise_pred_text, 2, -1, true);
          torch::Tensor noise_norm = torch::norm(noise_pred, 2, -1, true);
          torch::Tensor scale = (cond_norm / (noise_norm + 1e-8f))
                                    .clamp_min(cfg_renorm_min)
                                    .clamp_max(1.0f);
          noise_pred = noise_pred * scale;
        }
      }

      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    // Decode latents to images.
    torch::Tensor image_out;
    torch::Tensor unpacked_latents = unpack_latents(prepared_latents,
                                                    calculated_height,
                                                    calculated_width,
                                                    vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    unpacked_latents = unpacked_latents.to(options_.dtype());

    try {
      image_out = vae_->decode(unpacked_latents);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LongCatImageEdit] VAE decode failed: " << e.what();
      return std::vector<torch::Tensor>{{torch::zeros(
          {1, 3, calculated_height, calculated_width}, options_)}};
    } catch (...) {
      LOG(ERROR) << "[LongCatImageEdit] VAE decode failed with unknown "
                    "exception";
      return std::vector<torch::Tensor>{{torch::zeros(
          {1, 3, calculated_height, calculated_width}, options_)}};
    }

    image_out = vae_image_processor_->postprocess(image_out);
    image_out = image_out.cpu().to(torch::kFloat32).contiguous();

    return std::vector<torch::Tensor>{{image_out}};
  }

 private:
  // Model context (saved for use in load_model)
  DiTModelContext context_;

  // Member variables
  torch::TensorOptions options_;
  int64_t vae_scale_factor_;
  float vae_shift_factor_;
  float vae_scaling_factor_;

  // ProcessGroup for VLM (single-device)
  std::unique_ptr<ProcessGroup> vlm_tp_group_;

  // Model components
  VAEImageProcessor vae_image_processor_{nullptr};
  Qwen2VLImageProcessor vl_image_processor_;
  VAE vae_{nullptr};
  LongCatImagePosEmbed pos_embed_{nullptr};
  LongCatImageTransformer2DModel transformer_{nullptr};
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  Qwen2_5_VLForConditionalGeneration text_encoder_{nullptr};
  std::string prompt_template_encode_prefix_;
  std::string prompt_template_encode_suffix_;
};
TORCH_MODULE(LongCatImageEditPipeline);

// Register LongCat-Image-Edit as DiT model.
namespace {
const bool longcat_image_edit_dit_registered = []() {
  ModelRegistry::register_dit_model_factory(
      "LongCat-Image-Edit", [](const DiTModelContext& context) {
        LongCatImageEditPipeline model(context);
        model->eval();
        return std::make_unique<DiTModelImpl<LongCatImageEditPipeline>>(
            std::move(model), context.get_tensor_options());
      });
  return true;
}();
}  // namespace

}  // namespace xllm

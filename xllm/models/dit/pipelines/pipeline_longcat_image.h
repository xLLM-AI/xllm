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

#include <torch/torch.h>

#include <memory>
#include <string>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/tokenizer/tokenizer.h"
#if defined(USE_CUDA)
#include "core/layers/cuda/flashinfer_workspace.h"
#endif
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_longcat_image.h"
#include "models/model_registry.h"
#include "models/vlm/qwen2_5_vl.h"

namespace xllm {

// Forward declarations
class LongCatImagePipelineImpl;

// Utility constants
constexpr int64_t kLongCatRopeScaleBase = 10000;
// FlowMatch/Euler scheduler timestep scaling (model expects [0, 1000])
constexpr float kLongCatTimestepScale = 1000.0f;

// Prompt template constants for LongCat-Image
// Ref:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/longcat_image/pipeline_longcat_image.py
constexpr const char* PROMPT_TEMPLATE_ENCODE_PREFIX =
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive "
    "text prompt based on an image content, suitable for input to a "
    "text-to-image model.<|im_end|>\n<|im_start|>user\n";
constexpr const char* PROMPT_TEMPLATE_ENCODE_SUFFIX =
    "<|im_end|>\n<|im_start|>assistant\n";

inline float calculate_longcat_shift(int64_t image_seq_len,
                                     int64_t base_seq_len = 256,
                                     int64_t max_seq_len = 4096,
                                     float base_shift = 0.5f,
                                     float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

inline std::pair<torch::Tensor, int64_t> retrieve_longcat_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(num_inference_steps, device);
    scheduler_timesteps = scheduler->timesteps();
  }
  return {scheduler_timesteps, steps};
}

class LongCatImagePipelineImpl : public torch::nn::Module {
 public:
  LongCatImagePipelineImpl(const DiTModelContext& context) : context_(context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();

#if defined(USE_CUDA)
    // Initialize FlashinferWorkspace for CUDA attention operations.
    layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
        options_.device());
#endif

    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    LOG(INFO) << "Initializing LongCat-Image pipeline...";
    vae_image_processor_ =
        VAEImageProcessor(ModelContext(context.get_parallel_args(),
                                       context.get_model_args("vae"),
                                       context.get_quant_args("vae"),
                                       context.get_tensor_options()),
                          /*do_resize=*/true,
                          /*do_normalize=*/true,
                          /*do_center_crop=*/false,
                          /*do_convert_rgb=*/false,
                          /*do_resize_latents=*/false,
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
                                           29500,
                                           false,
                                           "127.0.0.1",
                                           "vlm_tp_group",
                                           options_.device());
      vlm_parallel_args.tp_group_ = vlm_tp_group_.get();
    }

    // LongCat-Image uses Qwen2_5_VL as text encoder, not CLIP+T5
    LOG(INFO) << "LongCat-Image uses Qwen2_5_VL as text encoder";
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
    prompt_template_encode_prefix_ = PROMPT_TEMPLATE_ENCODE_PREFIX;
    prompt_template_encode_suffix_ = PROMPT_TEMPLATE_ENCODE_SUFFIX;

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

    std::vector<torch::Tensor> output = forward_(
        prompts,                                  // prompt
        prompts_2,                                // prompt_2
        negative_prompts,                         // negative_prompt
        negative_prompts_2,                       // negative_prompt_2
        generation_params.true_cfg_scale,         // cfg scale
        generation_params.height,                 // height
        generation_params.width,                  // width
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
    LOG(INFO) << "LongCat-Image pipeline loading model from "
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

  // Prepare text position IDs for text tokens
  torch::Tensor prepare_text_ids(int64_t num_tokens,
                                 int64_t start_height = 0,
                                 int64_t start_width = 0) {
    // Create position IDs with correct integer values first
    // Use int64 to ensure precision, then convert to target dtype
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor text_ids_int = torch::zeros({num_tokens, 3}, int_options);

    // modality_id = 0 for text
    text_ids_int.select(1, 0).fill_(0);

    // position indices: [0, 1, 2, ..., num_tokens-1]
    torch::Tensor token_range = torch::arange(num_tokens, int_options);
    text_ids_int.select(1, 1) = token_range + start_height;
    text_ids_int.select(1, 2) = token_range + start_width;

    // CRITICAL FIX: Convert to float32 instead of options_.dtype() to avoid
    // precision loss bfloat16 cannot accurately represent 511, causing it to
    // round to 512 Since LongCatImagePosEmbed::forward() converts position IDs
    // to float32 anyway, we should use float32 directly to preserve precision
    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor text_ids = text_ids_int.to(float32_options);

    return text_ids;
  }

  // Prepare image position IDs for latent image tokens
  // Match diffusers: Position IDs should start from 512 (hardcoded
  // tokenizer_max_length) to avoid overlapping with text position IDs
  // Note: Python diffusers uses hardcoded 512, not max_position_embeddings()
  torch::Tensor prepare_latent_image_ids(int64_t batch_size,
                                         int64_t height,
                                         int64_t width) {
    constexpr int64_t TOKENIZER_MAX_LENGTH = 512;

    // Create position IDs with correct integer values first (int64), then
    // convert to target dtype This avoids precision loss when adding
    // start_offset
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor latent_image_ids_int =
        torch::zeros({height, width, 3}, int_options);

    // modality_id = 1 for image
    latent_image_ids_int.select(2, 0).fill_(1);

    // Create ranges in int64, add offset, then convert
    auto height_range = torch::arange(height, int_options).unsqueeze(1);
    auto width_range = torch::arange(width, int_options).unsqueeze(0);
    latent_image_ids_int.select(2, 1) += height_range + TOKENIZER_MAX_LENGTH;
    latent_image_ids_int.select(2, 2) += width_range + TOKENIZER_MAX_LENGTH;

    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor latent_image_ids = latent_image_ids_int.to(float32_options);
    latent_image_ids = latent_image_ids.view({height * width, 3});

    return latent_image_ids;
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

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, adjusted_height, adjusted_width};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2);
      return {latents.value(), latent_image_ids};
    }

    torch::TensorOptions latents_options = options_.dtype(torch::kFloat32);
    torch::Tensor latents_tensor =
        xllm::dit::randn_tensor(shape, seed, latents_options).to(options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor,
                                                batch_size,
                                                num_channels_latents,
                                                adjusted_height,
                                                adjusted_width);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2);
    return {packed_latents, latent_image_ids};
  }

  // Helper function: split quotation marks (simplified version)
  // For full regex-based implementation, see diffusers implementation
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

  // Encode prompt using Qwen2_5_VL text encoder
  torch::Tensor encode_prompt_qwen(std::vector<std::string>& prompt,
                                   int64_t num_images_per_prompt = 1,
                                   int64_t max_sequence_length = 512) {
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";
    CHECK(!text_encoder_.is_empty()) << "Text encoder not loaded";

    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> batch_all_tokens;
    batch_all_tokens.reserve(batch_size);

    // Step 1: Tokenize prompts with split_quotation handling
    for (const auto& each_prompt : prompt) {
      std::vector<int32_t> all_tokens;
      auto parts = split_quotation(each_prompt);

      for (const auto& [clean_prompt_sub, _] : parts) {
        std::vector<int32_t> tokens;
        if (tokenizer_->encode(clean_prompt_sub, &tokens, false)) {
          all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
        }
      }

      // Truncate if too long
      if (static_cast<int64_t>(all_tokens.size()) > max_sequence_length) {
        LOG(WARNING) << "Input truncated from " << all_tokens.size() << " to "
                     << max_sequence_length << " tokens";
        all_tokens.resize(max_sequence_length);
      }

      batch_all_tokens.push_back(std::move(all_tokens));
    }

    // Step 2: Pad tokens to max_sequence_length
    // Use pad_token_id from text_encoder model config (151643 for Qwen2) to
    // match diffusers. Fallback: tokenizer "<|endoftext|>", then 151643.
    int32_t pad_token_id =
        context_.get_model_args("text_encoder").pad_token_id();
    if (pad_token_id == 0) {
      auto pad_id = tokenizer_->token_to_id("<|endoftext|>");
      if (pad_id.has_value()) {
        pad_token_id = pad_id.value();
      }
      if (pad_token_id == 0) {
        pad_token_id = 151643;  // Qwen2 default, matches diffusers
      }
    }

    // Pad each sequence
    for (auto& tokens : batch_all_tokens) {
      int64_t pad_len =
          max_sequence_length - static_cast<int64_t>(tokens.size());
      if (pad_len > 0) {
        tokens.insert(tokens.end(), pad_len, pad_token_id);
      }
    }

    // Step 3: Tokenize prefix and suffix templates
    std::vector<int32_t> prefix_tokens;
    CHECK(tokenizer_->encode(
        prompt_template_encode_prefix_, &prefix_tokens, false));
    std::vector<int32_t> suffix_tokens;
    CHECK(tokenizer_->encode(
        prompt_template_encode_suffix_, &suffix_tokens, false));
    int64_t prefix_len = prefix_tokens.size();
    int64_t suffix_len = suffix_tokens.size();

    // Step 4: Concatenate prefix + tokens + suffix
    int64_t total_seq_len = prefix_len + max_sequence_length + suffix_len;
    std::vector<int32_t> input_ids_flat;
    std::vector<int32_t> attention_mask_flat;
    input_ids_flat.reserve(batch_size * total_seq_len);
    attention_mask_flat.reserve(batch_size * total_seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
      // Add prefix
      input_ids_flat.insert(
          input_ids_flat.end(), prefix_tokens.begin(), prefix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), prefix_len, 1);
      // Add tokens
      input_ids_flat.insert(input_ids_flat.end(),
                            batch_all_tokens[i].begin(),
                            batch_all_tokens[i].end());
      // Count non-padding tokens for attention mask
      int64_t non_pad_count = 0;
      for (const auto& token : batch_all_tokens[i]) {
        if (token != pad_token_id) {
          non_pad_count++;
        } else {
          break;  // Padding starts here
        }
      }
      // Mark real tokens as 1, padding as 0
      for (int64_t j = 0; j < max_sequence_length; ++j) {
        attention_mask_flat.push_back(j < non_pad_count ? 1 : 0);
      }
      // Add suffix
      input_ids_flat.insert(
          input_ids_flat.end(), suffix_tokens.begin(), suffix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), suffix_len, 1);
    }

    // Create input_ids tensor [batch_size, total_seq_len]
    torch::Tensor input_ids =
        torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    // Create attention_mask tensor [batch_size, total_seq_len]
    torch::Tensor attention_mask =
        torch::tensor(attention_mask_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    // Step 5: Get hidden states from text encoder
    // Following diffusers implementation exactly
    // The key is that attention_mask is used to create an additive mask for
    // the attention mechanism This is applied INSIDE the transformer layers,
    // not post-hoc

    // Flatten input_ids for forward pass: [batch_size, total_seq_len] ->
    // [batch_size * total_seq_len]
    torch::Tensor tokens_flat = input_ids.view({-1});

    // Create MROPE-style positions tensor: [3, seq_len]
    // HuggingFace Qwen2_5_VL uses MROPE (Multimodal RoPE) even for text-only:
    //   position_ids = attention_mask.long().cumsum(-1) - 1
    //   position_ids.masked_fill_(attention_mask == 0, 1)
    //   position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    // For text-only, all 3 MROPE dimensions (T, H, W) use the same positions
    auto mask_flat = attention_mask.view({-1});  // [batch * seq_len]
    auto positions_1d = mask_flat.to(torch::kInt64).cumsum(-1) - 1;
    positions_1d = positions_1d.masked_fill(mask_flat == 0, 1);

    // Expand to [3, seq_len] for MROPE - this triggers xllm's apply_mrope()
    // in llm_model_base.h when positions.dim() == 2
    torch::Tensor positions_2d =
        positions_1d.unsqueeze(0).expand({3, -1}).contiguous();

    // Prepare kv_caches for forward pass
    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    std::vector<KVCache> kv_caches(text_encoder_args.n_layers());

    ModelInputParams input_params =
        build_longcat_input_params(tokens_flat, positions_2d, attention_mask);
    auto model_output = text_encoder_->forward(
        tokens_flat, positions_2d, kv_caches, input_params);
    torch::Tensor hidden_states_flat = model_output.hidden_states;

    int64_t hidden_size = hidden_states_flat.size(-1);
    // Reshape: [batch_size * total_seq_len, hidden_size] -> [batch_size,
    // total_seq_len, hidden_size]
    torch::Tensor hidden_states_last =
        hidden_states_flat.view({batch_size, total_seq_len, hidden_size});

    // Step 6: Remove prefix and suffix tokens
    // This matches diffusers: prompt_embeds =
    // text_output.hidden_states[-1].detach()
    //                          prompt_embeds = prompt_embeds[:,
    //                          prefix_len:-suffix_len, :]
    // Result shape: [batch_size, max_sequence_length, hidden_size]
    torch::Tensor prompt_embeds =
        hidden_states_last.slice(1, prefix_len, total_seq_len - suffix_len);

    // Step 7: Repeat for num_images_per_prompt (matches diffusers
    // encode_prompt) duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    // Reshape: [batch_size, max_sequence_length * num_images_per_prompt,
    // hidden_size]
    //       -> [batch_size * num_images_per_prompt, max_sequence_length,
    //       hidden_size]
    prompt_embeds = prompt_embeds.view(
        {batch_size * num_images_per_prompt, max_sequence_length, -1});

    return prompt_embeds.to(options_);
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<torch::Tensor> prompt_embeds,
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
          prompt_list, num_images_per_prompt, max_sequence_length);
    }

    torch::Tensor text_ids = prepare_text_ids(prompt_embeds.value().size(1));
    return {prompt_embeds.value(), text_ids};
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
  VAE vae_{nullptr};
  LongCatImagePosEmbed pos_embed_{nullptr};
  LongCatImageTransformer2DModel transformer_{nullptr};
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  Qwen2_5_VLForConditionalGeneration text_encoder_{nullptr};
  std::string prompt_template_encode_prefix_;
  std::string prompt_template_encode_suffix_;

  // Build ModelInputParams for LongCat-Image text encoding
  ModelInputParams build_longcat_input_params(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      const torch::Tensor& attention_mask) {
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

    // Let Qwen2_5_VL build multimodal-aware embeddings from tokens and params.
    params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, params);

    if (attention_mask.defined() && attention_mask.size(0) > 0) {
      params.graph.attn_mask = attention_mask.view({-1}).to(torch::kFloat32);
    }

    return params;
  }

  std::vector<torch::Tensor> forward_(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      int64_t height = 768,
      int64_t width = 1344,
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
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0) / num_images_per_prompt;
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;

    // Enable CFG when guidance_scale > 1.0, regardless of negative_prompt
    // availability If no negative_prompt is provided, we'll use an empty
    // string
    bool do_classifier_free_guidance = (guidance_scale > 1.0f);

    // If CFG is enabled but no negative_prompt provided, use empty string
    if (do_classifier_free_guidance && !negative_prompt.has_value() &&
        !negative_prompt_embeds.has_value()) {
      negative_prompt = std::vector<std::string>{""};
    }

    // Encode prompt
    auto [encoded_prompt_embeds, text_ids] = encode_prompt(
        prompt, prompt_embeds, num_images_per_prompt, max_sequence_length);

    // Encode negative prompt
    torch::Tensor negative_encoded_embeds;
    torch::Tensor negative_text_ids;
    if (do_classifier_free_guidance) {
      auto [neg_emb, neg_ids] = encode_prompt(negative_prompt,
                                              negative_prompt_embeds,
                                              num_images_per_prompt,
                                              max_sequence_length);
      negative_encoded_embeds = neg_emb;
      negative_text_ids = neg_ids;
    }

    // Prepare latent variables
    int64_t num_channels_latents = 16;  // LongCat-Image uses 16 channels
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.value_or(0),
                        latents);

    // Prepare timesteps. Match diffusers: sigmas = np.linspace(1.0, 1.0/n,
    // n).
    std::vector<float> new_sigmas;
    double start = 1.0;
    double end = 1.0 / static_cast<double>(num_inference_steps);
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      double v = start + (end - start) * static_cast<double>(i) /
                             (num_inference_steps - 1);
      new_sigmas.push_back(static_cast<float>(v));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_longcat_shift(image_seq_len,
                                       scheduler_->base_image_seq_len(),
                                       scheduler_->max_image_seq_len(),
                                       scheduler_->base_shift(),
                                       scheduler_->max_shift());
    auto [timesteps, _] = retrieve_longcat_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);

    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb = torch::stack({rot_emb1, rot_emb2}, 0);

    torch::Tensor negative_image_rotary_emb;
    if (do_classifier_free_guidance) {
      auto [neg_rot1, neg_rot2] =
          pos_embed_->forward_cache(negative_text_ids,
                                    latent_image_ids,
                                    height / (vae_scale_factor_ * 2),
                                    width / (vae_scale_factor_ * 2));
      negative_image_rotary_emb = torch::stack({neg_rot1, neg_rot2}, 0);
    }

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(kLongCatTimestepScale);

      torch::Tensor noise_pred = transformer_->forward(
          prepared_latents, encoded_prompt_embeds, timestep, image_rotary_emb);

      if (do_classifier_free_guidance) {
        // Forward negative prompt. Use negative_image_rotary_emb (from
        // negative_text_ids) to match diffusers; positive uses
        // image_rotary_emb.
        torch::Tensor negative_noise_pred = transformer_->forward(
            prepared_latents,
            negative_encoded_embeds,
            timestep,
            negative_image_rotary_emb,
            i + 10000);  // Use different step_idx to avoid cache collision
        torch::Tensor noise_pred_text = noise_pred;
        // Classifier-free guidance
        noise_pred = negative_noise_pred +
                     guidance_scale * (noise_pred_text - negative_noise_pred);
        // CFG renorm
        if (enable_cfg_renorm) {
          torch::Tensor cond_norm = torch::norm(noise_pred_text, 2, -1, true);
          torch::Tensor noise_norm = torch::norm(noise_pred, 2, -1, true);
          torch::Tensor scale = (cond_norm / (noise_norm + 1e-8f))
                                    .clamp_min(cfg_renorm_min)
                                    .clamp_max(1.0f);
          noise_pred = noise_pred * scale;
        }
      }

      // Scheduler step
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    // Decode latents to images
    torch::Tensor image;
    torch::Tensor unpacked_latents =
        unpack_latents(prepared_latents, height, width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    // Ensure VAE input matches model dtype (VAE is loaded in bfloat16)
    unpacked_latents = unpacked_latents.to(options_.dtype());

    if (!vae_) {
      LOG(ERROR) << "[LongCatImage] VAE is null!";
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    }

    try {
      image = vae_->decode(unpacked_latents);
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LongCatImage] VAE decode failed with exception: "
                 << e.what();
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    } catch (...) {
      LOG(ERROR) << "[LongCatImage] VAE decode failed with unknown exception";
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    }

    image = vae_image_processor_->postprocess(image);

    // Ensure image is on CPU and in float32 format for encoding
    image = image.cpu().to(torch::kFloat32).contiguous();

    return std::vector<torch::Tensor>{{image}};
  }
};
TORCH_MODULE(LongCatImagePipeline);

// Register LongCat-Image as DiT model.
// We use manual registration instead of REGISTER_DIT_MODEL(ModelType, Class)
// because the macro concatenates ModelType into a C++ variable name (e.g.
// ModelType##_registered). "LongCat-Image" contains a hyphen, which is not a
// valid character in C++ identifiers, so the macro would produce invalid
// code.
namespace {
const bool longcat_image_dit_registered = []() {
  ModelRegistry::register_dit_model_factory(
      "LongCat-Image", [](const DiTModelContext& context) {
        LongCatImagePipeline model(context);
        model->eval();
        return std::make_unique<DiTModelImpl<LongCatImagePipeline>>(
            std::move(model), context.get_tensor_options());
      });
  return true;
}();
}  // namespace

}  // namespace xllm

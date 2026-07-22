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

#include <algorithm>
#include <cstring>
#include <memory>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#if defined(USE_NPU)
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#endif
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/autoencoders/autoencoder_kl_wan.h"
#include "models/dit/encoders/umt5_encoder.h"
#include "models/dit/processors/vae_video_processor.h"
#include "models/dit/schedulers/uni_pc_multi_step_scheduler.h"
#include "models/dit/transformers/transformer_wan.h"
#if defined(USE_NPU)
#include "models/dit/utils/dit_block_weight_manager.h"
#endif
#include "models/model_registry.h"

namespace xllm {

class WanImageToVideoPipelineImpl : public torch::nn::Module {
 public:
  WanImageToVideoPipelineImpl(const DiTModelContext& context)
      : parallel_args_(context.get_parallel_args()) {
    options_ = context.get_tensor_options();
    const auto& vae_args = context.get_model_args("vae");
    zdim_ = vae_args.z_dim();
    latents_mean_ = vae_args.latents_mean();
    latents_std_ = vae_args.latents_std();

    const auto& scheduler_args = context.get_model_args("scheduler");
    num_train_timesteps_ = scheduler_args.num_train_timesteps();

    // RainFusion config is static (derived from flags) — build it once here
    // and pass to the transformers via constructor. It never changes per
    // request, unlike SparseAttnState which is per-inference dynamic state.
    auto& dit_config = DiTConfig::get_instance();
    sparse_attn_config_.enabled = dit_config.dit_sparse_attention_enabled();
    sparse_attn_config_.sparsity =
        static_cast<float>(dit_config.dit_sparse_attention_sparsity());
    sparse_attn_config_.pool_size = dit_config.dit_sparse_attention_pool_size();
    sparse_attn_config_.sparse_start_step =
        dit_config.dit_sparse_attention_sparse_start_step();
    sparse_attn_config_.version = dit_config.dit_sparse_attention_version();
    sparse_attn_config_.mask_refresh_steps =
        dit_config.dit_sparse_attention_mask_refresh_steps();

    LOG(INFO) << "Initializing Wan2_2I2V pipeline...";
    vae_ = AutoencoderKLWan(context.get_model_context("vae"));
    transformer_ = WanTransformer3DModel(
        context.get_model_context("transformer"), sparse_attn_config_);
    transformer_2_ = WanTransformer3DModel(
        context.get_model_context("transformer_2"), sparse_attn_config_);
    umt5_ = UMT5EncoderModel(context.get_model_context("text_encoder"));
    scheduler_ =
        UniPCMultistepScheduler(context.get_model_context("scheduler"));
    video_processor_ = VAEVideoProcessor(context.get_model_context("vae"),
                                         true,
                                         true,
                                         false,
                                         false,
                                         false,
                                         4,
                                         vae_scale_factor_spatial_);
    register_module("vae", vae_);
    register_module("transformer", transformer_);
    register_module("transformer_2", transformer_2_);
    register_module("umt5", umt5_);
    register_module("scheduler", scheduler_);
    register_module("video_processor_", video_processor_);
  }

  ~WanImageToVideoPipelineImpl() = default;

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    int64_t seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto images = input.images.defined() ? std::make_optional(input.images)
                                         : std::nullopt;
    auto last_images = input.last_images.defined()
                           ? std::make_optional(input.last_images)
                           : std::nullopt;
    auto prompts = std::make_optional(input.prompts);

    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;

    auto output = forward_impl(images,
                               last_images,
                               prompts,
                               negative_prompts,
                               generation_params.height,
                               generation_params.width,
                               generation_params.num_frames,
                               generation_params.num_inference_steps,
                               generation_params.guidance_scale,
                               generation_params.guidance_scale_2,
                               generation_params.num_videos_per_prompt,
                               seed,
                               latents,
                               prompt_embeds,
                               negative_prompt_embeds,
                               generation_params.max_sequence_length);

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "Wan2_2I2VPipeline loading model from"
              << loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto transformer_2_loader = loader->take_component_loader("transformer_2");
    auto vae_loader = loader->take_component_loader("vae");
    auto umt5_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");

    LOG(INFO) << "Wan2_2I2VPipeline model components loaded, start to load "
                 "weights to sub models";
#if defined(USE_NPU)
    auto& load_config = LoadConfig::get_instance();
    use_rolling_load_ = load_config.enable_rolling_load() &&
                        ParallelConfig::get_instance().tp_size() == 1;
    if (use_rolling_load_) {
      transformer_->load_model(std::move(transformer_loader),
                               /*rolling=*/true);
      transformer_2_->load_model(std::move(transformer_2_loader),
                                 /*rolling=*/true);
      init_rolling_for(transformer_, rolling_transformer_);
      init_rolling_for(transformer_2_, rolling_transformer_2_);
    } else {
#endif
      transformer_->load_model(std::move(transformer_loader));
      transformer_->to(options_.device());
      transformer_2_->load_model(std::move(transformer_2_loader));
      transformer_2_->to(options_.device());
#if defined(USE_NPU)
    }
#endif
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    umt5_->load_model(std::move(umt5_loader));
    umt5_->to(options_.device());
    tokenizer_ = tokenizer_loader->tokenizer();
  }

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare_latents(
      torch::Tensor image,
      int64_t batch_size,
      int64_t num_channels_latents = 16,
      int64_t height = 480,
      int64_t width = 832,
      int64_t num_frames = 81,
      std::optional<torch::Tensor> last_image = std::nullopt,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t num_latent_frames =
        (num_frames - 1) / vae_scale_factor_temporal_ + 1;
    int64_t latent_height = height / vae_scale_factor_spatial_;
    int64_t latent_width = width / vae_scale_factor_spatial_;

    std::vector<int64_t> shape = {batch_size,
                                  num_channels_latents,
                                  num_latent_frames,
                                  latent_height,
                                  latent_width};
    torch::Tensor latents_tensor;

    if (latents.has_value()) {
      latents_tensor = latents.value().to(options_.device());
    } else {
      latents_tensor =
          xllm::dit::randn_tensor(shape, seed, options_, torch::kFloat32);
    }
    image = image.unsqueeze(2);
    torch::Tensor video_condition;

    if (expand_timesteps_) {
      video_condition = image;
    } else if (!last_image.has_value()) {
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 1, height, width},
          image.options());
      video_condition = torch::cat({image, zeros}, 2);
    } else {
      auto last_img = last_image.value().unsqueeze(2);
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 2, height, width},
          image.options());
      video_condition = torch::cat({image, zeros, last_img}, 2);
    }
    video_condition = video_condition.to(options_.device()).to(torch::kFloat32);

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(latents_tensor.device(), latents_tensor.dtype());
    torch::Tensor latents_std =
        1.0 / torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
                  .view({1, num_channels_latents, 1, 1, 1})
                  .to(latents_tensor.device(), latents_tensor.dtype());

    torch::Tensor latent_condition;
    latent_condition = vae_->encode(video_condition).latent_dist.mode();
    latent_condition = (latent_condition - latents_mean) * latents_std;
    if (latent_condition.size(0) == 1 && batch_size > 1) {
      latent_condition = latent_condition.repeat({batch_size, 1, 1, 1, 1});
    }

    if (expand_timesteps_) {
      torch::Tensor first_frame_mask = torch::ones(
          {1, 1, num_latent_frames, latent_height, latent_width},
          options_.dtype(torch::kFloat32).device(options_.device()));
      first_frame_mask.slice(2, 0, 1) = 0;
      return {latents_tensor, latent_condition, first_frame_mask};
    }

    torch::Tensor mask_lat_size =
        torch::ones({batch_size, 1, num_frames, latent_height, latent_width},
                    options_.dtype(torch::kFloat32).device(options_.device()));

    if (!last_image.has_value()) {
      for (int64_t i = 1; i < num_frames; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    } else {
      for (int64_t i = 1; i < num_frames - 1; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    }

    torch::Tensor first_frame_mask = mask_lat_size.slice(2, 0, 1);
    first_frame_mask = torch::repeat_interleave(
        first_frame_mask, vae_scale_factor_temporal_, 2);

    torch::Tensor rest_mask = mask_lat_size.slice(2, 1);
    mask_lat_size = torch::cat({first_frame_mask, rest_mask}, 2);

    mask_lat_size = mask_lat_size.view({batch_size,
                                        -1,
                                        vae_scale_factor_temporal_,
                                        latent_height,
                                        latent_width});
    mask_lat_size = mask_lat_size.transpose(1, 2);
    mask_lat_size = mask_lat_size.to(latent_condition.device());

    torch::Tensor combined_condition =
        torch::cat({mask_lat_size, latent_condition}, 1);

    return {latents_tensor, combined_condition, first_frame_mask};
  }

  torch::Tensor get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                     int64_t num_videos_per_prompt = 1,
                                     int64_t max_sequence_length = 512) {
    int64_t batch_size = prompt.size();

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);

    CHECK(tokenizer_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      ids.push_back(1);  // EOS token </s>
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(options_.device());

    torch::Tensor attention_mask =
        (1.0 - (input_ids > 0).to(options_.dtype()).unsqueeze(1).unsqueeze(2)) *
        (std::numeric_limits<float>::lowest());
    torch::Tensor prompt_embeds = umt5_->forward(input_ids, attention_mask);

    prompt_embeds = prompt_embeds.to(options_);

    auto seq_lens = (input_ids > 0).sum(1).to(torch::kLong);

    std::vector<torch::Tensor> trimmed_embeds;
    trimmed_embeds.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t seq_len = seq_lens[i].item<int64_t>();
      auto trimmed = prompt_embeds[i].slice(0, 0, seq_len);
      int64_t padding_len = max_sequence_length - trimmed.size(0);
      if (padding_len > 0) {
        auto zeros =
            torch::zeros({padding_len, trimmed.size(1)}, trimmed.options());
        trimmed = torch::cat({trimmed, zeros}, 0);
      }
      trimmed_embeds.emplace_back(std::move(trimmed));
    }
    prompt_embeds = torch::stack(trimmed_embeds, 0);

    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_videos_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_videos_per_prompt, seq_len, -1});

    return prompt_embeds;
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> negative_prompt,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> negative_prompt_embeds,
      bool do_classifier_free_guidance = true,
      int64_t num_videos_per_prompt = 1,
      int64_t max_sequence_length = 226) {
    torch::Tensor prompt_embeds_tensor;
    torch::Tensor negative_prompt_embeds_tensor;

    if (prompt_embeds.has_value()) {
      prompt_embeds_tensor = prompt_embeds.value();
    } else if (prompt.has_value()) {
      prompt_embeds_tensor = get_t5_prompt_embeds(
          prompt.value(), num_videos_per_prompt, max_sequence_length);
    }

    int64_t batch_size;
    if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds_tensor.size(0);
    } else if (prompt.has_value()) {
      batch_size = prompt.value().size();
    }

    if (do_classifier_free_guidance) {
      if (negative_prompt.has_value()) {
        negative_prompt_embeds_tensor =
            get_t5_prompt_embeds(negative_prompt.value(),
                                 num_videos_per_prompt,
                                 max_sequence_length);
      } else if (negative_prompt_embeds.has_value()) {
        negative_prompt_embeds_tensor = negative_prompt_embeds.value();
      } else {
        std::vector<std::string> empty_prompt = {""};
        negative_prompt_embeds_tensor = get_t5_prompt_embeds(
            empty_prompt, num_videos_per_prompt, max_sequence_length);
      }
    }
    return {prompt_embeds_tensor, negative_prompt_embeds_tensor};
  }

  torch::Tensor forward_impl(
      std::optional<torch::Tensor> images = std::nullopt,
      std::optional<torch::Tensor> last_images = std::nullopt,
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_frames = 16,
      int64_t num_inference_steps = 28,
      float guidance_scale = 5.0f,
      float guidance_scale_2 = -1.0f,
      int64_t num_videos_per_prompt = 1,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds.value().size(0);
    }

    int64_t total_batch_size = batch_size * num_videos_per_prompt;
    bool do_classifier_free_guidance = guidance_scale > 1.0f;

    if (num_frames % vae_scale_factor_temporal_ != 1) {
      LOG(WARNING) << "num_frames - 1 has to be divisible by "
                   << vae_scale_factor_temporal_
                   << ". Rounding to the nearest number.";
      num_frames =
          num_frames / vae_scale_factor_temporal_ * vae_scale_factor_temporal_ +
          1;
    }
    num_frames = std::max(num_frames, static_cast<int64_t>(1));

    int64_t patch_size_h = transformer_->patch_size()[1];
    int64_t patch_size_w = transformer_->patch_size()[2];

    int64_t dw = vae_scale_factor_spatial_ * patch_size_w;
    int64_t dh = vae_scale_factor_spatial_ * patch_size_h;

    // Call unified function for dimension adjustment
    AdjustVideoSize(images, height, width, dw, dh, false);

    if (boundary_ratio_ > 0.0f && guidance_scale_2 < 0.0f) {
      guidance_scale_2 = guidance_scale;
    }

    auto [encoded_prompt_embeds, encoded_negative_embeds] =
        encode_prompt(prompt,
                      negative_prompt,
                      prompt_embeds,
                      negative_prompt_embeds,
                      do_classifier_free_guidance,
                      num_videos_per_prompt,
                      max_sequence_length);
    scheduler_->set_timesteps(num_inference_steps,
                              options_.device(),
                              /*sigmas*/ std::nullopt,
                              /*mu*/ std::nullopt);
    torch::Tensor timesteps = scheduler_->timesteps();

    int64_t num_channels_latents = zdim_;
    torch::Tensor input_image;

    if (images.has_value()) {
      input_image = images.value();
    } else {
      LOG(WARNING) << "No input image provided for I2V pipeline. "
                   << "Using blank white image as fallback.";
      input_image = torch::ones({3, height, width}, torch::kFloat32);
    }
    torch::Tensor preprocessed_image = video_processor_->preprocess(
        input_image, height, width, "bicubic_no_aa");
    preprocessed_image =
        preprocessed_image.to(options_.device(), torch::kFloat32);

    if (preprocessed_image.dim() == 3) {
      preprocessed_image = preprocessed_image.unsqueeze(0);
    }

    std::optional<torch::Tensor> preprocessed_last_image;
    if (last_images.has_value()) {
      torch::Tensor last_img =
          video_processor_->preprocess(last_images.value(), height, width);
      last_img = last_img.to(options_.device(), torch::kFloat32);
      if (last_img.dim() == 3) {
        last_img = last_img.unsqueeze(0);
      }
      preprocessed_last_image = last_img;
    }

    torch::Tensor prepared_latents, latent_condition, first_frame_mask;
    std::tie(prepared_latents, latent_condition, first_frame_mask) =
        prepare_latents(preprocessed_image,
                        total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        num_frames,
                        preprocessed_last_image,
                        seed,
                        latents);

    float boundary_timestep =
        boundary_ratio_ > 0.0f ? boundary_ratio_ * num_train_timesteps_ : -1.0f;

    // RainFusion config is static and already bound to the transformers in the
    // constructor. Only the per-inference dynamic state lives here.
    xllm::dit::SparseAttnState sparse_attn_state;

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i];
      int64_t total_steps = timesteps.numel();

      WanTransformer3DModel current_model = nullptr;
      float current_guidance;

      if (boundary_timestep < 0 || t.item<float>() >= boundary_timestep) {
        current_model = transformer_;
        current_guidance = guidance_scale;
      } else {
        current_model = transformer_2_;
        current_guidance = guidance_scale_2;
      }

      sparse_attn_state.current_step = i;
      torch::Tensor latent_model_input;
      torch::Tensor timestep_input;

      if (expand_timesteps_) {
        latent_model_input = (1 - first_frame_mask) * latent_condition +
                             first_frame_mask * prepared_latents;
        latent_model_input = latent_model_input.to(prepared_latents.dtype());

        torch::Tensor temp_ts = (first_frame_mask[0][0]
                                     .slice(1, 0, first_frame_mask.size(2), 2)
                                     .slice(2, 0, first_frame_mask.size(3), 2) *
                                 t)
                                    .flatten();
        timestep_input =
            temp_ts.unsqueeze(0).expand({prepared_latents.size(0), -1});
      } else {
        latent_model_input =
            torch::cat({prepared_latents, latent_condition}, 1);
        latent_model_input = latent_model_input.to(prepared_latents.dtype());

        if (!timestep_input.defined()) {
          timestep_input = t.expand({prepared_latents.size(0)});
        }
      }
      torch::Tensor noise_pred;
      torch::Tensor noise_uncond;
#if defined(USE_NPU)
      auto& rolling = (current_model.get() == transformer_.get())
                          ? rolling_transformer_
                          : rolling_transformer_2_;
      auto rolling_forward = [&](const torch::Tensor& embeds) {
        return current_model->forward(
            latent_model_input,
            timestep_input,
            embeds,
            torch::Tensor(),
            sparse_attn_state,
            [&rolling](int32_t i) { rolling.wait_h2d(i); },
            [&rolling](int32_t i) { rolling.schedule_next_h2d(i); });
      };
#else
      auto rolling_forward = [&](const torch::Tensor& /*embeds*/) {
        LOG(FATAL) << "Rolling load requires USE_NPU";
        return torch::Tensor{};
      };
#endif

      if (do_classifier_free_guidance) {
        if (ParallelConfig::get_instance().cfg_size() == 2) {
          int32_t rank = parallel_args_.dit_cfg_group_->rank();
          noise_pred =
              use_rolling_load_
                  ? rolling_forward(rank == 0 ? encoded_prompt_embeds
                                              : encoded_negative_embeds)
                  : current_model->forward(latent_model_input,
                                           timestep_input,
                                           rank == 0 ? encoded_prompt_embeds
                                                     : encoded_negative_embeds,
                                           torch::Tensor(),
                                           sparse_attn_state);
          auto gathered = xllm::parallel_state::gather(
              noise_pred, parallel_args_.dit_cfg_group_, /*dim=*/0);
          auto chunks = torch::chunk(gathered, 2, 0);
          noise_pred = chunks[0];
          noise_uncond = chunks[1];
        } else {
          if (use_rolling_load_) {
            noise_pred = rolling_forward(encoded_prompt_embeds);
            noise_uncond = rolling_forward(encoded_negative_embeds);
          } else {
            noise_pred = current_model->forward(latent_model_input,
                                                timestep_input,
                                                encoded_prompt_embeds,
                                                torch::Tensor(),
                                                sparse_attn_state);
            noise_uncond = current_model->forward(latent_model_input,
                                                  timestep_input,
                                                  encoded_negative_embeds,
                                                  torch::Tensor(),
                                                  sparse_attn_state);
          }
        }

        noise_pred = noise_uncond.to(torch::kFloat32) +
                     static_cast<float>(current_guidance) *
                         (noise_pred.to(torch::kFloat32) -
                          noise_uncond.to(torch::kFloat32));
        noise_uncond.reset();
      } else {
        noise_pred = use_rolling_load_
                         ? rolling_forward(encoded_prompt_embeds)
                         : current_model->forward(latent_model_input,
                                                  timestep_input,
                                                  encoded_prompt_embeds,
                                                  torch::Tensor(),
                                                  sparse_attn_state);
      }
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    prepared_latents = prepared_latents.to(torch::kFloat32);

    if (expand_timesteps_) {
      prepared_latents = (1 - first_frame_mask) * latent_condition +
                         first_frame_mask * prepared_latents;
    }

    torch::Tensor video;

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(prepared_latents.device());
    torch::Tensor latents_std_raw =
        torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(prepared_latents.device());

    torch::Tensor latents_std = 1.0 / latents_std_raw;
    prepared_latents = prepared_latents / latents_std;
    prepared_latents = prepared_latents + latents_mean;
    video = vae_->decode(prepared_latents.to(torch::kFloat32)).sample;
    video = video_processor_->postprocess_video(video);
    return video;
  }

  void AdjustVideoSize(const std::optional<torch::Tensor>& images,
                       int64_t& height,
                       int64_t& width,
                       int64_t dw,
                       int64_t dh,
                       bool use_user_priority) {
    if (use_user_priority) {
      // User priority mode: directly use user-specified dimensions
      // Only enforce alignment to 16x multiple (dw, dh)
      if (height % dh != 0) {
        height = (height / dh) * dh;
        LOG(WARNING) << "Height adjusted to " << height << " (multiple of "
                     << dh << ")";
      }
      if (width % dw != 0) {
        width = (width / dw) * dw;
        LOG(WARNING) << "Width adjusted to " << width << " (multiple of " << dw
                     << ")";
      }
    } else {
      // Original logic: choose best candidate based on aspect ratio and area
      int64_t max_area = height * width;
      int64_t ih = images.has_value() ? images.value().size(-2) : height;
      int64_t iw = images.has_value() ? images.value().size(-1) : width;
      double ratio = static_cast<double>(iw) / ih;

      // Candidate 1: floor width first
      int64_t ow1 = static_cast<int64_t>(std::sqrt(max_area * ratio)) / dw * dw;
      int64_t oh1 = (max_area / ow1) / dh * dh;
      double ratio1 = static_cast<double>(ow1) / oh1;

      // Candidate 2: floor height first
      int64_t oh2 = static_cast<int64_t>(std::sqrt(max_area / ratio)) / dh * dh;
      int64_t ow2 = (max_area / oh2) / dw * dw;
      double ratio2 = static_cast<double>(ow2) / oh2;

      // Pick the candidate that preserves aspect ratio better
      int64_t calc_height, calc_width;
      if (std::max(ratio / ratio1, ratio1 / ratio) <
          std::max(ratio / ratio2, ratio2 / ratio)) {
        calc_width = ow1;
        calc_height = oh1;
      } else {
        calc_width = ow2;
        calc_height = oh2;
      }

      if (height != calc_height || width != calc_width) {
        height = calc_height;
        width = calc_width;
        LOG(INFO) << "Size adjusted by aspect ratio: height=" << height
                  << ", width=" << width;
      }
    }
  }

 private:
  // WAN rolling load state.
  bool use_rolling_load_ = false;
#if defined(USE_NPU)
  dit::DitRollingLoadManager rolling_transformer_;
  dit::DitRollingLoadManager rolling_transformer_2_;

  void init_rolling_for(WanTransformer3DModel model,
                        dit::DitRollingLoadManager& manager) {
    at::DeviceGuard guard(options_.device());
    auto loaders = model->get_block_weight_loaders();
    CHECK(!loaders.empty()) << "No block weight managers";

    size_t max_storage = 0;
    for (auto* loader : loaders) {
      max_storage = std::max(max_storage, loader->storage_size());
    }

    auto& load_config = LoadConfig::get_instance();
    int32_t num_slots =
        std::max(load_config.rolling_load_num_rolling_slots(), 2);
    auto buffer =
        std::make_shared<layer::RollingWeightBuffer>(num_slots, max_storage);
    manager.init(std::move(loaders), std::move(buffer), num_slots);
    manager.preload();
  }
#endif

  UniPCMultistepScheduler scheduler_{nullptr};
  AutoencoderKLWan vae_{nullptr};
  WanTransformer3DModel transformer_{nullptr};
  WanTransformer3DModel transformer_2_{nullptr};
  UMT5EncoderModel umt5_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_{nullptr};
  VAEVideoProcessor video_processor_{nullptr};

  int64_t vae_scale_factor_spatial_ = 8;
  int64_t vae_scale_factor_temporal_ = 4;
  float boundary_ratio_ = 0.9f;
  bool expand_timesteps_ = false;
  int64_t zdim_ = 16;
  float num_train_timesteps_ = 1000.0f;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
  torch::TensorOptions options_;
  const ParallelArgs parallel_args_;
  xllm::dit::SparseAttnConfig sparse_attn_config_;
};
TORCH_MODULE(WanImageToVideoPipeline);

REGISTER_DIT_MODEL(WanImageToVideoPipeline, WanImageToVideoPipeline);

}  // namespace xllm

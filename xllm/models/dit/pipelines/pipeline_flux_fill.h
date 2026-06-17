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
#include "models/dit/pipelines/pipeline_flux_base.h"
#include "models/dit/transformers/transformer_flux.h"
// pipeline_flux_fill compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_fill.py

namespace xllm {

class FluxFillPipelineImpl : public FluxPipelineBaseImpl {
 public:
  FluxFillPipelineImpl(const DiTModelContext& context) {
    auto model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    latent_channels_ = model_args.latent_channels();
    tokenizer_max_length_ =
        context.get_model_args("text_encoder").max_position_embeddings();
    LOG(INFO) << "Initializing FluxFill pipeline...";
    image_processor_ = VAEImageProcessor(context.get_model_context("vae"),
                                         true,
                                         true,
                                         false,
                                         false,
                                         false,
                                         latent_channels_);
    mask_processor_ = VAEImageProcessor(context.get_model_context("vae"),
                                        true,
                                        false,
                                        true,
                                        false,
                                        true,
                                        latent_channels_);
    vae_ = VAE(context.get_model_context("vae"));
    LOG(INFO) << "VAE initialized.";
    pos_embed_ = register_module(
        "pos_embed",
        FluxPosEmbed(ROPE_SCALE_BASE,
                     context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = FluxDiTModel(context.get_model_context("transformer"));
    t5_ = T5EncoderModel(context.get_model_context("text_encoder_2"));
    clip_text_model_ = CLIPTextModel(context.get_model_context("text_encoder"));
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
    register_module("vae", vae_);
    register_module("vae_image_processor", image_processor_);
    register_module("mask_processor", mask_processor_);
    register_module("transformer", transformer_);
    register_module("t5", t5_);
    register_module("scheduler", scheduler_);
    register_module("clip_text_model", clip_text_model_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;
    int64_t height = generation_params.height;
    int64_t width = generation_params.width;
    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);

    auto image = input.images.defined() ? std::make_optional(input.images)
                                        : std::nullopt;
    auto mask_image = input.mask_images.defined()
                          ? std::make_optional(input.mask_images)
                          : std::nullopt;
    auto masked_image_latents =
        input.masked_image_latents.defined()
            ? std::make_optional(input.masked_image_latents)
            : std::nullopt;

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;

    auto output = forward_impl(prompts,
                               prompts_2,
                               image,
                               mask_image,
                               masked_image_latents,
                               height,
                               width,
                               generation_params.strength,
                               generation_params.num_inference_steps,
                               generation_params.guidance_scale,
                               generation_params.num_images_per_prompt,
                               seed,
                               latents,
                               prompt_embeds,
                               pooled_prompt_embeds,
                               generation_params.max_sequence_length);

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxFillPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto t5_loader = loader->take_component_loader("text_encoder_2");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto tokenizer_2_loader = loader->take_component_loader("tokenizer_2");
    LOG(INFO) << "FluxFill model components loaded, start to load weights to "
                 "sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    t5_->load_model(std::move(t5_loader));
    t5_->to(options_.device());
    clip_text_model_->load_model(std::move(clip_loader));
    clip_text_model_->to(options_.device());
    tokenizer_ = tokenizer_loader->tokenizer();
    tokenizer_2_ = tokenizer_2_loader->tokenizer();
  }

 private:
  std::pair<torch::Tensor, torch::Tensor> prepare_mask_latents(
      torch::Tensor mask,
      torch::Tensor masked_image,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t num_images_per_prompt,
      int64_t height,
      int64_t width,
      int64_t seed) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor masked_image_latents;
    if (masked_image.size(1) == num_channels_latents) {
      masked_image_latents = masked_image;
    } else {
      masked_image_latents = vae_->encode(masked_image, seed);
    }

    masked_image_latents =
        (masked_image_latents - vae_shift_factor_) * vae_scaling_factor_;
    masked_image_latents = masked_image_latents.to(options_);

    batch_size = batch_size * num_images_per_prompt;
    if (mask.size(0) < batch_size) {
      CHECK(batch_size % mask.size(0) == 0)
          << "Masks batch size mismatch: mask cannot be duplicated to match "
             "total batch.";
      mask = mask.repeat({batch_size / mask.size(0), 1, 1, 1});
    }

    if (masked_image_latents.size(0) < batch_size) {
      CHECK(batch_size % masked_image_latents.size(0) == 0)
          << "Masked image batch size mismatch: cannot duplicate to match "
             "total batch.";
      masked_image_latents = masked_image_latents.repeat(
          {batch_size / masked_image_latents.size(0), 1, 1, 1});
    }

    masked_image_latents = pack_latents(
        masked_image_latents, batch_size, num_channels_latents, height, width);

    mask = mask.select(1, 0);
    mask = mask.view(
        {batch_size, height, vae_scale_factor_, width, vae_scale_factor_});
    mask = mask.permute({0, 2, 4, 1, 3});
    mask = mask.reshape(
        {batch_size, vae_scale_factor_ * vae_scale_factor_, height, width});
    mask = pack_latents(
        mask, batch_size, vae_scale_factor_ * vae_scale_factor_, height, width);
    mask = mask.to(options_);

    return {mask, masked_image_latents};
  }

  torch::Tensor encode_vae_image(const torch::Tensor& image, int64_t seed) {
    torch::Tensor latents = vae_->encode(image, seed);
    latents = (latents - vae_shift_factor_) * vae_scaling_factor_;
    return latents;
  }

  std::pair<torch::Tensor, int64_t> get_timesteps(int64_t num_inference_steps,
                                                  float strength) {
    int64_t init_timestep =
        std::min(static_cast<int64_t>(num_inference_steps * strength),
                 num_inference_steps);

    int64_t t_start = std::max(num_inference_steps - init_timestep, int64_t(0));
    int64_t start_idx = t_start * scheduler_->order();
    auto timesteps = scheduler_->timesteps().slice(0, start_idx).to(options_);
    scheduler_->set_begin_index(start_idx);
    return {timesteps, num_inference_steps - t_start};
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      torch::Tensor image,
      torch::Tensor timesteps,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, height, width};
    torch::Tensor latent_image_ids =
        prepare_latent_image_ids(batch_size, height / 2, width / 2);
    if (latents.has_value()) {
      return {latents.value().to(options_), latent_image_ids};
    }

    torch::Tensor image_latents;
    if (image.size(1) != latent_channels_) {
      image_latents = encode_vae_image(image, seed);
    } else {
      image_latents = image;
    }
    int64_t additional_image_per_prompt;
    if (batch_size > image_latents.size(0) &&
        batch_size % image_latents.size(0) == 0) {
      additional_image_per_prompt = batch_size / image_latents.size(0);
      image_latents =
          image_latents.repeat({additional_image_per_prompt, 1, 1, 1});
    } else if (batch_size > image_latents.size(0) &&
               batch_size % image_latents.size(0) != 0) {
      LOG(FATAL) << "Cannot match batch_size with input images.";
    } else {
      image_latents = torch::cat({image_latents}, 0);
    }
    auto noise = xllm::dit::randn_tensor(shape, seed, options_);
    latents = scheduler_->scale_noise(image_latents, timesteps, noise);
    latents = pack_latents(
        latents.value(), batch_size, num_channels_latents, height, width);
    return {latents.value(), latent_image_ids};
  }

  torch::Tensor forward_impl(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<torch::Tensor> image = std::nullopt,
      std::optional<torch::Tensor> mask_image = std::nullopt,
      std::optional<torch::Tensor> masked_image_latents = std::nullopt,
      int64_t height = 512,
      int64_t width = 512,
      float strength = 1.0f,
      int64_t num_inference_steps = 50,
      float guidance_scale = 30.0f,
      int64_t num_images_per_prompt = 1,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    torch::Tensor init_image =
        image_processor_->preprocess(image.value(), height, width);

    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }

    torch::Tensor text_ids;
    std::tie(prompt_embeds, pooled_prompt_embeds, text_ids) =
        encode_prompt(prompt,
                      prompt_2,
                      prompt_embeds,
                      pooled_prompt_embeds,
                      num_images_per_prompt,
                      max_sequence_length);

    std::vector<float> sigmas = [&](int64_t steps) {
      std::vector<float> result(steps);
      for (int64_t i = 0; i < steps; ++i)
        result[i] = 1.0f - static_cast<float>(i) / steps;
      return result;
    }(num_inference_steps);

    int64_t image_seq_len =
        (height / vae_scale_factor_ / 2) * (width / vae_scale_factor_ / 2);
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());

    retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), sigmas, mu);
    torch::Tensor timesteps;
    std::tie(timesteps, num_inference_steps) =
        get_timesteps(num_inference_steps, strength);
    CHECK(num_inference_steps >= 1);

    torch::Tensor latent_timestep =
        timesteps.index({torch::indexing::Slice(0, 1)})
            .repeat({batch_size * num_images_per_prompt});

    int64_t num_channels_latents = latent_channels_;
    torch::Tensor latent_image_ids;
    std::tie(latents, latent_image_ids) =
        prepare_latents(init_image,
                        latent_timestep,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        seed,
                        latents);

    if (masked_image_latents.has_value()) {
      masked_image_latents = masked_image_latents.value().to(options_);
    } else {
      mask_image =
          mask_processor_->preprocess(mask_image.value(), height, width);
      torch::Tensor masked_image = init_image * (1 - mask_image.value());

      height = init_image.size(-2);
      width = init_image.size(-1);

      torch::Tensor mask;
      std::tie(mask, masked_image_latents) =
          prepare_mask_latents(mask_image.value(),
                               masked_image,
                               batch_size,
                               num_channels_latents,
                               num_images_per_prompt,
                               height,
                               width,
                               seed);
      masked_image_latents =
          torch::cat({masked_image_latents.value(), mask}, -1);
    }

    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      guidance = torch::full(torch::IntArrayRef({1}), guidance_scale, options_);
      guidance = guidance.expand({latents.value().size(0)});
    }

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));

    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(options_.dtype());

    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      torch::Tensor t = timesteps[i];
      torch::Tensor timestep =
          t.expand({latents->size(0)}).to(options_.device());

      int64_t step_id = i + 1;
      torch::Tensor input_latents =
          torch::cat({latents.value(), masked_image_latents.value()}, 2);

      torch::Tensor noise_pred =
          transformer_->forward(input_latents,
                                prompt_embeds.value(),
                                pooled_prompt_embeds.value(),
                                timestep / 1000,
                                image_rotary_emb,
                                guidance,
                                step_id);
      auto prev_latents = scheduler_->step(noise_pred, t, latents.value());
      latents = prev_latents.detach().to(options_.device());
    }

    torch::Tensor output_image;
    torch::Tensor unpacked_latents =
        unpack_latents(latents.value(), height, width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;

    output_image = vae_->decode(unpacked_latents);
    output_image = image_processor_->postprocess(output_image);
    return output_image;
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor image_processor_{nullptr};
  VAEImageProcessor mask_processor_{nullptr};
  FluxDiTModel transformer_{nullptr};
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int64_t latent_channels_;
  FluxPosEmbed pos_embed_{nullptr};
};
TORCH_MODULE(FluxFillPipeline);

REGISTER_DIT_MODEL(fluxfill, FluxFillPipeline);
}  // namespace xllm

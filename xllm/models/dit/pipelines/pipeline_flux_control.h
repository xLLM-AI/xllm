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
#include "models/dit/pipelines/pipeline_flux_base.h"
#include "models/dit/transformers/transformer_flux.h"
// pipeline_flux_control compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_control.py

namespace xllm {

class FluxControlPipelineImpl : public FluxPipelineBaseImpl {
 public:
  FluxControlPipelineImpl(const DiTModelContext& context) {
    auto model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    latent_channels_ = model_args.latent_channels();
    tokenizer_max_length_ =
        context.get_model_args("text_encoder").max_position_embeddings();
    LOG(INFO) << "Initializing FluxControl pipeline...";
    image_processor_ = VAEImageProcessor(context.get_model_context("vae"),
                                         true,
                                         true,
                                         false,
                                         false,
                                         false,
                                         latent_channels_);
    vae_ = VAE(context.get_model_context("vae"));
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

    auto control_image = input.control_image;

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
                               control_image,
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
    LOG(INFO) << "FluxControlPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto t5_loader = loader->take_component_loader("text_encoder_2");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto tokenizer_2_loader = loader->take_component_loader("tokenizer_2");
    LOG(INFO)
        << "FluxControl model components loaded, start to load weights to "
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
    torch::Tensor latents_tensor =
        xllm::dit::randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor,
                                                batch_size,
                                                num_channels_latents,
                                                adjusted_height,
                                                adjusted_width);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2);
    return {packed_latents, latent_image_ids};
  }

  torch::Tensor prepare_image(torch::Tensor image,
                              int64_t width,
                              int64_t height,
                              int64_t batch_size,
                              int64_t num_images_per_prompt) {
    int image_batch_size = image.size(0);
    int repeat_times;
    image = image_processor_->preprocess(image, height, width);
    if (image_batch_size == 1) {
      repeat_times = batch_size;
    } else {
      repeat_times = num_images_per_prompt;
    }
    const auto B = image.size(0);
    const auto C = image.size(1);
    const auto H = image.size(2);
    const auto W = image.size(3);
    image = image.unsqueeze(1)
                .repeat({1, repeat_times, 1, 1, 1})
                .reshape({B * repeat_times, C, H, W})
                .to(options_);
    return image;
  }

  torch::Tensor forward_impl(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      torch::Tensor control_image = torch::Tensor(),
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
    int64_t actual_height = height;
    int64_t actual_width = width;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;
    // encode prompt
    auto [encoded_prompt_embeds, encoded_pooled_embeds, text_ids] =
        encode_prompt(prompt,
                      prompt_2,
                      prompt_embeds,
                      pooled_prompt_embeds,
                      num_images_per_prompt,
                      max_sequence_length);

    // prepare latent
    int64_t num_channels_latents = transformer_->in_channels() / 8;
    // control image to latents
    control_image = prepare_image(control_image,
                                  width,
                                  height,
                                  batch_size * num_images_per_prompt,
                                  num_images_per_prompt);
    if (control_image.dim() == 4) {
      auto enc = vae_->encode(control_image, seed);
      control_image = (enc - vae_shift_factor_) * vae_scaling_factor_;
      control_image = control_image.to(options_);
      auto shape = control_image.sizes();
      auto height_control_image = shape[2];
      auto width_control_image = shape[3];
      control_image = pack_latents(control_image,
                                   total_batch_size,
                                   num_channels_latents,
                                   height_control_image,
                                   width_control_image);
    }
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        actual_height,
                        actual_width,
                        seed,
                        latents);
    // prepare timestep
    std::vector<float> new_sigmas;
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                      (num_inference_steps - 1) *
                                      (1.0f - 1.0f / num_inference_steps));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());
    auto [timesteps, num_inference_steps_actual] = retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));
    // prepare guidance
    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      torch::TensorOptions options = options_.dtype(torch::kFloat32);
      guidance = torch::full(torch::IntArrayRef({1}), guidance_scale, options);
      guidance = guidance.expand({prepared_latents.size(0)});
    }
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());
    // image rotary positional embeddings outplace computation
    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(options_.dtype());
    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);
      int64_t step_id = i + 1;
      auto controlled_latents =
          torch::cat({prepared_latents, control_image}, 2);
      torch::Tensor noise_pred = transformer_->forward(controlled_latents,
                                                       encoded_prompt_embeds,
                                                       encoded_pooled_embeds,
                                                       timestep,
                                                       image_rotary_emb,
                                                       guidance,
                                                       step_id);
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }
    torch::Tensor image;
    // Unpack latents
    torch::Tensor unpacked_latents = unpack_latents(
        prepared_latents, actual_height, actual_width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    unpacked_latents = unpacked_latents.to(options_.dtype());
    image = vae_->decode(unpacked_latents);
    image = image_processor_->postprocess(image);
    return image;
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor image_processor_{nullptr};
  FluxDiTModel transformer_{nullptr};
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int64_t vae_latent_channels_;
  int64_t latent_channels_;
  FluxPosEmbed pos_embed_{nullptr};
};
TORCH_MODULE(FluxControlPipeline);

REGISTER_DIT_MODEL(flux_control, FluxControlPipeline);
}  // namespace xllm

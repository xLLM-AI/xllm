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
#include <chrono>
#include <cstdint>

#include "core/framework/dit_cache/dit_cache.h"
#include "models/dit/autoencoders/autoencoder_kl_flux2.h"
#include "models/dit/pipelines/pipeline_flux2_base.h"
#include "models/dit/processors/flux2_image_processor.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_flux2.h"

namespace xllm {

// Flux2 Klein Pipeline — lightweight variant of Flux2Pipeline.
//
// Key differences from Flux2Pipeline:
//   * Text encoder: Qwen3 (pure-text) instead of Mistral3 (multimodal).
//     The text encoder is expected to be run externally; this pipeline
//     accepts pre-computed prompt_embeds and negative_prompt_embeds.
//   * Guidance: classifier-free guidance (CFG) dual forward pass instead
//     of embedded guidance-distilled single pass.  When is_distilled_ is
//     true, CFG is skipped and guidance_scale is ignored.
//   * No system messages or prompt upsampling.
//   * DtoH sync optimisation: latent height/width are pre-computed and
//     passed to unpack_latents_with_ids to avoid torch.max().item().
class Flux2KleinPipelineImpl final : public Flux2PipelineBaseImpl {
 public:
  explicit Flux2KleinPipelineImpl(const DiTModelContext& context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    tokenizer_max_length_ = 512;
    default_sample_size_ = 128;
    is_distilled_ = false;

    flux2_image_processor_ = Flux2ImageProcessor(
        context.get_model_context("vae"), vae_scale_factor_ * 2);
    vae_ = AutoencoderKLFlux2(context.get_model_context("vae"));
    pos_embed_ = register_module(
        "pos_embed",
        Flux2PosEmbed(context.get_model_args("transformer").rope_theta(),
                      context.get_model_args("transformer").axes_dims_rope()));

    transformer_ = Flux2DiTModel(context.get_model_context("transformer"),
                                 context.get_parallel_args());
    num_single_layers_ =
        context.get_model_args("transformer").num_single_layers();
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
    register_module("vae", vae_);
    register_module("flux2_image_processor", flux2_image_processor_);
    register_module("transformer", transformer_);
    register_module("scheduler", scheduler_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    // int64_t seed = generation_params.seed_is_set ? generation_params.seed :
    // 42;
    int64_t seed = generation_params.seed >= 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;
    auto images = input.images.defined() ? std::make_optional(input.images)
                                         : std::nullopt;

    auto output = forward_impl(
        prompts,                                  // prompt
        generation_params.height,                 // height
        generation_params.width,                  // width
        generation_params.num_inference_steps,    // num_inference_steps
        generation_params.guidance_scale,         // guidance_scale
        generation_params.num_images_per_prompt,  // num_images_per_prompt
        seed,                                     // seed
        latents,                                  // latents
        prompt_embeds,                            // prompt_embeds
        negative_prompt_embeds,                   // negative_prompt_embeds
        images,                                   // images
        generation_params.max_sequence_length     // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
  }

 private:
  // Prepare noise latents and their 4-D position IDs.
  // Reuses base-class helpers: pack_latents, prepare_latent_image_ids.
  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {batch_size,
                                  num_channels_latents * 4,
                                  adjusted_height / 2,
                                  adjusted_width / 2};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids =
          prepare_latent_image_ids(latents.value());
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor =
        xllm::dit::randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(latents_tensor);
    return {packed_latents, latent_image_ids};
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_image_latents(
      const std::vector<torch::Tensor>& images,
      int64_t batch_size,
      int64_t seed) {
    std::vector<torch::Tensor> image_latents;
    image_latents.reserve(images.size());
    for (const torch::Tensor& image : images) {
      torch::Tensor image_latent = vae_->encode(image, seed);
      torch::Tensor patched_latent = patchify_latents(image_latent);
      torch::Tensor latents_bn_mean =
          vae_->get_bn_running_mean()
              .view({1, -1, 1, 1})
              .to(image_latent.device(), image_latent.dtype());
      torch::Tensor latents_bn_std =
          torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) +
                      vae_->get_batch_norm_eps());
      image_latent = (patched_latent - latents_bn_mean) / latents_bn_std;
      image_latents.emplace_back(image_latent);
    }
    torch::Tensor concatenated_latents = pack_latents_for_images(image_latents);
    torch::Tensor image_latent_ids = _prepare_image_ids(image_latents);

    concatenated_latents = concatenated_latents.unsqueeze(0);
    torch::Tensor repeated_latents =
        concatenated_latents.repeat({batch_size, 1, 1});
    torch::Tensor repeated_ids = image_latent_ids.repeat({batch_size, 1, 1});
    return {repeated_latents, repeated_ids};
  }

  torch::Tensor pack_latents_for_images(
      const std::vector<torch::Tensor>& image_latents) {
    std::vector<torch::Tensor> packed_latents;
    packed_latents.reserve(image_latents.size());
    for (const torch::Tensor& latent : image_latents) {
      torch::Tensor packed = pack_latents(latent);
      packed = packed.squeeze(0);
      packed_latents.emplace_back(packed);
    }
    return torch::cat(packed_latents, 0);
  }

  // Klein-optimised unpack: accepts pre-computed spatial dimensions to
  // avoid the DtoH synchronisation caused by torch::max().item() in the
  // base-class unpack_latents_with_ids.
  torch::Tensor unpack_latents_with_ids(const torch::Tensor& latents,
                                        const torch::Tensor& latent_ids,
                                        int64_t height,
                                        int64_t width) {
    int64_t batch_size = latents.size(0);
    int64_t channels = latents.size(2);

    std::vector<torch::Tensor> x_list;
    x_list.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      torch::Tensor data = latents[i];
      torch::Tensor pos = latent_ids[i];

      torch::Tensor h_ids = pos.select(1, 1).to(torch::kInt64);
      torch::Tensor w_ids = pos.select(1, 2).to(torch::kInt64);

      // Use provided height/width directly — no torch::max() DtoH sync.
      int64_t h = height;
      int64_t w = width;

      torch::Tensor flat_ids = h_ids * w + w_ids;

      torch::Tensor out = torch::zeros({h * w, channels}, data.options());
      out.scatter_(0, flat_ids.unsqueeze(1).expand({-1, channels}), data);

      out = out.view({h, w, channels}).permute({2, 0, 1});
      x_list.push_back(out);
    }

    return torch::stack(x_list, 0);
  }

  // Main inference logic.
  torch::Tensor forward_impl(
      std::optional<std::vector<std::string>> prompt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_inference_steps = 4,
      float guidance_scale = 1.0f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> images = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    LOG(INFO) << "guidance_scale: " << guidance_scale;
    LOG(INFO) << "num_inference_steps: " << num_inference_steps;
    LOG(INFO) << "num_images_per_prompt: " << num_images_per_prompt;
    LOG(INFO) << "seed: " << seed.has_value();
    if (seed.has_value()) {
      LOG(INFO) << "seed value: " << seed.value();
    }

    // // 1、加载 Python 侧保存的 tensor，替换传入的 prompt_embeds
    // {
    //   device_ = options_.device();
    //   // 已经加载的是xllm侧的qwen3的输出
    //   // auto encoded_prompt_embeds_load =
    //   StateDictFromSafeTensor::load("/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/12_test_flux2_klein_acc/00_diffusers_save_qwen3_prompt_embeds_output_tensor/02_cpp_qwen3_before_reshape_prompt_embeds_save.safetensors");
    //   auto encoded_prompt_embeds_load =
    //   StateDictFromSafeTensor::load("/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/12_test_flux2_klein_acc/00_diffusers_save_qwen3_prompt_embeds_output_tensor/01_qwen3_before_reshape_prompt_embeds_save.safetensors");
    //   auto encoded_prompt_embeds_shape =torch::ones({1, 3, 512,
    //   4096},torch::kBFloat16); bool is_conv_out_weight_loaded_ = false;
    //   weight::load_weight(*encoded_prompt_embeds_load,"prompt_embeds_save",encoded_prompt_embeds_shape,is_conv_out_weight_loaded_);
    //   prompt_embeds =
    //   encoded_prompt_embeds_shape.to(device_).to(torch::kBFloat16); LOG(INFO)
    //   << "prompt_embeds dtype: "
    //             << prompt_embeds->scalar_type()
    //             << " shape: " << prompt_embeds->sizes();

    //   // torch::Tensor python_loaded_prompt_embeds;
    //   // torch::load(python_loaded_prompt_embeds,
    //   //
    //   "/export/home/weinan5/wangshuibin/21_vllm-omni/02_python_dump_tensor/01_qwen3_prompt_embeds_output_tensor/01_qwen3_before_reshape_prompt_embeds.pt");
    //   // prompt_embeds = python_loaded_prompt_embeds;

    //   // prompt_embeds = python_loaded_prompt_embeds.to(torch::kFloat32);

    //   // 2、加载 Python 侧保存的 tensor，替换传入的 negative_prompt_embeds
    //   // if (negative_prompt_embeds.has_value()) {
    //   //   torch::Tensor python_loaded_neg_embeds;
    //   //   torch::load(python_loaded_neg_embeds,
    //   // "/path/to/dump_flux2_tensor/negative_prompt_embeds.pt");
    //   //   negative_prompt_embeds =
    //   python_loaded_neg_embeds.to(torch::kFloat32);
    //   // }
    // }

    // torch::save(prompt_embeds.value().to(torch::kCPU),
    // "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/12_test_flux2_klein_acc/03_cpp_dump_qwen3_output_tensor/03_03_valid_cpp_qwen3_prompt_embedding_base64_in_dit.pt");

    // ── 1. Prepare text embeddings ──
    int64_t batch_size = prompt_embeds.value().size(0);
    int64_t total_batch_size = batch_size * num_images_per_prompt;

    int64_t num_channels = prompt_embeds.value().size(1);
    int64_t seq_len = prompt_embeds.value().size(2);
    int64_t hidden_dim = prompt_embeds.value().size(3);

    // (B, num_channels, seq_len, hidden_dim) → (B, seq_len, num_channels *
    // hidden_dim)
    auto prompt_embeds_value =
        prompt_embeds.value()
            .permute({0, 2, 1, 3})
            .reshape({batch_size, seq_len, num_channels * hidden_dim});

    prompt_embeds_value =
        prompt_embeds_value.repeat({1, num_images_per_prompt, 1});
    prompt_embeds_value = prompt_embeds_value.view(
        {batch_size * num_images_per_prompt, seq_len, -1});

    device_ = options_.device();

    torch::Tensor encoded_prompt_embeds = prompt_embeds_value;
    torch::Tensor text_ids = prepare_text_ids(prompt_embeds_value);
    encoded_prompt_embeds =
        encoded_prompt_embeds.to(device_).to(torch::kBFloat16);
    text_ids = text_ids.to(device_).to(torch::kLong);

    // ── Tensor dump 01: prompt_embeds after reshape ──
    torch::save(encoded_prompt_embeds.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "01_cpp_prompt_embeds_after_reshape.pt");
    // ── Tensor dump 02: text_ids ──
    torch::save(text_ids.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "02_cpp_text_ids.pt");

    // Classifier-free guidance: prepare negative prompt embeddings.
    bool do_cfg = (guidance_scale > 1.0f) && !is_distilled_;
    torch::Tensor neg_prompt_embeds;
    torch::Tensor neg_text_ids;
    if (do_cfg) {
      if (negative_prompt_embeds.has_value()) {
        int64_t neg_num_ch = negative_prompt_embeds.value().size(1);
        int64_t neg_seq_len = negative_prompt_embeds.value().size(2);
        int64_t neg_hidden_dim = negative_prompt_embeds.value().size(3);

        neg_prompt_embeds =
            negative_prompt_embeds.value()
                .permute({0, 2, 1, 3})
                .reshape(
                    {batch_size, neg_seq_len, neg_num_ch * neg_hidden_dim});
        neg_prompt_embeds =
            neg_prompt_embeds.repeat({1, num_images_per_prompt, 1});
        neg_prompt_embeds = neg_prompt_embeds.view(
            {batch_size * num_images_per_prompt, neg_seq_len, -1});
      } else {
        // Fall back: use the positive prompt embeds as negative (user should
        // provide pre-computed empty-string embeddings for proper CFG).
        neg_prompt_embeds = encoded_prompt_embeds.clone();
      }
      neg_text_ids = prepare_text_ids(neg_prompt_embeds);
      neg_prompt_embeds = neg_prompt_embeds.to(device_).to(torch::kBFloat16);
      neg_text_ids = neg_text_ids.to(device_).to(torch::kLong);
    }

    // ── 2. Process condition images ──
    std::vector<torch::Tensor> condition_images_list;
    if (images.has_value()) {
      auto input_images = images.value();
      if (input_images.dim() == 3) {
        input_images = input_images.unsqueeze(0);
      }

      for (int64_t i = 0; i < input_images.size(0); ++i) {
        auto img = input_images[i];
        flux2_image_processor_->check_image_input(img);
        int64_t image_width = img.size(-1);
        int64_t image_height = img.size(-2);

        if (image_width * image_height > 1024 * 1024) {
          img = flux2_image_processor_->resize_to_target_area(img, 1024 * 1024);
          image_width = img.size(-1);
          image_height = img.size(-2);
        }
        int64_t multiple_of = vae_scale_factor_ * 2;
        image_width = (image_width / multiple_of) * multiple_of;
        image_height = (image_height / multiple_of) * multiple_of;
        img =
            flux2_image_processor_->preprocess(img, image_height, image_width);
        condition_images_list.push_back(img);
      }
    }

    // ── 3. Prepare noise latents ──
    int64_t num_channels_latents = transformer_->in_channels() / 4;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);
    prepared_latents = prepared_latents.to(torch::kBFloat16);
    latent_image_ids = latent_image_ids.to(torch::kLong);
    LOG(INFO) << "seed.value(): " << seed.value();

    // ── Tensor dump 03: prepared_latents (initial noise)
    torch::save(prepared_latents.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "03_cpp_prepared_latents.pt");
    // ── Tensor dump 04: latent_image_ids
    torch::save(latent_image_ids.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "04_cpp_latent_image_ids.pt");

    torch::Tensor image_latents;
    torch::Tensor image_latent_ids;
    if (!condition_images_list.empty()) {
      std::tie(image_latents, image_latent_ids) =
          prepare_image_latents(condition_images_list,
                                total_batch_size,
                                seed.has_value() ? seed.value() : 42);
    }

    // ── 4. Prepare timesteps ──
    // Klein uses linspace(1.0, 1/N, N) as default sigmas.
    std::vector<float> new_sigmas;
    new_sigmas.reserve(num_inference_steps);
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.emplace_back(1.0f - static_cast<float>(i) /
                                         (num_inference_steps - 1) *
                                         (1.0f - 1.0f / num_inference_steps));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = compute_empirical_mu(image_seq_len, num_inference_steps);
    auto [timesteps, num_inference_steps_actual] = flux2_retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);

    // ── Tensor dump 05: timesteps ──
    torch::save(timesteps.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "05_cpp_timesteps.pt");

    // Klein passes guidance=None to the transformer (no embedded guidance).
    // No guidance tensor is constructed.

    // ── 5. Pre-compute rotary position embeddings ──
    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));

    torch::Tensor image_rotary_emb =
        torch::stack({rot_emb1, rot_emb2}, 0).to(options_.dtype());

    // ── Tensor dump 06: image_rotary_emb ──
    torch::save(image_rotary_emb.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "06_cpp_image_rotary_emb.pt");

    // Pre-compute latent dimensions for unpack (avoids DtoH sync in
    // unpack_latents_with_ids from torch.max().item()).
    int64_t latent_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t latent_width = 2 * (width / (vae_scale_factor_ * 2));

    // ── 6. Denoising loop ──
    DiTCache::get_instance().set_context(
        {num_inference_steps, num_single_layers_});
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);

      // Build model input: noise latents [+ optional image condition tokens].
      torch::Tensor latent_model_input = prepared_latents.to(options_.dtype());
      torch::Tensor latent_image_ids_input = latent_image_ids;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({prepared_latents, image_latents}, 1)
                                 .to(options_.dtype());
        latent_image_ids_input =
            torch::cat({latent_image_ids, image_latent_ids}, 1);
        auto [rot_emb1_img, rot_emb2_img] =
            pos_embed_->forward_cache(text_ids,
                                      latent_image_ids_input,
                                      height / (vae_scale_factor_ * 2),
                                      width / (vae_scale_factor_ * 2));
        image_rotary_emb =
            torch::stack({rot_emb1_img, rot_emb2_img}, 0).to(options_.dtype());
      }

      // ── Tensor dump: transformer forward inputs (step 0 only) ──
      if (i == 0) {
        torch::save(latent_model_input.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "A1_cpp_transformer_input_latent_model_input.pt");
        torch::save(encoded_prompt_embeds.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "A2_cpp_transformer_input_encoded_prompt_embeds.pt");
        torch::save(timestep.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "A3_cpp_transformer_input_timestep.pt");
        torch::save(image_rotary_emb.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "A4_cpp_transformer_input_image_rotary_emb.pt");
      }

      // Conditional pass — Klein passes no guidance (guidance=None).
      torch::Tensor noise_pred =
          transformer_->forward(latent_model_input,
                                encoded_prompt_embeds,
                                timestep,
                                /*guidance=*/torch::Tensor(),
                                image_rotary_emb,
                                /*step_idx=*/i);

      // Truncate to noise-latent tokens (exclude image condition tokens).
      if (image_latents.defined()) {
        noise_pred = noise_pred.narrow(1, 0, prepared_latents.size(1));
      }
      torch::Tensor before_scheduler_noise_pred = noise_pred.to(torch::kCPU);
      std::string save_path_1 =
          "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
          "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
          "01_noise_pred/09_" +
          std::to_string(i + 1) + "_before_scheduler_noise_pred.pt";
      torch::save(before_scheduler_noise_pred, save_path_1);
      // Unconditional pass (CFG).
      if (do_cfg) {
        torch::Tensor neg_noise_pred =
            transformer_->forward(latent_model_input,
                                  neg_prompt_embeds,
                                  timestep,
                                  /*guidance=*/torch::Tensor(),
                                  image_rotary_emb,
                                  /*step_idx=*/i);
        if (image_latents.defined()) {
          neg_noise_pred =
              neg_noise_pred.narrow(1, 0, prepared_latents.size(1));
        }
        // CFG formula: uncond + scale * (cond - uncond)
        noise_pred =
            neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred);
      }

      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);

      // ── Tensor dump 07/08: noise_pred and latents after scheduler (step 0
      // only) ──
      if (i == 0) {
        torch::save(noise_pred.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "07_cpp_step1_noise_pred.pt");
      }

      prepared_latents = prev_latents.detach();

      // ── Tensor dump: latents after scheduler step (step 0 only) ──
      if (i == 0) {
        torch::save(prepared_latents.to(torch::kCPU),
                    "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                    "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                    "08_cpp_step1_after_scheduler_latents.pt");
      }
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    torch::Tensor dit_output_prepared_latents =
        prepared_latents.to(torch::kCPU);
    torch::save(
        dit_output_prepared_latents,
        "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
        "12_test_flux2_klein_acc/09_cpp_flux2_klein_dit_and_output_tensor/"
        "09_02_dit_output_prepared_latents.pt");
    // ── 7. Decode latents ──
    // ── Tensor dump 09: dit final output (prepared_latents after all steps) ──
    torch::save(prepared_latents.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "09_cpp_dit_final_output.pt");

    // Use pre-computed latent_height/latent_width to avoid DtoH sync.
    torch::Tensor unpacked_latents = unpack_latents_with_ids(prepared_latents,
                                                             latent_image_ids,
                                                             latent_height / 2,
                                                             latent_width / 2);
    auto latents_bn_mean =
        vae_->get_bn_running_mean()
            .view({1, -1, 1, 1})
            .to(unpacked_latents.device(), unpacked_latents.dtype());
    auto latents_bn_std =
        torch::sqrt(vae_->get_bn_running_var().view({1, -1, 1, 1}) +
                    vae_->get_batch_norm_eps())
            .to(unpacked_latents.device(), unpacked_latents.dtype());
    unpacked_latents = unpacked_latents * latents_bn_std + latents_bn_mean;
    unpacked_latents = unpatchify_latents(unpacked_latents);

    // ── Tensor dump 10: VAE input (after unpack + BN + unpatchify) ──
    torch::save(unpacked_latents.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "10_cpp_vae_input.pt");

    // ── Timer: vae_->decode ──
    aclrtSynchronizeDevice();
    auto vae_t0 = std::chrono::steady_clock::now();

    torch::Tensor image = vae_->decode(unpacked_latents);

    aclrtSynchronizeDevice();
    auto vae_t1 = std::chrono::steady_clock::now();
    double vae_decode_ms =
        std::chrono::duration<double, std::milli>(vae_t1 - vae_t0).count();
    LOG(INFO) << "[Timer] vae_->decode cost: " << vae_decode_ms << " ms";
    torch::Tensor dit_image_vae = image.to(torch::kCPU);
    torch::save(
        dit_image_vae,
        "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
        "12_test_flux2_klein_acc/09_cpp_flux2_klein_dit_and_output_tensor/"
        "09_03_dit_image_vae.pt");
    // ── Tensor dump 11: VAE output ──
    torch::save(image.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "11_cpp_vae_output.pt");

    image = flux2_image_processor_->postprocess(image);

    torch::Tensor dit_image = image.to(torch::kCPU);
    torch::save(dit_image,
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/"
                "09_cpp_flux2_klein_dit_and_output_tensor/09_04_dit_image.pt");
    // ── Tensor dump 12: final image (after postprocess) ──
    torch::save(image.to(torch::kCPU),
                "/export/home/weinan5/wangshuibin/15_JD_push_xllm_flux2/"
                "12_test_flux2_klein_acc/02_cpp_dump_klein_dit_vae_tensor/"
                "12_cpp_final_image.pt");

    return image;
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  AutoencoderKLFlux2 vae_{nullptr};
  Flux2ImageProcessor flux2_image_processor_{nullptr};
  Flux2DiTModel transformer_{nullptr};
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int32_t tokenizer_max_length_;
  int32_t default_sample_size_;
  int32_t vae_scale_factor_;
  int64_t num_single_layers_;
  bool is_distilled_;
  Flux2PosEmbed pos_embed_{nullptr};
};
TORCH_MODULE(Flux2KleinPipeline);

REGISTER_DIT_MODEL(flux2_klein, Flux2KleinPipeline);
REGISTER_DIT_MODEL(Flux2KleinPipeline, Flux2KleinPipeline);
};  // namespace xllm

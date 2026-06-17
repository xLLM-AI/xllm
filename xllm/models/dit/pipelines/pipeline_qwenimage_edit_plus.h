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

#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "core/common/global_flags.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/autoencoders/autoencoder_kl_qwenimage.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_qwen_image.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#include "processors/qwen2_vl_image_processor.h"
#include "util/tensor_helper.h"

#define CONDITION_IMAGE_SIZE 147456
#define VAE_IMAGE_SIZE 1048576

namespace xllm {
class QwenImageEditPlusPipelineImpl : public torch::nn::Module {
 public:
  QwenImageEditPlusPipelineImpl(const DiTModelContext& context)
      : parallel_args_(context.get_parallel_args()),
        vae_model_args_(context.get_model_args("vae")) {
    options_ = context.get_tensor_options();
    dtype_ = options_.dtype().toScalarType();
    device_ = options_.device();
    LOG(INFO) << "model info " << dtype_ << " ; " << options_.device();

    in_channels_ = context.get_model_args("transformer").in_channels();
    num_layers_ = context.get_model_args("transformer").num_layers();

    vae_scale_factor_ = static_cast<int64_t>(
        std::pow(2, vae_model_args_.temperal_downsample().size()));
    latent_channels_ = vae_model_args_.z_dim();
    tokenizer_max_length_ = 1024;

    prompt_template_encode_ =
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how "
        "the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while "
        "maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
        "assistant\n";
    prompt_template_encode_start_idx_ = 64;
    default_sample_size_ = 128;

    vae_ = AutoencoderKLQwenImage(context.get_model_context("vae"));
    transformer_ = QwenImageTransformer2DModel(
        context.get_model_context("transformer"), parallel_args_);
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));

    vae_image_processor_ =
        xllm::VAEImageProcessor(context.get_model_context("vae"),
                                /*do_resize=*/true,
                                /*do_normalize=*/true,
                                /*do_binarize=*/false,
                                /*do_convert_rgb=*/false,
                                /*do_convert_grayscale=*/false,
                                latent_channels_,
                                /*scale_factor=*/vae_scale_factor_ * 2);

    register_module("vae", vae_);
    register_module("scheduler", scheduler_);
    register_module("transformer", transformer_);
    register_module("vae_image_processor", vae_image_processor_);
  }

  std::vector<torch::Tensor> _extract_masked_hidden(
      const torch::Tensor& hidden_states,
      const torch::Tensor& mask) {
    auto bool_mask = mask.to(torch::kBool);
    auto valid_lengths = bool_mask.sum(1);

    auto valid_lengths_cpu = valid_lengths.to(torch::kCPU).contiguous();

    std::vector<int64_t> lengths_list;
    lengths_list.reserve(valid_lengths_cpu.numel());

    int64_t* lengths_ptr = valid_lengths_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < valid_lengths_cpu.numel(); ++i) {
      lengths_list.push_back(lengths_ptr[i]);
    }

    auto selected = hidden_states.index({bool_mask});
    auto split_result = torch::split(selected, lengths_list, 0);

    return std::vector<torch::Tensor>(split_result.begin(), split_result.end());
  }

  std::pair<torch::Tensor, torch::Tensor> _get_qwen_prompt_embeds(
      const std::vector<std::string>& prompt,
      const std::vector<torch::Tensor>& image,
      torch::TensorOptions& options) {
    std::string img_prompt_template =
        "Picture {}: <|vision_start|><|image_pad|><|vision_end|>";
    std::string base_img_prompt = "";

    torch::Tensor prompt_embeds;
    torch::Tensor prompt_embeds_mask;
    return std::make_pair(prompt_embeds, prompt_embeds_mask);
  }

  void _encode_prompt(const std::vector<torch::Tensor>& image,
                      const std::vector<std::string>& prompt,
                      torch::Tensor& prompt_embeds,
                      torch::Tensor& prompt_embeds_mask,
                      torch::TensorOptions& options,
                      int64_t num_images_per_prompt = 1,
                      int64_t max_sequence_length = 1024) {
    int64_t batch_size = prompt_embeds.defined() ? prompt_embeds.size(0) : 1;
    if (!prompt_embeds.defined()) {
      std::tie(prompt_embeds, prompt_embeds_mask) =
          _get_qwen_prompt_embeds(prompt, image, options);
    }

    CHECK(prompt_embeds.defined())
        << "currently, the prompt input is not supported for qwen image, "
        << "expected a valid prompt_embeds input, but got empty tensor ";

    auto seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    if (prompt_embeds_mask.defined()) {
      prompt_embeds_mask =
          prompt_embeds_mask.repeat({1, num_images_per_prompt, 1});
      prompt_embeds_mask = prompt_embeds_mask.view(
          {batch_size * num_images_per_prompt, seq_len});
    } else {
      prompt_embeds_mask = torch::ones(
          {prompt_embeds.size(0), prompt_embeds.size(1)},
          torch::TensorOptions().device(device_).dtype(torch::kLong));
    }
  }

  torch::Tensor _retrieve_latents(const AutoencoderKLOutput& encoder_output,
                                  int64_t seed = 42,
                                  const std::string& sample_mode = "sample") {
    if (sample_mode == "sample") {
      return encoder_output.latent_dist.sample(seed);
    } else if (sample_mode == "argmax") {
      return encoder_output.latent_dist.mode();
    } else {
      CHECK(false)
          << "sample_mode is expected to be 'sample' or 'argmax', but get: "
          << sample_mode;
      return torch::Tensor();
    }
  }

  torch::Tensor _pack_latents(torch::Tensor latents,
                              int64_t batch_size,
                              int64_t num_channels_latents,
                              int64_t height,
                              int64_t width) {
    latents = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents = latents.permute({0, 2, 4, 1, 3, 5});
    latents = latents.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return latents;
  }

  torch::Tensor _unpack_latents(torch::Tensor latents,
                                int64_t height,
                                int64_t width,
                                int64_t vae_scale_factor) {
    auto sizes = latents.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_patches = sizes[1];
    int64_t channels = sizes[2];

    height = 2 * (height / (vae_scale_factor * 2));
    width = 2 * (width / (vae_scale_factor * 2));

    latents =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents = latents.permute({0, 3, 1, 4, 2, 5});
    latents =
        latents.reshape({batch_size, channels / (2 * 2), 1, height, width});

    return latents;
  }

  torch::Tensor _encode_vae_image(torch::Tensor image,
                                  int64_t seed,
                                  torch::Device device) {
    auto image_latents = _retrieve_latents(vae_->encode(image), seed, "argmax");
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = latents_std.view({1, latent_channels_, 1, 1, 1})
                      .to(device, image_latents.dtype());
    image_latents = (image_latents - latents_mean) / latents_std;
    return image_latents;
  }

  std::pair<torch::Tensor, torch::Tensor> _prepare_latents(
      const std::vector<torch::Tensor>& images,
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      torch::TensorOptions& options,
      int64_t seed,
      torch::Tensor latents = torch::Tensor()) {
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    std::vector<int64_t> shape = {
        batch_size, 1, num_channels_latents, height, width};

    torch::Tensor image_latents;
    if (!images.empty()) {
      std::vector<torch::Tensor> all_image_latents;
      for (const auto& image : images) {
        auto current_image = image.to(options);
        torch::Tensor current_image_latents;

        if (current_image.size(1) != latent_channels_) {
          current_image_latents =
              _encode_vae_image(current_image, seed, device_);
        } else {
          current_image_latents = current_image;
        }

        current_image_latents = torch::cat({current_image_latents}, 0);
        int64_t image_latent_height = current_image_latents.size(3);
        int64_t image_latent_width = current_image_latents.size(4);

        current_image_latents = _pack_latents(current_image_latents,
                                              batch_size,
                                              num_channels_latents,
                                              image_latent_height,
                                              image_latent_width);
        all_image_latents.emplace_back(current_image_latents);
      }

      image_latents = torch::cat(all_image_latents, 1);
    }

    if (!latents.defined()) {
      latents = xllm::dit::randn_tensor(shape, seed, options);
      latents = _pack_latents(
          latents, batch_size, num_channels_latents, height, width);
    } else {
      latents = latents.to(options);
    }
    return std::make_pair(latents, image_latents);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& generation_params = input.generation_params;
    auto height = generation_params.height;
    auto width = generation_params.width;
    auto true_cfg_scale = generation_params.true_cfg_scale;
    auto num_inference_steps = generation_params.num_inference_steps;
    DiTCache::get_instance().set_infer_steps(num_inference_steps);
    DiTCache::get_instance().set_num_blocks(num_layers_);
    auto max_sequence_length = generation_params.max_sequence_length;
    auto seed = generation_params.seed >= 0 ? generation_params.seed : 42;

    auto prompts = input.prompts;
    auto negative_prompts = input.negative_prompts;
    auto latents = input.latents;
    if (latents.defined()) {
      latents = latents.to(options_.device(), dtype_);
    }

    auto prompt_embeds = input.prompt_embeds;
    if (prompt_embeds.defined()) {
      prompt_embeds = prompt_embeds.to(options_.device(), dtype_);
    }
    torch::Tensor prompt_embeds_mask;

    auto negative_prompt_embeds = input.negative_prompt_embeds;
    if (negative_prompt_embeds.defined()) {
      negative_prompt_embeds =
          negative_prompt_embeds.to(options_.device(), dtype_);
    }
    torch::Tensor negative_prompt_embeds_mask;

    if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
      input.debug_print();
    }

    if (input.images_list.empty()) {
      LOG(FATAL) << "QwenImageEditPlus pipeline expected to have "
                 << "image inputs in images_list";
    }

    std::vector<torch::Tensor> image_list;
    image_list.reserve(input.images_list.size());

    for (const auto& images : input.images_list) {
      auto img = images.to(options_.device(), dtype_);
      if (img.dim() != 4) {
        LOG(ERROR)
            << "image inputs are expected to be a 4 dim tensor, but got: "
            << img.dim() << "d tensor";
        continue;
      }
      if (img.size(0) > 1) {
        LOG(ERROR) << "currently QwenImageEdit doesn't support batch inference"
                   << "batch size: " << img.size(0);
      }
      image_list.emplace_back(img[0]);
    }

    if (image_list.empty()) {
      LOG(FATAL) << "No valid images found in images_list. ";
    }

    double height_size = static_cast<double>(image_list[0].size(1));
    double width_size = static_cast<double>(image_list[0].size(2));
    int64_t num_images_per_prompt = 1;

    double aspect_ratio = width_size / height_size;
    auto [calculated_width, calculated_height] =
        xllm::dit::calculate_dimensions(1024 * 1024, aspect_ratio);

    height = (height == 0) ? calculated_height : height;
    width = (width == 0) ? calculated_width : width;

    int64_t multiple_of = vae_scale_factor_ * 2;
    width = (width / multiple_of) * multiple_of;
    height = (height / multiple_of) * multiple_of;

    current_timestep_ = torch::Tensor();

    int64_t batch_size = prompts.size();
    std::vector<torch::Tensor> condition_images;
    std::vector<torch::Tensor> vae_images;
    std::vector<std::pair<int64_t, int64_t>> condition_image_sizes;
    std::vector<std::pair<int64_t, int64_t>> vae_image_sizes;
    if (!image_list.empty() && image_list[0].size(0) != latent_channels_) {
      for (size_t i = 0; i < image_list.size(); i++) {
        aspect_ratio =
            static_cast<double>(image_list[i].size(2)) / image_list[i].size(1);
        auto [condition_width, condition_height] =
            xllm::dit::calculate_dimensions(CONDITION_IMAGE_SIZE, aspect_ratio);
        auto [vae_width, vae_height] =
            xllm::dit::calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio);
        condition_image_sizes.push_back({condition_width, condition_height});
        vae_image_sizes.push_back({vae_width, vae_height});
        auto img = image_list[i].unsqueeze(0);
        auto condition_img = vae_image_processor_->resize(
            img,
            {condition_height, condition_width},
            /*resample=*/3,  // BICUBIC (approximate LANCZOS)
            /*antialias=*/true);
        auto vae_img =
            vae_image_processor_->preprocess(img, vae_height, vae_width)
                .unsqueeze(2);

        condition_images.push_back(condition_img);
        vae_images.push_back(vae_img);
      }
    }

    bool has_neg_prompt =
        negative_prompts.size() > 0 || negative_prompt_embeds.defined();

    bool do_true_cfg = (true_cfg_scale > 1.0) && has_neg_prompt;
    // inplace update prompt_embeds and prompt_embeds_mask
    _encode_prompt(condition_images,
                   prompts,
                   prompt_embeds,
                   prompt_embeds_mask,
                   options_,
                   num_images_per_prompt,
                   max_sequence_length);

    if (do_true_cfg) {
      // inplace update negative_prompt_embeds and negative_prompt_embeds_mask
      _encode_prompt(condition_images,
                     negative_prompts,
                     negative_prompt_embeds,
                     negative_prompt_embeds_mask,
                     options_,
                     num_images_per_prompt,
                     max_sequence_length);
    }

    int64_t num_channels_latents = in_channels_ / 4;
    torch::Tensor final_latents;
    torch::Tensor image_latents;

    std::tie(final_latents, image_latents) =
        _prepare_latents(vae_images,
                         batch_size * num_images_per_prompt,
                         num_channels_latents,
                         height,
                         width,
                         options_,
                         seed,
                         latents);

    std::vector<std::vector<int64_t>> main_shape = {
        {1, height / vae_scale_factor_ / 2, width / vae_scale_factor_ / 2}};

    for (const auto& [vae_width, vae_height] : vae_image_sizes) {
      main_shape.push_back({1,
                            vae_height / vae_scale_factor_ / 2,
                            vae_width / vae_scale_factor_ / 2});
    }

    std::vector<float> new_sigmas;
    double start = 1.0;
    double end = 1.0 / static_cast<double>(num_inference_steps);
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      double v = start + (end - start) * static_cast<double>(i) /
                             (num_inference_steps - 1);
      new_sigmas.push_back(static_cast<float>(v));
    }

    int64_t image_seq_len = final_latents.size(1);
    float mu = xllm::dit::calculate_shift(image_seq_len,
                                          scheduler_->base_image_seq_len(),
                                          scheduler_->max_image_seq_len(),
                                          scheduler_->base_shift(),
                                          scheduler_->max_shift());
    auto [timesteps, num_inference_steps_actual] =
        xllm::dit::retrieve_timesteps(
            scheduler_, num_inference_steps, device_, new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));

    num_timesteps_ = timesteps.size(0);
    torch::Tensor txt_seq_lens;
    if (prompt_embeds_mask.defined()) {
      txt_seq_lens = prompt_embeds_mask.sum(1);
    }
    torch::Tensor negative_txt_seq_lens;
    if (do_true_cfg && negative_prompt_embeds_mask.defined()) {
      negative_txt_seq_lens = negative_prompt_embeds_mask.sum(1);
    }
    scheduler_->set_begin_index(0);
    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      auto t = timesteps[i];
      current_timestep_ = t;

      auto latent_model_input = final_latents;
      if (image_latents.defined()) {
        latent_model_input = torch::cat({final_latents, image_latents}, 1);
      }

      auto timestep_expanded =
          t.expand({final_latents.size(0)}).to(final_latents.dtype());

      torch::Tensor noise_pred;
      torch::Tensor neg_noise_pred;
      torch::Tensor pos_neg_noise_preds;
      if (::xllm::ParallelConfig::get_instance().cfg_size() == 2 &&
          do_true_cfg) {
        auto rank = parallel_args_.dit_cfg_group_->rank();
        if (rank == 0) {
          noise_pred = transformer_->forward(latent_model_input,
                                             prompt_embeds,
                                             prompt_embeds_mask,
                                             timestep_expanded / 1000.0,
                                             main_shape,
                                             txt_seq_lens,
                                             /*use_cfg=*/false,
                                             /*step_index=*/i);
          noise_pred = noise_pred.slice(1, 0, final_latents.size(1));
          pos_neg_noise_preds =
              xllm::parallel_state::gather(noise_pred,
                                           parallel_args_.dit_cfg_group_,
                                           /*dim=*/0);
        } else {
          neg_noise_pred = transformer_->forward(latent_model_input,
                                                 negative_prompt_embeds,
                                                 negative_prompt_embeds_mask,
                                                 timestep_expanded / 1000.0,
                                                 main_shape,
                                                 negative_txt_seq_lens,
                                                 /*use_cfg=*/true,
                                                 /*step_index=*/i);

          neg_noise_pred = neg_noise_pred.slice(1, 0, final_latents.size(1));
          pos_neg_noise_preds =
              xllm::parallel_state::gather(neg_noise_pred,
                                           parallel_args_.dit_cfg_group_,
                                           /*dim=*/0);
        }
        auto noise_preds = torch::chunk(pos_neg_noise_preds, 2, 0);
        auto comb_pred =
            noise_preds[1] + true_cfg_scale * (noise_preds[0] - noise_preds[1]);
        auto cond_norm = torch::norm(noise_preds[0], 2, -1, true);
        auto noise_norm = torch::norm(comb_pred, 2, -1, true);
        noise_pred = comb_pred * (cond_norm / noise_norm);

      } else {
        noise_pred = transformer_->forward(latent_model_input,
                                           prompt_embeds,
                                           prompt_embeds_mask,
                                           timestep_expanded / 1000.0,
                                           main_shape,
                                           txt_seq_lens,
                                           /*use_cfg=*/false,
                                           /*step_index=*/i);
        noise_pred = noise_pred.slice(1, 0, final_latents.size(1));
        if (do_true_cfg) {
          neg_noise_pred = transformer_->forward(latent_model_input,
                                                 negative_prompt_embeds,
                                                 negative_prompt_embeds_mask,
                                                 timestep_expanded / 1000.0,
                                                 main_shape,
                                                 negative_txt_seq_lens,
                                                 /*use_cfg=*/true,
                                                 /*step_index=*/i);

          neg_noise_pred = neg_noise_pred.slice(1, 0, final_latents.size(1));

          auto comb_pred =
              neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred);
          auto cond_norm = torch::norm(noise_pred, 2, -1, true);
          auto noise_norm = torch::norm(comb_pred, 2, -1, true);
          noise_pred = comb_pred * (cond_norm / noise_norm);
        }
      }

      auto latents_dtype = final_latents.dtype();
      final_latents = scheduler_->step(noise_pred, t, final_latents);
      if (final_latents.dtype() != latents_dtype) {
        final_latents = final_latents.to(latents_dtype);
      }
    }
    current_timestep_ = torch::Tensor();

    torch::Tensor output_image;

    auto unpacked_latents =
        _unpack_latents(final_latents, height, width, vae_scale_factor_)
            .to(dtype_);
    auto latents_mean =
        torch::tensor(vae_model_args_.latents_mean(), torch::kDouble);
    latents_mean = latents_mean.view({1, latent_channels_, 1, 1, 1})
                       .to(device_, image_latents.dtype());
    auto latents_std =
        torch::tensor(vae_model_args_.latents_std(), torch::kDouble);
    latents_std = 1.0 / latents_std.view({1, latent_channels_, 1, 1, 1})
                            .to(device_, image_latents.dtype());

    unpacked_latents = unpacked_latents / latents_std + latents_mean;
    output_image = vae_->decode(unpacked_latents).sample.squeeze(2);
    output_image = vae_image_processor_->postprocess(output_image);
    auto output = std::vector<torch::Tensor>{{output_image}};
    DiTForwardOutput out;
    out.tensors = output;
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "QwenImageEditPlusPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto processor_loader = loader->take_component_loader("processor");
    LOG(INFO) << " QwenImageEditplus model components loaded, start to load "
                 "weights to sub models";

    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device(), dtype_);
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device(), dtype_);
  }

 private:
  // members from former QwenImagePipelineBaseImpl base class
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  AutoencoderKLQwenImage vae_{nullptr};
  xllm::VAEImageProcessor vae_image_processor_{nullptr};
  std::unique_ptr<Qwen2VLImageProcessor> qwen_image_processor_{nullptr};
  QwenImageTransformer2DModel transformer_{nullptr};
  std::unique_ptr<Tokenizer> qwen_tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_;
  xllm::FlowMatchEulerDiscreteScheduler scheduler_{nullptr};

  int64_t vae_scale_factor_;
  int64_t latent_channels_;
  int64_t tokenizer_max_length_;
  int64_t prompt_template_encode_start_idx_;
  int64_t default_sample_size_;
  int64_t in_channels_;
  int64_t num_timesteps_;
  int64_t num_layers_;
  const ParallelArgs parallel_args_;
  torch::Tensor current_timestep_;
  string prompt_template_encode_;
  const ModelArgs& vae_model_args_;
};

REGISTER_MODEL_ARGS(Qwen2Tokenizer, [&] {});
TORCH_MODULE(QwenImageEditPlusPipeline);

REGISTER_DIT_MODEL(QwenImageEditPlusPipeline, QwenImageEditPlusPipeline);
}  // namespace xllm

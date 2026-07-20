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

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_wan.h"
#include "models/dit/processors/vae_image_processor.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_joyimage_edit_plus.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#include "util/tensor_helper.h"

namespace xllm {

// NPU build: text encoding (Qwen3-VL) runs outside this pipeline; callers pass
// precomputed `prompt_embeds` (and `negative_prompt_embeds` for CFG). The
// in-pipeline VLM path is intentionally not compiled for NPU because the NPU
// Qwen3 language model returns only post-norm hidden states, whereas JoyImage
// requires the pre-norm last-layer output.
class JoyImageEditPlusPipelineImpl : public torch::nn::Module {
 public:
  JoyImageEditPlusPipelineImpl(const DiTModelContext& context)
      : context_(context), vae_model_args_(context.get_model_args("vae")) {
    options_ = context.get_tensor_options();
    dtype_ = options_.dtype().toScalarType();
    device_ = options_.device();

    in_channels_ = context.get_model_args("transformer").in_channels();
    num_layers_ = context.get_model_args("transformer").num_layers();
    auto patch = context.get_model_args("transformer").wan_patch_size();
    patch_t_ = patch[0];
    patch_h_ = patch[1];
    patch_w_ = patch[2];

    latent_channels_ = vae_model_args_.z_dim();
    vae_scale_factor_spatial_ = vae_model_args_.vae_scale_factor_spatial();
    if (vae_scale_factor_spatial_ <= 0) vae_scale_factor_spatial_ = 8;

    vae_ = AutoencoderKLWan(context.get_model_context("vae"));
    transformer_ = joyimage::JoyImageEditPlusTransformer3DModel(
        context.get_model_context("transformer"));
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
                                /*scale_factor=*/vae_scale_factor_spatial_);

    register_module("vae", vae_);
    register_module("scheduler", scheduler_);
    register_module("transformer", transformer_);
    register_module("vae_image_processor", vae_image_processor_);

    latents_mean_ = vae_model_args_.latents_mean();
    latents_std_ = vae_model_args_.latents_std();
  }

  torch::Tensor latents_mean_tensor(const torch::Tensor& ref) const {
    return torch::tensor(latents_mean_, torch::kFloat32)
        .view({1, latent_channels_, 1, 1, 1})
        .to(ref.device(), ref.dtype());
  }
  torch::Tensor latents_std_tensor(const torch::Tensor& ref) const {
    return torch::tensor(latents_std_, torch::kFloat32)
        .view({1, latent_channels_, 1, 1, 1})
        .to(ref.device(), ref.dtype());
  }

  // Patchify a [C, T, H, W] latent into [num_patches, C, pt, ph, pw] and
  // return {patches, (lt, lh, lw)}.
  std::pair<torch::Tensor, std::array<int64_t, 3>> patchify(
      const torch::Tensor& item) {
    int64_t c = item.size(0), t = item.size(1), h = item.size(2),
            w = item.size(3);
    int64_t lt = t / patch_t_, lh = h / patch_h_, lw = w / patch_w_;
    auto p = item.reshape({c, lt, patch_t_, lh, patch_h_, lw, patch_w_});
    p = p.permute({1, 3, 5, 0, 2, 4, 6})
            .reshape({-1, c, patch_t_, patch_h_, patch_w_});
    return {p, {lt, lh, lw}};
  }

  // Build the 6D padded latent tensor: target noise + reference latents.
  // Returns {padded_latents[B,N,C,pt,ph,pw], target_mask[B,N],
  //          shape_list (per-sample list of (t,h,w))}.
  std::tuple<torch::Tensor,
             torch::Tensor,
             std::vector<std::vector<std::array<int64_t, 3>>>>
  prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      const std::vector<std::vector<torch::Tensor>>& reference_images,
      const torch::Tensor& provided_latents) {
    std::vector<torch::Tensor> all_patches;
    std::vector<torch::Tensor> all_target_masks;
    std::vector<std::vector<std::array<int64_t, 3>>> all_shapes;
    int64_t max_patches = 0;

    int64_t h_target = height / vae_scale_factor_spatial_;
    int64_t w_target = width / vae_scale_factor_spatial_;

    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<torch::Tensor> items;
      // Target noise: [C, 1, h', w'].
      torch::Tensor noise;
      if (provided_latents.defined()) {
        noise = provided_latents[b].to(device_, dtype_);
      } else {
        noise = xllm::dit::randn_tensor(
            {num_channels_latents, 1, h_target, w_target}, seed + b, options_);
      }
      items.push_back(noise);

      // References: VAE-encode each, normalize, squeeze to [C, 1, h', w'].
      if (b < static_cast<int64_t>(reference_images.size())) {
        for (const auto& ref_img : reference_images[b]) {
          auto ref = ref_img.to(device_, dtype_);
          if (ref.dim() == 4) ref = ref.unsqueeze(2);  // [B,C,H,W]->[B,C,1,H,W]
          auto lat = vae_->encode(ref.to(dtype_)).latent_dist.mode();
          lat = lat.to(dtype_);
          lat = (lat - latents_mean_tensor(lat)) / latents_std_tensor(lat);
          items.push_back(lat.squeeze(0));  // [C, 1, h', w']
        }
      }

      std::vector<torch::Tensor> sample_patches;
      std::vector<torch::Tensor> sample_masks;
      std::vector<std::array<int64_t, 3>> sample_shapes;
      for (size_t j = 0; j < items.size(); ++j) {
        auto pr = patchify(items[j]);
        sample_shapes.push_back(pr.second);
        sample_patches.push_back(pr.first);
        auto n = pr.first.size(0);
        sample_masks.push_back(torch::full(
            {n},
            /*value=*/(j == 0),
            torch::TensorOptions().device(device_).dtype(torch::kBool)));
      }
      auto combined = torch::cat(sample_patches, 0);
      auto combined_mask = torch::cat(sample_masks, 0);
      all_patches.push_back(combined);
      all_target_masks.push_back(combined_mask);
      all_shapes.push_back(sample_shapes);
      max_patches = std::max(max_patches, combined.size(0));
    }

    auto padded = torch::zeros({batch_size,
                                max_patches,
                                num_channels_latents,
                                patch_t_,
                                patch_h_,
                                patch_w_},
                               options_);
    auto target_mask = torch::zeros(
        {batch_size, max_patches},
        torch::TensorOptions().device(device_).dtype(torch::kBool));
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t n = all_patches[b].size(0);
      padded.index_put_({b, torch::indexing::Slice(0, n)}, all_patches[b]);
      target_mask.index_put_({b, torch::indexing::Slice(0, n)},
                             all_target_masks[b]);
    }
    return std::make_tuple(padded, target_mask, all_shapes);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& gp = input.generation_params;
    int64_t num_inference_steps = gp.num_inference_steps;
    double guidance_scale =
        gp.true_cfg_scale > 0 ? gp.true_cfg_scale : gp.guidance_scale;
    int64_t seed = gp.seed >= 0 ? gp.seed : 42;

    // Collect reference images (one sample per batch entry).
    std::vector<torch::Tensor> raw_images;
    if (!input.images_list.empty()) {
      raw_images = input.images_list;
    } else if (input.images.defined()) {
      raw_images.push_back(input.images);
    } else {
      LOG(FATAL) << "JoyImageEditPlus requires reference images";
    }

    int64_t batch_size = input.batch_size;
    if (batch_size <= 0) {
      if (input.prompt_embeds.defined()) {
        batch_size = input.prompt_embeds.size(0);
      } else if (!input.prompts.empty()) {
        batch_size = static_cast<int64_t>(input.prompts.size());
      } else {
        batch_size = raw_images[0].dim() == 4 ? raw_images[0].size(0) : 1;
      }
    }

    // Determine output resolution from the last reference image if unset.
    int64_t height = gp.height;
    int64_t width = gp.width;
    if (height <= 0 || width <= 0) {
      const auto& last = raw_images.back();
      int64_t ih = last.size(last.dim() - 2);
      int64_t iw = last.size(last.dim() - 1);
      auto hw = joyimage_bucket(ih, iw);
      height = hw.first;
      width = hw.second;
    }
    height = (height / vae_scale_factor_spatial_) * vae_scale_factor_spatial_;
    width = (width / vae_scale_factor_spatial_) * vae_scale_factor_spatial_;

    // Reference images per sample, in two forms:
    //  - vae_refs: VAE-preprocessed to [-1,1] (for latent encoding), each
    //    bucket-resized to its own aspect bucket.
    std::vector<std::vector<torch::Tensor>> vae_refs(batch_size);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (const auto& imgs : raw_images) {
        auto img = imgs.dim() == 4 ? imgs[b] : imgs;  // [C,H,W]
        int64_t ih = img.size(1), iw = img.size(2);
        auto hw = joyimage_bucket(ih, iw);
        auto img4 = img.unsqueeze(0).to(device_);
        vae_refs[b].push_back(vae_image_processor_->preprocess(
            img4, hw.first, hw.second, /*resize_mode=*/"default"));
      }
    }
    // Prompt embeddings are produced by a separate Qwen3-VL embedding service
    // (two-stage serving) and passed in as `prompt_embeds` /
    // `negative_prompt_embeds`, mirroring the Qwen-Image-Edit NPU path. A
    // ones-mask is derived per sequence; CFG pads negatives/positives to equal
    // length (padded positions get a zero mask).
    bool do_cfg = guidance_scale > 1.0;
    CHECK(input.prompt_embeds.defined())
        << "JoyImageEditPlus (NPU) requires precomputed `prompt_embeds` from "
           "the Qwen3-VL embedding service.";
    torch::Tensor prompt_embeds =
        input.prompt_embeds.to(options_.device(), dtype_);
    torch::Tensor prompt_embeds_mask =
        torch::ones({prompt_embeds.size(0), prompt_embeds.size(1)},
                    torch::TensorOptions().device(device_).dtype(torch::kLong));

    torch::Tensor neg_embeds, neg_embeds_mask;
    if (do_cfg) {
      CHECK(input.negative_prompt_embeds.defined())
          << "JoyImageEditPlus CFG (guidance_scale > 1) requires precomputed "
             "`negative_prompt_embeds`.";
      neg_embeds = input.negative_prompt_embeds.to(options_.device(), dtype_);
      neg_embeds_mask = torch::ones(
          {neg_embeds.size(0), neg_embeds.size(1)},
          torch::TensorOptions().device(device_).dtype(torch::kLong));
      // Pad/concat [negative, positive] to equal sequence length.
      int64_t max_l = std::max(prompt_embeds.size(1), neg_embeds.size(1));
      prompt_embeds = pad_seq(prompt_embeds, max_l);
      neg_embeds = pad_seq(neg_embeds, max_l);
      prompt_embeds_mask = pad_seq(prompt_embeds_mask, max_l);
      neg_embeds_mask = pad_seq(neg_embeds_mask, max_l);
    }

    // Latents.
    int64_t num_channels_latents = in_channels_;
    auto lp = prepare_latents(batch_size,
                              num_channels_latents,
                              height,
                              width,
                              seed,
                              vae_refs,
                              input.latents);
    auto latents = std::get<0>(lp);
    auto target_mask = std::get<1>(lp);
    auto shape_list = std::get<2>(lp);
    auto clean_backup = latents.clone();

    // Timesteps (static shift; no dynamic shifting for Joy).
    scheduler_->set_timesteps(num_inference_steps, device_);
    scheduler_->set_begin_index(0);
    auto timesteps = scheduler_->timesteps();

    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      auto t = timesteps[i];
      // Restore reference patches.
      latents.index_put_({~target_mask}, clean_backup.index({~target_mask}));

      torch::Tensor noise_pred;
      if (do_cfg) {
        auto model_in = torch::cat({latents, latents}, 0);
        auto t_expand = t.repeat({model_in.size(0)});
        auto dbl_shape = shape_list;
        dbl_shape.insert(dbl_shape.end(), shape_list.begin(), shape_list.end());
        auto embeds = torch::cat({neg_embeds, prompt_embeds}, 0);
        auto mask = torch::cat({neg_embeds_mask, prompt_embeds_mask}, 0);
        auto pred =
            transformer_->forward(model_in, t_expand, embeds, mask, dbl_shape);
        auto chunks = pred.chunk(2, 0);
        auto uncond = chunks[0];
        auto cond = chunks[1];
        auto comb = uncond + guidance_scale * (cond - uncond);
        // Norm-rescale (diffusers): comb * (||cond|| / ||comb||) over channel
        // dim (2) of the 6D [B, N, C, pt, ph, pw] prediction.
        auto cond_norm =
            torch::norm(cond, 2, std::vector<int64_t>{2}, /*keepdim=*/true);
        auto noise_norm =
            torch::norm(comb, 2, std::vector<int64_t>{2}, /*keepdim=*/true);
        noise_pred = comb * (cond_norm / noise_norm.clamp_min(1e-6));
      } else {
        auto t_expand = t.repeat({batch_size});
        noise_pred = transformer_->forward(
            latents, t_expand, prompt_embeds, prompt_embeds_mask, shape_list);
      }

      latents = scheduler_->step(noise_pred, t, latents).to(latents.dtype());
    }

    // Restore refs and decode target patches per sample.
    latents.index_put_({~target_mask}, clean_backup.index({~target_mask}));
    std::vector<torch::Tensor> images;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto thw = shape_list[b][0];
      int64_t lt = thw[0], lh = thw[1], lw = thw[2];
      int64_t target_len = lt * lh * lw;
      auto patches = latents[b].slice(0, 0, target_len);  // [len, C, pt,ph,pw]
      int64_t c = patches.size(1);
      auto vid = patches.reshape({lt, lh, lw, c, patch_t_, patch_h_, patch_w_});
      vid = vid.permute({3, 0, 4, 1, 5, 2, 6})
                .reshape({1, c, lt * patch_t_, lh * patch_h_, lw * patch_w_});
      vid = vid * latents_std_tensor(vid) + latents_mean_tensor(vid);
      auto img = vae_->decode(vid.to(dtype_)).sample;       // [1,C,1,H,W]
      img = img.to(torch::kFloat32).squeeze(0).squeeze(1);  // [C,H,W]
      images.push_back(img.unsqueeze(0));
    }
    auto image = torch::cat(images, 0);
    image = vae_image_processor_->postprocess(image);

    DiTForwardOutput out;
    out.tensors = torch::chunk(image, batch_size, 0);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "JoyImageEditPlusPipeline loading from "
              << loader->model_root_path();
    // Only the DiT (transformer) and VAE run in this service. The text encoder
    // (Qwen3-VL) is a separate embedding service; its weights are not loaded
    // here.
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");

    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device(), dtype_);

    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device(), dtype_);
    transformer_->keep_fp32_modules();

    LOG(INFO) << "JoyImageEditPlusPipeline loaded.";
  }

 private:
  // Nearest 1024-base aspect bucket (h, w).
  std::pair<int64_t, int64_t> joyimage_bucket(int64_t h, int64_t w) {
    static const std::vector<std::pair<int64_t, int64_t>> kBuckets = {
        {512, 1792},  {512, 1856}, {512, 1920}, {512, 1984}, {512, 2048},
        {576, 1600},  {576, 1664}, {576, 1728}, {576, 1792}, {640, 1472},
        {640, 1536},  {640, 1600}, {704, 1344}, {704, 1408}, {704, 1472},
        {768, 1216},  {768, 1280}, {768, 1344}, {832, 1152}, {832, 1216},
        {896, 1088},  {896, 1152}, {960, 1024}, {960, 1088}, {1024, 960},
        {1024, 1024}, {1088, 896}, {1088, 960}, {1152, 832}, {1152, 896},
        {1216, 768},  {1216, 832}, {1280, 768}, {1344, 704}, {1344, 768},
        {1408, 704},  {1472, 640}, {1472, 704}, {1536, 640}, {1600, 576},
        {1600, 640},  {1664, 576}, {1728, 576}, {1792, 512}, {1792, 576},
        {1856, 512},  {1920, 512}, {1984, 512}, {2048, 512}};
    double target = static_cast<double>(h) / static_cast<double>(w);
    int64_t best_h = 1024, best_w = 1024;
    double best_diff = std::numeric_limits<double>::max();
    for (const auto& hw : kBuckets) {
      double diff =
          std::abs(static_cast<double>(hw.first) / hw.second - target);
      if (diff < best_diff) {
        best_diff = diff;
        best_h = hw.first;
        best_w = hw.second;
      }
    }
    return {best_h, best_w};
  }

  torch::Tensor pad_seq(const torch::Tensor& x, int64_t target_len) {
    int64_t cur = x.size(1);
    if (cur >= target_len) return x.slice(1, cur - target_len, cur);
    int64_t pad = target_len - cur;
    std::vector<int64_t> shape = x.sizes().vec();
    shape[1] = pad;
    auto zeros = torch::zeros(shape, x.options());
    return torch::cat({x, zeros}, 1);
  }

  DiTModelContext context_;
  const ModelArgs& vae_model_args_;
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;

  AutoencoderKLWan vae_{nullptr};
  joyimage::JoyImageEditPlusTransformer3DModel transformer_{nullptr};
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  xllm::VAEImageProcessor vae_image_processor_{nullptr};

  int64_t in_channels_;
  int64_t num_layers_;
  int64_t patch_t_, patch_h_, patch_w_;
  int64_t latent_channels_;
  int64_t vae_scale_factor_spatial_;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
};

TORCH_MODULE(JoyImageEditPlusPipeline);

// Only the transformer and VAE components are instantiated by this pipeline;
// their args loaders live in transformer_joyimage_edit_plus.h and
// autoencoder_kl_wan.h. The text_encoder / processor / tokenizer components in
// model_index.json belong to the separate Qwen3-VL embedding service and are
// not loaded here (the DiT loader simply skips components without a factory).

REGISTER_DIT_MODEL(JoyImageEditPlusPipeline, JoyImageEditPlusPipeline);
}  // namespace xllm

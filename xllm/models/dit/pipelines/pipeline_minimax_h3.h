/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_context/dit_model_context.h"
#include "core/framework/model_loader/dit_model_loader.h"
#if defined(USE_NPU)
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#endif
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_audio.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_video.h"
#include "models/dit/processors/minimax_h3_audio_processor.h"
#include "models/dit/processors/minimax_h3_visual_processor.h"
#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/transformer_minimax_h3.h"
#if defined(USE_NPU)
#include "models/dit/utils/dit_block_weight_manager.h"
#endif
#include "models/dit/utils/util.h"
#include "models/model_registry.h"

namespace xllm {

struct MiniMaxH3TimestepLayout {
  torch::Tensor timesteps;
  torch::Tensor indices;
};

class MiniMaxH3PipelineImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3PipelineImpl(const DiTModelContext& context)
      : options_(context.get_tensor_options()),
        tp_group_(context.get_parallel_args().dit_tp_group_),
        vae_group_(context.get_parallel_args().dit_vae_group_),
        is_tp_driver_(tp_group_ == nullptr || tp_group_->rank() == 0),
        is_vae_parallel_enabled_(vae_group_ != nullptr &&
                                 vae_group_->world_size() > 1),
        is_video_vae_worker_(is_vae_parallel_enabled_ || is_tp_driver_),
        task_type_(DiTConfig::get_instance().dit_h3_task_type()) {
    CHECK(task_type_ == "fl2va" || task_type_ == "ref2va")
        << "MiniMax-H3 dit_h3_task_type must be fl2va or ref2va";
    const ModelArgs& transformer_args = context.get_model_args("transformer");
    const ModelArgs& vae_args = context.get_model_args("vae");
    const int32_t vae_size = context.get_parallel_args().vae_size();
    CHECK_GT(vae_size, 0);
    CHECK(vae_size == 1 || vae_group_ != nullptr)
        << "MiniMax-H3 requires a VAE group when vae_size > 1";
    if (vae_group_ != nullptr) {
      CHECK_EQ(vae_size, vae_group_->world_size());
    }
    const std::vector<int64_t>& patch_size = transformer_args.wan_patch_size();
    CHECK_EQ(patch_size.size(), static_cast<size_t>(3))
        << "MiniMax-H3 transformer patch_size must have 3 dimensions";
    patch_h_ = patch_size[1];
    patch_w_ = patch_size[2];
    video_latent_channels_ = transformer_args.in_channels();
    audio_latent_channels_ = transformer_args.audio_in_channels();
    text_hidden_size_ = transformer_args.text_dim();
    vae_spatial_ratio_ =
        dit::product_checked(vae_args.spatial_downsample_factors());
    vae_temporal_ratio_ =
        dit::product_checked(vae_args.temporal_downsample_factors());
    CHECK_GT(patch_h_, 0);
    CHECK_GT(patch_w_, 0);
    CHECK_GT(video_latent_channels_, 0);
    CHECK_GT(audio_latent_channels_, 0);
    CHECK_GT(text_hidden_size_, 0);
    CHECK_GT(vae_spatial_ratio_, 0);
    CHECK_GT(vae_temporal_ratio_, 0);
    LOG(INFO) << "Initializing MiniMax-H3 " << task_type_ << " native pipeline";
    visual_processor_ =
        register_module("visual_processor",
                        MiniMaxH3VisualProcessor(
                            context.get_model_context("vae"),
                            /*canvas_multiple=*/vae_spatial_ratio_ * patch_h_));
    audio_processor_ = register_module(
        "audio_processor",
        MiniMaxH3AudioProcessor(context.get_model_context("audio_vae"),
                                /*channel_count=*/audio_channels_));
    transformer_ =
        register_module("transformer", MiniMaxH3Transformer3DModel(context));
    if (is_video_vae_worker_) {
      vae_ = register_module(
          "vae", AutoencoderKLMiniMaxH3(context.get_model_context("vae")));
    }
    if (is_tp_driver_) {
      audio_vae_ = register_module(
          "audio_vae",
          AutoencoderKLMiniMaxH3Audio(context.get_model_context("audio_vae")));
    }
  }

#if defined(USE_NPU)
  ~MiniMaxH3PipelineImpl() override {
    if (use_rolling_load_) {
      // Tail refills may still use host weights when the pipeline is destroyed.
      torch::DeviceGuard guard(options_.device());
      const aclError result = aclrtSynchronizeDevice();
      if (result != ACL_SUCCESS) {
        LOG(ERROR) << "Failed to synchronize MiniMax-H3 rolling load, result="
                   << result;
      }
    }
  }
#endif

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    CHECK_EQ(input.batch_size, 1)
        << "MiniMax-H3 native T2VA currently supports batch_size=1";
    // The request boundary already rejects num_videos_per_prompt != 1; this
    // guard catches internal callers that bypass it.
    CHECK_EQ(input.generation_params.num_videos_per_prompt, 1)
        << "MiniMax-H3 currently supports num_videos_per_prompt=1";
    const std::optional<NamedTensorConstRef> prompt_embed =
        input.tensor_sources.get_named_tensor("prompt_embed");
    CHECK(prompt_embed.has_value())
        << "MiniMax-H3 T2VA requires prompt_embed from the encoder service";
    const std::vector<int64_t>* prompt_token_tags =
        get_tensor_parameter<std::vector<int64_t>>(
            prompt_embed->get().parameters, "prompt_token_tags");
    CHECK(prompt_token_tags != nullptr)
        << "MiniMax-H3 T2VA requires prompt_token_tags from the encoder "
           "service";
    const torch::Tensor& prompt_embeds = prompt_embed->get().tensor;
    CHECK(prompt_embeds.is_floating_point())
        << "MiniMax-H3 prompt_embed must be floating point";
    CHECK(torch::isfinite(prompt_embeds).all().item<bool>())
        << "MiniMax-H3 prompt_embed must not contain NaN or Inf";
    CHECK_EQ(prompt_embeds.dim(), 3)
        << "MiniMax-H3 prompt_embed must have shape [1, T, 5120]";
    CHECK_EQ(prompt_embeds.size(0), 1);
    CHECK_EQ(prompt_embeds.size(2), text_hidden_size_);
    torch::Tensor text_token_tags =
        torch::tensor(*prompt_token_tags, device_options(torch::kInt64));
    CHECK_EQ(text_token_tags.size(0), prompt_embeds.size(1));
    CHECK(torch::all((text_token_tags == video_tag_) |
                     (text_token_tags == text_tag_))
              .item<bool>())
        << "MiniMax-H3 prompt_token_tags may contain only video/text tags";
    const DiTGenerationParams& generation_params = input.generation_params;
    const double video_fps = generation_params.video_fps;
    CHECK(std::isfinite(video_fps) && video_fps > 0.0)
        << "MiniMax-H3 fps must be finite and positive";
    CHECK_EQ(generation_params.audio_sampling_rate,
             audio_processor_->sampling_rate())
        << "MiniMax-H3 sampling_rate must match the audio VAE config";
    validate_media_sources(input.media_sources);
    const int64_t requested_height = generation_params.height;
    const int64_t requested_width = generation_params.width;
    CHECK_GT(requested_height, 0);
    CHECK_GT(requested_width, 0);
    const int64_t height =
        visual_processor_->align_canvas_pixels(requested_height);
    const int64_t width =
        visual_processor_->align_canvas_pixels(requested_width);
    if (requested_height != height || requested_width != width) {
      LOG(WARNING) << "MiniMax-H3 canvas rounded from " << requested_width
                   << "x" << requested_height << " to " << width << "x"
                   << height;
    }
    const int64_t requested_frames = generation_params.num_frames;
    CHECK_GT(requested_frames, 0);
    const int64_t num_frames =
        visual_processor_->align_target_frames(requested_frames);
    const double duration = static_cast<double>(num_frames) / video_fps;
    CHECK_GE(duration, min_duration_seconds_)
        << "MiniMax-H3 duration must be at least 5 seconds";
    CHECK_LE(duration, max_duration_seconds_)
        << "MiniMax-H3 duration must be at most 15 seconds";
    if (requested_frames != num_frames) {
      LOG(WARNING) << "MiniMax-H3 num_frames rounded from " << requested_frames
                   << " to " << num_frames;
    }

    const int64_t latent_frames =
        visual_processor_->encode_latent_frames(num_frames);
    const int64_t latent_height = height / vae_spatial_ratio_;
    const int64_t latent_width = width / vae_spatial_ratio_;
    const int64_t audio_latents =
        audio_processor_->audio_latents_for_video(num_frames, video_fps);
    const int64_t seed =
        generation_params.seed_is_set ? generation_params.seed : 42;

    const bool is_ref2va = task_type_ == "ref2va";
    std::vector<std::string> keyframe_anchors;
    size_t image_count = 0;
    for (const MediaNamedTensor& source : input.media_sources.entries()) {
      if (source.name == "image") {
        ++image_count;
      }
    }
    const bool has_first_keyframe = !is_ref2va && image_count > 0;
    const bool has_last_keyframe = !is_ref2va && image_count == 2;
    if (has_first_keyframe) {
      keyframe_anchors.emplace_back("first");
    }
    if (has_last_keyframe) {
      keyframe_anchors.emplace_back("last");
    }
    const bool has_keyframes = !keyframe_anchors.empty();
    const bool has_video_tags =
        torch::any(text_token_tags == video_tag_).item<bool>();
    if (!is_ref2va) {
      CHECK_EQ(has_keyframes, has_video_tags)
          << "MiniMax-H3 FL2VA requires image-aware encoder output and "
             "matching images inputs";
    }

    torch::Generator generator =
        torch::make_generator<torch::CPUGeneratorImpl>();
    generator.set_current_seed(seed);
    ReferenceData reference_data;
    PackedLayout layout;
    torch::Tensor condition_rows;
    torch::Tensor audio_condition_rows;
    if (is_ref2va) {
      reference_data =
          prepare_reference_data(input, num_frames, video_fps, generator);
      condition_rows = reference_data.video_rows;
      audio_condition_rows = reference_data.audio_rows;
      layout = build_ref_layout(text_token_tags,
                                reference_data.blocks,
                                latent_frames,
                                latent_height,
                                latent_width,
                                audio_latents);
    } else {
      layout = build_layout(text_token_tags,
                            latent_frames,
                            latent_height,
                            latent_width,
                            audio_latents,
                            keyframe_anchors);
      if (has_keyframes) {
        condition_rows = prepare_fl_frame_data(
            input, height, width, generator, keyframe_anchors);
        broadcast_from_driver(condition_rows);
        CHECK_EQ(condition_rows.dim(), 3)
            << "MiniMax-H3 FL2VA condition rows must be [B,S,D]";
        CHECK_EQ(condition_rows.size(0), 1)
            << "MiniMax-H3 FL2VA condition batch size mismatch";
        CHECK_EQ(condition_rows.size(1), layout.num_condition_video_rows)
            << "MiniMax-H3 FL2VA condition row count mismatch";
      }
    }

    std::pair<torch::Tensor, torch::Tensor> noise = prepare_noise(
        latent_frames, latent_height, latent_width, audio_latents, generator);
    torch::Tensor video_rows = patchify_video(noise.first);
    if (condition_rows.defined()) {
      video_rows = torch::cat({condition_rows, video_rows}, 1);
    }
    torch::Tensor audio_rows = patchify_audio(noise.second);
    if (audio_condition_rows.defined() && audio_condition_rows.size(1) > 0) {
      audio_rows = torch::cat({audio_condition_rows, audio_rows}, 1);
    }
    const MiniMaxH3TransformerConditioning conditioning =
        transformer_->prepare_conditioning(prompt_embeds, layout.position_ids);

    const int64_t num_inference_steps = generation_params.num_inference_steps;
    scheduler_.set_timesteps(num_inference_steps, options_.device());
    audio_scheduler_.set_timesteps(num_inference_steps, options_.device());
    CHECK_EQ(scheduler_.timesteps().numel(),
             audio_scheduler_.timesteps().numel());
    // Keep layout decisions on host scalars.  The schedules are fixed for the
    // denoising pass, so copying them once avoids a device synchronization on
    // every diffusion step.
    const torch::Tensor video_timesteps_cpu =
        scheduler_.timesteps().to(torch::kCPU);
    const torch::Tensor audio_timesteps_cpu =
        audio_scheduler_.timesteps().to(torch::kCPU);
    std::function<void(int32_t)> before_layer_cb;
    std::function<void(int32_t)> after_layer_cb;
#if defined(USE_NPU)
    if (use_rolling_load_) {
      before_layer_cb = [this](int32_t index) {
        rolling_transformer_.wait_h2d(index);
      };
      after_layer_cb = [this](int32_t index) {
        rolling_transformer_.schedule_next_h2d(index);
      };
    }
#endif
    for (int64_t step = 0; step < scheduler_.timesteps().numel(); ++step) {
      torch::Tensor video_timestep = scheduler_.timesteps()[step];
      torch::Tensor audio_timestep = audio_scheduler_.timesteps()[step];
      const double video_timestep_value =
          video_timesteps_cpu[step].item<double>();
      const double audio_timestep_value =
          audio_timesteps_cpu[step].item<double>();
      MiniMaxH3TimestepLayout timestep_layout =
          build_minimax_h3_timestep_layout(layout,
                                           video_timestep,
                                           audio_timestep,
                                           video_timestep_value,
                                           audio_timestep_value);
      MiniMaxH3TransformerOutput prediction =
          transformer_->forward(video_rows,
                                audio_rows,
                                conditioning,
                                timestep_layout.timesteps,
                                timestep_layout.indices,
                                layout.position_ids,
                                layout.token_tags,
                                layout.video_indices,
                                layout.audio_indices,
                                layout.text_indices,
                                before_layer_cb,
                                after_layer_cb);
      video_rows = step_generated_rows(video_rows,
                                       prediction.sample,
                                       layout.num_condition_video_rows,
                                       scheduler_,
                                       video_timestep);
      audio_rows = step_generated_rows(audio_rows,
                                       prediction.audio_sample,
                                       layout.num_condition_audio_rows,
                                       audio_scheduler_,
                                       audio_timestep);
    }

    if (!is_video_vae_worker_) {
      synchronize_tp_ranks();
      return DiTForwardOutput();
    }

    torch::Tensor video_latent;
    if (is_tp_driver_) {
      torch::Tensor generated_video_rows =
          generated_rows(video_rows, layout.num_condition_video_rows);
      video_latent = unpatchify_video(
          generated_video_rows, latent_frames, latent_height, latent_width);
      video_latent = vae_->denormalize_latents(video_latent);
    } else {
      video_latent = torch::empty(
          {1,
           video_latent_channels_,
           latent_frames,
           latent_height,
           latent_width},
          options_.dtype(torch::kFloat32).device(options_.device()));
    }
    if (is_vae_parallel_enabled_) {
      if (tp_group_ != nullptr && tp_group_->world_size() > 1) {
        tp_group_->broadcast(video_latent, /*root_rank=*/0);
      }
      vae_group_->broadcast(video_latent, /*root_rank=*/0);
    }

    torch::Tensor video = vae_->decode_to_pixels(video_latent);
    if (!is_tp_driver_) {
      synchronize_tp_ranks();
      return DiTForwardOutput();
    }

    torch::Tensor generated_audio_rows =
        generated_rows(audio_rows, layout.num_condition_audio_rows);
    torch::Tensor audio_latent =
        unpatchify_audio(generated_audio_rows, audio_latents);

    audio_latent = audio_vae_->denormalize_latents(audio_latent);

    torch::Tensor audio = audio_vae_->decode(audio_latent.to(torch::kFloat32));
    const int64_t requested_samples =
        audio_processor_->samples_for_video(num_frames, video_fps);
    audio = audio.slice(2, 0, requested_samples);

    DiTForwardOutput output;
    output.tensors = {video};
    output.audio_tensors = {audio};
    synchronize_tp_ranks();
    return output;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 loader must not be null";
    LOG(INFO) << "MiniMax-H3 " << task_type_ << " loading model from "
              << loader->model_root_path();
    const std::string transformer_component =
        task_type_ == "ref2va" ? "transformer_ref" : "transformer";
    CHECK(loader->has_component(transformer_component));
    CHECK(loader->has_component("vae"));
    CHECK(loader->has_component("audio_vae"));

    transformer_->load_model(
        loader->take_component_loader(transformer_component));
    if (is_video_vae_worker_) {
      vae_->load_model(loader->take_component_loader("vae"));
    }
    if (is_tp_driver_) {
      audio_vae_->load_model(loader->take_component_loader("audio_vae"));
    }
    if (loader->has_component("scheduler")) {
      scheduler_.load_model(loader->take_component_loader("scheduler"));
    }
    if (loader->has_component("audio_scheduler")) {
      audio_scheduler_.load_model(
          loader->take_component_loader("audio_scheduler"));
    }

    use_rolling_load_ = transformer_->rolling_load_enabled();
    transformer_->apply_reference_precision(options_.device());
#if defined(USE_NPU)
    if (use_rolling_load_) {
      init_rolling_transformer();
    }
#endif
    if (is_video_vae_worker_) {
      vae_->to(options_.device(), torch::kFloat32);
    }
    if (is_tp_driver_) {
      audio_vae_->to(options_.device(), torch::kFloat32);
    }
    LOG(INFO) << "MiniMax-H3 " << task_type_ << " native components loaded";
  }

 private:
  void validate_media_sources(const DiTMediaSources& media_sources) const {
    size_t image_count = 0;
    size_t video_count = 0;
    size_t audio_count = 0;
    size_t video_audio_count = 0;
    const std::vector<MediaNamedTensor>& sources = media_sources.entries();
    for (size_t index = 0; index < sources.size(); ++index) {
      const std::string& name = sources[index].name;
      if (name == "image") {
        ++image_count;
      } else if (name == "prompt_video") {
        ++video_count;
      } else if (name == "prompt_audio") {
        ++audio_count;
      } else if (name == "prompt_audio_in_video") {
        CHECK_GT(index, static_cast<size_t>(0));
        CHECK_EQ(sources[index - 1].name, "prompt_video")
            << "MiniMax-H3 embedded audio must follow its prompt_video";
        ++video_audio_count;
      } else {
        CHECK(false) << "MiniMax-H3 unsupported media source name: " << name;
      }
    }
    if (task_type_ == "fl2va") {
      CHECK_LE(image_count, static_cast<size_t>(2));
      CHECK_EQ(video_count, static_cast<size_t>(0));
      CHECK_EQ(audio_count, static_cast<size_t>(0));
      CHECK_EQ(video_audio_count, static_cast<size_t>(0));
      return;
    }
    CHECK_LE(image_count, static_cast<size_t>(9));
    CHECK_LE(video_count, static_cast<size_t>(1));
    CHECK_LE(audio_count, static_cast<size_t>(1));
    CHECK_LE(video_audio_count, video_count);
  }

#if defined(USE_NPU)
  void init_rolling_transformer() {
    torch::DeviceGuard guard(options_.device());
    transformer_->prepare_for_rolling_load();
    auto loaders = transformer_->get_block_weight_loaders();
    CHECK(!loaders.empty()) << "MiniMax-H3 rolling load has no block loaders";

    size_t max_storage = 0;
    for (const dit::BlockWeightLoader* loader : loaders) {
      max_storage = std::max(max_storage, loader->storage_size());
    }
    constexpr int32_t kMinRollingSlots = 2;
    const int32_t num_slots =
        std::max(LoadConfig::get_instance().rolling_load_num_rolling_slots(),
                 kMinRollingSlots);
    auto buffer =
        std::make_shared<layer::RollingWeightBuffer>(num_slots, max_storage);
    rolling_transformer_.init(std::move(loaders), std::move(buffer), num_slots);
    rolling_transformer_.preload();
  }
#endif

  struct PackedLayout {
    torch::Tensor position_ids;
    torch::Tensor token_tags;
    torch::Tensor video_indices;
    torch::Tensor audio_indices;
    torch::Tensor text_indices;
    int64_t sequence_length;
    int64_t audio_start;
    int64_t video_start;
    int64_t num_condition_video_rows;
    int64_t num_condition_audio_rows = 0;
  };

  enum class ReferenceKind { IMAGE, VIDEO, AUDIO };

  struct ReferenceBlock {
    ReferenceKind kind;
    int64_t latent_frames = 0;
    int64_t latent_height = 0;
    int64_t latent_width = 0;
    int64_t audio_latents = 0;
  };

  // Latent rows one frame occupies in the packed sequence.
  int64_t rows_per_frame(int64_t latent_height, int64_t latent_width) const {
    return latent_height / patch_h_ * (latent_width / patch_w_);
  }

  // Row counts a reference block occupies in the packed sequence, given the
  // pipeline's patch size and audio channel count.
  int64_t reference_video_rows(const ReferenceBlock& block) const {
    return block.latent_frames *
           rows_per_frame(block.latent_height, block.latent_width);
  }

  int64_t audio_rows(int64_t audio_latents) const {
    return audio_latents * audio_channels_;
  }

  // Row count sum (video, audio) across reference blocks, in one pass.
  std::pair<int64_t, int64_t> reference_row_counts(
      const std::vector<ReferenceBlock>& blocks) const {
    int64_t video_rows = 0;
    int64_t audio_row_count = 0;
    for (const ReferenceBlock& block : blocks) {
      if (block.kind != ReferenceKind::AUDIO) {
        video_rows += reference_video_rows(block);
      }
      audio_row_count += audio_rows(block.audio_latents);
    }
    return {video_rows, audio_row_count};
  }

  // Unbatches a media source tensor ([1, ...]) to its single sample, checking
  // the expected leading dimensions.
  static torch::Tensor unbatched_source(const MediaNamedTensor& source,
                                        int64_t dim,
                                        const char* what) {
    CHECK_EQ(source.tensor.dim(), dim)
        << "MiniMax-H3 " << what << " must have " << dim << " dimensions";
    CHECK_EQ(source.tensor.size(0), 1)
        << "MiniMax-H3 " << what << " batch size must be 1";
    return source.tensor.select(0, 0);
  }

  // Latent count for a reference waveform clamped to the reference budget.
  int64_t clamped_reference_latents(const torch::Tensor& waveform,
                                    int64_t max_audio_samples) const {
    return audio_processor_->reference_audio_latents(
        std::min(waveform.size(1), max_audio_samples));
  }

  // Advances rows by one scheduler step over the generated (non-condition)
  // rows; condition rows keep their keyframe noising.
  template <typename Scheduler>
  torch::Tensor step_generated_rows(const torch::Tensor& rows,
                                    const torch::Tensor& prediction,
                                    int64_t num_condition_rows,
                                    Scheduler& scheduler,
                                    const torch::Tensor& timestep) {
    if (num_condition_rows > 0) {
      torch::Tensor generated_rows = rows.slice(1, num_condition_rows);
      torch::Tensor generated_prediction =
          prediction.slice(1, num_condition_rows);
      generated_rows.copy_(
          scheduler.step(generated_prediction, timestep, generated_rows));
      return rows;
    }
    return scheduler.step(prediction, timestep, rows);
  }

  static torch::Tensor generated_rows(const torch::Tensor& rows,
                                      int64_t num_condition_rows) {
    return num_condition_rows > 0 ? rows.slice(1, num_condition_rows) : rows;
  }

  struct ReferenceData {
    std::vector<ReferenceBlock> blocks;
    torch::Tensor video_rows;
    torch::Tensor audio_rows;
  };

  MiniMaxH3TimestepLayout build_minimax_h3_timestep_layout(
      const PackedLayout& layout,
      const torch::Tensor& video_timestep,
      const torch::Tensor& audio_timestep,
      double video_timestep_value,
      double audio_timestep_value) const {
    CHECK_EQ(video_timestep.numel(), 1);
    CHECK_EQ(audio_timestep.numel(), 1);
    CHECK_EQ(video_timestep.device(), audio_timestep.device());
    CHECK_EQ(video_timestep.scalar_type(), audio_timestep.scalar_type());
    CHECK_GE(layout.num_condition_video_rows, 0);
    CHECK_GE(layout.num_condition_audio_rows, 0);
    CHECK_LE(layout.num_condition_video_rows, layout.video_indices.numel());
    CHECK_LE(layout.num_condition_audio_rows, layout.audio_indices.numel());

    const bool has_condition = layout.num_condition_video_rows > 0 ||
                               layout.num_condition_audio_rows > 0;
    const torch::TensorOptions index_options =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .device(video_timestep.device());
    if (!has_condition) {
      CHECK_GE(layout.audio_start, 0);
      CHECK_GE(layout.video_start, layout.audio_start);
      CHECK_LE(layout.video_start, layout.sequence_length);
      if (video_timestep_value == audio_timestep_value) {
        return {video_timestep.reshape({1}),
                torch::zeros({layout.sequence_length}, index_options)};
      }

      const bool video_first = video_timestep_value < audio_timestep_value;
      torch::Tensor timesteps =
          video_first ? torch::stack({video_timestep, audio_timestep})
                      : torch::stack({audio_timestep, video_timestep});
      const int64_t video_index = video_first ? 0 : 1;
      const int64_t audio_index = video_first ? 1 : 0;
      torch::Tensor indices =
          torch::full({layout.sequence_length}, video_index, index_options);
      indices.slice(0, layout.audio_start, layout.video_start)
          .fill_(audio_index);
      return {timesteps, indices};
    }

    const int64_t sequence_length = layout.video_indices.numel() +
                                    layout.audio_indices.numel() +
                                    layout.text_indices.numel();
    CHECK_EQ(sequence_length, layout.sequence_length);
    const float generated_video_value =
        static_cast<float>(video_timestep_value);
    const float condition_video_value =
        static_cast<float>(std::max(video_timestep_value, keyframe_noise_aug_));
    std::vector<float> unique_values{generated_video_value};
    auto add_unique_value = [&unique_values](float value) {
      if (std::find(unique_values.begin(), unique_values.end(), value) ==
          unique_values.end()) {
        unique_values.push_back(value);
      }
    };
    if (layout.num_condition_video_rows > 0) {
      add_unique_value(condition_video_value);
    }
    if (layout.num_condition_audio_rows > 0) {
      add_unique_value(/*condition_audio_timestep=*/1.0f);
    }
    std::sort(unique_values.begin(), unique_values.end());
    const auto timestep_index = [&unique_values](float value) {
      const auto it =
          std::lower_bound(unique_values.begin(), unique_values.end(), value);
      CHECK(it != unique_values.end() && *it == value);
      return static_cast<int64_t>(it - unique_values.begin());
    };
    torch::TensorOptions timestep_options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(video_timestep.device());
    torch::Tensor timesteps = torch::tensor(unique_values, timestep_options);
    torch::Tensor indices = torch::full({sequence_length},
                                        timestep_index(generated_video_value),
                                        index_options);
    if (layout.num_condition_video_rows > 0) {
      indices.index_fill_(
          0,
          layout.video_indices.slice(0, 0, layout.num_condition_video_rows),
          timestep_index(condition_video_value));
    }
    if (layout.num_condition_audio_rows > 0) {
      indices.index_fill_(
          0,
          layout.audio_indices.slice(0, 0, layout.num_condition_audio_rows),
          timestep_index(/*condition_audio_timestep=*/1.0f));
    }
    return {timesteps, indices};
  }

  bool tp_active() const {
    return tp_group_ != nullptr && tp_group_->world_size() > 1;
  }

  void broadcast_from_driver(torch::Tensor& tensor) const {
    if (tp_active()) {
      tp_group_->broadcast(tensor, /*root_rank=*/0);
    }
  }

  void synchronize_tp_ranks() const {
    if (!tp_active()) {
      return;
    }
    tp_group_->barrier();
  }

  PackedLayout build_layout(
      const torch::Tensor& text_tags,
      int64_t latent_frames,
      int64_t latent_height,
      int64_t latent_width,
      int64_t audio_latents,
      const std::vector<std::string>& keyframe_anchors) const {
    const int64_t text_tokens = text_tags.numel();
    torch::Tensor cpu_text_tags = text_tags.to(torch::kCPU, torch::kLong);
    const int64_t frame_rows = rows_per_frame(latent_height, latent_width);
    const int64_t condition_rows =
        static_cast<int64_t>(keyframe_anchors.size()) * frame_rows;
    const int64_t audio_rows = this->audio_rows(audio_latents);
    const int64_t video_rows = latent_frames * frame_rows;
    const int64_t condition_start = text_tokens;
    const int64_t audio_start = condition_start + condition_rows;
    const int64_t video_start = audio_start + audio_rows;
    const int64_t sequence_length = video_start + video_rows;
    torch::TensorOptions double_options = cpu_options(torch::kFloat64);
    torch::Tensor positions =
        torch::zeros({sequence_length, 3}, double_options);
    positions.slice(0, 0, text_tokens)
        .select(1, 0)
        .copy_(torch::arange(text_tokens, double_options));

    std::pair<torch::Tensor, torch::Tensor> target_grid =
        build_frame_grid(latent_height, latent_width);
    const torch::Tensor& frame_grid = target_grid.first;
    const torch::Tensor& width_grid = target_grid.second;

    for (size_t index = 0; index < keyframe_anchors.size(); ++index) {
      double anchor_time = static_cast<double>(text_tokens);
      if (keyframe_anchors[index] == "last") {
        // The "last" anchor sits after every frame except the first, whose
        // rope contribution (frame_span 1.0 * rope_frame_scale_) is excluded.
        anchor_time += rope_frame_time(latent_frames) - rope_frame_scale_;
      } else {
        CHECK_EQ(keyframe_anchors[index], "first")
            << "MiniMax-H3 keyframe anchor must be first or last";
      }
      fill_video_positions(
          positions,
          condition_start + static_cast<int64_t>(index) * frame_rows,
          /*latent_frames=*/1,
          frame_grid,
          anchor_time);
    }

    fill_audio_positions(positions,
                         audio_start,
                         audio_rows,
                         audio_latents,
                         static_cast<double>(text_tokens),
                         width_grid);

    fill_video_positions(positions,
                         video_start,
                         latent_frames,
                         frame_grid,
                         static_cast<double>(text_tokens));

    torch::TensorOptions long_options = cpu_options(torch::kLong);
    torch::Tensor text_indices = torch::arange(text_tokens, long_options);
    torch::Tensor audio_indices =
        torch::arange(audio_start, video_start, long_options);
    torch::Tensor video_indices =
        condition_rows > 0
            ? torch::cat(
                  {torch::arange(condition_start, audio_start, long_options),
                   torch::arange(video_start, sequence_length, long_options)})
            : torch::arange(video_start, sequence_length, long_options);
    torch::Tensor tags = torch::empty({sequence_length}, long_options);
    tags.slice(0, 0, text_tokens).copy_(cpu_text_tags);
    tags.slice(0, audio_start, video_start).fill_(audio_tag_);
    if (condition_rows > 0) {
      tags.slice(0, condition_start, audio_start).fill_(video_tag_);
    }
    tags.slice(0, video_start, sequence_length).fill_(video_tag_);

    return {positions.to(options_.device(), torch::kFloat32),
            tags.to(options_.device()),
            video_indices.to(options_.device()),
            audio_indices.to(options_.device()),
            text_indices.to(options_.device()),
            sequence_length,
            audio_start,
            video_start,
            condition_rows};
  }

  PackedLayout build_ref_layout(const torch::Tensor& text_tags,
                                const std::vector<ReferenceBlock>& references,
                                int64_t latent_frames,
                                int64_t latent_height,
                                int64_t latent_width,
                                int64_t audio_latents) const {
    const int64_t text_tokens = text_tags.numel();
    torch::Tensor cpu_text_tags = text_tags.to(torch::kCPU, torch::kLong);
    const int64_t target_rows_per_frame =
        rows_per_frame(latent_height, latent_width);
    const int64_t target_video_rows = latent_frames * target_rows_per_frame;
    const int64_t target_audio_rows = audio_rows(audio_latents);
    const auto [reference_video_rows, reference_audio_rows] =
        reference_row_counts(references);
    const int64_t sequence_length = text_tokens + reference_video_rows +
                                    reference_audio_rows + target_audio_rows +
                                    target_video_rows;
    torch::TensorOptions double_options = cpu_options(torch::kFloat64);
    torch::TensorOptions long_options = cpu_options(torch::kLong);
    torch::Tensor positions =
        torch::zeros({sequence_length, 3}, double_options);
    positions.slice(0, 0, text_tokens)
        .select(1, 0)
        .copy_(torch::arange(text_tokens, double_options));

    std::pair<torch::Tensor, torch::Tensor> target_grid =
        build_frame_grid(latent_height, latent_width);
    std::vector<torch::Tensor> video_index_blocks;
    std::vector<torch::Tensor> audio_index_blocks;
    int64_t cursor = text_tokens;
    double rotary_time = static_cast<double>(text_tokens);
    for (const ReferenceBlock& reference : references) {
      if (reference.kind == ReferenceKind::IMAGE) {
        std::pair<torch::Tensor, torch::Tensor> grid =
            build_frame_grid(reference.latent_height, reference.latent_width);
        const int64_t rows = reference.latent_frames * grid.first.size(0);
        video_index_blocks.emplace_back(
            torch::arange(cursor, cursor + rows, long_options));
        fill_video_positions(positions,
                             cursor,
                             reference.latent_frames,
                             grid.first,
                             rotary_time);
        cursor += rows;
        rotary_time += 1.0;
      } else if (reference.kind == ReferenceKind::AUDIO) {
        const int64_t rows = audio_rows(reference.audio_latents);
        audio_index_blocks.emplace_back(
            torch::arange(cursor, cursor + rows, long_options));
        fill_audio_positions(positions,
                             cursor,
                             rows,
                             reference.audio_latents,
                             rotary_time,
                             target_grid.second);
        cursor += rows;
        rotary_time += static_cast<double>(reference.audio_latents);
      } else {
        std::pair<torch::Tensor, torch::Tensor> grid =
            build_frame_grid(reference.latent_height, reference.latent_width);
        const int64_t block_audio_rows = audio_rows(reference.audio_latents);
        const int64_t video_rows = reference.latent_frames * grid.first.size(0);
        if (block_audio_rows > 0) {
          audio_index_blocks.emplace_back(
              torch::arange(cursor, cursor + block_audio_rows, long_options));
        }
        video_index_blocks.emplace_back(
            torch::arange(cursor + block_audio_rows,
                          cursor + block_audio_rows + video_rows,
                          long_options));
        fill_audio_positions(positions,
                             cursor,
                             block_audio_rows,
                             reference.audio_latents,
                             rotary_time,
                             grid.second);
        const int64_t video_start = cursor + block_audio_rows;
        fill_video_positions(positions,
                             video_start,
                             reference.latent_frames,
                             grid.first,
                             rotary_time);
        cursor += block_audio_rows + video_rows;
        rotary_time += std::max(static_cast<double>(reference.audio_latents),
                                rope_frame_time(reference.latent_frames));
      }
    }

    const int64_t audio_start = cursor;
    const int64_t video_start = audio_start + target_audio_rows;
    fill_audio_positions(positions,
                         audio_start,
                         target_audio_rows,
                         audio_latents,
                         rotary_time,
                         target_grid.second);
    fill_video_positions(
        positions, video_start, latent_frames, target_grid.first, rotary_time);
    video_index_blocks.emplace_back(
        torch::arange(video_start, sequence_length, long_options));
    audio_index_blocks.emplace_back(
        torch::arange(audio_start, video_start, long_options));
    torch::Tensor video_indices = torch::cat(video_index_blocks);
    torch::Tensor audio_indices = torch::cat(audio_index_blocks);
    torch::Tensor text_indices = torch::arange(text_tokens, long_options);
    torch::Tensor tags = torch::empty({sequence_length}, long_options);
    tags.index_put_({text_indices}, cpu_text_tags);
    tags.index_put_({audio_indices}, audio_tag_);
    tags.index_put_({video_indices}, video_tag_);
    return {positions.to(options_.device(), torch::kFloat32),
            tags.to(options_.device()),
            video_indices.to(options_.device()),
            audio_indices.to(options_.device()),
            text_indices.to(options_.device()),
            sequence_length,
            audio_start,
            video_start,
            reference_video_rows,
            reference_audio_rows};
  }

  // Scalar [1] fp32 timestep on the compute device carrying the keyframe
  // noise-augmentation value, for scheduler_.scale_noise calls.
  torch::Tensor keyframe_noise_timestep() const {
    return torch::full(
        {1}, keyframe_noise_aug_, device_options(torch::kFloat32));
  }

  double frame_span(int64_t frame) const {
    return frame % visual_processor_->latents_per_clip() == 0
               ? 1.0
               : static_cast<double>(vae_temporal_ratio_);
  }

  double rope_frame_time(int64_t num_frames) const {
    double total = 0.0;
    for (int64_t frame = 0; frame < num_frames; ++frame) {
      total += rope_frame_scale_ * frame_span(frame);
    }
    return total;
  }

  std::pair<torch::Tensor, torch::Tensor> build_frame_grid(
      int64_t latent_height,
      int64_t latent_width) const {
    const double sqrt_area =
        std::sqrt(static_cast<double>(latent_height * latent_width));
    torch::Tensor height_grid =
        spatial_grid(latent_height, patch_h_, sqrt_area);
    torch::Tensor width_grid = spatial_grid(latent_width, patch_w_, sqrt_area);
    const int64_t grid_height = latent_height / patch_h_;
    const int64_t grid_width = latent_width / patch_w_;
    torch::Tensor grid = torch::stack({height_grid.view({grid_height, 1})
                                           .expand({grid_height, grid_width})
                                           .reshape({-1}),
                                       width_grid.view({1, grid_width})
                                           .expand({grid_height, grid_width})
                                           .reshape({-1})},
                                      1);
    return {grid, width_grid};
  }

  // Fills the (rotary time, height, width) position columns of one video
  // block, whose latent rows are [latent_frames, frame_grid.size(0)].
  void fill_video_positions(torch::Tensor& positions,
                            int64_t video_start,
                            int64_t latent_frames,
                            const torch::Tensor& frame_grid,
                            double rotary_time) const {
    const int64_t frame_rows = frame_grid.size(0);
    positions.slice(0, video_start, video_start + latent_frames * frame_rows)
        .select(1, 0)
        .copy_(torch::repeat_interleave(
            temporal_grid(latent_frames, rotary_time), frame_rows));
    positions.slice(0, video_start, video_start + latent_frames * frame_rows)
        .slice(1, 1)
        .copy_(frame_grid.repeat({latent_frames, 1}));
  }

  void fill_audio_positions(torch::Tensor& positions,
                            int64_t start,
                            int64_t count,
                            int64_t num_latents,
                            double rotary_time,
                            const torch::Tensor& width_grid) const {
    if (count == 0) {
      return;
    }
    torch::TensorOptions double_options = cpu_options(torch::kFloat64);
    positions.slice(0, start, start + count)
        .select(1, 0)
        .copy_((rotary_time + torch::arange(num_latents, double_options))
                   .repeat({audio_channels_}));
    positions.slice(0, start, start + count)
        .select(1, 2)
        .copy_(torch::cat(
            {torch::full(
                 {num_latents}, width_grid[0].item<double>(), double_options),
             torch::full({num_latents},
                         width_grid[width_grid.size(0) - 1].item<double>(),
                         double_options)}));
  }

  torch::Tensor spatial_grid(int64_t dimension,
                             int64_t patch,
                             double sqrt_area) const {
    const int64_t count = dimension / patch;
    const double ratio = static_cast<double>(dimension) / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    torch::TensorOptions options = cpu_options(torch::kFloat64);
    return (left + torch::arange(count, options) *
                       (ratio / static_cast<double>(count))) *
           static_cast<double>(vae_spatial_ratio_ * patch);
  }

  torch::Tensor temporal_grid(int64_t latent_frames, double origin) const {
    std::vector<double> values;
    values.reserve(static_cast<size_t>(latent_frames));
    double position = origin;
    for (int64_t frame = 0; frame < latent_frames; ++frame) {
      values.emplace_back(position);
      position += rope_frame_scale_ * frame_span(frame);
    }
    return torch::tensor(values, torch::dtype(torch::kFloat64));
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_noise(
      int64_t latent_frames,
      int64_t latent_height,
      int64_t latent_width,
      int64_t audio_latents,
      torch::Generator& generator) const {
    const int64_t video_values =
        video_latent_channels_ * latent_frames * latent_height * latent_width;
    const int64_t audio_values =
        audio_channels_ * audio_latent_channels_ * audio_latents;
    torch::Tensor noise =
        dit::randn_tensor({video_values + audio_values}, generator, options_);
    torch::Tensor video = noise.slice(0, 0, video_values)
                              .view({1,
                                     video_latent_channels_,
                                     latent_frames,
                                     latent_height,
                                     latent_width});
    torch::Tensor audio =
        noise.slice(0, video_values)
            .view({audio_channels_, audio_latent_channels_, audio_latents});
    return {video, audio};
  }

  // Keep TP ranks on the same random stream without allocating unused device
  // noise on follower ranks. The driver creates and broadcasts these tensors.
  static void advance_generator(const std::vector<int64_t>& shape,
                                torch::Generator& generator) {
    torch::randn(
        shape,
        generator,
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  }

  ReferenceData prepare_reference_data(const DiTForwardInput& input,
                                       int64_t num_frames,
                                       double video_fps,
                                       torch::Generator& generator) {
    ReferenceData result;
    std::vector<torch::Tensor> visual_latents;
    std::vector<torch::Tensor> audio_rows;
    std::vector<torch::Tensor> reference_audio_inputs;
    reference_audio_inputs.reserve(input.media_sources.size());
    const int64_t max_audio_samples =
        audio_processor_->max_reference_samples(num_frames, video_fps);

    const std::vector<MediaNamedTensor>& sources =
        input.media_sources.entries();
    for (size_t index = 0; index < sources.size(); ++index) {
      const MediaNamedTensor& source = sources[index];
      if (source.name == "prompt_audio_in_video") {
        continue;
      }
      if (source.name == "image") {
        const torch::Tensor& image = unbatched_source(source, 4, "image");
        const std::pair<int64_t, int64_t> size =
            visual_processor_->resolve_reference_image_size(image.size(1),
                                                            image.size(2));
        result.blocks.push_back({ReferenceKind::IMAGE,
                                 1,
                                 size.first / vae_spatial_ratio_,
                                 size.second / vae_spatial_ratio_,
                                 0});
        if (is_video_vae_worker_) {
          torch::Tensor prepared = visual_processor_->prepare_reference_image(
              image, size.first, size.second);
          visual_latents.emplace_back(vae_->encode_reference_condition(
              prepared.to(options_.device(), torch::kUInt8)
                  .unsqueeze(0)
                  .unsqueeze(2)));
        }
        continue;
      }
      if (source.name == "prompt_video") {
        const torch::Tensor& video_input =
            unbatched_source(source, 5, "reference video");
        const double* prompt_video_fps =
            get_tensor_parameter<double>(source.parameters, "prompt_video_fps");
        CHECK(prompt_video_fps != nullptr)
            << "MiniMax-H3 reference video requires prompt_video_fps";
        CHECK(std::isfinite(*prompt_video_fps) && *prompt_video_fps > 0.0)
            << "MiniMax-H3 prompt_video_fps must be finite and positive";
        const int64_t normalized_frames =
            visual_processor_->normalized_video_frame_count(
                video_input.size(0), *prompt_video_fps, video_fps, num_frames);
        const int64_t encode_frames =
            visual_processor_->reference_video_encode_frames(normalized_frames);
        const std::pair<int64_t, int64_t> size =
            visual_processor_->resolve_reference_canvas(video_input.size(2),
                                                        video_input.size(3));
        const bool has_audio =
            index + 1 < sources.size() &&
            sources[index + 1].name == "prompt_audio_in_video";
        torch::Tensor soundtrack;
        int64_t reference_audio_latents = 0;
        if (has_audio) {
          soundtrack =
              unbatched_source(sources[index + 1], 3, "reference soundtrack");
          reference_audio_latents =
              clamped_reference_latents(soundtrack, max_audio_samples);
        }
        const int64_t reference_video_latent_frames =
            visual_processor_->encode_latent_frames(encode_frames);
        result.blocks.push_back({ReferenceKind::VIDEO,
                                 reference_video_latent_frames,
                                 size.first / vae_spatial_ratio_,
                                 size.second / vae_spatial_ratio_,
                                 reference_audio_latents});
        if (is_video_vae_worker_) {
          torch::Tensor video =
              visual_processor_->prepare_reference_video(video_input,
                                                         *prompt_video_fps,
                                                         video_fps,
                                                         num_frames,
                                                         size.first,
                                                         size.second);
          visual_latents.emplace_back(vae_->encode_reference_condition(
              video.permute({1, 0, 2, 3})
                  .unsqueeze(0)
                  .to(options_.device(), torch::kUInt8)));
        }
        if (has_audio && is_tp_driver_) {
          reference_audio_inputs.emplace_back(std::move(soundtrack));
        }
        continue;
      }
      CHECK_EQ(source.name, "prompt_audio");
      torch::Tensor audio_input =
          unbatched_source(source, 3, "reference audio");
      const int64_t reference_audio_latents =
          clamped_reference_latents(audio_input, max_audio_samples);
      result.blocks.push_back(
          {ReferenceKind::AUDIO, 0, 0, 0, reference_audio_latents});
      if (is_tp_driver_) {
        reference_audio_inputs.emplace_back(std::move(audio_input));
      }
    }

    if (is_tp_driver_ && !reference_audio_inputs.empty()) {
      audio_rows.reserve(reference_audio_inputs.size());
      for (torch::Tensor waveform : reference_audio_inputs) {
        waveform =
            audio_processor_->trim_reference_audio(waveform, max_audio_samples)
                .to(options_.device(), torch::kFloat32);
        torch::Tensor latent = audio_vae_->encode_mode(waveform.unsqueeze(1));
        audio_rows.emplace_back(audio_vae_->normalize_latents(latent)
                                    .reshape({1, -1, audio_latent_channels_})
                                    .contiguous());
      }
    }

    std::vector<torch::Tensor> packed_visual_rows;
    size_t visual_index = 0;
    const torch::Tensor keyframe_timestep = keyframe_noise_timestep();
    for (const ReferenceBlock& block : result.blocks) {
      if (block.kind == ReferenceKind::AUDIO) {
        continue;
      }
      const std::vector<int64_t> shape = {1,
                                          video_latent_channels_,
                                          block.latent_frames,
                                          block.latent_height,
                                          block.latent_width};
      if (is_video_vae_worker_) {
        torch::Tensor noise = dit::randn_tensor(shape, generator, options_);
        CHECK_EQ(visual_latents[visual_index].sizes(), c10::IntArrayRef(shape));
        packed_visual_rows.emplace_back(patchify_video(scheduler_.scale_noise(
            visual_latents[visual_index], keyframe_timestep, noise)));
      } else {
        advance_generator(shape, generator);
      }
      ++visual_index;
    }
    const auto [video_row_count, audio_row_count] =
        reference_row_counts(result.blocks);
    if (is_video_vae_worker_ && !packed_visual_rows.empty()) {
      result.video_rows = torch::cat(packed_visual_rows, 1).contiguous();
    } else {
      result.video_rows = torch::empty(
          {1, video_row_count, video_latent_channels_ * patch_h_ * patch_w_},
          device_options(torch::kFloat32));
    }
    if (is_tp_driver_ && !audio_rows.empty()) {
      result.audio_rows = torch::cat(audio_rows, 1).contiguous();
    } else {
      result.audio_rows =
          torch::empty({1, audio_row_count, audio_latent_channels_},
                       device_options(torch::kFloat32));
    }
    if (video_row_count > 0) {
      broadcast_from_driver(result.video_rows);
    }
    if (audio_row_count > 0) {
      broadcast_from_driver(result.audio_rows);
    }
    CHECK_EQ(result.video_rows.size(1), video_row_count);
    CHECK_EQ(result.audio_rows.size(1), audio_row_count);
    return result;
  }

  torch::Tensor prepare_fl_frame_data(
      const DiTForwardInput& input,
      int64_t height,
      int64_t width,
      torch::Generator& generator,
      const std::vector<std::string>& keyframe_anchors) {
    if (!is_video_vae_worker_) {
      const int64_t latent_height = height / vae_spatial_ratio_;
      const int64_t latent_width = width / vae_spatial_ratio_;
      const int64_t frame_rows = rows_per_frame(latent_height, latent_width);
      for (size_t index = 0; index < keyframe_anchors.size(); ++index) {
        advance_generator(
            {1, video_latent_channels_, 1, latent_height, latent_width},
            generator);
      }
      return torch::empty(
          {1,
           static_cast<int64_t>(keyframe_anchors.size()) * frame_rows,
           video_latent_channels_ * patch_h_ * patch_w_},
          options_.dtype(torch::kFloat32).device(options_.device()));
    }

    std::vector<torch::Tensor> keyframes;
    keyframes.reserve(keyframe_anchors.size());
    const std::vector<torch::Tensor> image_sources =
        input.media_sources.get({"image", "image"});
    for (size_t index = 0; index < image_sources.size(); ++index) {
      keyframes.emplace_back(visual_processor_->prepare_keyframe_image(
          image_sources[index].select(0, 0),
          height,
          width,
          /*is_follower=*/index > 0));
    }
    CHECK_EQ(keyframes.size(), keyframe_anchors.size());

    std::vector<torch::Tensor> rows;
    rows.reserve(keyframes.size());
    const torch::Tensor keyframe_timestep = keyframe_noise_timestep();
    for (const torch::Tensor& keyframe : keyframes) {
      torch::Tensor pixels = keyframe.to(options_.device(), torch::kUInt8)
                                 .unsqueeze(0)
                                 .unsqueeze(2);
      torch::Tensor condition = vae_->encode_keyframe_condition(pixels);
      torch::Tensor noise =
          dit::randn_tensor(condition.sizes().vec(), generator, options_);
      torch::Tensor noised =
          scheduler_.scale_noise(condition, keyframe_timestep, noise);
      rows.emplace_back(patchify_video(noised));
    }
    return torch::cat(rows, 1).contiguous();
  }

  torch::Tensor patchify_video(const torch::Tensor& latent) const {
    const int64_t batch_size = latent.size(0);
    const int64_t channels = latent.size(1);
    const int64_t frames = latent.size(2);
    const int64_t height = latent.size(3);
    const int64_t width = latent.size(4);
    return latent
        .reshape({batch_size,
                  channels,
                  frames,
                  height / patch_h_,
                  patch_h_,
                  width / patch_w_,
                  patch_w_})
        .permute({0, 2, 3, 5, 1, 4, 6})
        .reshape({batch_size,
                  frames * height / patch_h_ * width / patch_w_,
                  channels * patch_h_ * patch_w_})
        .contiguous();
  }

  torch::Tensor patchify_audio(const torch::Tensor& latent) const {
    return latent.permute({0, 2, 1})
        .reshape({1, -1, audio_latent_channels_})
        .contiguous();
  }

  torch::Tensor unpatchify_video(const torch::Tensor& rows,
                                 int64_t frames,
                                 int64_t height,
                                 int64_t width) const {
    return rows
        .reshape({1,
                  frames,
                  height / patch_h_,
                  width / patch_w_,
                  video_latent_channels_,
                  patch_h_,
                  patch_w_})
        .permute({0, 4, 1, 2, 5, 3, 6})
        .reshape({1, video_latent_channels_, frames, height, width})
        .contiguous();
  }

  torch::Tensor unpatchify_audio(const torch::Tensor& rows,
                                 int64_t audio_latents) const {
    return rows
        .reshape({audio_channels_, audio_latents, audio_latent_channels_})
        .permute({0, 2, 1})
        .contiguous();
  }

  int64_t audio_channels_ = 2;
  int64_t video_tag_ = 0;
  int64_t text_tag_ = 1;
  int64_t audio_tag_ = 2;
  double min_duration_seconds_ = 5.0;
  double max_duration_seconds_ = 15.0;
  double keyframe_noise_aug_ = 0.999;
  double rope_frame_scale_ = 5.0 / 3.0;

  torch::TensorOptions options_;

  // Options helpers: device options for a scalar dtype, and CPU options for
  // the fp64/fp32/int64 layout-building grids.
  torch::TensorOptions device_options(torch::ScalarType dtype) const {
    return options_.dtype(dtype).device(options_.device());
  }

  static torch::TensorOptions cpu_options(torch::ScalarType dtype) {
    return torch::TensorOptions().dtype(dtype);
  }

  ProcessGroup* tp_group_ = nullptr;
  ProcessGroup* vae_group_ = nullptr;
  bool is_tp_driver_ = true;
  bool use_rolling_load_ = false;
  bool is_vae_parallel_enabled_ = false;
  bool is_video_vae_worker_ = true;
  std::string task_type_;
  int64_t patch_h_;
  int64_t patch_w_;
  int64_t video_latent_channels_;
  int64_t audio_latent_channels_;
  int64_t text_hidden_size_;
  int64_t vae_spatial_ratio_;
  int64_t vae_temporal_ratio_;
  MiniMaxH3VisualProcessor visual_processor_{nullptr};
  MiniMaxH3AudioProcessor audio_processor_{nullptr};
  MiniMaxH3Transformer3DModel transformer_{nullptr};
  AutoencoderKLMiniMaxH3 vae_{nullptr};
  AutoencoderKLMiniMaxH3Audio audio_vae_{nullptr};
  MiniMaxH3Scheduler scheduler_;
  MiniMaxH3Scheduler audio_scheduler_{/*shift=*/3.0};
#if defined(USE_NPU)
  dit::DitRollingLoadManager rolling_transformer_;
#endif
};
TORCH_MODULE(MiniMaxH3Pipeline);

REGISTER_DIT_MODEL(MiniMaxH3ModularPipeline, MiniMaxH3Pipeline);

}  // namespace xllm

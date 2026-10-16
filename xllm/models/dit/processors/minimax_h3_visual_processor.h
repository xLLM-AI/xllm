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
#include <utility>
#include <vector>

#include "core/framework/model_context/model_context.h"
#include "models/dit/processors/vae_image_processor.h"
#include "models/dit/utils/util.h"

namespace xllm {

class MiniMaxH3VisualProcessorImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VisualProcessorImpl(const ModelContext& context,
                                        int64_t canvas_multiple)
      : image_processor_(
            register_module("image_processor",
                            VAEImageProcessor(context,
                                              /*do_resize=*/true,
                                              /*do_normalize=*/false))),
        canvas_multiple_(canvas_multiple) {
    CHECK_GT(canvas_multiple_, 0);
    const ModelArgs& model_args = context.get_model_args();
    clip_length_ = model_args.clip_length();
    token_drop_ = model_args.token_drop();
    const int64_t temporal_ratio =
        dit::product_checked(model_args.temporal_downsample_factors());
    CHECK_GT(clip_length_, 0);
    latents_per_clip_ = (clip_length_ - 1) / temporal_ratio + 1;
    CHECK_GT(latents_per_clip_, 0);
    CHECK_LT(latents_per_clip_, clip_length_);
    CHECK_GE(token_drop_, 0);
    CHECK_LT(token_drop_, latents_per_clip_);
  }

  // Latent frames one whole clip contributes; the pipeline's frame-span
  // arithmetic shares this value.
  int64_t latents_per_clip() const { return latents_per_clip_; }

  std::pair<int64_t, int64_t> resolve_reference_canvas(
      int64_t source_height,
      int64_t source_width) const {
    const double ratio = reference_aspect_ratio(
        source_height, source_width, "MiniMax-H3 references");

    double width =
        ratio >= 1.0 ? static_cast<double>(reference_canvas_short_edge_) * ratio
                     : static_cast<double>(reference_canvas_short_edge_);
    double height =
        ratio >= 1.0
            ? static_cast<double>(reference_canvas_short_edge_)
            : static_cast<double>(reference_canvas_short_edge_) / ratio;
    const double area = width * height;
    if (area > reference_canvas_max_pixels_) {
      const double scale = std::sqrt(reference_canvas_max_pixels_ / area);
      width *= scale;
      height *= scale;
    }
    return {round_to_multiple(height, canvas_multiple_),
            round_to_multiple(width, canvas_multiple_)};
  }

  std::pair<int64_t, int64_t> resolve_reference_image_size(
      int64_t source_height,
      int64_t source_width) const {
    reference_aspect_ratio(
        source_height, source_width, "MiniMax-H3 reference images");
    const double scale =
        static_cast<double>(reference_image_short_edge_) /
        static_cast<double>(std::min(source_height, source_width));
    return {round_to_multiple(source_height * scale, canvas_multiple_),
            round_to_multiple(source_width * scale, canvas_multiple_)};
  }

  int64_t normalized_video_frame_count(int64_t source_frames,
                                       double source_fps,
                                       double target_fps,
                                       int64_t target_frames) const {
    CHECK_GT(source_frames, 0);
    CHECK_GT(source_fps, 0.0);
    CHECK_GT(target_fps, 0.0);
    CHECK_GT(target_frames, 0);
    int64_t frames = source_frames;
    if (source_fps != target_fps) {
      frames = round_half_up(static_cast<double>(source_frames) * target_fps /
                             source_fps);
    }
    return std::min(frames, target_frames);
  }

  int64_t reference_video_encode_frames(int64_t normalized_frames) const {
    CHECK_GT(normalized_frames, 0);
    const int64_t snapped =
        std::max<int64_t>(
            1, (normalized_frames - latents_per_clip_) / clip_length_) *
            clip_length_ +
        latents_per_clip_;
    return std::min(normalized_frames, snapped);
  }

  // Smallest multiple of the canvas multiple (VAE spatial ratio * patch)
  // that is >= num_pixels; requests smaller than one multiple keep one.
  int64_t align_canvas_pixels(int64_t num_pixels) const {
    CHECK_GT(num_pixels, 0);
    const int64_t multiple = canvas_multiple_;
    return ((num_pixels - 1) / multiple + 1) * multiple;
  }

  // Smallest frame count >= num_frames that the causal video encoder can
  // encode in whole clips (num_frames % clip_length == latents_per_clip).
  int64_t align_target_frames(int64_t num_frames) const {
    CHECK_GT(num_frames, 0);
    const int64_t remainder = num_frames % clip_length_;
    const int64_t increment =
        (latents_per_clip_ - remainder + clip_length_) % clip_length_;
    return num_frames + increment;
  }

  // Latent frame count the encoder produces for an aligned frame count:
  // whole clips contribute latents_per_clip_ frames, the first clip drops
  // the token_drop_ warm-up latents, and a single-frame clip keeps one frame.
  int64_t encode_latent_frames(int64_t num_frames) const {
    CHECK_GT(num_frames, 0);
    if (num_frames == 1) {
      return 1;
    }
    return (num_frames - latents_per_clip_) / clip_length_ * latents_per_clip_ +
           (latents_per_clip_ - token_drop_);
  }

  torch::Tensor prepare_reference_image(torch::Tensor image,
                                        int64_t target_height,
                                        int64_t target_width) {
    validate_rgb_image(image, "MiniMax-H3 reference image");
    CHECK_GT(target_height, 0);
    CHECK_GT(target_width, 0);
    image = image.to(torch::kCPU, torch::kUInt8).contiguous();
    if (image.size(1) == target_height && image.size(2) == target_width) {
      return image;
    }
    return to_uint8_pixels(image_processor_->resize(
        image, target_height, target_width, "lanczos"));
  }

  torch::Tensor prepare_reference_video(torch::Tensor video,
                                        double source_fps,
                                        double target_fps,
                                        int64_t target_frames,
                                        int64_t target_height,
                                        int64_t target_width) {
    CHECK_EQ(video.dim(), 4) << "MiniMax-H3 reference video must be [T,C,H,W]";
    CHECK_EQ(video.size(1), 3);
    CHECK_GT(video.size(0), 0);
    CHECK_GT(source_fps, 0.0);
    CHECK_GT(target_fps, 0.0);
    CHECK_GT(target_frames, 0);
    CHECK_GT(target_height, 0);
    CHECK_GT(target_width, 0);
    video = video.to(torch::kCPU, torch::kUInt8);
    if (source_fps != target_fps) {
      const double scale = target_fps / source_fps;
      const int64_t final_slot =
          round_half_up(static_cast<double>(video.size(0)) * scale);
      std::vector<int64_t> indices;
      indices.reserve(static_cast<size_t>(final_slot));
      for (int64_t frame = 0; frame < video.size(0); ++frame) {
        const int64_t start = round_half_up(static_cast<double>(frame) * scale);
        const int64_t end =
            frame + 1 < video.size(0)
                ? round_half_up(static_cast<double>(frame + 1) * scale)
                : final_slot;
        for (int64_t slot = start; slot < end; ++slot) {
          indices.emplace_back(frame);
        }
      }
      CHECK(!indices.empty())
          << "MiniMax-H3 reference video is empty after FPS resampling";
      video = video.index_select(
          0, torch::tensor(indices, torch::dtype(torch::kLong)));
    }

    video = video.slice(0, 0, std::min(video.size(0), target_frames));
    const int64_t encode_frames = reference_video_encode_frames(video.size(0));
    video = video.slice(0, 0, encode_frames);
    if (video.size(2) == target_height && video.size(3) == target_width) {
      return video.contiguous();
    }

    return to_uint8_pixels(image_processor_->resize(
        video, target_height, target_width, "lanczos"));
  }

  torch::Tensor prepare_keyframe_image(torch::Tensor image,
                                       int64_t height,
                                       int64_t width,
                                       bool is_follower) {
    validate_rgb_image(image, "MiniMax-H3 keyframe image");
    image = image.unsqueeze(0).to(torch::kFloat32);
    const int64_t source_height = image.size(2);
    const int64_t source_width = image.size(3);
    if (source_height == height && source_width == width) {
      return image.squeeze(0).to(torch::kUInt8).contiguous();
    }
    if (!is_follower) {
      return to_uint8_pixels(
          image_processor_->resize(image, height, width, "bilinear")
              .squeeze(0));
    }

    const double scale = std::max(
        static_cast<double>(width) / static_cast<double>(source_width),
        static_cast<double>(height) / static_cast<double>(source_height));
    const int64_t resized_width =
        std::max(width,
                 static_cast<int64_t>(
                     std::llround(static_cast<double>(source_width) * scale)));
    const int64_t resized_height =
        std::max(height,
                 static_cast<int64_t>(
                     std::llround(static_cast<double>(source_height) * scale)));
    torch::Tensor resized = image_processor_->resize(
        image, resized_height, resized_width, "bilinear");
    const int64_t left = std::max<int64_t>(0, (resized_width - width) / 2);
    const int64_t top = std::max<int64_t>(0, (resized_height - height) / 2);
    return to_uint8_pixels(resized.slice(2, top, top + height)
                               .slice(3, left, left + width)
                               .squeeze(0));
  }

 private:
  static double reference_aspect_ratio(int64_t source_height,
                                       int64_t source_width,
                                       const char* description) {
    CHECK_GT(source_height, 0);
    CHECK_GT(source_width, 0);
    const double ratio =
        static_cast<double>(source_width) / static_cast<double>(source_height);
    CHECK_GE(ratio, 0.25) << description << " must be within 1:4 and 4:1";
    CHECK_LE(ratio, 4.0) << description << " must be within 1:4 and 4:1";
    return ratio;
  }

  static void validate_rgb_image(const torch::Tensor& image,
                                 const char* description) {
    CHECK_EQ(image.dim(), 3) << description << " must be [C,H,W]";
    CHECK_EQ(image.size(0), 3) << description << " must have 3 channels";
  }

  // Rounds to nearest, ties away from zero -- the frame-slot arithmetic of
  // FPS resampling relies on this exact rounding (floor(x + 0.5) for x >= 0).
  static int64_t round_half_up(double value) {
    return static_cast<int64_t>(std::floor(value + 0.5));
  }

  // Converts resized floating-point pixels back to the CPU uint8 layout the
  // reference encoders consume: round to nearest, clamp to [0, 255].
  static torch::Tensor to_uint8_pixels(const torch::Tensor& pixels) {
    return pixels.round()
        .clamp(0, 255)
        .to(torch::kCPU, torch::kUInt8)
        .contiguous();
  }

  int64_t reference_image_short_edge_ = 2048;
  int64_t reference_canvas_short_edge_ = 768;
  double reference_canvas_max_pixels_ = 768.0 * 1344.0;

  static int64_t round_to_multiple(double value, int64_t multiple) {
    return std::max<int64_t>(
        multiple,
        static_cast<int64_t>(std::nearbyint(value / multiple)) * multiple);
  }

  VAEImageProcessor image_processor_{nullptr};
  int64_t canvas_multiple_;
  int64_t clip_length_ = 0;
  int64_t latents_per_clip_ = 0;
  int64_t token_drop_ = 0;
};
TORCH_MODULE(MiniMaxH3VisualProcessor);

}  // namespace xllm

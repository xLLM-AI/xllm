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

#include "processors/qwen2_vl_video_processor.h"

#include "processors/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int32_t, int32_t>;

std::optional<Size> smart_resize(int32_t height,
                                 int32_t width,
                                 int32_t factor = 28,
                                 int32_t min_pixels = 56 * 56,
                                 int32_t max_pixels = 14 * 14 * 4 * 1280) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }

  int32_t h_bar =
      static_cast<int32_t>(std::rint(height / static_cast<double>(factor))) *
      factor;
  int32_t w_bar =
      static_cast<int32_t>(std::rint(width / static_cast<double>(factor))) *
      factor;

  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
    h_bar = static_cast<int32_t>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
    h_bar = static_cast<int32_t>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}

}  // namespace

Qwen2VLVideoProcessor::Qwen2VLVideoProcessor(const ModelArgs& args) {
  image_mean_ = torch::tensor(args.mm_image_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  image_std_ = torch::tensor(args.mm_image_normalize_std(),
                             torch::dtype(torch::kFloat32));
  patch_size_ = args.mm_image_patch_size();
  temporal_patch_size_ = args.mm_image_temporal_patch_size();
  merge_size_ = args.mm_image_merge_size();
  size_ = {{"longest_edge", 12845056}, {"shortest_edge", 3136}};

  if (do_rescale_ && do_normalize_) {
    image_mean_.mul_(1.0 / rescale_factor_);
    image_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
}

torch::Tensor Qwen2VLVideoProcessor::sample_frames(
    const VideoMetadata& metadata,
    int32_t temporal_patch_size,
    int32_t min_frames,
    int32_t max_frames,
    int32_t num_frames,
    double set_fps) const {
  if (set_fps > 0.0 && num_frames > 0) {
    LOG(FATAL) << "num_frames and fps are mutually exclusive arguments, please "
                  "use only one!";
  }

  double fps = set_fps;
  int32_t total_num_frames = metadata.total_num_frames;

  if (num_frames > 0) {
    double double_num_frames =
        std::round(static_cast<double>(num_frames) / temporal_patch_size) *
        temporal_patch_size;
    num_frames = static_cast<int32_t>(double_num_frames);
  } else if (fps > 0.0) {
    if (metadata.fps <= 0.0) {
      LOG(FATAL)
          << "Asked to sample `fps` frames per second but no video metadata "
             "was provided which is required when sampling with `fps`. ";
    }

    max_frames =
        (std::min(max_frames, total_num_frames) / temporal_patch_size) *
        temporal_patch_size;
    double double_num_frames =
        static_cast<double>(total_num_frames) / metadata.fps * fps;
    double_num_frames = std::min(
        std::min(std::max(double_num_frames, static_cast<double>(min_frames)),
                 static_cast<double>(max_frames)),
        static_cast<double>(total_num_frames));
    double_num_frames = std::floor(double_num_frames / temporal_patch_size) *
                        temporal_patch_size;

    num_frames = static_cast<int32_t>(double_num_frames);
  }

  if (num_frames > total_num_frames) {
    LOG(FATAL) << "Video can't be sampled. The inferred num_frames="
               << num_frames << " exceeds total_num_frames=" << total_num_frames
               << ".";
  }

  if (num_frames > 0) {
    std::vector<int64_t> indices;
    indices.reserve(num_frames);
    for (int32_t i = 0; i < num_frames; ++i) {
      int64_t k = static_cast<int64_t>(
          (static_cast<int64_t>(i) * total_num_frames) / num_frames);
      if (k >= total_num_frames) {
        k = total_num_frames - 1;
      }
      indices.push_back(k);
    }
    return torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong));
  }

  return torch::arange(0,
                       static_cast<int64_t>(total_num_frames),
                       torch::TensorOptions().dtype(torch::kLong));
}

bool Qwen2VLVideoProcessor::process(const torch::Tensor& origin_video,
                                    const VideoMetadata& metadata,
                                    MMDataItem& output_item) const {
  torch::Tensor pixel_values;
  torch::Tensor thw;
  VideoMetadata output_metadata = metadata;
  if (!process_video(origin_video, output_metadata, pixel_values, thw)) {
    return false;
  }

  double fps = output_metadata.sampled_fps > 0.0 ? output_metadata.sampled_fps
                                                 : output_metadata.fps;
  double seconds_per_grid = static_cast<double>(temporal_patch_size_) / fps;
  torch::Tensor second_per_grid_ts = torch::tensor(
      {seconds_per_grid}, torch::TensorOptions().dtype(torch::kFloat32));
  output_item = MMDataItem(MMType::VIDEO,
                           MMDict{{"pixel_values_videos", pixel_values},
                                  {"video_grid_thw", thw},
                                  {"second_per_grid_ts", second_per_grid_ts}},
                           output_metadata);
  return true;
}

bool Qwen2VLVideoProcessor::process_video(const torch::Tensor& origin_video,
                                          VideoMetadata& metadata,
                                          torch::Tensor& pixel_values,
                                          torch::Tensor& thw) const {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices;
  if (do_sample_frame_) {
    indices = sample_frames(metadata,
                            temporal_patch_size_,
                            min_frames_,
                            max_frames_,
                            /*num_frames=*/-1,
                            /*set_fps=*/2.0);
  } else {
    indices = torch::arange(0,
                            static_cast<int64_t>(origin_video.size(0)),
                            torch::TensorOptions().dtype(torch::kLong));
  }

  auto video = origin_video.index_select(/*dim=*/0, indices);
  int64_t sampled_total_frames = video.size(0);

  metadata.frame_indices = indices;
  metadata.timestamps.clear();
  metadata.timestamps.reserve(static_cast<size_t>(sampled_total_frames));
  double fps_for_ts = (metadata.fps > 0.0) ? metadata.fps : 24.0;
  for (int64_t i = 0; i < sampled_total_frames; ++i) {
    int64_t frame_idx = metadata.frame_indices[i].item<int64_t>();
    metadata.timestamps.push_back(static_cast<double>(frame_idx) / fps_for_ts);
  }

  if (metadata.total_num_frames > 0 && metadata.fps > 0.0) {
    metadata.sampled_fps = double(sampled_total_frames) /
                           double(metadata.total_num_frames) * metadata.fps;
  } else {
    metadata.sampled_fps = fps_for_ts;
  }

  auto shape = video.sizes();
  auto channel = shape[1];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             size_.at("shortest_edge"),
                             size_.at("longest_edge"));
    if (!size) {
      return false;
    }
    std::tie(resized_height, resized_width) = *size;
  }

  auto out_video = video;
  if (do_resize_) {
    out_video = transforms::resize(
        out_video, {resized_height, resized_width}, resample_, true);
  }
  if (do_normalize_) {
    out_video = transforms::normalize(out_video, image_mean_, image_std_);
  }
  if (do_rescale_) {
    out_video = transforms::rescale(out_video, rescale_factor_);
  }

  auto pad_t =
      (temporal_patch_size_ - (out_video.size(0) % temporal_patch_size_)) %
      temporal_patch_size_;
  if (pad_t != 0) {
    auto last = out_video.index({out_video.size(0) - 1})
                    .unsqueeze(0)
                    .repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;
  auto grid_t = shape[0] / temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 temporal_patch_size_,
                                 channel,
                                 grid_h / merge_size_,
                                 merge_size_,
                                 patch_size_,
                                 grid_w / merge_size_,
                                 merge_size_,
                                 patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).reshape({-1, 3});

  return true;
}

}  // namespace xllm

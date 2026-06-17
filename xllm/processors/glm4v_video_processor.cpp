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

#include "processors/glm4v_video_processor.h"

#include <cfenv>
#include <cmath>
#include <unordered_set>

#include "processors/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int32_t, int32_t>;

std::optional<Size> smart_resize(int32_t num_frames,
                                 int32_t height,
                                 int32_t width,
                                 int32_t temporal_factor,
                                 int32_t factor = 28,
                                 int32_t min_pixels = 56 * 56,
                                 int32_t max_pixels = 14 * 14 * 4 * 1280) {
  if (height < factor || width < factor) {
    LOG(ERROR) << "Height or width must be larger than factor";
    return std::nullopt;
  }
  if (num_frames < temporal_factor) {
    LOG(ERROR) << "num_frames must be larger than temporal_factor, num_frames: "
               << num_frames << ", temporal_factor: " << temporal_factor;
    return std::nullopt;
  }

  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }
  int32_t t_bar = static_cast<int32_t>(std::rint(
                      num_frames / static_cast<double>(temporal_factor))) *
                  temporal_factor;
  int32_t h_bar =
      static_cast<int32_t>(std::rint(height / static_cast<double>(factor))) *
      factor;
  int32_t w_bar =
      static_cast<int32_t>(std::rint(width / static_cast<double>(factor))) *
      factor;

  if (t_bar * h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((num_frames * height * width) /
                            static_cast<double>(max_pixels));
    h_bar = static_cast<int32_t>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (t_bar * h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels /
                            static_cast<double>(height * width * num_frames));
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

Glm4VVideoProcessor::Glm4VVideoProcessor(const ModelArgs& args) {
  video_mean_ = torch::tensor(args.mm_video_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  video_std_ = torch::tensor(args.mm_video_normalize_std(),
                             torch::dtype(torch::kFloat32));

  video_min_pixels_ = args.mm_video_shortest_edge();
  video_max_pixels_ = args.mm_video_longest_edge();

  video_patch_size_ = args.mm_video_patch_size();
  video_temporal_patch_size_ = args.mm_video_temporal_patch_size();
  video_merge_size_ = args.mm_video_merge_size();

  if (do_rescale_ && do_normalize_) {
    video_mean_.mul_(1.0 / rescale_factor_);
    video_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
}

torch::Tensor Glm4VVideoProcessor::sample_frames(
    const VideoMetadata& metadata) const {
  const int32_t total_frames = metadata.total_num_frames;
  if (total_frames <= 0) {
    return torch::empty({0}, torch::dtype(torch::kLong));
  }

  if (metadata.fps <= 0.0) {
    LOG(FATAL) << "invalid metadata.fps <= 0";
  }

  const int32_t max_frame_idx = total_frames - 1;

  double duration = metadata.duration;
  if (duration <= 0.0) {
    duration =
        std::round(static_cast<double>(max_frame_idx) / metadata.fps) + 1.0;
  }

  constexpr double DYN_FPS_30 = 3.0;
  constexpr double DYN_FPS_300 = 1.0;
  constexpr double DYN_FPS_2400 = 0.5;
  constexpr int32_t MAX_FRAME_COUNT_DYNAMIC = 640;
  constexpr double MAX_DURATION = 2400.0;

  const double effective_duration = std::min(duration, MAX_DURATION);

  double target_fps = 0.0;
  if (effective_duration <= 30.0) {
    target_fps = DYN_FPS_30;
  } else if (effective_duration <= 300.0) {
    target_fps = DYN_FPS_300;
  } else {
    target_fps = DYN_FPS_2400;
  }

  int32_t extract_t =
      static_cast<int32_t>(effective_duration * target_fps *
                           static_cast<double>(video_temporal_patch_size_));
  extract_t = std::min(extract_t, MAX_FRAME_COUNT_DYNAMIC);

  const double duration_per_frame = 1.0 / metadata.fps;
  std::vector<double> timestamps(total_frames);
  for (int32_t i = 0; i < total_frames; ++i) {
    timestamps[i] = static_cast<double>(i) * duration_per_frame;
  }
  const int32_t max_second = static_cast<int32_t>(duration);

  torch::Tensor frame_indices;

  if (total_frames < extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  } else {
    std::vector<int64_t> tmp;
    tmp.reserve(static_cast<size_t>(total_frames));
    double current_second = 0.0;
    const double inv_fps =
        1.0 / (static_cast<double>(video_temporal_patch_size_) * target_fps);

    for (int32_t frame_index = 0; frame_index < total_frames; frame_index++) {
      if (timestamps[frame_index] >= current_second) {
        current_second += inv_fps;
        tmp.push_back(frame_index);
        if (current_second >= static_cast<double>(max_second)) {
          break;
        }
      }
    }
    frame_indices =
        torch::tensor(tmp, torch::TensorOptions().dtype(torch::kLong));
  }
  int64_t len = frame_indices.size(0);
  if (len < extract_t) {
    int64_t start, end;
    if (len == 0) {
      start = 0;
      end = std::max<int64_t>(total_frames - 1, 0);
    } else {
      start = frame_indices[0].item<int64_t>();
      end = frame_indices[len - 1].item<int64_t>();
    }
    frame_indices =
        torch::linspace(start, end, extract_t, torch::dtype(torch::kLong));
  } else if (len > extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  }

  len = frame_indices.size(0);
  std::unordered_set<int64_t> seen;
  seen.reserve(static_cast<size_t>(len) * 2);
  std::vector<int64_t> uniq;
  uniq.reserve(static_cast<size_t>(len));

  for (int64_t i = 0; i < len; ++i) {
    auto idx = frame_indices[i].item<int64_t>();
    if (seen.insert(idx).second) {
      uniq.push_back(idx);
    }
  }

  if (!uniq.empty() && (uniq.size() & 1)) {
    uniq.push_back(uniq.back());
  }

  return torch::tensor(uniq, torch::TensorOptions().dtype(torch::kLong));
}

bool Glm4VVideoProcessor::process(const torch::Tensor& origin_video,
                                  const VideoMetadata& metadata,
                                  MMDataItem& output_item) const {
  torch::Tensor pixel_values;
  torch::Tensor thw;
  VideoMetadata output_metadata = metadata;
  if (!process_video(origin_video, output_metadata, pixel_values, thw)) {
    return false;
  }
  output_item = MMDataItem(
      MMType::VIDEO,
      MMDict{{"pixel_values_videos", pixel_values}, {"video_grid_thw", thw}},
      output_metadata);
  return true;
}

bool Glm4VVideoProcessor::process_video(const torch::Tensor& origin_video,
                                        VideoMetadata& metadata,
                                        torch::Tensor& pixel_values,
                                        torch::Tensor& thw) const {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices;
  if (do_sample_frame_) {
    indices = sample_frames(metadata);
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
  auto time_len = shape[0];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  if (do_resize_) {
    auto size = smart_resize(time_len,
                             resized_height,
                             resized_width,
                             video_temporal_patch_size_,
                             video_patch_size_ * video_merge_size_,
                             video_min_pixels_,
                             video_max_pixels_);
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
    out_video = transforms::normalize(out_video, video_mean_, video_std_);
  }
  if (do_rescale_) {
    out_video = transforms::rescale(out_video, rescale_factor_);
  }

  auto pad_t = (video_temporal_patch_size_ -
                (out_video.size(0) % video_temporal_patch_size_)) %
               video_temporal_patch_size_;
  if (pad_t != 0) {
    auto last = out_video.index({out_video.size(0) - 1})
                    .unsqueeze(0)
                    .repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / video_patch_size_;
  auto grid_w = resized_width / video_patch_size_;
  auto grid_t = shape[0] / video_temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 video_temporal_patch_size_,
                                 channel,
                                 grid_h / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_,
                                 grid_w / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape({grid_t * grid_h * grid_w,
                             channel * video_temporal_patch_size_ *
                                 video_patch_size_ * video_patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).reshape({-1, 3});

  return true;
}

}  // namespace xllm

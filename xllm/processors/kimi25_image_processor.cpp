/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#include "processors/kimi25_image_processor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

#include "glog/logging.h"
#include "processors/transforms.h"

namespace xllm {
namespace {

KimiK25MediaConfig make_kimi_k25_media_config(const ModelArgs& args) {
  KimiK25MediaConfig config;
  config.patch_size =
      args.mm_km_patch_size() > 0 ? args.mm_km_patch_size() : 14;
  config.merge_kernel_size =
      args.mm_km_merge_kernel_size() > 0 ? args.mm_km_merge_kernel_size() : 2;
  config.temporal_merge_kernel_size =
      args.mm_km_temporal_merge_kernel_size() > 0
          ? args.mm_km_temporal_merge_kernel_size()
          : 4;

  config.in_patch_limit =
      args.mm_km_in_patch_limit() > 0 ? args.mm_km_in_patch_limit() : 16384;
  config.in_patch_limit_each_frame =
      args.mm_km_in_patch_limit_each_frame() > 0
          ? args.mm_km_in_patch_limit_each_frame()
          : -1;
  config.patch_limit_on_one_side = args.mm_km_patch_limit_on_one_side() > 0
                                       ? args.mm_km_patch_limit_on_one_side()
                                       : 512;
  config.in_patch_limit_video = args.mm_km_in_patch_limit_video() > 0
                                    ? args.mm_km_in_patch_limit_video()
                                    : -1;
  config.fixed_output_tokens = args.mm_km_fixed_output_tokens();

  config.sample_fps =
      args.mm_km_sample_fps() > 0 ? args.mm_km_sample_fps() : 2.0;
  config.max_num_frames_each_video =
      args.mm_km_max_num_frames_each_video() > 0
          ? args.mm_km_max_num_frames_each_video()
          : 2;
  config.min_frames = 4;
  config.timestamp_mode = args.mm_km_timestamp_mode();

  if (!args.mm_km_image_mean().empty()) {
    config.image_mean.assign(args.mm_km_image_mean().begin(),
                             args.mm_km_image_mean().end());
  } else {
    config.image_mean = {0.5, 0.5, 0.5};
  }

  if (!args.mm_km_image_std().empty()) {
    config.image_std.assign(args.mm_km_image_std().begin(),
                            args.mm_km_image_std().end());
  } else {
    config.image_std = {0.5, 0.5, 0.5};
  }

  config.do_convert_rgb = true;
  config.do_normalize = true;
  config.do_resize = true;

  config.resample = 3;
  return config;
}

torch::Tensor km_normalize(const torch::Tensor& image,
                           const std::vector<double>& mean,
                           const std::vector<double>& std) {
  if (image.dim() != 3) {
    LOG(FATAL)
        << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  }

  int numChannels = image.size(0);
  if (mean.size() != numChannels || std.size() != numChannels) {
    LOG(FATAL) << "Mean and std vectors must have the same number "
               << "of elements as the number of channels in the "
               << "image.";
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  result = result / 255.0;

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto m_tensor = torch::tensor(mean, options).reshape({-1, 1, 1});
  auto s_tensor = torch::tensor(std, options).reshape({-1, 1, 1});

  auto s_inv_tensor = 1.0 / s_tensor;

  result = result.sub(m_tensor);
  return result.mul_(s_inv_tensor);
}

}  // namespace

KimiK25ImageProcessor::KimiK25ImageProcessor(const ModelArgs& args)
    : config_(make_kimi_k25_media_config(args)) {}

KimiK25VideoProcessor::KimiK25VideoProcessor(const ModelArgs& args)
    : config_(make_kimi_k25_media_config(args)) {}

bool KimiK25ImageProcessor::process_image(
    const std::vector<torch::Tensor>& images,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<torch::Tensor>& thw) const {
  pixel_values.clear();
  thw.clear();
  pixel_values.reserve(images.size());
  thw.reserve(images.size());
  for (const torch::Tensor& image : images) {
    torch::Tensor image_pixel_values;
    torch::Tensor image_thw;
    if (!process_image(image, image_pixel_values, image_thw)) {
      return false;
    }
    pixel_values.push_back(std::move(image_pixel_values));
    thw.push_back(std::move(image_thw));
  }
  return true;
}

bool KimiK25ImageProcessor::process(
    const std::vector<torch::Tensor>& images,
    std::vector<MMDataItem>& output_items) const {
  std::vector<torch::Tensor> pixel_values;
  std::vector<torch::Tensor> thw;
  if (!process_image(images, pixel_values, thw)) {
    return false;
  }

  output_items.clear();
  output_items.reserve(images.size());
  const size_t image_size = images.size();
  for (size_t index = 0; index < image_size; ++index) {
    output_items.emplace_back(MMType::IMAGE,
                              MMDict{{"pixel_values", pixel_values[index]},
                                     {"image_grid_thw", thw[index]}});
  }
  return true;
}

std::vector<VideoChunkMetadata> KimiK25VideoProcessor::split_video_chunks(
    const VideoMetadata& video_meta) const {
  std::vector<VideoChunkMetadata> chunks;

  torch::Tensor frame_indices = sample_frames(video_meta);
  std::vector<int64_t> indices(
      frame_indices.data_ptr<int64_t>(),
      frame_indices.data_ptr<int64_t>() + frame_indices.numel());

  int chunk_size = config_.temporal_merge_kernel_size;
  int num_chunks = (indices.size() + chunk_size - 1) / chunk_size;

  for (int i = 0; i < num_chunks; ++i) {
    VideoChunkMetadata chunk;
    chunk.chunk_id = i;

    int start = i * chunk_size;
    int end = std::min((i + 1) * chunk_size, static_cast<int>(indices.size()));
    chunk.frame_indices =
        std::vector<int64_t>(indices.begin() + start, indices.begin() + end);
    chunk.num_frames = chunk.frame_indices.size();

    double start_time =
        static_cast<double>(chunk.frame_indices[0]) / video_meta.fps;
    chunk.start_timestamp = start_time;

    chunk.timestamp_text = timestamp_as_str(start_time, config_.timestamp_mode);

    chunks.push_back(chunk);
  }

  return chunks;
}

torch::Tensor KimiK25VideoProcessor::sample_frames(
    const VideoMetadata& metadata) const {
  int32_t total_frames = metadata.total_num_frames;
  double sample_fps = std::min(config_.sample_fps, metadata.fps);

  int32_t sampled_frames =
      std::max(static_cast<int32_t>(
                   std::round(total_frames * sample_fps / metadata.fps)),
               1);

  torch::Tensor frame_inds_float =
      torch::linspace(0, static_cast<double>(total_frames - 1), sampled_frames);
  torch::Tensor frame_inds = frame_inds_float.round().to(torch::kLong);

  return frame_inds;
}

std::string KimiK25VideoProcessor::timestamp_as_str(
    double timestamp,
    const std::string& timestamp_mode) const {
  int32_t hours = static_cast<int32_t>(timestamp) / 3600;
  int32_t minutes = (static_cast<int32_t>(timestamp) % 3600) / 60;
  int32_t seconds = static_cast<int32_t>(timestamp) % 60;
  int32_t milliseconds = static_cast<int32_t>(
      (timestamp - static_cast<int32_t>(timestamp)) * 1000);

  char buffer[32];
  if (timestamp_mode == "hh:mm:ss.fff") {
    snprintf(buffer,
             sizeof(buffer),
             "%02d:%02d:%02d.%03d",
             hours,
             minutes,
             seconds,
             milliseconds);
  } else if (timestamp_mode == "mm:ss.fff") {
    snprintf(buffer,
             sizeof(buffer),
             "%02d:%02d.%03d",
             minutes,
             seconds,
             milliseconds);
  } else if (timestamp_mode == "mm:ss") {
    snprintf(buffer, sizeof(buffer), "%02d:%02d", minutes, seconds);
  } else {
    LOG(ERROR) << "Invalid timestamp mode: " << timestamp_mode;
    return "";
  }
  return std::string(buffer);
}

namespace {

std::optional<NavitResizeResult> navit_resize_image(
    int width,
    int height,
    int patch_size,
    int merge_kernel_size,
    int in_patch_limit,
    int patch_limit_on_one_side,
    std::optional<int> fixed_output_tokens) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Aspect ratio exceeds 200";
    return std::nullopt;
  }

  NavitResizeResult result;
  double s1 =
      std::sqrt(in_patch_limit /
                (std::max(1.0, static_cast<double>(width / patch_size)) *
                 std::max(1.0, static_cast<double>(height / patch_size))));
  double s2 = static_cast<double>(patch_limit_on_one_side * patch_size) / width;
  double s3 =
      static_cast<double>(patch_limit_on_one_side * patch_size) / height;
  double scale = std::min({1.0, s1, s2, s3});

  int new_w = std::max(1, static_cast<int>(width * scale));
  int new_h = std::max(1, static_cast<int>(height * scale));
  new_w = std::min(new_w, patch_limit_on_one_side * patch_size);
  new_h = std::min(new_h, patch_limit_on_one_side * patch_size);

  int factor = merge_kernel_size * patch_size;

  int pad_height = (factor - new_h % factor) % factor;
  int pad_width = (factor - new_w % factor) % factor;

  int num_tokens;
  if (fixed_output_tokens.has_value()) {
    num_tokens = fixed_output_tokens.value();
  } else {
    int token_height = (new_h + pad_height) / factor;
    int token_width = (new_w + pad_width) / factor;

    if (token_height * merge_kernel_size > patch_limit_on_one_side) {
      LOG(ERROR) << "token_height " << token_height << " * merge_kernel_size "
                 << merge_kernel_size << " > patch_limit_on_one_side "
                 << patch_limit_on_one_side;
      return std::nullopt;
    }
    if (token_width * merge_kernel_size > patch_limit_on_one_side) {
      LOG(ERROR) << "token_width " << token_width << " * merge_kernel_size "
                 << merge_kernel_size << " > patch_limit_on_one_side "
                 << patch_limit_on_one_side;
      return std::nullopt;
    }

    num_tokens = token_height * token_width;
  }

  result.sampled_nframes = 1;
  result.num_tokens = num_tokens;
  result.new_width = new_w;
  result.new_height = new_h;
  result.pad_width = pad_width;
  result.pad_height = pad_height;

  return result;
}

std::optional<NavitResizeResult> navit_resize_video(
    int width,
    int height,
    int nframes,
    double avg_fps,
    double sample_fps,
    int patch_size,
    int merge_kernel_size,
    int in_patch_limit_each_frame,
    int patch_limit_on_one_side,
    std::optional<int> in_patch_limit_total,
    std::optional<int> max_num_frames_each_video,
    std::optional<int> fixed_output_tokens_each_frame) {
  sample_fps = std::min(sample_fps, avg_fps);
  int sampled_nframes =
      std::max(static_cast<int>(std::round(nframes * sample_fps / avg_fps)), 1);
  if (max_num_frames_each_video.has_value()) {
    sampled_nframes =
        std::min(sampled_nframes, max_num_frames_each_video.value());
  }

  if (in_patch_limit_total.has_value()) {
    int per_frame_limit = static_cast<int>(std::round(
        static_cast<double>(in_patch_limit_total.value()) / sampled_nframes));
    in_patch_limit_each_frame =
        std::min(per_frame_limit, in_patch_limit_each_frame);
  }

  auto ret = navit_resize_image(width,
                                height,
                                patch_size,
                                merge_kernel_size,
                                in_patch_limit_each_frame,
                                patch_limit_on_one_side,
                                fixed_output_tokens_each_frame);

  if (ret) {
    ret->sampled_nframes = sampled_nframes;
  }
  return ret;
}

std::optional<NavitResizeResult> navit_resize(const KimiK25MediaConfig& config,
                                              int32_t height,
                                              int32_t width,
                                              bool is_video,
                                              int32_t nframes = 1,
                                              double avg_fps = 24.0) {
  if (!is_video) {
    std::optional<int> fixed_output_tokens =
        config.fixed_output_tokens >= 0
            ? std::optional<int>(config.fixed_output_tokens)
            : std::nullopt;
    return navit_resize_image(width,
                              height,
                              config.patch_size,
                              config.merge_kernel_size,
                              config.in_patch_limit,
                              config.patch_limit_on_one_side,
                              fixed_output_tokens);
  } else {
    std::optional<int> in_patch_limit_total_opt =
        config.in_patch_limit_video > 0
            ? std::optional<int>(config.in_patch_limit_video)
            : std::nullopt;

    double sample_fps = std::numeric_limits<double>::infinity();
    std::optional<int> max_num_frames_each_video_opt = std::nullopt;

    int32_t in_patch_limit_each_frame = config.in_patch_limit_each_frame > 0
                                            ? config.in_patch_limit_each_frame
                                            : config.in_patch_limit;
    std::optional<int> fixed_output_tokens_opt =
        config.fixed_output_tokens >= 0
            ? std::optional<int>(config.fixed_output_tokens)
            : std::nullopt;

    return navit_resize_video(width,
                              height,
                              nframes,
                              avg_fps,
                              sample_fps,
                              config.patch_size,
                              config.merge_kernel_size,
                              in_patch_limit_each_frame,
                              config.patch_limit_on_one_side,
                              in_patch_limit_total_opt,
                              max_num_frames_each_video_opt,
                              fixed_output_tokens_opt);
  }
}

std::pair<torch::Tensor, torch::Tensor> navit_patchify(torch::Tensor pixels,
                                                       int32_t patch_size) {
  int64_t t = pixels.size(0);
  int64_t c = pixels.size(1);
  int64_t h = pixels.size(2);
  int64_t w = pixels.size(3);

  torch::Tensor patches = pixels.permute({0, 2, 3, 1});

  patches = patches.view(
      {t, h / patch_size, patch_size, w / patch_size, patch_size, c});

  patches = patches.permute({0, 1, 3, 5, 2, 4});

  patches = patches.reshape(
      {t * (h / patch_size) * (w / patch_size), c, patch_size, patch_size});

  torch::Tensor grid_thw =
      torch::tensor({t, h / patch_size, w / patch_size}, torch::kInt64);

  return {patches, grid_thw};
}

}  // namespace

bool KimiK25ImageProcessor::process_image(torch::Tensor image,
                                          torch::Tensor& pixel_values,
                                          torch::Tensor& thw) const {
  if (image.dim() != 3) {
    LOG(ERROR) << "Image must be CHW format";
    return false;
  }

  auto shape = image.sizes();
  int64_t height = shape[1];
  int64_t width = shape[2];

  if (config_.do_resize) {
    auto resize_result = navit_resize(config_,
                                      static_cast<int32_t>(height),
                                      static_cast<int32_t>(width),
                                      /*is_video=*/false);
    if (!resize_result) {
      return false;
    }

    int32_t new_h = resize_result->new_height;
    int32_t new_w = resize_result->new_width;
    int32_t pad_height = resize_result->pad_height;
    int32_t pad_width = resize_result->pad_width;

    image = transforms::resize(image, {new_h, new_w}, config_.resample, false);

    if (pad_height > 0 || pad_width > 0) {
      std::vector<int64_t> padding = {
          0,
          pad_width,
          0,
          pad_height,
          0,
          0};  // [left, right, top, bottom, front, back]
      image = torch::nn::functional::pad(
          image,
          torch::nn::functional::PadFuncOptions(padding)
              .mode(torch::kConstant)
              .value(0));
    }

    height = new_h + pad_height;
    width = new_w + pad_width;
  }

  if (config_.do_normalize) {
    image = km_normalize(image, config_.image_mean, config_.image_std);
  }

  image = image.unsqueeze(0);

  auto [patches, grid_thw_tensor] = navit_patchify(image, config_.patch_size);
  pixel_values = patches;

  thw = grid_thw_tensor.unsqueeze(0);

  return true;
}

bool KimiK25VideoProcessor::process(const torch::Tensor& origin_video,
                                    const VideoMetadata& metadata,
                                    MMDataItem& output_item) const {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  VideoMetadata output_metadata = metadata;
  torch::Tensor pixel_values;
  torch::Tensor thw;

  std::vector<VideoChunkMetadata> chunks = split_video_chunks(output_metadata);

  double seconds_per_grid = 0;
  for (const VideoChunkMetadata& chunk : chunks) {
    torch::Tensor pixel_values_chunk;
    torch::Tensor thw_chunk;

    torch::Tensor chunk_frames = origin_video.index_select(
        /*dim=*/0, torch::tensor(chunk.frame_indices, torch::kLong));

    if (!process_video_chunk(
            chunk_frames, output_metadata, pixel_values_chunk, thw_chunk)) {
      return false;
    }

    double single_seconds_per_grid =
        static_cast<double>(chunk.num_frames) / output_metadata.fps;
    seconds_per_grid += single_seconds_per_grid;

    if (pixel_values.numel() == 0) {
      pixel_values = pixel_values_chunk;
      thw = thw_chunk;
    } else {
      pixel_values = torch::cat({pixel_values, pixel_values_chunk}, 0);
      thw = torch::cat({thw, thw_chunk}, 0);
    }
  }

  torch::Tensor second_per_grid_ts = torch::tensor({seconds_per_grid}, opts);

  output_item = MMDataItem(MMType::VIDEO,
                           MMDict{{"pixel_values_videos", pixel_values},
                                  {"video_grid_thw", thw},
                                  {"second_per_grid_ts", second_per_grid_ts}},
                           output_metadata);
  return true;
}

bool KimiK25VideoProcessor::process_video_chunk(torch::Tensor video_chunk,
                                                const VideoMetadata& video_meta,
                                                torch::Tensor& pixel_values,
                                                torch::Tensor& thw) const {
  if (video_chunk.dim() != 4) {
    LOG(ERROR) << "Video chunk must be TCHW format";
    return false;
  }

  auto shape = video_chunk.sizes();
  int64_t t = shape[0];
  int64_t h = shape[2];
  int64_t w = shape[3];

  std::vector<torch::Tensor> processed_frames;
  processed_frames.reserve(static_cast<size_t>(t));

  auto frames = video_chunk.unbind(0);
  for (auto& frame : frames) {
    if (config_.do_resize) {
      auto resize_result = navit_resize(config_,
                                        static_cast<int32_t>(h),
                                        static_cast<int32_t>(w),
                                        /*is_video=*/true,
                                        static_cast<int32_t>(t),
                                        /*avg_fps=*/1.0);
      if (!resize_result) {
        return false;
      }

      int32_t new_h = resize_result->new_height;
      int32_t new_w = resize_result->new_width;
      int32_t pad_height = resize_result->pad_height;
      int32_t pad_width = resize_result->pad_width;

      frame = transforms::resize(frame, {new_h, new_w}, config_.resample, true);

      if (pad_height > 0 || pad_width > 0) {
        std::vector<int64_t> padding = {
            0,
            pad_width,
            0,
            pad_height,
            0,
            0};  // [left, right, top, bottom, front, back]
        frame = torch::nn::functional::pad(
            frame,
            torch::nn::functional::PadFuncOptions(padding)
                .mode(torch::kConstant)
                .value(0));
      }

      h = new_h + pad_height;
      w = new_w + pad_width;
    }

    if (config_.do_normalize) {
      frame = km_normalize(frame, config_.image_mean, config_.image_std);
    }

    processed_frames.emplace_back(frame);
  }

  torch::Tensor processed_video = torch::stack(processed_frames);

  auto [patches, grid_thw_tensor] =
      navit_patchify(processed_video, config_.patch_size);
  pixel_values = patches;

  thw = grid_thw_tensor.unsqueeze(0);

  return true;
}

KimiK25PromptProcessor::KimiK25PromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size() > 0
                    ? args.mm_image_merge_size()
                    : std::max<int32_t>(args.mm_spatial_merge_size(), 2);
  vision_start_token_id_ = args.vision_start_token_id();
  vision_token_id_ = args.vision_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
}

void KimiK25PromptProcessor::process(std::string& prompt,
                                     const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = res.value();
  }

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) {
    return;
  }

  std::vector<int32_t> image_token_counts =
      get_media_token_counts(image_grid_thw);
  std::vector<int32_t> video_token_counts =
      get_media_token_counts(video_grid_thw);

  int32_t total_token = 0;
  for (int32_t token_count : image_token_counts) {
    total_token += token_count;
  }
  for (int32_t token_count : video_token_counts) {
    total_token += token_count;
  }

  std::string data;
  data.reserve(prompt.size() + total_token * media_pad_token_.size());

  int32_t image_index = 0;
  int32_t video_index = 0;
  int32_t* index = nullptr;
  const std::vector<int32_t>* token_counts = nullptr;
  std::string prefix;

  size_t begin = 0;
  auto pair = find_media_prompt(prompt, begin);

  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);

    if (pair.first == TokenType::IMAGE) {
      token_counts = &image_token_counts;
      index = &image_index;
      prefix = image_prompt_prefix_;
    } else if (pair.first == TokenType::VIDEO) {
      token_counts = &video_token_counts;
      index = &video_index;
      prefix = video_prompt_prefix_;
    } else {
      LOG(FATAL) << "Unexpected Kimi media token type.";
    }

    CHECK(token_counts != nullptr);
    CHECK_LT(*index, static_cast<int32_t>(token_counts->size()))
        << "media placeholder count does not match processed media count";

    data.append(prefix);
    int32_t token_num = (*token_counts)[*index];
    while (token_num > 0) {
      data.append(media_pad_token_);
      --token_num;
    }
    data.append(media_prompt_suffix_);

    ++(*index);
    begin =
        pair.second + (pair.first == TokenType::IMAGE ? image_prompt_.size()
                                                      : video_prompt_.size());
    pair = find_media_prompt(prompt, begin);
  }

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }

  prompt = std::move(data);
}

void KimiK25PromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto start = token_ids.begin();
  size_t global_mm_index = 0;
  auto& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto vision_start_it =
        std::find(start, token_ids.end(), vision_start_token_id_);
    if (vision_start_it == token_ids.end()) {
      break;
    }

    auto vision_content_it =
        std::find(vision_start_it, token_ids.end(), vision_token_id_);
    CHECK(vision_content_it != token_ids.end())
        << "missing media content token after media begin";
    auto vision_end_it =
        std::find(vision_content_it, token_ids.end(), vision_end_token_id_);
    CHECK(vision_end_it != token_ids.end())
        << "missing media end token after media content";
    auto media_pad_begin = std::next(vision_content_it);
    int32_t offset =
        static_cast<int32_t>(std::distance(token_ids.begin(), media_pad_begin));
    int32_t length =
        static_cast<int32_t>(std::distance(media_pad_begin, vision_end_it));

    CHECK_LT(global_mm_index, mm_items.size())
        << "media span count exceeds mm item count";
    MMDataItem& item = mm_items[global_mm_index];
    item.mutable_state().mutable_token_pos() = {offset, length};
    ++global_mm_index;
    start = std::next(vision_end_it);
  }
}

std::vector<int32_t> KimiK25PromptProcessor::get_media_token_counts(
    const torch::Tensor& grid_thw) const {
  std::vector<int32_t> token_counts;
  if (!grid_thw.defined()) {
    return token_counts;
  }

  int32_t merge_length = merge_size_ * merge_size_;
  CHECK_GT(merge_length, 0)
      << "merge_length must be positive, merge_size_=" << merge_size_;
  int64_t count = grid_thw.sizes()[0];
  token_counts.reserve(static_cast<size_t>(count));
  for (int64_t idx = 0; idx < count; ++idx) {
    token_counts.emplace_back(grid_thw[idx].prod().item<int32_t>() /
                              merge_length);
  }
  return token_counts;
}

std::pair<KimiK25PromptProcessor::TokenType, size_t>
KimiK25PromptProcessor::find_media_prompt(const std::string& prompt,
                                          size_t begin) const {
  size_t img_pos = prompt.find(image_prompt_, begin);
  size_t vid_pos = prompt.find(video_prompt_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos) {
    return {TokenType::INVALID, std::string::npos};
  }
  if (vid_pos == std::string::npos) {
    return {TokenType::IMAGE, img_pos};
  }
  if (img_pos == std::string::npos) {
    return {TokenType::VIDEO, vid_pos};
  }
  return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                           : std::make_pair(TokenType::VIDEO, vid_pos);
}

}  // namespace xllm

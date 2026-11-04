/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xllm {

inline constexpr int64_t kMiniMaxH3TextId = -5;
inline constexpr int64_t kMiniMaxH3ImageVideoConditionId = -11;
inline constexpr int64_t kMiniMaxH3AudioReferenceConditionId = -17;
inline constexpr int64_t kMiniMaxH3AudioFirstId = -15;
inline constexpr int64_t kMiniMaxH3AudioId = -14;
inline constexpr int64_t kMiniMaxH3VideoFirstId = -3;
inline constexpr int64_t kMiniMaxH3VideoId = -2;
inline constexpr int64_t kMiniMaxH3VideoLastId = -4;
inline constexpr int64_t kMiniMaxH3PadId = -1;

enum class H3ReferenceBlockKind : int32_t {
  IMAGE = 0,
  AUDIO = 1,
  VIDEO = 2,
  VIDEO_AUDIO = 3,
};

enum class H3VideoSpanRole : int32_t {
  REFERENCE = 0,
  TARGET = 1,
};

struct H3Slice {
  int64_t start = 0;
  int64_t stop = 0;
};

struct H3LatentGrid {
  int64_t temporal = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct H3ReferenceBlock {
  H3ReferenceBlockKind kind = H3ReferenceBlockKind::IMAGE;
  int64_t ref_audio_t = 0;
  int64_t latent_t = 0;
  int64_t latent_h = 0;
  int64_t latent_w = 0;
};

struct H3TargetLatents {
  int64_t audio_t = 0;
  int64_t audio_channels = 2;
  int64_t latent_t = 0;
  int64_t latent_h = 0;
  int64_t latent_w = 0;
};

struct H3PhysicalReferenceBlock {
  H3ReferenceBlock block;
  int64_t packed_offset = 0;
  H3Slice row_slice;
  std::optional<H3Slice> audio_slice;
  std::optional<H3Slice> visual_slice;
  double temporal_origin = 0.0;
  double temporal_extent = 0.0;
};

struct H3VideoSpan {
  int64_t start = 0;
  H3LatentGrid latent_grid;
  H3VideoSpanRole role = H3VideoSpanRole::REFERENCE;
};

struct H3VideoLatentShape {
  int64_t channels = 0;
  int64_t temporal = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct H3PackedLayout {
  int64_t used_length = 0;
  int64_t aligned_length = 0;
  torch::Tensor condition_hidden;
  torch::Tensor input_ids;
  torch::Tensor image_mask;
  torch::Tensor audio_mask;
  torch::Tensor img_pos;
  torch::Tensor audio_pos;
  torch::Tensor text_pos;
  torch::Tensor update_mask;
  torch::Tensor audio_update_mask;
  torch::Tensor position_ids;
  torch::Tensor token_tags;
  torch::Tensor cu_seqlens;
  torch::Tensor document_id;
  H3LatentGrid target_latent_grid;
  H3Slice text_slice;
  H3Slice target_audio_slice;
  H3Slice target_video_slice;
  H3Slice padding_slice;
  double target_temporal_origin = 0.0;
  std::vector<H3PhysicalReferenceBlock> reference_blocks;
  std::vector<H3VideoSpan> video_spans;
};

namespace minimax_h3_packing_detail {

inline constexpr int64_t kSequenceAlignment = 64;
inline constexpr int64_t kPatchHeight = 2;
inline constexpr int64_t kPatchWidth = 2;
inline constexpr int64_t kTextDimension = 5120;
inline constexpr double kPositionInterpolation = 32.0;
inline constexpr double kFrameRescale = 5.0 / 3.0;
inline constexpr int64_t kTemporalGroup = 5;
inline constexpr int64_t kFramesPerToken[kTemporalGroup] = {1, 4, 4, 4, 4};

[[noreturn]] inline void fail(const std::string& message) {
  throw std::invalid_argument("MiniMax-H3 packing " + message);
}

inline int64_t checked_add(int64_t left,
                           int64_t right,
                           const std::string& field) {
  if (left < 0 || right < 0 ||
      left > std::numeric_limits<int64_t>::max() - right) {
    fail(field + " row count overflows int64");
  }
  return left + right;
}

inline int64_t checked_multiply(int64_t left,
                                int64_t right,
                                const std::string& field) {
  if (left < 0 || right < 0 ||
      (right != 0 && left > std::numeric_limits<int64_t>::max() / right)) {
    fail(field + " row count overflows int64");
  }
  return left * right;
}

inline void require_positive(int64_t value, const std::string& field) {
  if (value <= 0) {
    fail(field + " must be positive");
  }
}

inline void require_even_spatial(int64_t value, const std::string& field) {
  require_positive(value, field);
  if (value % 2 != 0) {
    fail(field + " must be divisible by 2");
  }
}

inline int64_t visual_rows(int64_t temporal,
                           int64_t height,
                           int64_t width,
                           const std::string& field) {
  const int64_t frame_rows = checked_multiply(
      height / kPatchHeight, width / kPatchWidth, field + " spatial");
  return checked_multiply(temporal, frame_rows, field + " temporal");
}

inline int64_t audio_rows(int64_t temporal,
                          int64_t channels,
                          const std::string& field) {
  return checked_multiply(temporal, channels, field);
}

inline double video_temporal_span(int64_t temporal) {
  double span = 0.0;
  for (int64_t index = 0; index < temporal; ++index) {
    span += kFrameRescale *
            static_cast<double>(kFramesPerToken[index % kTemporalGroup]);
  }
  return span;
}

inline std::vector<double> spatial_axis(int64_t dimension,
                                        int64_t patch,
                                        double sqrt_area) {
  // Preserve numpy.linspace(endpoint=False) operation order. NumPy may
  // vectorize its multiply/add, so Golden comparisons permit at most one ULP.
  const double ratio = static_cast<double>(dimension) / sqrt_area;
  const double left = (1.0 - ratio) * 1.0 / 2.0;
  const double right = left + ratio * 1.0;
  const int64_t count = dimension / patch;
  const double step = (right - left) / static_cast<double>(count);
  std::vector<double> axis;
  axis.reserve(static_cast<size_t>(count));
  for (int64_t index = 0; index < count; ++index) {
    double value = static_cast<double>(index);
    value *= step;
    value += left;
    value *= kPositionInterpolation;
    axis.emplace_back(value);
  }
  return axis;
}

inline void fill_audio_positions(const H3Slice& slice,
                                 int64_t temporal,
                                 int64_t channels,
                                 double origin,
                                 const std::vector<double>& width_axis,
                                 double* positions) {
  for (int64_t channel = 0; channel < channels; ++channel) {
    const double width = channel == 0 ? width_axis.front() : width_axis.back();
    for (int64_t time = 0; time < temporal; ++time) {
      const int64_t row = slice.start + channel * temporal + time;
      positions[row * 3] = origin + static_cast<double>(time);
      positions[row * 3 + 2] = width;
    }
  }
}

inline void fill_visual_positions(const H3Slice& slice,
                                  int64_t temporal,
                                  int64_t height,
                                  int64_t width,
                                  double origin,
                                  double* positions) {
  const double sqrt_area =
      std::sqrt(static_cast<double>(height) * static_cast<double>(width));
  const std::vector<double> height_axis =
      spatial_axis(height, kPatchHeight, sqrt_area);
  const std::vector<double> width_axis =
      spatial_axis(width, kPatchWidth, sqrt_area);
  double cumulative_span = 0.0;
  int64_t row = slice.start;
  for (int64_t time = 0; time < temporal; ++time) {
    const double temporal_position = origin + cumulative_span;
    for (double height_position : height_axis) {
      for (double width_position : width_axis) {
        positions[row * 3] = temporal_position;
        positions[row * 3 + 1] = height_position;
        positions[row * 3 + 2] = width_position;
        ++row;
      }
    }
    cumulative_span +=
        kFrameRescale *
        static_cast<double>(kFramesPerToken[time % kTemporalGroup]);
  }
}

inline void append_range(const H3Slice& slice, std::vector<int64_t>* rows) {
  rows->reserve(rows->size() + static_cast<size_t>(slice.stop - slice.start));
  for (int64_t row = slice.start; row < slice.stop; ++row) {
    rows->emplace_back(row);
  }
}

inline torch::Tensor index_tensor(const std::vector<int64_t>& values) {
  return torch::tensor(
      values, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}

inline void validate_condition(const torch::Tensor& hidden,
                               const torch::Tensor& tags) {
  if (!hidden.defined() || hidden.scalar_type() != torch::kBFloat16 ||
      !hidden.is_contiguous() || hidden.dim() != 3 || hidden.size(0) != 1 ||
      hidden.size(1) <= 0 || hidden.size(2) != kTextDimension) {
    fail(
        "condition hidden must be contiguous BF16 with shape [1,N,5120] and "
        "N > 0");
  }
  if (!tags.defined() || tags.scalar_type() != torch::kInt64 ||
      !tags.is_contiguous() || tags.dim() != 2 || tags.size(0) != 1 ||
      tags.size(1) != hidden.size(1)) {
    fail(
        "condition tags must be contiguous int64 with shape [1,N] matching "
        "condition hidden");
  }
  if (!torch::logical_or(tags == 0, tags == 1).all().item<bool>()) {
    fail("condition tags values must be 0 or 1");
  }
}

inline void validate_target(const H3TargetLatents& target) {
  require_positive(target.audio_t, "target audio_t");
  if (target.audio_channels != 2) {
    fail("target audio must be stereo");
  }
  require_positive(target.latent_t, "target latent_t");
  require_even_spatial(target.latent_h, "target latent_h");
  require_even_spatial(target.latent_w, "target latent_w");
}

inline void validate_reference_block(const H3ReferenceBlock& block,
                                     size_t index) {
  const std::string path = "reference_blocks[" + std::to_string(index) + "]";
  switch (block.kind) {
    case H3ReferenceBlockKind::IMAGE:
      require_even_spatial(block.latent_h, path + ".latent_h");
      require_even_spatial(block.latent_w, path + ".latent_w");
      return;
    case H3ReferenceBlockKind::AUDIO:
      require_positive(block.ref_audio_t, path + ".ref_audio_t");
      return;
    case H3ReferenceBlockKind::VIDEO:
      if (block.ref_audio_t != 0) {
        fail(path + ".ref_audio_t must be 0 for video kind");
      }
      break;
    case H3ReferenceBlockKind::VIDEO_AUDIO:
      require_positive(block.ref_audio_t, path + ".ref_audio_t");
      break;
    default:
      fail(path + ".kind is unsupported for Ref2VA");
  }
  require_positive(block.latent_t, path + ".latent_t");
  require_even_spatial(block.latent_h, path + ".latent_h");
  require_even_spatial(block.latent_w, path + ".latent_w");
}

inline int64_t reference_audio_rows(const H3ReferenceBlock& block) {
  if (block.kind == H3ReferenceBlockKind::AUDIO ||
      block.kind == H3ReferenceBlockKind::VIDEO_AUDIO) {
    return audio_rows(block.ref_audio_t, /*channels=*/2, "reference audio");
  }
  return 0;
}

inline int64_t reference_visual_rows(const H3ReferenceBlock& block) {
  if (block.kind == H3ReferenceBlockKind::IMAGE) {
    return visual_rows(
        /*temporal=*/1, block.latent_h, block.latent_w, "reference image");
  }
  if (block.kind == H3ReferenceBlockKind::VIDEO ||
      block.kind == H3ReferenceBlockKind::VIDEO_AUDIO) {
    return visual_rows(
        block.latent_t, block.latent_h, block.latent_w, "reference video");
  }
  return 0;
}

}  // namespace minimax_h3_packing_detail

inline torch::Tensor minimax_h3_patchify_video_latent(
    const torch::Tensor& latent) {
  using namespace minimax_h3_packing_detail;
  if (!latent.defined() || latent.dim() != 5) {
    fail("video latent must have rank 5 [B,C,T,H,W]");
  }
  for (int64_t dimension = 0; dimension < latent.dim(); ++dimension) {
    require_positive(latent.size(dimension), "video latent dimension");
  }
  if (latent.size(3) % kPatchHeight != 0 || latent.size(4) % kPatchWidth != 0) {
    fail("video latent H and W must be divisible by patch [1,2,2]");
  }

  const int64_t batch = latent.size(0);
  const int64_t channels = latent.size(1);
  const int64_t temporal = latent.size(2);
  const int64_t height = latent.size(3) / kPatchHeight;
  const int64_t width = latent.size(4) / kPatchWidth;
  torch::Tensor packed = latent.reshape(
      {batch, channels, temporal, 1, height, kPatchHeight, width, kPatchWidth});
  packed = packed.permute({0, 2, 4, 6, 1, 3, 5, 7});
  return packed
      .reshape({batch * temporal * height * width,
                channels * kPatchHeight * kPatchWidth})
      .contiguous();
}

inline torch::Tensor minimax_h3_unpatchify_video_tokens(
    const torch::Tensor& rows,
    const H3VideoLatentShape& latent_shape) {
  using namespace minimax_h3_packing_detail;
  if (!rows.defined() || rows.dim() != 2) {
    fail("video token rows must have rank 2");
  }
  require_positive(latent_shape.channels, "video latent channels");
  require_positive(latent_shape.temporal, "video latent temporal");
  require_even_spatial(latent_shape.height, "video latent height");
  require_even_spatial(latent_shape.width, "video latent width");
  const int64_t expected_dimension = checked_multiply(
      latent_shape.channels, /*right=*/4, "video token dimension");
  if (rows.size(1) != expected_dimension) {
    fail("video token row dimension does not equal latent channels * 4");
  }
  const int64_t rows_per_sample = visual_rows(latent_shape.temporal,
                                              latent_shape.height,
                                              latent_shape.width,
                                              "video token rows");
  if (rows.size(0) <= 0 || rows.size(0) % rows_per_sample != 0) {
    fail("video token row count must be divisible by T*(H/2)*(W/2)");
  }

  const int64_t batch = rows.size(0) / rows_per_sample;
  const int64_t height = latent_shape.height / kPatchHeight;
  const int64_t width = latent_shape.width / kPatchWidth;
  torch::Tensor packed = rows.reshape({batch,
                                       latent_shape.temporal,
                                       height,
                                       width,
                                       latent_shape.channels,
                                       1,
                                       kPatchHeight,
                                       kPatchWidth});
  packed = packed.permute({0, 4, 1, 5, 2, 6, 3, 7});
  return packed
      .reshape({batch,
                latent_shape.channels,
                latent_shape.temporal,
                latent_shape.height,
                latent_shape.width})
      .contiguous();
}

inline torch::Tensor minimax_h3_pack_audio_latent(const torch::Tensor& latent) {
  using namespace minimax_h3_packing_detail;
  if (!latent.defined() || latent.dim() != 3) {
    fail("audio latent must have rank 3 [channels,latent_dim,T]");
  }
  for (int64_t dimension = 0; dimension < latent.dim(); ++dimension) {
    require_positive(latent.size(dimension), "audio latent dimension");
  }
  return latent.permute({0, 2, 1})
      .reshape({latent.size(0) * latent.size(2), latent.size(1)})
      .contiguous();
}

inline torch::Tensor minimax_h3_unpack_audio_tokens(const torch::Tensor& rows,
                                                    int64_t audio_channels,
                                                    int64_t temporal) {
  using namespace minimax_h3_packing_detail;
  if (!rows.defined() || rows.dim() != 2) {
    fail("audio token rows must have rank 2");
  }
  require_positive(audio_channels, "audio channels");
  require_positive(temporal, "audio temporal length");
  require_positive(rows.size(1), "audio token row dimension");
  const int64_t expected_rows =
      audio_rows(temporal, audio_channels, "audio token rows");
  if (rows.size(0) != expected_rows) {
    fail("audio token row count must equal channels * temporal length");
  }
  return rows.reshape({audio_channels, temporal, rows.size(1)})
      .permute({0, 2, 1})
      .contiguous();
}

inline H3PackedLayout minimax_h3_build_ref2va_packed_layout(
    const torch::Tensor& condition_hidden,
    const torch::Tensor& condition_tags,
    const H3TargetLatents& target,
    const std::vector<H3ReferenceBlock>& reference_blocks,
    std::optional<int64_t> sequence_length = std::nullopt) {
  using namespace minimax_h3_packing_detail;
  validate_condition(condition_hidden, condition_tags);
  validate_target(target);
  if (reference_blocks.empty()) {
    fail("Ref2VA requires at least one visual reference block");
  }

  int64_t reference_rows = 0;
  bool has_visual_reference = false;
  for (size_t index = 0; index < reference_blocks.size(); ++index) {
    const H3ReferenceBlock& block = reference_blocks[index];
    validate_reference_block(block, index);
    const int64_t block_audio_rows = reference_audio_rows(block);
    const int64_t block_visual_rows = reference_visual_rows(block);
    reference_rows =
        checked_add(reference_rows, block_audio_rows, "reference blocks");
    reference_rows =
        checked_add(reference_rows, block_visual_rows, "reference blocks");
    has_visual_reference = has_visual_reference || block_visual_rows > 0;
  }
  if (!has_visual_reference) {
    fail(
        "audio-only Ref2VA is unsupported; at least one image or video "
        "reference is required");
  }

  const int64_t text_length = condition_hidden.size(1);
  const int64_t target_audio_rows =
      audio_rows(target.audio_t, target.audio_channels, "target audio");
  const int64_t target_video_rows = visual_rows(
      target.latent_t, target.latent_h, target.latent_w, "target video");
  int64_t used_length = checked_add(text_length, reference_rows, "sequence");
  used_length = checked_add(used_length, target_audio_rows, "sequence");
  used_length = checked_add(used_length, target_video_rows, "sequence");

  int64_t aligned_length = 0;
  if (sequence_length.has_value()) {
    aligned_length = *sequence_length;
    if (aligned_length < used_length) {
      fail("explicit sequence length is smaller than used rows");
    }
    if (aligned_length % kSequenceAlignment != 0) {
      fail("explicit sequence length must be divisible by 64");
    }
  } else {
    aligned_length =
        checked_add(used_length, kSequenceAlignment - 1, "aligned sequence");
    aligned_length = aligned_length / kSequenceAlignment * kSequenceAlignment;
  }
  if (aligned_length > std::numeric_limits<int32_t>::max()) {
    fail("aligned sequence length exceeds int32 cu_seqlens capacity");
  }

  H3PackedLayout layout;
  layout.used_length = used_length;
  layout.aligned_length = aligned_length;
  layout.condition_hidden = condition_hidden;
  layout.text_slice = {.start = 0, .stop = text_length};
  layout.reference_blocks.reserve(reference_blocks.size());
  layout.video_spans.reserve(reference_blocks.size() + 1);

  int64_t packed_cursor = text_length;
  double temporal_cursor = static_cast<double>(text_length);
  for (const H3ReferenceBlock& block : reference_blocks) {
    const int64_t block_audio_rows = reference_audio_rows(block);
    const int64_t block_visual_rows = reference_visual_rows(block);
    H3PhysicalReferenceBlock physical;
    physical.block = block;
    physical.packed_offset = packed_cursor;
    physical.temporal_origin = temporal_cursor;

    if (block.kind == H3ReferenceBlockKind::IMAGE) {
      physical.visual_slice = H3Slice{
          .start = packed_cursor,
          .stop = checked_add(packed_cursor, block_visual_rows, "image slice")};
      packed_cursor = physical.visual_slice->stop;
      physical.temporal_extent = 1.0;
    } else if (block.kind == H3ReferenceBlockKind::AUDIO) {
      physical.audio_slice = H3Slice{
          .start = packed_cursor,
          .stop = checked_add(packed_cursor, block_audio_rows, "audio slice")};
      packed_cursor = physical.audio_slice->stop;
      physical.temporal_extent = static_cast<double>(block.ref_audio_t);
    } else {
      physical.audio_slice =
          H3Slice{.start = packed_cursor,
                  .stop = checked_add(
                      packed_cursor, block_audio_rows, "video audio slice")};
      physical.visual_slice =
          H3Slice{.start = physical.audio_slice->stop,
                  .stop = checked_add(physical.audio_slice->stop,
                                      block_visual_rows,
                                      "video visual slice")};
      packed_cursor = physical.visual_slice->stop;
      const double video_span = video_temporal_span(block.latent_t);
      physical.temporal_extent =
          std::max(static_cast<double>(block.ref_audio_t), video_span);
      layout.video_spans.emplace_back(
          H3VideoSpan{.start = physical.visual_slice->start,
                      .latent_grid = {.temporal = block.latent_t,
                                      .height = block.latent_h / kPatchHeight,
                                      .width = block.latent_w / kPatchWidth},
                      .role = H3VideoSpanRole::REFERENCE});
    }
    physical.row_slice = {.start = physical.packed_offset,
                          .stop = packed_cursor};
    temporal_cursor += physical.temporal_extent;
    layout.reference_blocks.emplace_back(std::move(physical));
  }

  layout.target_temporal_origin = temporal_cursor;
  layout.target_audio_slice = {
      .start = packed_cursor,
      .stop =
          checked_add(packed_cursor, target_audio_rows, "target audio slice")};
  layout.target_video_slice = {
      .start = layout.target_audio_slice.stop,
      .stop = checked_add(layout.target_audio_slice.stop,
                          target_video_rows,
                          "target video slice")};
  layout.padding_slice = {.start = layout.target_video_slice.stop,
                          .stop = aligned_length};
  layout.target_latent_grid = {.temporal = target.latent_t,
                               .height = target.latent_h / kPatchHeight,
                               .width = target.latent_w / kPatchWidth};
  layout.video_spans.emplace_back(
      H3VideoSpan{.start = layout.target_video_slice.start,
                  .latent_grid = layout.target_latent_grid,
                  .role = H3VideoSpanRole::TARGET});

  const torch::TensorOptions int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  const torch::TensorOptions bool_options =
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  const torch::TensorOptions float64_options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  layout.input_ids =
      torch::full({aligned_length}, kMiniMaxH3PadId, int64_options);
  layout.image_mask = torch::zeros({aligned_length}, bool_options);
  layout.audio_mask = torch::zeros({aligned_length}, bool_options);
  layout.position_ids = torch::zeros({aligned_length, 3}, float64_options);
  layout.token_tags = torch::full({aligned_length}, -1, int64_options);
  layout.document_id = torch::zeros({aligned_length}, int32_options);

  int64_t* input_ids = layout.input_ids.data_ptr<int64_t>();
  bool* image_mask = layout.image_mask.data_ptr<bool>();
  bool* audio_mask = layout.audio_mask.data_ptr<bool>();
  double* positions = layout.position_ids.data_ptr<double>();
  int64_t* token_tags = layout.token_tags.data_ptr<int64_t>();
  int32_t* document_id = layout.document_id.data_ptr<int32_t>();
  const torch::Tensor condition_tags_cpu = condition_tags.to(torch::kCPU);
  const int64_t* condition_tag_values = condition_tags_cpu.data_ptr<int64_t>();
  for (int64_t row = 0; row < text_length; ++row) {
    input_ids[row] = kMiniMaxH3TextId;
    positions[row * 3] = static_cast<double>(row);
    token_tags[row] = condition_tag_values[row];
  }

  std::vector<int64_t> reference_image_positions;
  std::vector<int64_t> reference_audio_positions;
  reference_image_positions.reserve(static_cast<size_t>(reference_rows));
  reference_audio_positions.reserve(static_cast<size_t>(reference_rows));
  const double target_sqrt_area =
      std::sqrt(static_cast<double>(target.latent_h) *
                static_cast<double>(target.latent_w));
  const std::vector<double> target_width_axis =
      spatial_axis(target.latent_w, kPatchWidth, target_sqrt_area);

  for (const H3PhysicalReferenceBlock& physical : layout.reference_blocks) {
    const H3ReferenceBlock& block = physical.block;
    if (physical.audio_slice.has_value()) {
      const H3Slice audio_slice = *physical.audio_slice;
      for (int64_t row = audio_slice.start; row < audio_slice.stop; ++row) {
        input_ids[row] = kMiniMaxH3AudioReferenceConditionId;
        audio_mask[row] = true;
        token_tags[row] = 2;
      }
      append_range(audio_slice, &reference_audio_positions);
      if (block.ref_audio_t > 0) {
        std::vector<double> audio_width_axis = target_width_axis;
        if (block.kind == H3ReferenceBlockKind::VIDEO_AUDIO) {
          const double sqrt_area =
              std::sqrt(static_cast<double>(block.latent_h) *
                        static_cast<double>(block.latent_w));
          audio_width_axis =
              spatial_axis(block.latent_w, kPatchWidth, sqrt_area);
        }
        fill_audio_positions(audio_slice,
                             block.ref_audio_t,
                             /*channels=*/2,
                             physical.temporal_origin,
                             audio_width_axis,
                             positions);
      }
    }
    if (physical.visual_slice.has_value()) {
      const H3Slice visual_slice = *physical.visual_slice;
      for (int64_t row = visual_slice.start; row < visual_slice.stop; ++row) {
        input_ids[row] = kMiniMaxH3ImageVideoConditionId;
        image_mask[row] = true;
        token_tags[row] = 0;
      }
      append_range(visual_slice, &reference_image_positions);
      const int64_t temporal =
          block.kind == H3ReferenceBlockKind::IMAGE ? 1 : block.latent_t;
      fill_visual_positions(visual_slice,
                            temporal,
                            block.latent_h,
                            block.latent_w,
                            physical.temporal_origin,
                            positions);
    }
  }

  for (int64_t row = layout.target_audio_slice.start;
       row < layout.target_audio_slice.stop;
       ++row) {
    input_ids[row] = kMiniMaxH3AudioId;
    audio_mask[row] = true;
    token_tags[row] = 2;
  }
  input_ids[layout.target_audio_slice.start] = kMiniMaxH3AudioFirstId;
  fill_audio_positions(layout.target_audio_slice,
                       target.audio_t,
                       target.audio_channels,
                       layout.target_temporal_origin,
                       target_width_axis,
                       positions);

  for (int64_t row = layout.target_video_slice.start;
       row < layout.target_video_slice.stop;
       ++row) {
    input_ids[row] = kMiniMaxH3VideoId;
    image_mask[row] = true;
    token_tags[row] = 0;
  }
  input_ids[layout.target_video_slice.start] = kMiniMaxH3VideoFirstId;
  input_ids[layout.target_video_slice.stop - 1] = kMiniMaxH3VideoLastId;
  fill_visual_positions(layout.target_video_slice,
                        target.latent_t,
                        target.latent_h,
                        target.latent_w,
                        layout.target_temporal_origin,
                        positions);

  std::vector<int64_t> image_positions = reference_image_positions;
  append_range(layout.target_video_slice, &image_positions);
  std::vector<int64_t> audio_positions = reference_audio_positions;
  append_range(layout.target_audio_slice, &audio_positions);
  layout.img_pos = index_tensor(image_positions);
  layout.audio_pos = index_tensor(audio_positions);
  std::vector<int64_t> text_positions;
  text_positions.reserve(static_cast<size_t>(text_length));
  append_range(layout.text_slice, &text_positions);
  layout.text_pos = index_tensor(text_positions);

  layout.update_mask = torch::zeros(
      {static_cast<int64_t>(image_positions.size())}, bool_options);
  bool* update_mask = layout.update_mask.data_ptr<bool>();
  for (int64_t row = static_cast<int64_t>(reference_image_positions.size());
       row < static_cast<int64_t>(image_positions.size());
       ++row) {
    update_mask[row] = true;
  }
  layout.audio_update_mask = torch::zeros(
      {static_cast<int64_t>(audio_positions.size())}, bool_options);
  bool* audio_update_mask = layout.audio_update_mask.data_ptr<bool>();
  for (int64_t row = static_cast<int64_t>(reference_audio_positions.size());
       row < static_cast<int64_t>(audio_positions.size());
       ++row) {
    audio_update_mask[row] = true;
  }

  for (int64_t row = layout.padding_slice.start;
       row < layout.padding_slice.stop;
       ++row) {
    document_id[row] = 1;
  }
  layout.cu_seqlens = torch::tensor({static_cast<int32_t>(0),
                                     static_cast<int32_t>(used_length),
                                     static_cast<int32_t>(aligned_length)},
                                    int32_options);
  return layout;
}

}  // namespace xllm

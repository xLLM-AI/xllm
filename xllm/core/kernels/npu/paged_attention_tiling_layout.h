/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace xllm::kernel::npu {

// Dynamic fields in a CustomPagedAttention tiling buffer that may be refreshed
// before graph replay. The ATB adapter owns this layout; callers and update
// kernels consume the descriptor without interpreting the ATB ABI themselves.
struct PagedAttentionTilingLayout {
  int64_t buffer_words = 0;
  int64_t header_words = 0;
  int64_t row_stride_words = 0;
  int64_t max_kv_seq_len_offset = 0;
  int64_t kv_split_length_offset = 0;
  int64_t kv_split_core_count_offset = 0;
  int64_t row_kv_seq_len_offset = 0;
  uint32_t tiling_key = 0;
};

inline std::optional<int64_t> paged_attention_tiling_required_words(
    const PagedAttentionTilingLayout& layout,
    int64_t num_rows) {
  if (layout.buffer_words <= 0 || layout.header_words <= 0 ||
      layout.row_stride_words <= 0 || num_rows <= 0 ||
      layout.max_kv_seq_len_offset < 0 || layout.kv_split_length_offset < 0 ||
      layout.kv_split_core_count_offset < 0 ||
      layout.row_kv_seq_len_offset < layout.header_words ||
      num_rows > (std::numeric_limits<int64_t>::max() -
                  layout.row_kv_seq_len_offset - 1) /
                     layout.row_stride_words) {
    return std::nullopt;
  }
  const int64_t words = layout.row_kv_seq_len_offset +
                        (num_rows - 1) * layout.row_stride_words + 1;
  if (layout.max_kv_seq_len_offset >= layout.buffer_words ||
      layout.kv_split_length_offset >= layout.buffer_words ||
      layout.kv_split_core_count_offset >= layout.buffer_words ||
      words > layout.buffer_words) {
    return std::nullopt;
  }
  return words;
}

inline bool operator==(const PagedAttentionTilingLayout& lhs,
                       const PagedAttentionTilingLayout& rhs) {
  return lhs.buffer_words == rhs.buffer_words &&
         lhs.header_words == rhs.header_words &&
         lhs.row_stride_words == rhs.row_stride_words &&
         lhs.max_kv_seq_len_offset == rhs.max_kv_seq_len_offset &&
         lhs.kv_split_length_offset == rhs.kv_split_length_offset &&
         lhs.kv_split_core_count_offset == rhs.kv_split_core_count_offset &&
         lhs.row_kv_seq_len_offset == rhs.row_kv_seq_len_offset &&
         lhs.tiling_key == rhs.tiling_key;
}

// ATB CustomPagedAttention currently documents a 1 MiB device tiling buffer.
// Keep this allocation contract local to the adapter instead of duplicating it
// in model or TileLang code.
inline constexpr int64_t custom_paged_attention_tiling_capacity_words() {
  return 1024 * 256;
}

inline constexpr int64_t custom_paged_attention_block_alignment() { return 16; }

// Recognize the current CustomPagedAttention host-tiling schema and expose
// only the fields needed by graph replay. These offsets mirror the adapter ABI
// in third_party/xllm_ops/atb_customize/ops/custom_paged_attention/
// kernel_implement/tiling/custom_paged_attention_tiling_dependency.{h,cpp}.
// Keep them centralized here; unknown or truncated schemas are not guessed and
// must use the conservative replay path. The tiling key selects a kernel within
// this schema, so it is metadata rather than a schema-version discriminator.
inline std::optional<PagedAttentionTilingLayout>
parse_custom_paged_attention_tiling_layout(
    const std::vector<uint32_t>& tiling_words) {
  constexpr int64_t kTilingKeyOffset = 16;
  constexpr int64_t kHeaderWordsOffset = 17;
  constexpr int64_t kRowStrideWordsOffset = 18;
  constexpr int64_t kMaxKvSeqLenOffset = 22;
  constexpr int64_t kKvSplitLengthOffset = 23;
  constexpr int64_t kKvSplitCoreCountOffset = 24;
  constexpr int64_t kRowKvSeqLenFieldOffset = 1;
  constexpr int64_t kMinimumHeaderWords = kKvSplitCoreCountOffset + 1;
  constexpr int64_t kAtbV1HeaderWords = 44;
  constexpr int64_t kAtbV1RowStrideWords = 17;

  if (tiling_words.size() <= static_cast<size_t>(kRowStrideWordsOffset)) {
    return std::nullopt;
  }
  const int64_t header_words = tiling_words[kHeaderWordsOffset];
  const int64_t row_stride_words = tiling_words[kRowStrideWordsOffset];
  if (header_words != kAtbV1HeaderWords ||
      row_stride_words != kAtbV1RowStrideWords ||
      header_words < kMinimumHeaderWords ||
      header_words > static_cast<int64_t>(tiling_words.size())) {
    return std::nullopt;
  }

  PagedAttentionTilingLayout layout;
  layout.buffer_words = static_cast<int64_t>(tiling_words.size());
  layout.header_words = header_words;
  layout.row_stride_words = row_stride_words;
  layout.max_kv_seq_len_offset = kMaxKvSeqLenOffset;
  layout.kv_split_length_offset = kKvSplitLengthOffset;
  layout.kv_split_core_count_offset = kKvSplitCoreCountOffset;
  layout.row_kv_seq_len_offset = header_words + kRowKvSeqLenFieldOffset;
  layout.tiling_key = tiling_words[kTilingKeyOffset];
  return layout;
}

}  // namespace xllm::kernel::npu

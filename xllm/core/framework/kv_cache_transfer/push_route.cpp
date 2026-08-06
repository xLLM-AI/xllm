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

#include "framework/kv_cache_transfer/push_route.h"

#include <cstddef>

namespace xllm {

bool use_push_owner(const int32_t src_tp_size, const int32_t dst_tp_size) {
  return src_tp_size > 0 && dst_tp_size > 0 && src_tp_size > dst_tp_size;
}

std::vector<int32_t> get_dst_ranks(const int32_t src_tp_rank,
                                   const int32_t src_tp_size,
                                   const int32_t dst_tp_size,
                                   const int32_t dst_dp_rank) {
  std::vector<int32_t> dst_ranks;
  if (src_tp_size <= 0 || dst_tp_size <= 0 || src_tp_rank < 0 ||
      src_tp_rank >= src_tp_size || dst_dp_rank < 0) {
    return dst_ranks;
  }

  if (use_push_owner(src_tp_size, dst_tp_size)) {
    dst_ranks.reserve(1);
    int32_t dst_rank = dst_dp_rank * dst_tp_size + src_tp_rank % dst_tp_size;
    dst_ranks.emplace_back(dst_rank);
    return dst_ranks;
  }

  int32_t start_rank = src_tp_rank % dst_tp_size + dst_tp_size * dst_dp_rank;
  int32_t end_rank = dst_tp_size * (dst_dp_rank + 1);
  const size_t dst_rank_num =
      static_cast<size_t>((end_rank - 1 - start_rank) / src_tp_size + 1);
  dst_ranks.reserve(dst_rank_num);
  for (int32_t i = start_rank; i < end_rank; i += src_tp_size) {
    dst_ranks.emplace_back(i);
  }
  return dst_ranks;
}

std::vector<int32_t> get_src_tp_ranks(const int32_t dst_tp_rank,
                                      const int32_t src_tp_size,
                                      const int32_t dst_tp_size) {
  std::vector<int32_t> src_tp_ranks;
  if (src_tp_size <= 0 || dst_tp_size <= 0 || dst_tp_rank < 0 ||
      dst_tp_rank >= dst_tp_size) {
    return src_tp_ranks;
  }

  if (src_tp_size <= dst_tp_size) {
    src_tp_ranks.reserve(1);
    src_tp_ranks.emplace_back(dst_tp_rank % src_tp_size);
    return src_tp_ranks;
  }

  const size_t src_rank_num =
      static_cast<size_t>((src_tp_size - 1 - dst_tp_rank) / dst_tp_size + 1);
  src_tp_ranks.reserve(src_rank_num);
  for (int32_t src_tp_rank = dst_tp_rank; src_tp_rank < src_tp_size;
       src_tp_rank += dst_tp_size) {
    src_tp_ranks.emplace_back(src_tp_rank);
  }
  return src_tp_ranks;
}

}  // namespace xllm

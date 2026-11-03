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
#include <cstdint>

// Plain scalar ABI shared by tiling, the direct launcher and CPU mapping tests.
namespace t5qkv {
constexpr uint32_t kTileElements =
    32768;  // 64 KiB; two DMA buffers = 128 KiB UB.
constexpr uint32_t kMaxVectorCores = 48;
struct Plan {
  uint32_t batch_, heads_, length_, dim_;
  uint32_t tile_elements_, query_tiles_, cache_tiles_, total_tiles_, blocks_;
};
inline bool valid_shape(int64_t b, int64_t h, int64_t l, int64_t d) {
  if (b < 1 || b > 64 || h < 1 || h > 128 || l < 0 || l > 8192 || d < 1 ||
      d > 256) {
    return false;
  }
  return b * h * d * (2 * l + 3) <= (int64_t{1} << 28);
}
inline Plan make_plan(uint32_t b,
                      uint32_t h,
                      uint32_t l,
                      uint32_t d,
                      uint32_t cores = kMaxVectorCores) {
  Plan p{b, h, l, d, kTileElements, 0, 0, 0, 0};
  const uint64_t q = uint64_t{b} * h * d;
  const uint64_t kv = q * (l + 1);
  p.query_tiles_ = (q + p.tile_elements_ - 1) / p.tile_elements_;
  p.cache_tiles_ = (kv + p.tile_elements_ - 1) / p.tile_elements_;
  p.total_tiles_ = p.query_tiles_ + 2 * p.cache_tiles_;
  // Small inputs still use one block per useful tile. Tile boundaries are 32B
  // aligned.
  p.blocks_ = p.total_tiles_ < cores ? p.total_tiles_ : cores;
  return p;
}
}  // namespace t5qkv

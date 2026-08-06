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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

struct MemoryRange {
  uintptr_t begin;
  uintptr_t end;
  std::string name;
};

void check_index(const torch::Tensor& out_token_index, int64_t beam_width) {
  TORCH_CHECK(out_token_index.defined(),
              "cache_select: out_token_index must be defined");
  TORCH_CHECK(out_token_index.device().type() == c10::DeviceType::PrivateUse1,
              "cache_select: out_token_index must be on an MLU device");
  TORCH_CHECK(out_token_index.scalar_type() == torch::kInt32,
              "cache_select: out_token_index must have int32 dtype");
  TORCH_CHECK(out_token_index.is_contiguous(),
              "cache_select: out_token_index must be contiguous");
  TORCH_CHECK(
      out_token_index.dim() == 2 && out_token_index.size(1) == 1,
      "cache_select: out_token_index must have shape [batch * beam, 1]");
  TORCH_CHECK(out_token_index.size(0) > 0,
              "cache_select: out_token_index must contain at least one row");
  TORCH_CHECK(
      out_token_index.size(0) % beam_width == 0,
      "cache_select: out_token_index rows must divide evenly by beam_width");
}

void check_cache(const torch::Tensor& cache, const std::string& name) {
  TORCH_CHECK(cache.defined(), "cache_select: ", name, " must be defined");
  TORCH_CHECK(cache.device().type() == c10::DeviceType::PrivateUse1,
              "cache_select: ",
              name,
              " must be on an MLU device");
  TORCH_CHECK(cache.scalar_type() == torch::kBFloat16,
              "cache_select: ",
              name,
              " must have bfloat16 dtype");
  TORCH_CHECK(
      cache.is_contiguous(), "cache_select: ", name, " must be contiguous");
  TORCH_CHECK(cache.dim() == 5, "cache_select: ", name, " must be 5D");
}

MemoryRange get_range(const torch::Tensor& cache, const std::string& name) {
  const auto count = static_cast<uint64_t>(cache.numel());
  const auto element_size = static_cast<uint64_t>(cache.element_size());
  TORCH_CHECK(count <= std::numeric_limits<uintptr_t>::max() / element_size,
              "cache_select: ",
              name,
              " byte size overflows address range");
  const auto begin = reinterpret_cast<uintptr_t>(cache.data_ptr());
  const auto bytes = static_cast<uintptr_t>(count * element_size);
  TORCH_CHECK(begin <= std::numeric_limits<uintptr_t>::max() - bytes,
              "cache_select: ",
              name,
              " address range overflows");
  return {begin, begin + bytes, name};
}

void check_ranges(const std::vector<torch::Tensor>& unshared_k_caches,
                  const std::vector<torch::Tensor>& unshared_v_caches) {
  std::vector<MemoryRange> ranges;
  ranges.reserve(unshared_k_caches.size() + unshared_v_caches.size());
  for (size_t layer = 0; layer < unshared_k_caches.size(); ++layer) {
    ranges.push_back(
        get_range(unshared_k_caches[layer],
                  "unshared_k_caches[" + std::to_string(layer) + "]"));
    ranges.push_back(
        get_range(unshared_v_caches[layer],
                  "unshared_v_caches[" + std::to_string(layer) + "]"));
  }
  for (size_t i = 0; i < ranges.size(); ++i) {
    for (size_t j = i + 1; j < ranges.size(); ++j) {
      TORCH_CHECK(
          ranges[i].end <= ranges[j].begin || ranges[j].end <= ranges[i].begin,
          "cache_select: ",
          ranges[i].name,
          " and ",
          ranges[j].name,
          " must not overlap");
    }
  }
}

void check_args(const torch::Tensor& out_token_index,
                const std::vector<torch::Tensor>& unshared_k_caches,
                const std::vector<torch::Tensor>& unshared_v_caches,
                int64_t decode_step,
                int64_t beam_width) {
  TORCH_CHECK(beam_width > 0, "cache_select: beam_width must be positive");
  TORCH_CHECK(!unshared_k_caches.empty(),
              "cache_select: unshared K/V cache vectors must not be empty");
  TORCH_CHECK(
      unshared_k_caches.size() == unshared_v_caches.size(),
      "cache_select: unshared K/V cache vectors must have equal length");
  check_index(out_token_index, beam_width);

  for (size_t layer = 0; layer < unshared_k_caches.size(); ++layer) {
    check_cache(unshared_k_caches[layer],
                "unshared_k_caches[" + std::to_string(layer) + "]");
    check_cache(unshared_v_caches[layer],
                "unshared_v_caches[" + std::to_string(layer) + "]");
  }

  const auto& reference = unshared_k_caches.front();
  const int64_t batch = out_token_index.size(0) / beam_width;
  TORCH_CHECK(batch > 0, "cache_select: cache batch must be positive");
  TORCH_CHECK(reference.size(0) == batch,
              "cache_select: cache batch must match out_token_index");
  TORCH_CHECK(reference.size(1) == beam_width,
              "cache_select: cache beam dimension must equal beam_width");
  TORCH_CHECK(reference.size(2) > 0,
              "cache_select: cache kv_head dimension must be positive");
  TORCH_CHECK(reference.size(3) > 0,
              "cache_select: cache max_decode_steps must be positive");
  TORCH_CHECK(reference.size(4) > 0,
              "cache_select: cache head_dim must be positive");
  TORCH_CHECK(decode_step >= 0 && decode_step < reference.size(3),
              "cache_select: decode_step must be within cache history");

  for (size_t layer = 0; layer < unshared_k_caches.size(); ++layer) {
    TORCH_CHECK(unshared_k_caches[layer].sizes() == reference.sizes(),
                "cache_select: all K caches must have the same shape");
    TORCH_CHECK(unshared_v_caches[layer].sizes() == reference.sizes(),
                "cache_select: K and V caches must have identical shapes");
  }
  check_ranges(unshared_k_caches, unshared_v_caches);
}

void reparent_cache(torch::Tensor& cache,
                    const torch::Tensor& src_rows,
                    int64_t batch,
                    int64_t beam_width,
                    int64_t decode_step) {
  auto history = cache.narrow(/*dim=*/3, /*start=*/0, decode_step + 1)
                     .view({batch * beam_width,
                            cache.size(2),
                            decode_step + 1,
                            cache.size(4)});
  auto selected = history.index_select(/*dim=*/0, src_rows);
  history.copy_(selected);
}

}  // namespace

void cache_select(const torch::Tensor& out_token_index,
                  std::vector<torch::Tensor>& unshared_k_caches,
                  std::vector<torch::Tensor>& unshared_v_caches,
                  int64_t decode_step,
                  int64_t beam_width) {
  check_args(out_token_index,
             unshared_k_caches,
             unshared_v_caches,
             decode_step,
             beam_width);

  const int64_t batch = out_token_index.size(0) / beam_width;
  auto parent = torch::div(out_token_index.reshape({-1}), beam_width, "floor")
                    .to(torch::kInt64);
  auto batch_ids =
      torch::arange(batch, parent.options()).repeat_interleave(beam_width);
  auto src_rows = batch_ids * beam_width + parent;

  for (size_t layer = 0; layer < unshared_k_caches.size(); ++layer) {
    reparent_cache(
        unshared_k_caches[layer], src_rows, batch, beam_width, decode_step);
    reparent_cache(
        unshared_v_caches[layer], src_rows, batch, beam_width, decode_step);
  }
}

}  // namespace xllm::kernel::mlu

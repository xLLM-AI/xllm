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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace xllm {

// A logical page covers the same block IDs in every tensor. Each tensor maps
// an integral number of hardware pages, so distinct logical pages never share
// a physical mapping. For MLA 512/64 this gives eight latent pages and one
// RoPE page, rather than treating the two tensors as equally wide.
class VirtualKVCachePageLayout final {
 public:
  VirtualKVCachePageLayout(std::vector<size_t> block_bytes, size_t page_size)
      : tensor_block_bytes_(std::move(block_bytes)), page_size_(page_size) {
    CHECK(!tensor_block_bytes_.empty());
    CHECK_GT(page_size_, 0);
    for (size_t bytes : tensor_block_bytes_) {
      CHECK_GT(bytes, 0);
      block_bytes_ = std::gcd(block_bytes_, bytes);
    }
    CHECK_EQ(page_size_ % block_bytes_, 0)
        << "Virtual KV page size must be divisible by the common block size";
  }

  size_t block_bytes() const { return block_bytes_; }
  size_t blocks_per_page() const { return page_size_ / block_bytes_; }
  size_t pages_per_tensor(size_t index) const {
    return tensor_block_bytes_.at(index) / block_bytes_;
  }

  size_t physical_pages(const std::vector<size_t>& tensor_counts) const {
    CHECK_EQ(tensor_counts.size(), tensor_block_bytes_.size());
    size_t pages = 0;
    for (size_t index = 0; index < tensor_counts.size(); ++index) {
      const size_t per_tensor = pages_per_tensor(index);
      CHECK_LE(tensor_counts[index],
               (std::numeric_limits<size_t>::max() - pages) / per_tensor);
      pages += tensor_counts[index] * per_tensor;
    }
    return pages;
  }

  // The first dimension is the block ID; ND and NPU NZ layouts both keep
  // all bytes belonging to one block contiguous in the remaining dimensions.
  static size_t tensor_block_bytes(const std::vector<int64_t>& dims,
                                   size_t element_size) {
    CHECK(!dims.empty());
    CHECK_GT(dims.front(), 0);
    CHECK_GT(element_size, 0);
    size_t bytes = element_size;
    for (size_t index = 1; index < dims.size(); ++index) {
      CHECK_GT(dims[index], 0);
      CHECK_LE(static_cast<uint64_t>(dims[index]),
               std::numeric_limits<size_t>::max() / bytes);
      bytes *= static_cast<size_t>(dims[index]);
    }
    return bytes;
  }

 private:
  std::vector<size_t> tensor_block_bytes_;
  size_t page_size_;
  size_t block_bytes_ = 0;
};

}  // namespace xllm

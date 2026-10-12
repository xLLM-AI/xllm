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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace xllm::detail {

// Validates the byte range before creating device storage. element_size comes
// from a supported tensor dtype; this calculation needs no device runtime.
inline size_t checked_tensor_view_numel(size_t region_size,
                                        size_t element_size,
                                        size_t offset,
                                        const std::vector<int64_t>& dims) {
  CHECK_LE(offset, region_size) << "Tensor byte offset is out of bounds";
  CHECK_EQ(offset % element_size, 0) << "Tensor byte offset is not aligned";
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "Tensor dimensions must be nonnegative";
  }
  size_t num_elems =
      std::find(dims.begin(), dims.end(), 0) == dims.end() ? 1 : 0;
  for (int64_t dim : dims) {
    if (dim == 0 || num_elems == 0) {
      continue;
    }
    CHECK_LE(num_elems,
             std::numeric_limits<size_t>::max() / static_cast<size_t>(dim));
    num_elems *= static_cast<size_t>(dim);
  }
  CHECK_LE(num_elems, (region_size - offset) / element_size)
      << "Tensor view exceeds its virtual memory region";
  return num_elems;
}

}  // namespace xllm::detail

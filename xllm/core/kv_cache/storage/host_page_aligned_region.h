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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {

#if defined(USE_MLU)
namespace mlu {
class MLUHostMemoryRegion;
}
#endif

class HostPageAlignedRegion final {
 public:
  void* base_ptr = nullptr;
  size_t total_bytes = 0;

  HostPageAlignedRegion();
  explicit HostPageAlignedRegion(size_t bytes);
  HostPageAlignedRegion(const HostPageAlignedRegion&) = delete;
  HostPageAlignedRegion& operator=(const HostPageAlignedRegion&) = delete;
  HostPageAlignedRegion(HostPageAlignedRegion&& other) noexcept;
  HostPageAlignedRegion& operator=(HostPageAlignedRegion&& other) noexcept;
  ~HostPageAlignedRegion();

 private:
#if defined(USE_MLU)
  std::unique_ptr<mlu::MLUHostMemoryRegion> mlu_region_;
#endif
};

// Allocate a page-aligned, mlock'd (and NPU-registered) host tensor over a
// HostPageAlignedRegion. The region owns the memory; the tensor is a view.
void create_host_page_aligned_tensor(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     torch::Tensor* tensor,
                                     HostPageAlignedRegion* region);

}  // namespace xllm

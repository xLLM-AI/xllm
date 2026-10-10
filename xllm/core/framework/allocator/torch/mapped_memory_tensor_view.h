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
#include <vector>

namespace xllm {

class MappedMemoryRegion;

// Creates a non-owning tensor view on a byte range. The region and its mappings
// must outlive every use of the tensor; destroying the tensor does not free or
// unmap the region. offset is measured in bytes.
torch::Tensor create_mapped_memory_tensor_view(
    const MappedMemoryRegion& region,
    torch::Dtype dtype,
    size_t offset,
    const std::vector<int64_t>& dims);

}  // namespace xllm

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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm::mlu {

torch::Tensor alloc_zero_tensor(const std::vector<int64_t>& dims,
                                torch::ScalarType dtype,
                                const torch::Device& device);

// Allocates block-aligned backing storage suitable for MLU RDMA registration
// while preserving the logical tensor shape. dims[0] is the block count.
torch::Tensor alloc_rdma_registerable_zero_tensor(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    const torch::Device& device);

// Returns the required RDMA registration length after validating the tensor's
// device, layout, storage offset, and backing storage capacity.
size_t get_rdma_registerable_nbytes(const torch::Tensor& tensor);

}  // namespace xllm::mlu

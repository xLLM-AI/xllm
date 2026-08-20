/* Copyright 2025-2026 The xLLM Authors.

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

#include "platform/dcu/dcu_tensor_alloc.h"

#include <glog/logging.h>
#include <hip/hip_runtime.h>

#include <limits>

namespace xllm::dcu {
namespace {

size_t get_nbytes(const std::vector<int64_t>& dims,
                  const torch::ScalarType dtype) {
  size_t count = 1;
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "tensor dim must be non-negative";
    size_t dim_size = static_cast<size_t>(dim);
    if (dim_size > static_cast<size_t>(0)) {
      CHECK_LE(count, std::numeric_limits<size_t>::max() / dim_size)
          << "tensor element count overflow";
    }
    count *= dim_size;
  }
  size_t elem_size = static_cast<size_t>(torch::elementSize(dtype));
  CHECK_GT(elem_size, static_cast<size_t>(0)) << "tensor dtype size is zero";
  CHECK_LE(count, std::numeric_limits<size_t>::max() / elem_size)
      << "tensor byte size overflow";
  return count * elem_size;
}

void free_tensor(void* ptr, int32_t device_id) {
  if (ptr == nullptr) {
    return;
  }

  hipError_t ret = hipSetDevice(device_id);
  CHECK(ret == hipSuccess) << "hipSetDevice failed: " << hipGetErrorString(ret)
                           << ", device_id=" << device_id;
  ret = hipFree(ptr);
  CHECK(ret == hipSuccess) << "hipFree failed: " << hipGetErrorString(ret)
                           << ", ptr=" << ptr;
}

}  // namespace

torch::Tensor alloc_zero_tensor(const std::vector<int64_t>& dims,
                                torch::ScalarType dtype,
                                const torch::Device& device) {
  CHECK(device.has_index()) << "DCU device index is required";
  int32_t device_id = static_cast<int32_t>(device.index());

  hipError_t ret = hipSetDevice(device_id);
  CHECK(ret == hipSuccess) << "hipSetDevice failed: " << hipGetErrorString(ret)
                           << ", device_id=" << device_id;

  size_t nbytes = get_nbytes(dims, dtype);
  void* ptr = nullptr;
  ret = hipMalloc(&ptr, nbytes);
  CHECK(ret == hipSuccess) << "hipMalloc failed: " << hipGetErrorString(ret)
                           << ", nbytes=" << nbytes;
  ret = hipMemset(ptr, 0, nbytes);
  CHECK(ret == hipSuccess) << "hipMemset failed: " << hipGetErrorString(ret)
                           << ", nbytes=" << nbytes;

  auto deleter = [device_id](void* data) { free_tensor(data, device_id); };
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(
          /*requires_grad=*/false);
  return torch::from_blob(ptr, dims, deleter, options);
}

}  // namespace xllm::dcu

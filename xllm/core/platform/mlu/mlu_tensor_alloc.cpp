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

#include "platform/mlu/mlu_tensor_alloc.h"

#include <cnrt.h>
#include <glog/logging.h>

#include <limits>

#include "platform/mlu/mlu_rdma_memory_plan.h"

namespace xllm::mlu {

namespace {

size_t get_nbytes(const std::vector<int64_t>& dims,
                  const torch::ScalarType dtype) {
  size_t count = 1;
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "tensor dim must be non-negative";
    const size_t dim_size = static_cast<size_t>(dim);
    if (dim_size > static_cast<size_t>(0)) {
      CHECK_LE(count, std::numeric_limits<size_t>::max() / dim_size)
          << "tensor element count overflow";
    }
    count *= dim_size;
  }
  const size_t elem_size = static_cast<size_t>(torch::elementSize(dtype));
  CHECK_GT(elem_size, static_cast<size_t>(0)) << "tensor dtype size is zero";
  CHECK_LE(count, std::numeric_limits<size_t>::max() / elem_size)
      << "tensor byte size overflow";
  return count * elem_size;
}

void check_mlu_device(const torch::Device& device) {
  CHECK(device.type() == c10::DeviceType::PrivateUse1)
      << "MLU tensor must use the PrivateUse1 device type";
  CHECK(device.has_index()) << "MLU device index is required";
}

void free_tensor(void* ptr, int32_t device_id) {
  if (ptr == nullptr) {
    return;
  }

  cnrtRet_t ret = cnrtSetDevice(device_id);
  CHECK(ret == cnrtSuccess)
      << "cnrtSetDevice failed, ret=" << static_cast<int32_t>(ret)
      << ", device_id=" << device_id;
  ret = cnrtFree(ptr);
  CHECK(ret == cnrtSuccess)
      << "cnrtFree failed, ret=" << static_cast<int32_t>(ret)
      << ", ptr=" << ptr;
}

torch::Tensor alloc_zero_tensor_impl(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     const torch::Device& device,
                                     size_t logical_nbytes,
                                     size_t allocation_nbytes) {
  CHECK_GE(allocation_nbytes, logical_nbytes)
      << "allocation must cover logical tensor bytes";
  CHECK(device.has_index()) << "MLU device index is required";
  const int32_t device_id = static_cast<int32_t>(device.index());

  cnrtRet_t ret = cnrtSetDevice(device_id);
  CHECK(ret == cnrtSuccess)
      << "cnrtSetDevice failed, ret=" << static_cast<int32_t>(ret)
      << ", device_id=" << device_id;

  void* ptr = nullptr;
  ret = cnrtMalloc(&ptr, allocation_nbytes);
  CHECK(ret == cnrtSuccess)
      << "cnrtMalloc failed, ret=" << static_cast<int32_t>(ret)
      << ", nbytes=" << allocation_nbytes;
  ret = cnrtMemset(ptr, 0, allocation_nbytes);
  CHECK(ret == cnrtSuccess)
      << "cnrtMemset failed, ret=" << static_cast<int32_t>(ret)
      << ", nbytes=" << allocation_nbytes;

  auto deleter = [device_id](void* data) { free_tensor(data, device_id); };
  auto options =
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
  if (allocation_nbytes == logical_nbytes) {
    return torch::from_blob(ptr, dims, deleter, options);
  }

  const size_t element_size = static_cast<size_t>(torch::elementSize(dtype));
  CHECK_GT(element_size, static_cast<size_t>(0)) << "tensor dtype size is zero";
  CHECK_EQ(allocation_nbytes % element_size, static_cast<size_t>(0))
      << "allocation bytes must be divisible by tensor element size";
  const size_t allocation_numel = allocation_nbytes / element_size;
  const size_t logical_numel = logical_nbytes / element_size;
  CHECK_LE(allocation_numel,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      << "allocation element count exceeds int64 range";

  torch::Tensor storage_tensor = torch::from_blob(
      ptr, {static_cast<int64_t>(allocation_numel)}, deleter, options);
  return storage_tensor
      .narrow(/*dim=*/0,
              /*start=*/0,
              /*length=*/static_cast<int64_t>(logical_numel))
      .view(dims);
}

}  // namespace

torch::Tensor alloc_zero_tensor(const std::vector<int64_t>& dims,
                                torch::ScalarType dtype,
                                const torch::Device& device) {
  const size_t logical_nbytes = get_nbytes(dims, dtype);
  return alloc_zero_tensor_impl(
      dims, dtype, device, logical_nbytes, logical_nbytes);
}

torch::Tensor alloc_rdma_registerable_zero_tensor(
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    const torch::Device& device) {
  check_mlu_device(device);
  CHECK(!dims.empty()) << "RDMA registerable tensor shape must not be empty";
  CHECK_GT(dims[0], 0) << "tensor block dimension must be positive";

  const size_t logical_nbytes = get_nbytes(dims, dtype);
  const RdmaMemoryPlan plan =
      make_rdma_memory_plan(logical_nbytes, static_cast<size_t>(dims[0]));
  return alloc_zero_tensor_impl(
      dims, dtype, device, logical_nbytes, plan.registered_bytes);
}

size_t get_rdma_registerable_nbytes(const torch::Tensor& tensor) {
  CHECK(tensor.defined()) << "RDMA registerable tensor must be defined";
  check_mlu_device(tensor.device());
  CHECK(tensor.is_contiguous())
      << "RDMA registerable tensor must be contiguous";
  CHECK_GT(tensor.dim(), 0) << "RDMA registerable tensor dim must be positive";
  CHECK_GT(tensor.size(0), 0) << "tensor block dimension must be positive";
  CHECK_EQ(tensor.storage_offset(), 0)
      << "RDMA registerable tensor storage offset must be zero";

  const size_t logical_nbytes = static_cast<size_t>(tensor.nbytes());
  const RdmaMemoryPlan plan = make_rdma_memory_plan(
      logical_nbytes, static_cast<size_t>(tensor.size(0)));
  const size_t available_bytes = tensor.storage().nbytes();
  CHECK_GE(available_bytes, plan.registered_bytes)
      << "MLU RDMA registration exceeds tensor storage capacity: "
      << "logical_bytes=" << plan.logical_bytes
      << ", required_bytes=" << plan.registered_bytes
      << ", available_bytes=" << available_bytes
      << ", block_bytes=" << plan.block_bytes;
  return plan.registered_bytes;
}

}  // namespace xllm::mlu

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

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"

namespace xllm {

torch::Tensor MappedMemoryRegion::to_torch_tensor() const {
  int64_t num_elems = static_cast<int64_t>(size_ / torch::elementSize(dtype_));
  return to_torch_tensor(/*offset=*/0, {num_elems});
}

torch::Tensor MappedMemoryRegion::to_torch_tensor(
    size_t offset,
    const std::vector<int64_t>& dims) const {
  CHECK_LE(offset, size_) << "Tensor byte offset is out of bounds";
  size_t element_size = torch::elementSize(dtype_);
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
  CHECK_LE(num_elems, (size_ - offset) / element_size)
      << "Tensor view exceeds its virtual memory region";
  uintptr_t addr = vir_ptr_to_uintptr(vaddr_) + offset;

#if defined(USE_NPU)
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype_).device(dev_);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);

  // PyTorch 2.7+: StorageImpl now takes DataPtr instead of raw allocator
  storage = fptr(c10::StorageImpl::use_byte_size_t(),
                 c10::SymInt(tensor_nbytes),
                 std::move(c10_data_ptr),
                 allocator,
                 true);

  tensor.set_(storage, 0, dims);

  return tensor;
#else
  // For non-NPU devices, use torch::from_blob
  auto options =
      torch::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  return torch::from_blob(reinterpret_cast<void*>(addr), dims, options);
#endif
}

}  // namespace xllm

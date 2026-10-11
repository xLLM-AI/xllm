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

#include "core/framework/allocator/torch/mapped_memory_tensor_view.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"

namespace xllm {

torch::Tensor create_mapped_memory_tensor_view(
    const MappedMemoryRegion& region,
    torch::Dtype dtype,
    size_t offset,
    const std::vector<int64_t>& dims) {
  CHECK_LE(offset, region.size()) << "Tensor byte offset is out of bounds";
  const size_t element_size = torch::elementSize(dtype);
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
  CHECK_LE(num_elems, (region.size() - offset) / element_size)
      << "Tensor view exceeds its virtual memory region";
  const uintptr_t addr = vir_ptr_to_uintptr(region.vaddr()) + offset;
  const torch::TensorOptions options = torch::TensorOptions()
                                           .dtype(dtype)
                                           .device(region.device())
                                           .requires_grad(false);

#if defined(USE_NPU)
  const torch::DeviceType device_type = torch::DeviceType::PrivateUse1;
  torch::Tensor tensor = torch::empty({0}, options);
  void* address = reinterpret_cast<void*>(addr);
  torch::DataPtr data_ptr(address, address, [](void*) {}, tensor.device());

  const size_t tensor_nbytes = num_elems * element_size;
  const auto storage_factory = c10::GetStorageImplCreate(device_type);
  auto* allocator = c10::GetAllocator(device_type);
  torch::Storage storage = storage_factory(c10::StorageImpl::use_byte_size_t(),
                                           c10::SymInt(tensor_nbytes),
                                           std::move(data_ptr),
                                           allocator,
                                           /*resizable=*/true);
  tensor.set_(storage, /*storage_offset=*/0, dims);
  return tensor;
#else
  return torch::from_blob(reinterpret_cast<void*>(addr), dims, options);
#endif
}

}  // namespace xllm

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

#include "core/framework/allocator/torch/mapped_virtual_memory_tensor_view.h"

#include "core/framework/allocator/torch/mapped_virtual_memory_tensor_view_internal.h"
#include "core/framework/allocator/virtual_memory/mapped_virtual_memory_region.h"
#include "core/util/tensor_helper.h"

namespace xllm {

torch::Tensor create_mapped_virtual_memory_tensor_view(
    const MappedVirtualMemoryRegion& region,
    torch::ScalarType dtype,
    size_t offset,
    const std::vector<int64_t>& dims) {
  const size_t element_size = torch::elementSize(dtype);
  detail::checked_tensor_view_numel(region.size(), element_size, offset, dims);
  const uintptr_t addr = vir_ptr_to_uintptr(region.vaddr()) + offset;
  return get_tensor_from_blob(
      dims, dtype, reinterpret_cast<void*>(addr), region.device());
}

}  // namespace xllm

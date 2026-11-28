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

#include <atb/atb_infer.h>
#include <atb/utils.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <cstdint>

#include "csrc/atb/atb_operation.h"
#include "csrc/atb/utils.h"

namespace xllm_kernel {
namespace {

aclDataType cache_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ACL_FLOAT16;
    case torch::kBFloat16:
      return ACL_BF16;
    case torch::kFloat32:
      return ACL_FLOAT;
    case torch::kInt8:
      return ACL_INT8;
    case torch::kUInt8:
      return ACL_UINT8;
    case torch::kInt32:
      return ACL_INT32;
    case torch::kInt64:
      return ACL_INT64;
    default:
      LOG(FATAL) << "Unsupported ATB cache dtype: " << dtype;
  }
}

atb::Tensor cache_tensor(const torch::Tensor& tensor) {
  const int64_t format = at_npu::native::get_npu_format(tensor);
  atb::Tensor result;
  result.desc.format =
      is_base_format(format) ? ACL_FORMAT_ND : static_cast<aclFormat>(format);
  result.desc.dtype = cache_dtype(tensor.scalar_type());
  result.desc.shape.dimNum = tensor.dim();
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    result.desc.shape.dims[dim] = tensor.size(dim);
  }
  result.deviceData = tensor.data_ptr();
  result.dataSize = atb::Utils::GetTensorSize(result);
  return result;
}

torch::Tensor reshape_paged_cache_atb(const torch::Tensor& slot_mapping,
                                      const torch::Tensor& keys,
                                      const torch::Tensor& values,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache) {
  if (keys.size(0) == 0) {
    return key_cache;
  }
  CHECK(key_cache.is_contiguous());
  CHECK(value_cache.is_contiguous());
  const torch::DeviceGuard guard(keys.device());
  torch::Tensor packed_keys =
      at_npu::native::npu_format_cast(keys.contiguous(), ACL_FORMAT_ND);
  torch::Tensor packed_values =
      at_npu::native::npu_format_cast(values.contiguous(), ACL_FORMAT_ND);
  torch::Tensor packed_slots =
      at_npu::native::npu_format_cast(slot_mapping.contiguous(), ACL_FORMAT_ND);
  atb::Tensor key = cache_tensor(key_cache);
  atb::Tensor value = cache_tensor(value_cache);

  atb::infer::ReshapeAndCacheParam param;
  param.compressType =
      atb::infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;
  param.kvCacheCfg = key.desc.format == ACL_FORMAT_FRACTAL_NZ &&
                             value.desc.format == ACL_FORMAT_FRACTAL_NZ
                         ? atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ
                         : atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE;
  atb::VariantPack pack;
  pack.inTensors = {cache_tensor(packed_keys),
                    cache_tensor(packed_values),
                    key,
                    value,
                    cache_tensor(packed_slots)};
  pack.outTensors = {key, value};
  run_atb_operation(
      "XllmKernelReshapePagedCache",
      param,
      static_cast<int32_t>(param.kvCacheCfg),
      std::move(pack),
      {packed_keys, packed_values, packed_slots, key_cache, value_cache});
  return key_cache;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "reshape_paged_cache_atb(Tensor slot_mapping, Tensor keys, Tensor "
      "values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("reshape_paged_cache_atb", TORCH_FN(reshape_paged_cache_atb));
}

}  // namespace
}  // namespace xllm_kernel

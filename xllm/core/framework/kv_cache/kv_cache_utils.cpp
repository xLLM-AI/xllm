/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_cache_utils.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/kv_cache_shape.h"
#if defined(USE_MLU)
#include "platform/mlu/mlu_tensor_alloc.h"
#endif
#if defined(USE_NPU)
#include "acl/acl.h"
#endif

namespace xllm {
namespace {

#if defined(USE_NPU)
size_t get_tensor_nbytes(const std::vector<int64_t>& dims,
                         torch::ScalarType dtype) {
  size_t count = 1;
  for (int64_t dim : dims) {
    CHECK_GE(dim, 0) << "tensor dim must be non-negative";
    const size_t dim_size = static_cast<size_t>(dim);
    if (dim_size > 0) {
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

void free_acl_tensor(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  const auto acl_ret = aclrtFree(ptr);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtFree failed, ret=" << std::hex << acl_ret << ", ptr=" << ptr;
}

torch::Tensor alloc_npu_huge_page_tensor(const std::vector<int64_t>& dims,
                                         torch::ScalarType dtype,
                                         aclFormat format) {
  void* buffer = nullptr;
  const size_t nbytes = get_tensor_nbytes(dims, dtype);
  auto acl_ret = aclrtMalloc(&buffer, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtMalloc KV cache failed, ret=" << std::hex << acl_ret
      << ", nbytes=" << nbytes;

  constexpr c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  auto tensor = torch::empty(
      {0}, torch::TensorOptions().dtype(dtype).device(device_type));
  torch::DataPtr data_ptr(buffer, buffer, free_acl_tensor, tensor.device());

  auto* storage_create = c10::GetStorageImplCreate(device_type);
  auto* allocator = c10::GetAllocator(device_type);
  torch::Storage storage = storage_create(c10::StorageImpl::use_byte_size_t(),
                                          c10::SymInt(nbytes),
                                          std::move(data_ptr),
                                          allocator,
                                          true);

  tensor.set_(storage, 0, dims);
  auto* tensor_storage = static_cast<torch_npu::NPUStorageImpl*>(
      tensor.storage().unsafeGetStorageImpl());
  tensor_storage->npu_desc_.npu_format_ = format;
  return tensor;
}
#endif

}  // namespace

bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval) {
  if (full_attention_interval <= 1) {
    return false;
  }
  return (layer_idx + 1) % full_attention_interval != 0;
}

bool use_npu_nz_kv_cache_layout(const std::string& model_type) {
  return (model_type == "deepseek_v3" || model_type == "deepseek_v3_mtp") &&
         ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
}

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  KVCacheTensors tensors;
#if defined(USE_MLU)
  if (create_options.enable_raw_device_allocator()) {
    tensors.key_cache = mlu::alloc_zero_tensor(kv_cache_shape.key_cache_shape(),
                                               create_options.dtype(),
                                               create_options.device());
    if (kv_cache_shape.has_value_cache_shape()) {
      tensors.value_cache =
          mlu::alloc_zero_tensor(kv_cache_shape.value_cache_shape(),
                                 create_options.dtype(),
                                 create_options.device());
    }
  } else {
    tensors.key_cache = torch::zeros(
        kv_cache_shape.key_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
    if (!kv_cache_shape.value_cache_shape().empty()) {
      tensors.value_cache = torch::zeros(
          kv_cache_shape.value_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device()));
    }
  }
#elif defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.key_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.key_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
    tensors.value_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.value_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
  } else {
    tensors.key_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.key_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
    tensors.value_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.value_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
  }
#else
  tensors.key_cache = torch::zeros(
      kv_cache_shape.key_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));

  // deepseek_v3 model has no value cache on mlu device
  if (!kv_cache_shape.value_cache_shape().empty()) {
    tensors.value_cache = torch::zeros(
        kv_cache_shape.value_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
  }
#endif
  return tensors;
}

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_index_cache_shape())
      << "index_cache_shape must be initialized.";
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

#if defined(USE_MLU)
  if (create_options.enable_raw_device_allocator()) {
    tensors.index_cache =
        mlu::alloc_zero_tensor(kv_cache_shape.index_cache_shape(),
                               create_options.dtype(),
                               create_options.device());
  } else {
    tensors.index_cache = torch::zeros(
        kv_cache_shape.index_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
  }
#elif defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.index_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.index_cache_shape(),
                                   create_options.dtype(),
                                   npu_format_type);
  } else {
    tensors.index_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.index_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        npu_format_type);
  }
#else
  tensors.index_cache = torch::zeros(
      kv_cache_shape.index_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
#endif
  return tensors;
}

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
#if !defined(USE_MLU)
  CHECK(!create_options.enable_kv_cache_quant())
      << "KV cache quantization is only supported on MLU backend.";
#endif

  QuantizedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  std::vector<int64_t> key_scale_shape(key_cache_shape.begin(),
                                       key_cache_shape.end() - 1);

  // float32 scale tensor for quantized KV cache (int8)
  tensors.key_cache_scale = torch::zeros(
      key_scale_shape,
      torch::dtype(torch::kFloat32).device(create_options.device()));
  if (!kv_cache_shape.value_cache_shape().empty()) {
    const std::vector<int64_t>& value_cache_shape =
        kv_cache_shape.value_cache_shape();
    std::vector<int64_t> value_scale_shape(value_cache_shape.begin(),
                                           value_cache_shape.end() - 1);
    tensors.value_cache_scale = torch::zeros(
        value_scale_shape,
        torch::dtype(torch::kFloat32).device(create_options.device()));
  }

  return tensors;
}

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_conv_cache_shape())
      << "conv_cache_shape must be initialized.";
  CHECK(kv_cache_shape.has_ssm_cache_shape())
      << "ssm_cache_shape must be initialized.";
  LinearAttentionKVCacheTensors tensors;

#if defined(USE_NPU)
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.conv_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.conv_cache_shape(),
                                   create_options.dtype(),
                                   ACL_FORMAT_ND);
    tensors.conv_cache.zero_();
    tensors.ssm_cache =
        alloc_npu_huge_page_tensor(kv_cache_shape.ssm_cache_shape(),
                                   create_options.ssm_dtype(),
                                   ACL_FORMAT_ND);
    tensors.ssm_cache.zero_();
  } else {
    tensors.conv_cache = at_npu::native::npu_format_cast(
        torch::zeros(kv_cache_shape.conv_cache_shape(),
                     torch::dtype(create_options.dtype())
                         .device(create_options.device())),
        ACL_FORMAT_ND);
    tensors.ssm_cache = at_npu::native::npu_format_cast(
        torch::zeros(kv_cache_shape.ssm_cache_shape(),
                     torch::dtype(create_options.ssm_dtype())
                         .device(create_options.device())),
        ACL_FORMAT_ND);
  }
#else
  tensors.conv_cache = torch::zeros(
      kv_cache_shape.conv_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
  tensors.ssm_cache = torch::zeros(
      kv_cache_shape.ssm_cache_shape(),
      torch::dtype(create_options.ssm_dtype()).device(create_options.device()));
#endif

  return tensors;
}

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type) {
  return use_npu_nz_kv_cache_layout(model_type) ? ACL_FORMAT_FRACTAL_NZ
                                                : ACL_FORMAT_ND;
}
#endif

}  // namespace xllm

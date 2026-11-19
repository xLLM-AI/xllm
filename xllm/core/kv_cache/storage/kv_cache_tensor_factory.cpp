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

#include "core/kv_cache/storage/kv_cache_tensor_factory.h"

#include <glog/logging.h>

#include "core/kv_cache/layout/kv_cache_layout_policy.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/tensor_byte_size.h"
#include "core/util/utils.h"
#if defined(USE_NPU)
#include "acl/acl_rt.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/framework/model/model_type.h"
#endif

namespace xllm {

namespace {
#if defined(USE_NPU)
void free_acl_tensor(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  const auto acl_ret = aclrtFree(ptr);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtFree failed, ret=" << std::hex << acl_ret << ", ptr=" << ptr;
}
#endif

// Per-token fp32 scale tensor, shaped as the cache shape with the trailing
// (head_dim) dimension removed.
torch::Tensor alloc_scale(const std::vector<int64_t>& cache_shape,
                          const torch::Device& device) {
  std::vector<int64_t> scale_shape(cache_shape.begin(), cache_shape.end() - 1);
  return torch::zeros(scale_shape,
                      torch::dtype(torch::kFloat32).device(device));
}

torch::Tensor alloc_cache_tensor(KVCacheTensorRole role,
                                 const std::vector<int64_t>& shape,
                                 torch::ScalarType dtype,
                                 const KVCacheCreateOptions& create_options) {
  std::shared_ptr<KVTensorAllocator> allocator =
      create_options.tensor_allocator();
  // Linear state is indexed by request slots and remains resident independently
  // of the scheduler's paged KV blocks.
  const bool resident_state =
      create_options.enable_vmm_kvcache() &&
      (role == KVCacheTensorRole::CONV || role == KVCacheTensorRole::SSM);
  if (allocator == nullptr || resident_state) {
    allocator = default_kv_tensor_allocator();
  }
  return allocator->allocate(role, shape, dtype, create_options.device());
}

#if defined(USE_NPU)
torch::Tensor alloc_npu_zeroed_tensor(
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    aclFormat format,
    const KVCacheCreateOptions& create_options) {
  if (create_options.enable_kv_cache_huge_page_allocator()) {
    auto tensor = alloc_npu_huge_page_tensor(shape, dtype, format);
    tensor.zero_();
    return tensor;
  }
  return at_npu::native::npu_format_cast(
      torch::zeros(shape, torch::dtype(dtype).device(create_options.device())),
      format);
}
#endif

}  // namespace

#if defined(USE_NPU)
torch::Tensor alloc_npu_huge_page_tensor(const std::vector<int64_t>& dims,
                                         torch::ScalarType dtype,
                                         aclFormat format) {
  void* buffer = nullptr;
  const size_t nbytes = detail::get_tensor_nbytes(dims, dtype);
  // aclrtMalloc rejects zero-byte buffers; preserve the requested empty shape.
  if (nbytes == 0) {
    auto empty_tensor = torch::empty(dims,
                                     torch::dtype(dtype).device(torch::Device(
                                         c10::DeviceType::PrivateUse1)));
    auto* empty_storage = static_cast<torch_npu::NPUStorageImpl*>(
        empty_tensor.storage().unsafeGetStorageImpl());
    empty_storage->npu_desc_.npu_format_ = format;
    return empty_tensor;
  }
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

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  KVCacheTensors tensors;
#if defined(USE_MLU)
  tensors.key_cache = alloc_cache_tensor(KVCacheTensorRole::KEY,
                                         kv_cache_shape.key_cache_shape(),
                                         create_options.dtype(),
                                         create_options);
  if (kv_cache_shape.has_value_cache_shape()) {
    tensors.value_cache = alloc_cache_tensor(KVCacheTensorRole::VALUE,
                                             kv_cache_shape.value_cache_shape(),
                                             create_options.dtype(),
                                             create_options);
  }
#elif defined(USE_NPU)
  if (create_options.static_layout() != nullptr) {
    tensors.key_cache = alloc_cache_tensor(KVCacheTensorRole::KEY,
                                           kv_cache_shape.key_cache_shape(),
                                           create_options.dtype(),
                                           create_options);
    tensors.value_cache = alloc_cache_tensor(KVCacheTensorRole::VALUE,
                                             kv_cache_shape.value_cache_shape(),
                                             create_options.dtype(),
                                             create_options);
    return tensors;
  }
  CHECK(!create_options.enable_vmm_kvcache() ||
        !use_npu_nz_kv_cache_layout(create_options.model_type()))
      << "Virtual memory does not support the NPU NZ KV cache layout.";
  const bool mla_packed_c8 = create_options.mla_packed_c8();
  // The factory rejects NZ + virtual memory before any tensor is created.
  // Keep the model-selected descriptor here so non-VM DeepSeek caches retain
  // FRACTAL_NZ, while supported virtual-memory caches select ND naturally.
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  auto allocate = [&](KVCacheTensorRole role,
                      const std::vector<int64_t>& shape,
                      torch::ScalarType dtype) -> torch::Tensor {
    if (create_options.tensor_allocator() != nullptr) {
      return at_npu::native::npu_format_cast(
          alloc_cache_tensor(role, shape, dtype, create_options),
          npu_format_type);
    }
    if (create_options.enable_kv_cache_huge_page_allocator()) {
      return alloc_npu_huge_page_tensor(shape, dtype, npu_format_type);
    }
    return at_npu::native::npu_format_cast(
        torch::empty(shape,
                     torch::dtype(dtype).device(create_options.device())),
        npu_format_type);
  };
  tensors.key_cache =
      allocate(KVCacheTensorRole::KEY,
               kv_cache_shape.key_cache_shape(),
               mla_packed_c8 ? torch::kChar : create_options.dtype());
  // Packed main KV stores NoPE, RoPE and scale bytes once; K and V alias.
  if (mla_packed_c8) {
    tensors.value_cache = tensors.key_cache;
  } else if (kv_cache_shape.has_value_cache_shape()) {
    tensors.value_cache = allocate(KVCacheTensorRole::VALUE,
                                   kv_cache_shape.value_cache_shape(),
                                   create_options.dtype());
  }
#else
  tensors.key_cache = alloc_cache_tensor(KVCacheTensorRole::KEY,
                                         kv_cache_shape.key_cache_shape(),
                                         create_options.dtype(),
                                         create_options);

  // deepseek_v3 model has no value cache on mlu device
  if (!kv_cache_shape.value_cache_shape().empty()) {
    tensors.value_cache = alloc_cache_tensor(KVCacheTensorRole::VALUE,
                                             kv_cache_shape.value_cache_shape(),
                                             create_options.dtype(),
                                             create_options);
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
  if (kv_cache_shape.has_kpool_tail_shape()) {
    CHECK(!create_options.enable_indexer_cache_quant());
    CHECK_EQ(create_options.dtype(), torch::kBFloat16);
#if defined(USE_NPU)
    const aclFormat npu_format_type =
        get_npu_kv_cache_format(create_options.model_type());
    tensors.kpool_tail =
        alloc_npu_zeroed_tensor(kv_cache_shape.kpool_tail_shape(),
                                torch::kBFloat16,
                                npu_format_type,
                                create_options);
    // A finite gate marks a circular-tail row as initialized. Invalid or
    // unwritten rows stay at -inf so pool completion can reject stale slots.
    fill_kpool_tail_gate_sentinel(tensors.kpool_tail);
#else
    tensors.kpool_tail = alloc_cache_tensor(KVCacheTensorRole::KPOOL_TAIL,
                                            kv_cache_shape.kpool_tail_shape(),
                                            torch::kBFloat16,
                                            create_options);
#endif
  }
  const bool mla_packed_c8 = create_options.mla_packed_c8();
  if (create_options.enable_kv_cache_quant() && !mla_packed_c8) {
    QuantizedKVCacheTensors quantized_tensors =
        create_quantized_kv_cache_tensors(kv_cache_shape, create_options);
    tensors.kv_cache_tensors = quantized_tensors.kv_cache_tensors;
    tensors.key_cache_scale = quantized_tensors.key_cache_scale;
    tensors.value_cache_scale = quantized_tensors.value_cache_scale;
  } else {
    // Packed C8 uses byte storage with embedded scales.
    tensors.kv_cache_tensors =
        create_kv_cache_tensors(kv_cache_shape, create_options);
  }

#if defined(USE_MLU)
  const torch::ScalarType index_dtype =
      create_options.enable_indexer_cache_quant() ? torch::kChar
                                                  : create_options.dtype();
  tensors.index_cache = alloc_cache_tensor(KVCacheTensorRole::INDEX,
                                           kv_cache_shape.index_cache_shape(),
                                           index_dtype,
                                           create_options);
#elif defined(USE_NPU)
  const torch::ScalarType index_dtype =
      create_options.enable_indexer_cache_quant() ? torch::kChar
                                                  : create_options.dtype();
  // Keep INDEX's descriptor consistent with the main KV tensors. NZ + virtual
  // memory is rejected before reaching this allocation path.
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  if (create_options.tensor_allocator() != nullptr) {
    tensors.index_cache = at_npu::native::npu_format_cast(
        alloc_cache_tensor(KVCacheTensorRole::INDEX,
                           kv_cache_shape.index_cache_shape(),
                           index_dtype,
                           create_options),
        npu_format_type);
  } else if (create_options.enable_kv_cache_huge_page_allocator()) {
    tensors.index_cache = alloc_npu_huge_page_tensor(
        kv_cache_shape.index_cache_shape(), index_dtype, npu_format_type);
  } else {
    tensors.index_cache = at_npu::native::npu_format_cast(
        torch::empty(kv_cache_shape.index_cache_shape(),
                     torch::dtype(index_dtype).device(create_options.device())),
        npu_format_type);
  }
#else
  tensors.index_cache = alloc_cache_tensor(KVCacheTensorRole::INDEX,
                                           kv_cache_shape.index_cache_shape(),
                                           create_options.dtype(),
                                           create_options);
  if (!create_options.enable_vmm_kvcache() &&
      create_options.tensor_allocator() != nullptr) {
    tensors.index_cache.zero_();
  }
#endif
  if (create_options.enable_indexer_cache_quant()) {
#if !defined(USE_MLU) && !defined(USE_NPU)
    CHECK(false) << "Indexer cache INT8 is unsupported on this backend.";
#endif
    if (kv_cache_shape.has_index_cache_scale_shape()) {
#if defined(USE_MLU)
      tensors.index_cache_scale =
          alloc_cache_tensor(KVCacheTensorRole::INDEX_SCALE,
                             kv_cache_shape.index_cache_scale_shape(),
                             torch::kFloat32,
                             create_options);
#elif defined(USE_NPU)
      if (create_options.tensor_allocator() != nullptr) {
        tensors.index_cache_scale =
            alloc_cache_tensor(KVCacheTensorRole::INDEX_SCALE,
                               kv_cache_shape.index_cache_scale_shape(),
                               torch::kFloat16,
                               create_options)
                .zero_();
      } else {
        tensors.index_cache_scale = torch::zeros(
            kv_cache_shape.index_cache_scale_shape(),
            torch::dtype(torch::kFloat16).device(create_options.device()));
      }
#else
      tensors.index_cache_scale = torch::zeros(
          kv_cache_shape.index_cache_scale_shape(),
          torch::dtype(torch::kFloat32).device(create_options.device()));
#endif
    } else {
#if defined(USE_MLU)
      std::vector<int64_t> scale_shape(
          kv_cache_shape.index_cache_shape().begin(),
          kv_cache_shape.index_cache_shape().end() - 1);
      tensors.index_cache_scale =
          alloc_cache_tensor(KVCacheTensorRole::INDEX_SCALE,
                             scale_shape,
                             torch::kFloat32,
                             create_options);
#else
      tensors.index_cache_scale = alloc_scale(
          kv_cache_shape.index_cache_shape(), create_options.device());
#endif
    }
  }
  return tensors;
}

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
#if !defined(USE_MLU)
  // SFA C8 uses indexed byte storage; this generic factory keeps the
  // conventional key/value scale contract and is only supported on MLU.
  CHECK(!create_options.enable_kv_cache_quant())
      << util::kNonMluKvCacheQuantRejectMsg;
#endif

  QuantizedKVCacheTensors tensors;
  KVCacheCreateOptions quantized_options = create_options;
  quantized_options.dtype(torch::kChar);
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, quantized_options);

  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  std::vector<int64_t> key_scale_shape(key_cache_shape.begin(),
                                       key_cache_shape.end() - 1);

  // float32 scale tensor for quantized KV cache (int8)
  tensors.key_cache_scale = alloc_cache_tensor(KVCacheTensorRole::KEY_SCALE,
                                               key_scale_shape,
                                               torch::kFloat32,
                                               create_options);
  if (!kv_cache_shape.value_cache_shape().empty()) {
    const std::vector<int64_t>& value_cache_shape =
        kv_cache_shape.value_cache_shape();
    std::vector<int64_t> value_scale_shape(value_cache_shape.begin(),
                                           value_cache_shape.end() - 1);
    tensors.value_cache_scale =
        alloc_cache_tensor(KVCacheTensorRole::VALUE_SCALE,
                           value_scale_shape,
                           torch::kFloat32,
                           create_options);
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
  if (create_options.static_layout() != nullptr) {
    tensors.conv_cache = alloc_cache_tensor(KVCacheTensorRole::CONV,
                                            kv_cache_shape.conv_cache_shape(),
                                            create_options.dtype(),
                                            create_options);
    tensors.ssm_cache = alloc_cache_tensor(KVCacheTensorRole::SSM,
                                           kv_cache_shape.ssm_cache_shape(),
                                           create_options.ssm_dtype(),
                                           create_options);
    return tensors;
  }
  tensors.conv_cache =
      alloc_npu_zeroed_tensor(kv_cache_shape.conv_cache_shape(),
                              create_options.dtype(),
                              ACL_FORMAT_ND,
                              create_options);
  tensors.ssm_cache = alloc_npu_zeroed_tensor(kv_cache_shape.ssm_cache_shape(),
                                              create_options.ssm_dtype(),
                                              ACL_FORMAT_ND,
                                              create_options);
#else
  tensors.conv_cache = alloc_cache_tensor(KVCacheTensorRole::CONV,
                                          kv_cache_shape.conv_cache_shape(),
                                          create_options.dtype(),
                                          create_options);
  tensors.ssm_cache = alloc_cache_tensor(KVCacheTensorRole::SSM,
                                         kv_cache_shape.ssm_cache_shape(),
                                         create_options.ssm_dtype(),
                                         create_options);
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

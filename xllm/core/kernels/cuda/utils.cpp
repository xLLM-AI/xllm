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

#include "utils.h"

#include <ATen/DLConvertor.h>
#include <c10/core/Device.h>
#include <cuda_runtime.h>
#if !defined(USE_DCU)
#include <dlfcn.h>
#include <dlpack/dlpack.h>
#endif

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/env_var.h"
#include "core/util/utils.h"

namespace {
const std::unordered_map<torch::ScalarType, std::string_view>
    filename_safe_dtype_map = {
        {torch::kFloat16, "f16"},
        {torch::kBFloat16, "bf16"},
        {torch::kFloat8_e4m3fn, "e4m3"},
        {torch::kFloat8_e5m2, "e5m2"},
        {torch::kInt8, "i8"},
        {torch::kUInt8, "u8"},
        {torch::kInt32, "i32"},
        {torch::kUInt32, "u32"},
        {torch::kInt64, "i64"},
        {torch::kUInt64, "u64"},
};

#if !defined(USE_DCU)
void ensure_tvm_ffi_global_symbols() {
  // Ensure that TVMFFIEnvGetStream and other TVM FFI symbols are globally
  // visible for offline inference
  static std::once_flag once;
  std::call_once(once, []() {
    auto has_required_symbol = [](void* handle) -> bool {
      return handle != nullptr &&
             dlsym(handle, "TVMFFIEnvGetStream") != nullptr;
    };

    constexpr const char* kLibNames[] = {
        "libtvm_ffi.so",
        "libtvm_ffi.so.0",
    };

    // Prefer promoting an already loaded handle to RTLD_GLOBAL.
    for (const char* lib : kLibNames) {
      void* handle = dlopen(lib, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
      if (has_required_symbol(handle)) {
        VLOG(1) << "[tvmffi] promoted existing handle to RTLD_GLOBAL: " << lib;
        return;
      }
    }

    // Respect explicit path when provided.
    const char* explicit_lib = std::getenv("TVM_FFI_LIB");
    if (explicit_lib != nullptr && explicit_lib[0] != '\0') {
      void* handle = dlopen(explicit_lib, RTLD_NOW | RTLD_GLOBAL);
      if (has_required_symbol(handle)) {
        VLOG(1) << "[tvmffi] loaded explicit TVM_FFI_LIB with RTLD_GLOBAL: "
                << explicit_lib;
        return;
      }
    }

    // Fallback to dynamic linker search paths.
    for (const char* lib : kLibNames) {
      void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
      if (has_required_symbol(handle)) {
        VLOG(1) << "[tvmffi] loaded with RTLD_GLOBAL: " << lib;
        return;
      }
    }

    const char* err = dlerror();
    LOG(WARNING) << "[tvmffi] failed to make TVMFFI symbols globally visible. "
                 << "flashinfer op loading may fail. dlerror="
                 << (err ? err : "unknown");
  });
}

DLDataType torch_scalar_type_to_dl_data_type_impl(torch::ScalarType scalar_type,
                                                  int64_t element_bits) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = static_cast<uint8_t>(element_bits);
  switch (scalar_type) {
    case torch::ScalarType::UInt1:
    case torch::ScalarType::UInt2:
    case torch::ScalarType::UInt3:
    case torch::ScalarType::UInt4:
    case torch::ScalarType::UInt5:
    case torch::ScalarType::UInt6:
    case torch::ScalarType::UInt7:
    case torch::ScalarType::Byte:
    case torch::ScalarType::UInt16:
    case torch::ScalarType::UInt32:
    case torch::ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 6
    case torch::ScalarType::Int1:
    case torch::ScalarType::Int2:
    case torch::ScalarType::Int3:
    case torch::ScalarType::Int4:
    case torch::ScalarType::Int5:
    case torch::ScalarType::Int6:
    case torch::ScalarType::Int7:
    case torch::ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
#endif
    case torch::ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case torch::ScalarType::ComplexHalf:
    case torch::ScalarType::ComplexFloat:
    case torch::ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case torch::ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case torch::ScalarType::Float8_e5m2:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2;
      break;
    case torch::ScalarType::Float8_e5m2fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2fnuz;
      break;
    case torch::ScalarType::Float8_e4m3fn:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      break;
    case torch::ScalarType::Float8_e4m3fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fnuz;
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
    case torch::ScalarType::Float8_e8m0fnu:
      dtype.code = DLDataTypeCode::kDLFloat8_e8m0fnu;
      break;
    case torch::ScalarType::Float4_e2m1fn_x2:
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      dtype.lanes = 2;
      dtype.bits = 4;
      break;
#endif
    default:
      LOG(FATAL) << "Unsupported scalar type: " << torch::toString(scalar_type);
      break;
  }
  return dtype;
}

DLDataType get_data_type_for_dlpack_v1(const torch::Tensor& t) {
  const int64_t element_bits = static_cast<int64_t>(t.element_size() * 8);
  return torch_scalar_type_to_dl_data_type_impl(t.scalar_type(), element_bits);
}

DLDevice torch_device_to_dl_device_for_dlpack_v1(torch::Device device) {
  DLDevice ctx;

  ctx.device_id =
      (device.is_cuda() || device.is_privateuseone())
          ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
          : 0;

  switch (device.type()) {
    case torch::DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case torch::DeviceType::CUDA:
#ifdef USE_ROCM
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case torch::DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case torch::DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case torch::DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case torch::DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case torch::DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      LOG(FATAL) << "Cannot pack tensors on " << device.str();
      break;
  }

  return ctx;
}

torch::Device dl_device_to_torch_device_for_dlpack_v1(DLDevice device) {
  switch (device.device_type) {
    case DLDeviceType::kDLCPU:
    case DLDeviceType::kDLCUDAHost:
      return torch::Device(torch::kCPU);
    case DLDeviceType::kDLCUDA:
      return torch::Device(torch::kCUDA,
                           static_cast<c10::DeviceIndex>(device.device_id));
    case DLDeviceType::kDLROCM:
      return torch::Device(torch::kHIP,
                           static_cast<c10::DeviceIndex>(device.device_id));
    default:
      LOG(FATAL) << "Unsupported DLPack device type: "
                 << std::to_string(device.device_type);
      return torch::Device(torch::kCPU);
  }
}

template <class T>
struct ATenDLMTensor {
  torch::Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fill_version(T* tensor) {}

template <>
void fill_version<DLManagedTensorVersioned>(DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* to_dlpack_impl(const torch::Tensor& src) {
  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  atDLMTensor->tensor.dl_tensor.device =
      torch_device_to_dl_device_for_dlpack_v1(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = get_data_type_for_dlpack_v1(src);
  atDLMTensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides =
      const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fill_version(&atDLMTensor->tensor);
  return &(atDLMTensor->tensor);
}

int32_t torch_dlpack_managed_tensor_allocator(
    DLTensor* prototype,
    DLManagedTensorVersioned** out,
    void* error_ctx,
    void (*set_error)(void* error_ctx, const char* kind, const char* message)) {
  try {
    if (prototype == nullptr || out == nullptr) {
      LOG(FATAL) << "prototype and out must not be null";
      return -1;
    }

    std::vector<int64_t> shape(prototype->shape,
                               prototype->shape + prototype->ndim);
    torch::TensorOptions options =
        torch::TensorOptions()
            .dtype(at::toScalarType(prototype->dtype))
            .device(dl_device_to_torch_device_for_dlpack_v1(prototype->device));
    torch::Tensor tensor = torch::empty(shape, options);
    *out = to_dlpack_impl<DLManagedTensorVersioned>(tensor);
    return 0;
  } catch (const std::exception& e) {
    if (set_error != nullptr) {
      set_error(error_ctx, "MemoryError", e.what());
    }
    return -1;
  }
}

void ensure_tvm_ffi_tensor_allocator() {
  static std::once_flag once;
  std::call_once(once, []() {
    DLPackManagedTensorAllocator previous_allocator = nullptr;
    const int32_t rc = TVMFFIEnvSetDLPackManagedTensorAllocator(
        torch_dlpack_managed_tensor_allocator,
        /*write_to_global_context=*/1,
        &previous_allocator);
    if (rc != 0) {
      LOG(FATAL) << "[tvmffi] failed to register Torch DLPack allocator, rc="
                 << rc;
    }
  });
}
}  // namespace

namespace xllm::kernel::cuda {

bool should_use_tensor_core(torch::ScalarType kv_cache_dtype,
                            int64_t num_attention_heads,
                            int64_t num_kv_heads) {
  // Calculate GQA group size
  int64_t gqa_group_size = num_attention_heads / num_kv_heads;

  // For Flashinfer, a GQA group size of at least 4 is needed to efficiently
  // use Tensor Core for decode phase, as it fuses the head group with the token
  // dimension in MMA.
  if (kv_cache_dtype == torch::ScalarType::Float8_e4m3fn ||
      kv_cache_dtype == torch::ScalarType::Float8_e5m2) {
    return true;
  } else if (kv_cache_dtype == torch::ScalarType::Half ||
             kv_cache_dtype == torch::ScalarType::BFloat16) {
    return gqa_group_size >= 4;
  }

  return false;
}

bool support_pdl() { return Platform::is_enable_pdl(); }

std::string path_to_uri_so_lib(const std::string& uri) {
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask) {
  bool support_fa3_backend =
      (pos_encoding_mode == 0) && !use_fp16_qk_reduction && !use_custom_mask;

  if (Platform::is_support_sm90a() && support_fa3_backend) {
    return "fa3";
  }
  return "fa2";
}

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction) {
  std::ostringstream oss;
#if defined(USE_MACA)
  // use batch_prefill_tvm_ffi_with_kv_cache api for maca
  oss << "batch_prefill_tvm_ffi_with_kv_cache_"
#else
  oss << "batch_prefill_with_kv_cache_"
#endif
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False") << "_"
      << "f16qk_" << (use_fp16_qk_reduction ? "True" : "False");

  if (backend == "fa3") oss << "_sm90";

  return oss.str();
}

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap) {
  std::ostringstream oss;

#if defined(USE_MACA)
  // use batch_decode_tvm_ffi_with_kv_cache api for maca
  oss << "batch_decode_tvm_ffi_with_kv_cache_"
#else
  oss << "batch_decode_with_kv_cache_"
#endif
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False");

  return oss.str();
}

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  static std::unordered_map<std::string, torch::Tensor> cache_buffer_map;
  int32_t seq_len_pow2 = xllm::util::ceil_pow2(seq_len);

  std::string key = std::string("range_") + std::to_string(seq_len_pow2);
  auto it = cache_buffer_map.find(key);
  if (it != cache_buffer_map.end()) {
    return it->second.slice(0, 0, seq_len);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor buffer = torch::arange(seq_len_pow2, options);
  cache_buffer_map.insert(std::make_pair(key, buffer));
  return buffer.slice(0, 0, seq_len);
}

std::tuple<torch::Tensor, double> split_scale_param(
    const torch::Tensor& scale) {
  if (!scale.defined()) {
    return std::make_tuple(torch::Tensor(), 1.0);
  }

  if (scale.dim() == 0) {
    return std::make_tuple(torch::Tensor(), scale.item<double>());
  }

  return std::make_tuple(scale, 1.0);
}

DLDataType to_dl_data_type(torch::ScalarType scalar_type) {
  const int64_t element_bits =
      static_cast<int64_t>(torch::elementSize(scalar_type) * 8);
  return torch_scalar_type_to_dl_data_type_impl(scalar_type, element_bits);
}

// below are tvm-ffi related functions
ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor) {
  if (!torch_tensor.defined()) {
    LOG(FATAL) << "torch_tensor is not defined";
  }

  auto dlpack = to_dlpack_impl<DLManagedTensorVersioned>(torch_tensor);
  return ffi::Tensor::FromDLPackVersioned(dlpack);
}

ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional) {
  if (!optional.has_value()) {
    return ffi::Optional<ffi::Tensor>();
  }
  return ffi::Optional<ffi::Tensor>(to_ffi_tensor(optional.value()));
}

ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors) {
  std::vector<ffi::Tensor> ffi_tensors;
  ffi_tensors.reserve(torch_tensors.size());
  for (const auto& torch_tensor : torch_tensors) {
    ffi_tensors.emplace_back(to_ffi_tensor(torch_tensor));
  }
  return ffi::Array<ffi::Tensor>(ffi_tensors);
}

ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional) {
  if (!optional.has_value()) {
    return ffi::Optional<ffi::Array<ffi::Tensor>>();
  }
  return ffi::Optional<ffi::Array<ffi::Tensor>>(
      to_ffi_array_tensors(optional.value()));
}

ffi::Module get_module(const std::string& uri) {
  static thread_local std::unordered_map<std::string, ffi::Module> module_cache;

  auto it = module_cache.find(uri);
  if (it != module_cache.end()) {
    return it->second;
  }

  ensure_tvm_ffi_global_symbols();
  ensure_tvm_ffi_tensor_allocator();
  std::string so_file_path = path_to_uri_so_lib(uri);
  auto mod = ffi::Module::LoadFromFile(so_file_path);
  module_cache.emplace(uri, mod);
  return mod;
}

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name) {
  static thread_local std::unordered_map<std::string, ffi::Function> func_cache;

  std::string key = uri + "|" + func_name;

  auto it = func_cache.find(key);
  if (it != func_cache.end()) {
    return it->second;
  }
  VLOG(10) << "get_function:  uri: " << uri << " func_name: " << func_name;
  auto func_opt = get_module(uri)->GetFunction(func_name);
  if (!func_opt.defined()) {
    LOG(FATAL) << "TVM function not found. uri=" << uri
               << " func_name=" << func_name
               << " so_path=" << path_to_uri_so_lib(uri)
               << ". This usually indicates a mismatched or incomplete kernel "
                  "library build.";
  }
  auto func = func_opt.value();
  func_cache.emplace(key, func);
  return func;
}
#endif  // !USE_DCU
}  // namespace xllm::kernel::cuda

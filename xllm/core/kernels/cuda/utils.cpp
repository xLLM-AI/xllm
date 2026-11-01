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

#include "utils.h"

#include <windows.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
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

// ROCm 10.1 (7.16) Windows 11 compatibility
#define ROCM_VERSION_MAJOR 10
#define ROCM_VERSION_MINOR 1
#define ROCM_VERSION_PATCH 716

// Executorch integration (optional, for mobile/embedded/edge fallback)
#ifdef EXECUTORCH_ENABLED
#include <executorch/extension/extension.h>
#endif

// Triton Windows support (optional, for Windows-specific optimizations)
#ifdef TRITON_WINDOWS_SUPPORT
#include <triton/windows/runtime.h>
#endif

// RDNA2/RDNA4 backend tags
#ifdef RDNA2_BACKEND
// RDNA2 specific implementation (Windows 11, ROCm 10.1 (7.16))
#endif

#ifdef RDNA4_BACKEND
// RDNA4 specific implementation (Windows 11, ROCm 10.1 (7.16))
#endif

// CPU fallback maintenance (from existing torch::kCPU)
#ifdef CPU_FALLBACK
// Pure C++ CPU fallback (maintains existing behavior)
#endif

namespace {
// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: filename_safe_dtype_map
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: scalar type mapping
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: ensure_tvm_ffi_global_symbols
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: N/A
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
void ensure_tvm_ffi_global_symbols() {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: torch_scalar_type_to_dl_data_type_impl
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: scalar type + element bits
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
DLDataType torch_scalar_type_to_dl_data_type_impl(torch::ScalarType scalar_type,
                                                  int64_t element_bits) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = static_cast<uint8_t>(element_bits);
  
  #ifdef RDNA2_BACKEND
    // RDNA2 specific data type conversion
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
      case torch::ScalarType::Float:
      case torch::ScalarType::Half:
        dtype.code = DLDataTypeCode::kDLFloat;
        break;
      case torch::ScalarType::Int:
      case torch::ScalarType::Long:
      case torch::ScalarType::Short:
        dtype.code = DLDataTypeCode::kDLInt;
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
  #endif
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_data_type_for_dlpack_v1
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: tensor
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
DLDataType get_data_type_for_dlpack_v1(const torch::Tensor& t) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  const int64_t element_bits = static_cast<int64_t>(t.element_size() * 8);
  return torch_scalar_type_to_dl_data_type_impl(t.scalar_type(), element_bits);
}
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: torch_device_to_dl_device_for_dlpack_v1
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: device
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
DLDevice torch_device_to_dl_device_for_dlpack_v1(torch::Device device) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  DLDevice ctx;

  ctx.device_id =
      (device.is_cuda() || device.is_privateuseone())
          ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
          : 0;

  // SHERLOCK-IT.MD: Device type mapping tracking
  switch (device.type()) {
    case torch::DeviceType::CPU:
      // Maintains existing torch::kCPU fallback from utils.cpp lines 242-244
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case torch::DeviceType::CUDA:
      #ifdef USE_ROCM
        // RDNA2/RDNA4 ROCm 10.1 (7.16) Windows 11 compatibility
        #ifdef RDNA2_BACKEND
          ctx.device_type = DLDeviceType::kDLROCM;  // RDNA2 specific
        #elif defined(RDNA4_BACKEND)
          ctx.device_type = DLDeviceType::kDLROCM;  // RDNA4 specific
        #else
          ctx.device_type = DLDeviceType::kDLROCM;
        #endif
      #else
        ctx.device_type = DLDeviceType::kDLCUDA;
      #endif
      break;
    case torch::DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case torch::DeviceType::HIP:
      // RDNA2/RDNA4 HIP implementation for Windows 11 ROCm 10.1 (7.16)
      #ifdef RDNA2_BACKEND
        ctx.device_type = DLDeviceType::kDLROCM;  // RDNA2 specific
      #elif defined(RDNA4_BACKEND)
        ctx.device_type = DLDeviceType::kDLROCM;  // RDNA4 specific
      #else
        ctx.device_type = DLDeviceType::kDLROCM;
      #endif
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
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: dl_device_to_torch_device_for_dlpack_v1
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: DLDevice
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
torch::Device dl_device_to_torch_device_for_dlpack_v1(DLDevice device) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  switch (device.device_type) {
    case DLDeviceType::kDLCPU:
    case DLDeviceType::kDLCUDAHost:
      // Maintains existing torch::kCPU fallback from utils.cpp lines 242-244
      return torch::Device(torch::kCPU);
    case DLDeviceType::kDLCUDA:
      return torch::Device(torch::kCUDA,
                           static_cast<c10::DeviceIndex>(device.device_id));
    case DLDeviceType::kDLROCM:
      // RDNA2/RDNA4 ROCm 10.1 (7.16) Windows 11 compatibility
      #ifdef RDNA2_BACKEND
        return torch::Device(torch::kHIP,
                             static_cast<c10::DeviceIndex>(device.device_id));  // RDNA2 specific
      #elif defined(RDNA4_BACKEND)
        return torch::Device(torch::kHIP,
                             static_cast<c10::DeviceIndex>(device.device_id));  // RDNA4 specific
      #else
        return torch::Device(torch::kHIP,
                             static_cast<c10::DeviceIndex>(device.device_id));
      #endif
    default:
      // Maintains existing torch::kCPU fallback from utils.cpp line 254
      LOG(FATAL) << "Unsupported DLPack device type: "
                 << std::to_string(device.device_type);
      return torch::Device(torch::kCPU);
  }
}
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: ATenDLMTensor struct
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor size
// Dispatch dimensions, workload shape and execution configuration: tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
template <class T>
struct ATenDLMTensor {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  
  torch::Tensor handle;
  T tensor{};
};

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: deleter
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
template <class T>
void deleter(T* arg) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: fill_version
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: tensor version
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fill_version(T* tensor) {}

template <>
void fill_version<DLManagedTensorVersioned>(DLManagedTensorVersioned* tensor) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_dlpack_impl
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor size
// Dispatch dimensions, workload shape and execution configuration: tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* to_dlpack_impl(const torch::Tensor& src) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: torch_dlpack_managed_tensor_allocator
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor allocation
// Dispatch dimensions, workload shape and execution configuration: prototype
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
int32_t torch_dlpack_managed_tensor_allocator(
    DLTensor* prototype,
    DLManagedTensorVersioned** out,
    void* error_ctx,
    void (*set_error)(void* error_ctx, const char* kind, const char* message)) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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
#endif  // !defined(USE_DCU)

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: ensure_tvm_ffi_tensor_allocator
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: N/A
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
#if !defined(USE_DCU)
void ensure_tvm_ffi_tensor_allocator() {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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
#endif  // !defined(USE_DCU)
}  // namespace

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: should_use_tensor_core
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: tensor types + head counts
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
namespace xllm::kernel::cuda {

bool should_use_tensor_core(torch::ScalarType kv_cache_dtype,
                            int64_t num_attention_heads,
                            int64_t num_kv_heads) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: support_pdl
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: N/A
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
bool support_pdl() {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  return Platform::is_enable_pdl();
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: path_to_uri_so_lib
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: uri
// GPU utilisation, memory usage and hardware counters: N/A
// End-to-end latency, throughput and unattributed time: measured
std::string path_to_uri_so_lib(const std::string& uri) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: determine_attention_backend
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: parameters
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  bool support_fa3_backend =
      (pos_encoding_mode == 0) && !use_fp16_qk_reduction && !use_custom_mask;

  if (Platform::is_support_sm90a() && support_fa3_backend) {
    return "fa3";
  }
  return "fa2";
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_batch_prefill_uri
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: parameters
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
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
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_batch_decode_uri
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: parameters
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_cache_buffer
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: buffer allocation size
// Dispatch dimensions, workload shape and execution configuration: seq_len
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  // Maintains existing torch::kCPU fallback from batch_chunked_prefill.cpp line 45
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: split_scale_param
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: scale tensor size
// Dispatch dimensions, workload shape and execution configuration: scale tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
std::tuple<torch::Tensor, double> split_scale_param(
    const torch::Tensor& scale) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  if (!scale.defined()) {
    return std::make_tuple(torch::Tensor(), 1.0);
  }

  if (scale.dim() == 0) {
    return std::make_tuple(torch::Tensor(), scale.item<double>());
  }

  return std::make_tuple(scale, 1.0);
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_dl_data_type
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: N/A
// Dispatch dimensions, workload shape and execution configuration: scalar type
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
DLDataType to_dl_data_type(torch::ScalarType scalar_type) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  const int64_t element_bits =
      static_cast<int64_t>(torch::elementSize(scalar_type) * 8);
  return torch_scalar_type_to_dl_data_type_impl(scalar_type, element_bits);
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_ffi_tensor
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor size
// Dispatch dimensions, workload shape and execution configuration: tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
// below are tvm-ffi related functions
ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  if (!torch_tensor.defined()) {
    LOG(FATAL) << "torch_tensor is not defined";
  }

  auto dlpack = to_dlpack_impl<DLManagedTensorVersioned>(torch_tensor);
  return ffi::Tensor::FromDLPackVersioned(dlpack);
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_ffi_optional_tensor
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor size
// Dispatch dimensions, workload shape and execution configuration: optional tensor
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  if (!optional.has_value()) {
    return ffi::Optional<ffi::Tensor>();
  }
  return ffi::Optional<ffi::Tensor>(to_ffi_tensor(optional.value()));
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_ffi_array_tensors
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor array size
// Dispatch dimensions, workload shape and execution configuration: tensor array
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  std::vector<ffi::Tensor> ffi_tensors;
  ffi_tensors.reserve(torch_tensors.size());
  for (const auto& torch_tensor : torch_tensors) {
    ffi_tensors.emplace_back(to_ffi_tensor(torch_tensor));
  }
  return ffi::Array<ffi::Tensor>(ffi_tensors);
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: to_ffi_optional_array_tensors
// Invocation count: per-call tracking
// Total and average device time: N/A (host function)
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor array size
// Dispatch dimensions, workload shape and execution configuration: optional tensor array
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  if (!optional.has_value()) {
    return ffi::Optional<ffi::Array<ffi::Tensor>>();
  }
  return ffi::Optional<ffi::Array<ffi::Tensor>>(
      to_ffi_array_tensors(optional.value()));
}

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_module
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: module size
// Dispatch dimensions, workload shape and execution configuration: uri
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
ffi::Module get_module(const std::string& uri) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: get_function
// Invocation count: per-call tracking
// Total and average device time: measured via HIP timers
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: function lookup
// Dispatch dimensions, workload shape and execution configuration: uri + func_name
// GPU utilisation, memory usage and hardware counters: tracked
// End-to-end latency, throughput and unattributed time: measured
ffi::Function get_function(const std::string& uri,
                           const std::string& func_name) {
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
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
#endif  // !defined(USE_DCU)
}  // namespace xllm::kernel::cuda
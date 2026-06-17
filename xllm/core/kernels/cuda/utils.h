/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <ATen/DynamicLibrary.h>
#if defined(USE_DCU)
#include <c10/hip/HIPGuard.h>
#else
#include <c10/cuda/CUDAGuard.h>
#endif
#include <glog/logging.h>
#include <torch/torch.h>
#if !defined(USE_DCU)
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>
#endif

#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#if defined(__CUDACC__) || defined(_NVHPC_CUDA) || defined(__HIPCC__)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#endif

#if !defined(USE_DCU)
namespace ffi = tvm::ffi;
#endif

namespace xllm::kernel::cuda {

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

enum class ActivationType : int8_t {
  GELU = 0,
  RELU = 1,
  SILU = 2,
  SWIGLU = 3,
  GEGLU = 4,
  SWIGLU_BIAS = 5,
  RELU2 = 6,
  IDENTITY = 7,
  INVALID_TYPE = 8
};

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device);

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define DISPATCH_CASE_HALF_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
// NOLINTEND(cppcoreguidelines-macro-usage)

bool should_use_tensor_core(torch::ScalarType kv_cache_dtype,
                            int64_t num_attention_heads,
                            int64_t num_kv_heads);

bool support_pdl();

std::string path_to_uri_so_lib(const std::string& uri);

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask);

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
                                  bool use_fp16_qk_reduction);

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap);

std::tuple<torch::Tensor, double> split_scale_param(const torch::Tensor& scale);

#if !defined(USE_DCU)
DLDataType to_dl_data_type(torch::ScalarType scalar_type);

// below are tvm-ffi related functions
ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor);

ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional);

ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors);

ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional);

ffi::Module get_module(const std::string& uri);

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name);

inline void bind_tvmffi_stream_to_current_torch_stream(
    const torch::Device& device) {
  const auto cur = c10::cuda::getCurrentCUDAStream(device.index());
  // DLPack device type for CUDA is 2 (kDLCUDA).
  void* original_stream = nullptr;
  const int rc = TVMFFIEnvSetStream(
      /*device_type=*/2,
      /*device_id=*/device.index(),
      reinterpret_cast<void*>(cur.stream()),
      &original_stream);
  if (rc != 0) {
    LOG(WARNING) << "[tvmffi.stream] failed to set stream, rc=" << rc
                 << " dev=" << device.index();
  }
}
#endif  // !defined(USE_DCU)
}  // namespace xllm::kernel::cuda

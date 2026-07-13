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

#include "kernels/dcu/hipblaslt_fp8_adapter.h"

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <glog/logging.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace xllm {
namespace kernel {
namespace dcu {
namespace hipblaslt_fp8 {
namespace {

class HipblasLtHandleCache final {
 public:
  static HipblasLtHandleCache& instance() {
    // HIP/PyTorch runtime teardown may run before C++ static destructors.
    // Keep hipBLASLt handles alive until process exit and let the OS reclaim
    // them to avoid shutdown-time hipblasLtDestroy crashes.
    static HipblasLtHandleCache* cache = new HipblasLtHandleCache();
    return *cache;
  }

  hipblasLtHandle_t get_current_device_handle() {
    int device_id = 0;
    CHECK_EQ(hipGetDevice(&device_id), hipSuccess)
        << "dcu::hipblaslt_fp8: failed to query current device";
    CHECK_GE(device_id, 0) << "dcu::hipblaslt_fp8: invalid device id";

    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LT(static_cast<size_t>(device_id), handles_.size())
        << "dcu::hipblaslt_fp8: device id exceeds device count";
    hipblasLtHandle_t& handle = handles_[static_cast<size_t>(device_id)];
    if (handle == nullptr) {
      CHECK_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS)
          << "dcu::hipblaslt_fp8: hipblasLtCreate failed";
    }
    return handle;
  }

 private:
  HipblasLtHandleCache() {
    int device_count = 0;
    CHECK_EQ(hipGetDeviceCount(&device_count), hipSuccess)
        << "dcu::hipblaslt_fp8: failed to query device count";
    CHECK_GT(device_count, 0) << "dcu::hipblaslt_fp8: no DCU device found";
    handles_.resize(static_cast<size_t>(device_count), nullptr);
  }

  std::mutex mutex_;
  std::vector<hipblasLtHandle_t> handles_;
};

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

hipDataType data_type_from_scalar(torch::ScalarType dtype) {
  if (dtype == torch::kFloat8_e4m3fn) {
    return HIP_R_8F_E4M3;
  }
  if (dtype == torch::kFloat8_e5m2) {
    return HIP_R_8F_E5M2;
  }
  if (dtype == torch::kFloat16) {
    return HIP_R_16F;
  }
  if (dtype == torch::kBFloat16) {
    return HIP_R_16BF;
  }
  if (dtype == torch::kFloat32) {
    return HIP_R_32F;
  }
  LOG(FATAL) << "dcu::hipblaslt_fp8: unsupported dtype " << dtype;
  return HIP_R_32F;
}

int to_blas_int(int64_t value, const char* name) {
  CHECK_GT(value, 0) << "dcu::hipblaslt_fp8: " << name << " must be positive";
  CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int>::max()))
      << "dcu::hipblaslt_fp8: " << name << " exceeds hipBLASLt int limit";
  return static_cast<int>(value);
}

torch::Tensor normalize_scale(const torch::Tensor& scale,
                              int64_t expected_size,
                              const char* name) {
  CHECK(scale.defined()) << "dcu::hipblaslt_fp8: " << name
                         << " must be defined";
  CHECK(scale.is_cuda()) << "dcu::hipblaslt_fp8: " << name << " must be on DCU";
  CHECK(scale.scalar_type() == torch::kFloat32)
      << "dcu::hipblaslt_fp8: " << name << " must be float32";

  if (scale.numel() == 1) {
    return scale.reshape({1}).expand({expected_size}).contiguous();
  }

  torch::Tensor normalized = scale;
  if (scale.dim() == 2) {
    CHECK_EQ(scale.size(1), 1)
        << "dcu::hipblaslt_fp8: " << name << " must be [N] or [N,1], got "
        << scale.sizes();
    normalized = scale.squeeze(1);
  }
  CHECK_EQ(normalized.dim(), 1)
      << "dcu::hipblaslt_fp8: " << name
      << " must be scalar, [1], [N], or [N,1], got " << scale.sizes();
  CHECK_EQ(normalized.size(0), expected_size)
      << "dcu::hipblaslt_fp8: " << name << " size mismatch, expected "
      << expected_size << ", got " << normalized.size(0);
  return normalized.contiguous();
}

void check_fp8_gemm_inputs(const torch::Tensor& activation,
                           const torch::Tensor& weight,
                           const std::optional<torch::Tensor>& output) {
  CHECK(activation.defined())
      << "dcu::hipblaslt_fp8: activation must be defined";
  CHECK(weight.defined()) << "dcu::hipblaslt_fp8: weight must be defined";
  CHECK(activation.is_cuda())
      << "dcu::hipblaslt_fp8: activation must be on DCU";
  CHECK(weight.device() == activation.device())
      << "dcu::hipblaslt_fp8: weight must be on the same DCU device";
  CHECK(activation.is_contiguous())
      << "dcu::hipblaslt_fp8: activation must be contiguous";
  CHECK(weight.is_contiguous())
      << "dcu::hipblaslt_fp8: weight must be contiguous";
  CHECK_EQ(activation.dim(), 2)
      << "dcu::hipblaslt_fp8: activation must be [M,K], got "
      << activation.sizes();
  CHECK_EQ(weight.dim(), 2)
      << "dcu::hipblaslt_fp8: weight must be [N,K], got " << weight.sizes();
  CHECK_EQ(weight.size(1), activation.size(1))
      << "dcu::hipblaslt_fp8: weight K " << weight.size(1)
      << " != activation K " << activation.size(1);
  CHECK(is_fp8_dtype(activation.scalar_type()))
      << "dcu::hipblaslt_fp8: activation must be FP8";
  CHECK(weight.scalar_type() == activation.scalar_type())
      << "dcu::hipblaslt_fp8: activation/weight FP8 dtype must match";

  if (output.has_value() && output.value().defined()) {
    const torch::Tensor& output_tensor = output.value();
    CHECK(output_tensor.is_cuda())
        << "dcu::hipblaslt_fp8: output must be on DCU";
    CHECK(output_tensor.device() == activation.device())
        << "dcu::hipblaslt_fp8: output must be on the same DCU device";
    CHECK(output_tensor.is_contiguous())
        << "dcu::hipblaslt_fp8: output must be contiguous";
    CHECK_EQ(output_tensor.dim(), 2)
        << "dcu::hipblaslt_fp8: output must be [M,N], got "
        << output_tensor.sizes();
    CHECK_EQ(output_tensor.size(0), activation.size(0))
        << "dcu::hipblaslt_fp8: output M mismatch";
    CHECK_EQ(output_tensor.size(1), weight.size(0))
        << "dcu::hipblaslt_fp8: output N mismatch";
  }
}

}  // namespace

torch::Tensor fp8_gemm_nt(const torch::Tensor& activation,
                          const torch::Tensor& weight,
                          const torch::Tensor& activation_scale,
                          const torch::Tensor& weight_scale,
                          torch::ScalarType output_dtype,
                          const std::optional<torch::Tensor>& bias,
                          const std::optional<torch::Tensor>& output) {
  check_fp8_gemm_inputs(activation, weight, output);
  CHECK(output_dtype == torch::kFloat16 || output_dtype == torch::kBFloat16)
      << "dcu::hipblaslt_fp8: output dtype must be fp16 or bf16";

  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output = torch::empty({activation.size(0), weight.size(0)},
                                 activation.options().dtype(output_dtype));
  }

  torch::Tensor activation_scale_1d =
      normalize_scale(activation_scale, activation.size(0), "activation_scale");
  torch::Tensor weight_scale_1d =
      normalize_scale(weight_scale, weight.size(0), "weight_scale");

  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      activation.device());
  hipblasLtHandle_t handle =
      HipblasLtHandleCache::instance().get_current_device_handle();

  const int64_t m = activation.size(0);
  const int64_t n = weight.size(0);
  const int64_t k = activation.size(1);
  const int blas_m = to_blas_int(n, "N");
  const int blas_n = to_blas_int(m, "M");
  const int blas_k = to_blas_int(k, "K");

  const size_t lda = static_cast<size_t>(k);
  const size_t ldb = static_cast<size_t>(k);
  const size_t ldc = static_cast<size_t>(n);
  const size_t ldd = static_cast<size_t>(n);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const hipDataType fp8_type = data_type_from_scalar(activation.scalar_type());
  const hipDataType output_type = data_type_from_scalar(output_dtype);
  const hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  hipblasStatus_t status = hipblasLtGemmExQ(handle,
                                            HIPBLAS_OP_T,
                                            HIPBLAS_OP_N,
                                            blas_m,
                                            blas_n,
                                            blas_k,
                                            &alpha,
                                            weight.data_ptr(),
                                            fp8_type,
                                            lda,
                                            activation.data_ptr(),
                                            fp8_type,
                                            ldb,
                                            &beta,
                                            result_output.data_ptr(),
                                            output_type,
                                            ldc,
                                            result_output.data_ptr(),
                                            output_type,
                                            ldd,
                                            HIPBLAS_COMPUTE_32F,
                                            weight_scale_1d.data_ptr(),
                                            scale_mode,
                                            activation_scale_1d.data_ptr(),
                                            scale_mode,
                                            nullptr,
                                            output_type,
                                            stream);
  CHECK_EQ(status, HIPBLAS_STATUS_SUCCESS)
      << "dcu::hipblaslt_fp8: hipblasLtGemmExQ failed, M=" << m << ", N=" << n
      << ", K=" << k << ", status=" << status;

  if (bias.has_value() && bias.value().defined()) {
    CHECK(bias.value().device() == activation.device())
        << "dcu::hipblaslt_fp8: bias must be on the same DCU device";
    CHECK_EQ(bias.value().numel(), weight.size(0))
        << "dcu::hipblaslt_fp8: bias size must match output N";
    result_output.add_(bias.value());
  }
  return result_output;
}

}  // namespace hipblaslt_fp8
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm

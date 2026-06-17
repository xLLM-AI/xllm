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

#include <ATen/hip/HIPContextLight.h>
#include <c10/hip/HIPStream.h>
#include <glog/logging.h>
#include <hip/hip_bf16.h>
#include <hipblaslt/hipblaslt.h>

#include <mutex>

#include "core/kernels/dcu/dcu_ops_api.h"

namespace xllm::kernel::dcu {

namespace {

void check_hipblaslt_status(hipblasStatus_t status, const char* op) {
  CHECK(status == HIPBLAS_STATUS_SUCCESS)
      << "hipBLASLt " << op << " failed: status=" << status;
}

// Per-device hipBLASLt handle singleton, matching lmslim's BlasltHandle.
class BlasltHandle final {
 public:
  static hipblasLtHandle_t get() {
    int32_t dev;
    CHECK_EQ(hipGetDevice(&dev), hipSuccess);
    CHECK_GE(dev, 0);
    CHECK_LT(dev, kMaxGpu);
    CHECK_EQ(hipSetDevice(dev), hipSuccess);
    std::call_once(init_flags_[dev], [&]() {
      const hipblasStatus_t status = hipblasLtCreate(&handles_[dev]);
      check_hipblaslt_status(status, "Create");
    });
    CHECK(handles_[dev] != nullptr);
    return handles_[dev];
  }

 private:
  static constexpr int32_t kMaxGpu = 16;
  static inline std::once_flag init_flags_[kMaxGpu];
  static inline hipblasLtHandle_t handles_[kMaxGpu] = {nullptr};
};

// Create a matmul descriptor for a single GEMM call.
// Thread-safe: each call owns its own descriptor with per-call scale/bias
// pointers, avoiding races on shared mutable state.
hipblasLtMatmulDesc_t create_matmul_desc() {
  hipblasLtMatmulDesc_t desc;
  check_hipblaslt_status(
      hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32F),
      "MatmulDescCreate");

  hipblasLtMatmulMatrixScale_t mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)),
      "SetAttribute(A_SCALE_MODE)");
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)),
      "SetAttribute(B_SCALE_MODE)");

  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
      "SetAttribute(TRANSA)");
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(
          desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
      "SetAttribute(TRANSB)");
  return desc;
}

hipDataType scalar_type_to_hip_type(torch::ScalarType dt) {
  if (dt == torch::kHalf) {
    return HIP_R_16F;
  }
  if (dt == torch::kBFloat16) {
    return HIP_R_16BF;
  }
  return HIP_R_32F;
}

}  // namespace

torch::Tensor scaled_matmul(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& a_scale,
                            const torch::Tensor& b_scale,
                            torch::ScalarType output_dtype,
                            const std::optional<torch::Tensor>& bias,
                            const std::optional<torch::Tensor>& /*c*/,
                            const std::string& /*act_mode*/,
                            int64_t quant_bit_size,
                            double /*alpha*/,
                            double /*beta*/,
                            bool /*use_hp_active*/,
                            int64_t a_quant_bit_size,
                            const std::optional<torch::Tensor>& /*a_calib*/,
                            const std::optional<torch::Tensor>& /*b_calib*/,
                            const std::optional<torch::Tensor>& output) {
  CHECK(quant_bit_size == 8 && a_quant_bit_size == 8)
      << "scaled_matmul only supports w8a8 quantization";

  CHECK(output_dtype == at::ScalarType::Half ||
        output_dtype == at::ScalarType::BFloat16)
      << "output dtype must be half or bfloat16";

  // Input validation
  CHECK_EQ(a.dim(), 2);
  CHECK_EQ(b.dim(), 2);
  CHECK_EQ(a.scalar_type(), torch::kInt8);
  CHECK_EQ(b.scalar_type(), torch::kInt8);
  CHECK_EQ(a.size(1), b.size(1));
  CHECK(a.is_contiguous());
  CHECK(b.is_contiguous());
  CHECK(a_scale.has_value()) << "a_scale is required for scaled_matmul";
  CHECK(a_scale->defined()) << "a_scale must be defined";
  CHECK(b_scale.defined()) << "b_scale must be defined";

  int64_t m = a.size(0);
  int64_t n = b.size(0);
  int64_t k = a.size(1);

  // Allocate output
  torch::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = output.value();
    CHECK_EQ(output_tensor.dim(), 2);
    CHECK_EQ(output_tensor.size(0), m);
    CHECK_EQ(output_tensor.size(1), n);
    CHECK_EQ(output_tensor.scalar_type(), output_dtype);
    CHECK_EQ(output_tensor.device(), a.device());
    CHECK(output_tensor.is_contiguous());
  } else {
    output_tensor = torch::empty(
        {m, n}, torch::TensorOptions().dtype(output_dtype).device(a.device()));
  }

  // Prepare scale tensors: squeeze to 1D if needed
  torch::Tensor scale_a = b_scale;          // weight scale [N] or [N,1]
  torch::Tensor scale_b = a_scale.value();  // activation scale [M] or [M,1]
  if (scale_a.dim() > 1) {
    scale_a = scale_a.squeeze(-1);
  }
  if (scale_b.dim() > 1) {
    scale_b = scale_b.squeeze(-1);
  }

  hipblasLtHandle_t handle = BlasltHandle::get();
  hipblasLtMatmulDesc_t matmul_desc = create_matmul_desc();

  // Matrix layouts (column-major convention, matching lmslim)
  hipblasLtMatrixLayout_t mat_a_layout, mat_b_layout, mat_c_layout,
      output_layout;

  // A = weight [N,K]: col-major layout rows=K, cols=N, lda=K
  check_hipblaslt_status(
      hipblasLtMatrixLayoutCreate(&mat_a_layout, HIP_R_8I, k, n, k),
      "MatrixLayoutCreate(mat_a)");
  // B = activation [M,K]: col-major layout rows=K, cols=M, ldb=K
  check_hipblaslt_status(
      hipblasLtMatrixLayoutCreate(&mat_b_layout, HIP_R_8I, k, m, k),
      "MatrixLayoutCreate(mat_b)");
  // C (INT32 scratch): col-major rows=N, cols=M, ldc=N
  check_hipblaslt_status(
      hipblasLtMatrixLayoutCreate(&mat_c_layout, HIP_R_32I, n, m, n),
      "MatrixLayoutCreate(mat_c)");
  // D (final output): col-major rows=N, cols=M, ldd=N
  check_hipblaslt_status(
      hipblasLtMatrixLayoutCreate(
          &output_layout, scalar_type_to_hip_type(output_dtype), n, m, n),
      "MatrixLayoutCreate(output)");

  // Set scale pointers
  void* scale_a_ptr = reinterpret_cast<void*>(scale_a.data_ptr());
  void* scale_b_ptr = reinterpret_cast<void*>(scale_b.data_ptr());
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(matmul_desc,
                                      HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                      &scale_a_ptr,
                                      sizeof(void*)),
      "SetAttribute(A_SCALE_POINTER)");
  check_hipblaslt_status(
      hipblasLtMatmulDescSetAttribute(matmul_desc,
                                      HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                      &scale_b_ptr,
                                      sizeof(void*)),
      "SetAttribute(B_SCALE_POINTER)");

  // Bias epilogue
  if (bias.has_value()) {
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    hipDataType bias_type = scalar_type_to_hip_type(output_dtype);
    void* bias_ptr = reinterpret_cast<void*>(bias.value().data_ptr());
    check_hipblaslt_status(
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                        &epilogue,
                                        sizeof(epilogue)),
        "SetAttribute(EPILOGUE)");
    check_hipblaslt_status(
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                                        HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                        &bias_type,
                                        sizeof(bias_type)),
        "SetAttribute(BIAS_DATA_TYPE)");
    check_hipblaslt_status(
        hipblasLtMatmulDescSetAttribute(matmul_desc,
                                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                        &bias_ptr,
                                        sizeof(void*)),
        "SetAttribute(BIAS_POINTER)");
  }

  // Algo heuristic search
  hipblasLtMatmulHeuristicResult_t heuristic;
  hipblasLtMatmulPreference_t pref;
  check_hipblaslt_status(hipblasLtMatmulPreferenceCreate(&pref),
                         "MatmulPreferenceCreate");
  int32_t max_workspace = 0;
  check_hipblaslt_status(hipblasLtMatmulPreferenceSetAttribute(
                             pref,
                             HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                             &max_workspace,
                             sizeof(max_workspace)),
                         "MatmulPreferenceSetAttribute(MAX_WORKSPACE)");

  int32_t returned_algo_count = 0;
  hipblasStatus_t heuristic_status =
      hipblasLtMatmulAlgoGetHeuristic(handle,
                                      matmul_desc,
                                      mat_a_layout,
                                      mat_b_layout,
                                      mat_c_layout,
                                      output_layout,
                                      pref,
                                      /*requestedAlgoCount=*/1,
                                      &heuristic,
                                      &returned_algo_count);

  CHECK(heuristic_status == HIPBLAS_STATUS_SUCCESS && returned_algo_count > 0)
      << "No hipBLASLt algo found for W8A8 matmul m=" << m << " n=" << n
      << " k=" << k;

  // Execute: D = alpha * op(A) * op(B) + beta * C
  const int32_t alpha_val = 1;
  const int32_t beta_val = 0;
  void* c_fake = reinterpret_cast<void*>(0x1);
  hipStream_t stream = c10::hip::getCurrentHIPStream();

  hipblasStatus_t exec_status = hipblasLtMatmul(handle,
                                                matmul_desc,
                                                &alpha_val,
                                                b.data_ptr(),
                                                mat_a_layout,
                                                a.data_ptr(),
                                                mat_b_layout,
                                                &beta_val,
                                                c_fake,
                                                mat_c_layout,
                                                output_tensor.data_ptr(),
                                                output_layout,
                                                &heuristic.algo,
                                                nullptr,
                                                0,
                                                stream);
  CHECK(exec_status == HIPBLAS_STATUS_SUCCESS)
      << "hipBLASLt matmul failed: status=" << exec_status << " m=" << m
      << " n=" << n << " k=" << k << " dtype=" << output_dtype;

  // Cleanup
  check_hipblaslt_status(hipblasLtMatrixLayoutDestroy(mat_a_layout),
                         "MatrixLayoutDestroy(mat_a)");
  check_hipblaslt_status(hipblasLtMatrixLayoutDestroy(mat_b_layout),
                         "MatrixLayoutDestroy(mat_b)");
  check_hipblaslt_status(hipblasLtMatrixLayoutDestroy(mat_c_layout),
                         "MatrixLayoutDestroy(mat_c)");
  check_hipblaslt_status(hipblasLtMatrixLayoutDestroy(output_layout),
                         "MatrixLayoutDestroy(output)");
  check_hipblaslt_status(hipblasLtMatmulPreferenceDestroy(pref),
                         "MatmulPreferenceDestroy");
  check_hipblaslt_status(hipblasLtMatmulDescDestroy(matmul_desc),
                         "MatmulDescDestroy");

  return output_tensor;
}

}  // namespace xllm::kernel::dcu

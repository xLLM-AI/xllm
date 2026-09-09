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
#pragma once
namespace xllm::kernel::ilu {
#undef check_tensor_contiguous
#define check_tensor_contiguous(x, type) \
  TORCH_CHECK(x.scalar_type() == type);  \
  TORCH_CHECK(x.is_cuda());              \
  TORCH_CHECK(x.is_contiguous());

#undef check_tensor_half_bf_float
#define check_tensor_half_bf_float(x)                       \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||    \
              x.scalar_type() == at::ScalarType::Float ||   \
              x.scalar_type() == at::ScalarType::BFloat16); \
  TORCH_CHECK(x.is_cuda());

// from torchCheckMsgImpl
inline const char* ixformer_check_msg_impl(const char* msg) { return msg; }
// // If there is just 1 user-provided C-string argument, use it.

#define IXFORMER_CHECK_MSG(cond, type, ...)                 \
  (ixformer_check_msg_impl(                                 \
      "Expected " #cond                                     \
      " to be true, but got false.  "                       \
      "(Could this error message be improved?  If so, "     \
      "please report an enhancement request to ixformer.)", \
      ##__VA_ARGS__))

#define IXFORMER_CHECK(cond, ...)                                            \
  {                                                                          \
    if (!(cond)) {                                                           \
      std::cerr << __FILE__ << " (" << __LINE__ << ")"                       \
                << "-" << __FUNCTION__ << " : "                              \
                << IXFORMER_CHECK_MSG(cond, "", ##__VA_ARGS__) << std::endl; \
      throw std::runtime_error("IXFORMER_CHECK ERROR");                      \
    }                                                                        \
  }

#undef CUINFER_CHECK
#define CUINFER_CHECK(func)                                                \
  do {                                                                     \
    cuinferStatus_t status = (func);                                       \
    if (status != CUINFER_STATUS_SUCCESS) {                                \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ \
                << ": " << cuinferGetErrorString(status) << std::endl;     \
      throw std::runtime_error("CUINFER_CHECK ERROR");                     \
    }                                                                      \
  } while (0)

}  // namespace xllm::kernel::ilu

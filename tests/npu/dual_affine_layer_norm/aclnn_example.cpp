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

#include <acl/acl.h>
#include <aclnn_dual_affine_layer_norm.h>

#include <array>
#include <cstdint>
#include <iostream>

#define CHECK_ACL(call)                                                   \
  do {                                                                    \
    const int32_t status = (call);                                        \
    if (status != 0) {                                                    \
      std::cerr << #call << ": " << status << " " << aclGetRecentErrMsg() \
                << std::endl;                                             \
      return 1;                                                           \
    }                                                                     \
  } while (false)

int main() {
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  constexpr int64_t rows = 3, cols = 7;
  const int64_t matrix_shape[] = {rows, cols};
  const int64_t matrix_stride[] = {cols, 1};
  const int64_t vector_shape[] = {cols}, vector_stride[] = {1};
  std::array<void*, 7> buffers{};
  std::array<aclTensor*, 7> tensors{};
  for (size_t i = 0; i < buffers.size(); ++i) {
    const bool matrix = i == 0 || i >= 5;
    const size_t elements = matrix ? rows * cols : cols;
    CHECK_ACL(
        aclrtMalloc(&buffers[i], elements * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    std::array<uint16_t, rows * cols> host{};
    // x=0, gamma1=gamma2=1, beta1=0.5, beta2=-0.5 (raw FP16).
    host.fill(i == 1 || i == 3 ? 0x3c00
              : i == 2         ? 0x3800
              : i == 4         ? 0xb800
                               : 0);
    CHECK_ACL(aclrtMemcpy(buffers[i],
                          elements * 2,
                          host.data(),
                          elements * 2,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    tensors[i] = aclCreateTensor(matrix ? matrix_shape : vector_shape,
                                 matrix ? 2 : 1,
                                 ACL_FLOAT16,
                                 matrix ? matrix_stride : vector_stride,
                                 0,
                                 ACL_FORMAT_ND,
                                 matrix ? matrix_shape : vector_shape,
                                 matrix ? 2 : 1,
                                 buffers[i]);
    if (tensors[i] == nullptr) {
      return 1;
    }
  }
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL(aclnnDualAffineLayerNormGetWorkspaceSize(tensors[0],
                                                     tensors[1],
                                                     tensors[2],
                                                     tensors[3],
                                                     tensors[4],
                                                     1e-5,
                                                     tensors[5],
                                                     tensors[6],
                                                     &workspace_size,
                                                     &executor));
  if (workspace_size != 0) {
    return 1;
  }
  CHECK_ACL(aclnnDualAffineLayerNorm(nullptr, 0, executor, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));
  for (size_t i = 5; i < 7; ++i) {
    std::array<uint16_t, rows * cols> output{};
    CHECK_ACL(aclrtMemcpy(output.data(),
                          sizeof(output),
                          buffers[i],
                          sizeof(output),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (const uint16_t value : output) {
      if (value != (i == 5 ? 0x3800 : 0xb800)) {
        return 1;
      }
    }
  }
  for (size_t i = 0; i < buffers.size(); ++i) {
    CHECK_ACL(aclDestroyTensor(tensors[i]));
    CHECK_ACL(aclrtFree(buffers[i]));
  }
  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(0));
  CHECK_ACL(aclFinalize());
  std::cout << "ACLNN two-stage example passed; workspace=0" << std::endl;
}

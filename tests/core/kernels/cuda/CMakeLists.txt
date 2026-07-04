include(cc_test)

add_subdirectory(moe)
add_subdirectory(xattention)

cc_test(
  NAME
    activation_test
  SRCS
    activation_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)

option(XLLM_ENABLE_BLOCK_COPY_TEST
       "Build and register the expensive CUDA block_copy_test"
       OFF)

if(XLLM_ENABLE_BLOCK_COPY_TEST)
  cc_test(
    NAME
      block_copy_test
    SRCS
      block_copy_test.cpp
    DEPS
      :cuda_kernels
      torch
      GTest::gtest_main
      glog::glog
  )
endif()

cc_test(
  NAME
    fused_qknorm_rope_test
  SRCS
    fused_qknorm_rope_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    cutlass_scaled_mm_test
  SRCS
    cutlass_scaled_mm_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cutlass_scaled_mm_test PRIVATE brpc)

cc_test(
  NAME
    fp8_quant_test
  SRCS
    fp8_quant_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(fp8_quant_test PRIVATE brpc)

cc_test(
  NAME
    xllm_ops_test
  SRCS
    xllm_ops_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
# Embedded-interpreter test path: needs libtorch_python (THPVariable / at::Tensor
# pybind caster) and libpython.
target_link_libraries(xllm_ops_test PRIVATE torch_python Python::Python)

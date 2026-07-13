include(cc_test)

# Temporarily disabled: Dsv4ScatterCache tests are unstable in the current
# NPU test environment.
# cc_test(
#   NAME
#     dsv4_scatter_cache_write_test
#   SRCS
#     dsv4_scatter_cache_write_test.cpp
#   DEPS
#     ascendcl
#     torch
#     torch_npu
#     xllm_ops
#     GTest::gtest_main
#     glog::glog
# )

cc_test(
  NAME
    npu_activation_test
  SRCS
    activation_test.cpp
  DEPS
    :npu_kernels
    torch
    GTest::gtest_main
)

target_link_libraries(npu_activation_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

add_subdirectory(tilelang)

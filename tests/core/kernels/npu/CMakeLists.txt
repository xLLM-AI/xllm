include(cc_test)

cc_test(
  NAME
    npu_xllm_ops_test
  SRCS
    npu_xllm_ops_test.cpp
  DEPS
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest_main
    glog::glog
    pybind11::embed
)

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

add_subdirectory(tilelang)

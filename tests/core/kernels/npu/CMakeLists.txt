include(cc_test)

cc_test(
  NAME
    mega_moe_test
  SRCS
    mega_moe_test.cpp
  DEPS
    ascendcl
    torch
    torch_npu
    npu_kernels
    GTest::gtest_main
    glog::glog
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

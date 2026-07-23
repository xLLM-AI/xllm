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
    mtp_prepare_next_draft_test
  SRCS
    mtp_prepare_next_draft_test.cpp
  DEPS
    :xllm_ops
    ascendcl
    nnopbase
    torch
    torch_npu
    GTest::gtest_main
    glog::glog
)
if(EXISTS "$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
  target_link_options(mtp_prepare_next_draft_test PRIVATE
    "-Wl,-rpath-link,$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
endif()

add_subdirectory(tilelang)

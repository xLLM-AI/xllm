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

# build_cp_context is CompositeExplicitAutograd pure host index math, so this
# runs on CPU with no NPU; it only needs the xllm_ops registrations linked in.
cc_test(
  NAME
    cp_context_builder_test
  SRCS
    cp_context_builder_test.cpp
  DEPS
    :xllm_ops
    ascendcl
    torch
    torch_npu
    kernels
    npu_kernels
    GTest::gtest_main
    glog::glog
)
if(EXISTS "$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
  target_link_options(cp_context_builder_test PRIVATE
    "-Wl,-rpath-link,$ENV{PYTORCH_INSTALL_PATH}/../torch.libs")
endif()

add_subdirectory(tilelang)

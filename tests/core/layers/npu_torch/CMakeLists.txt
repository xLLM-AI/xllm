include(cc_test)

cc_test(
  NAME
    npu_deepseek_v4_indexer_test
  SRCS
    deepseek_v4_indexer_tests.cpp
  DEPS
    :npu_torch_layers
    :state_dict
    :xllm_ops
    glog::glog
    torch
    GTest::gtest_main
)

target_link_libraries(npu_deepseek_v4_indexer_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

target_link_options(npu_deepseek_v4_indexer_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NAME
    npu_linear_w8a8_dynamic_test
  SRCS
    linear_w8a8_dynamic_tests.cpp
    tests_utils.cpp
  DEPS
    :npu_torch_layers
    :common_layers
    :parallel_state
    :model
    :state_dict
    :xllm_ops
    glog::glog
    torch
    GTest::gtest_main
)

target_link_libraries(npu_linear_w8a8_dynamic_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

target_link_options(npu_linear_w8a8_dynamic_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

# Pure-CPU metadata unit test for DSA-CP (M0). No device ops; validates the
# host-side token-partition math in DSAMetadataBuilder::build_cp_local_metadata.
cc_test(
  NAME
    npu_dsa_cp_metadata_test
  SRCS
    dsa_cp_metadata_tests.cpp
  DEPS
    :common_layers
    :parallel_state
    :model
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

target_link_libraries(npu_dsa_cp_metadata_test
                      PRIVATE
                      ascendcl
                      hccl
                      c_sec
                      nnopbase
                      atb)

target_link_options(npu_dsa_cp_metadata_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

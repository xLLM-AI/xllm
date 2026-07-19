include(cc_test)

cc_test(
  NAME
    mega_moe_policy_test
  SRCS
    mega_moe_policy_test.cpp
  DEPS
    :mega_moe_policy
    GTest::gtest_main
)

cc_test(
  NAME
    mega_moe_runtime_test
  SRCS
    mega_moe_runtime_test.cpp
  DEPS
    :mega_moe_runtime
    torch
    GTest::gtest_main
)

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

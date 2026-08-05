include(cc_test)

cc_test(
  NAME
    eplb_options_test
  SRCS
    eplb_options_test.cpp
  DEPS
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_options_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    eplb_policy_test
  SRCS
    eplb_policy_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_policy_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    eplb_utils_test
  SRCS
    eplb_utils_test.cpp
  DEPS
    glog::glog
    torch
    GTest::gtest_main
)
target_link_libraries(eplb_utils_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    eplb_aggregator_test
  SRCS
    eplb_aggregator_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_aggregator_test
                      PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    expert_weight_buffer_shm_test
  SRCS
    expert_weight_buffer_shm_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(expert_weight_buffer_shm_test
                      PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    eplb_manager_test
  SRCS
    eplb_manager_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_manager_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

cc_test(
  NAME
    eplb_executor_test
  SRCS
    eplb_executor_test.cpp
  DEPS
    torch
    :eplb
    GTest::gtest_main
)
target_link_libraries(eplb_executor_test PRIVATE $<$<BOOL:${USE_NPU}>:c_sec>)

include(cc_test)

cc_test(
  NAME
    host_kv_transfer_test
  SRCS
    host_transfer/transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    push_route_test
  SRCS
    push_route_test.cpp
  DEPS
    :push_route
    GTest::gtest_main
)

cc_test(
  NAME
    pd_topology_guard_test
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

cc_test(
  NAME
    kv_transfer_completion_test
  SRCS
    kv_transfer_completion_test.cpp
  DEPS
    :kv_transfer_completion
    GTest::gtest_main
)

cc_test(
  NAME
    reshard_planner_test
  SRCS
    reshard_planner_test.cpp
  DEPS
    :reshard_planner
    GTest::gtest_main
)

if(USE_MLU)
cc_test(
  NAME
    compact_load_test
  SRCS
    host_transfer/compact_load_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    compact_offload_test
  SRCS
    host_transfer/compact_offload_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    kv_cache_store_test
  SRCS
    kv_cache_store_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    basic_host_kv_transfer_test
  SRCS
    host_transfer/basic_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    hierarchy_kv_cache_transfer_test
  SRCS
    hierarchy_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)
endif()

include(cc_test)

cc_test(
  NAME
    kv_cache_page_state_test
  NO_NPU_RUNTIME
  SRCS
    kv_cache_page_state_test.cpp
  DEPS
    :kv_cache_page_state
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    concurrent_block_manager_test
  SRCS
    concurrent_block_manager_test.cpp
  DEPS
    :block
    :config
    :prefix_cache
    GTest::gtest_main
)

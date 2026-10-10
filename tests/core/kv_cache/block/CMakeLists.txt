include(cc_test)

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

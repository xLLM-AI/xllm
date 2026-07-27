include(cc_test)

cc_test(
  NAME
    processor_cache_test
  SRCS
    processor_cache_test.cpp
  DEPS
    :multimodal
    :util
    :config
    GTest::gtest_main
    torch
)

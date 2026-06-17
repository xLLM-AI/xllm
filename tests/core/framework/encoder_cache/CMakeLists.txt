include(cc_test)

cc_test(
  NAME
    encoder_cache_test
  SRCS
    encoder_cache_test.cpp
  DEPS
    :encoder_cache
    :util
    :config
    GTest::gtest_main
    torch
)

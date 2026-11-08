include(cc_test)

cc_test(
  NAME
    encoder_cache_test
  CPU_ONLY
  SRCS
    encoder_cache_test.cpp
  DEPS
    :content_hash
    :encoder_cache
    :util
    :config
    GTest::gtest_main
    torch
)

include(cc_test)

cc_test(
  NAME
    mm_codec_test
  CPU_ONLY
  SRCS
    mm_codec_test.cpp
  DEPS
    :multimodal
    GTest::gtest_main
)

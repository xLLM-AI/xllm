include(cc_test)

cc_test(
  NAME
    fast_tokenizer_test
  SRCS
    fast_tokenizer_test.cpp
  DEPS
    :tokenizer
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    rwkv_tokenizer_test
  SRCS
    rwkv_tokenizer_test.cpp
  DEPS
    :tokenizer
    glog::glog
    GTest::gtest_main
)

include(cc_test)

cc_test(
  NAME
    rec_vocab_dict_test
  SRCS
    rec_vocab_dict_test.cpp
  DEPS
    :config
    :state_dict
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    state_dict_utils_test
  SRCS
    state_dict_utils_test.cpp
  DEPS
    :state_dict
    GTest::gtest_main
    torch
)

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

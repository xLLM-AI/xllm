include(cc_test)


cc_test(
  NAME
    state_dict_utils_test
  CPU_ONLY
  SRCS
    state_dict_utils_test.cpp
  DEPS
    :config
    :state_dict
    GTest::gtest_main
    glog::glog
)

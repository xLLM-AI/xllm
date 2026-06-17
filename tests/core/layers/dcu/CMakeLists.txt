include(cc_test)

cc_test(
  NAME
    activation_layer_dcu_test
  SRCS
    activation_test.cpp
  DEPS
    :common_layers
    :platform
    glog::glog
    torch
    GTest::gtest_main
)

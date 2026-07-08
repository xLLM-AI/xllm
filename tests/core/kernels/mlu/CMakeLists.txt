include(cc_test)

cc_test(
  NAME
    beam_search_mlu_test
  SRCS
    beam_search_test.cpp
  DEPS
    :mlu_kernels
    torch
    torch_mlu
    GTest::gtest_main
    glog::glog
)

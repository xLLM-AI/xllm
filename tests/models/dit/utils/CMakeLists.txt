include(cc_test)

cc_test(
  NAME
    cola_utils_test
  SRCS
    cola_utils_test.cpp
  DEPS
    torch
    GTest::gtest_main
)

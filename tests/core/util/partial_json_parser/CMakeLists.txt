include(cc_test)

cc_test(
  CPU_ONLY
  NAME
    partial_json_parser_test
  SRCS
    loads_test.cpp
    examples_test.cpp
    property_based_test.cpp
  DEPS
    :partial_json_parser
    GTest::gtest
    GTest::gtest_main
)

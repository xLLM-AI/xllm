include(cc_test)

cc_test(
  CPU_ONLY
  NAME
    partial_json_parser_test
  SRCS
    test_examples.cpp
    test_property_based.cpp
  DEPS
    :partial_json_parser
    GTest::gtest
    GTest::gtest_main
)

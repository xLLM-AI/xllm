include(cc_test)

cc_test(
  NAME
    config_json_test
  CPU_ONLY
  SRCS
    config_json_test.cpp
  DATA
    config_test.json
  DEPS
    :config
    :util
    GTest::gtest_main
)

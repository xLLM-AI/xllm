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

# Isolate real gflags parsing from config_json_test's legacy FLAGS_ writeback.
cc_test(
  NAME
    scheduler_config_input_test
  CPU_ONLY
  SRCS
    scheduler_config_input_test.cpp
  DEPS
    :config
    :util
    GTest::gtest_main
)

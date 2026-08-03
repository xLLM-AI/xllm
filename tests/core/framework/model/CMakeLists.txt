include(cc_test)

cc_test(
  NAME
    mtp_draft_model_args_test
  SRCS
    mtp_draft_model_args_test.cpp
  DEPS
    :model
    GTest::gtest_main
)

include(cc_test)

cc_test(
  NAME
    linear_state_restore_test
  SRCS
    linear_state_restore_test.cpp
  DEPS
    :linear_state_restore
    GTest::gtest_main
)
target_link_libraries(linear_state_restore_test PUBLIC Python::Python)

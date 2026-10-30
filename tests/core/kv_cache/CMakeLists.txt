include(cc_test)

add_subdirectory(layout)
add_subdirectory(storage)
add_subdirectory(block)
add_subdirectory(events)
add_subdirectory(prefix)
add_subdirectory(transfer)

cc_test(
  NAME
    linear_state_restore_test
  CPU_ONLY
  SRCS
    linear_state_restore_test.cpp
  DEPS
    :linear_state_restore
    :model
    GTest::gtest_main
)
target_link_libraries(linear_state_restore_test PUBLIC Python::Python)

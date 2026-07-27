include(cc_test)

# Standalone test for AutoFlipController::decide_target's mode-selection
# state machine (algorithm mirrored inline). Depends only on gtest so it
# runs without a live Engine / Scheduler / brpc stub.
cc_test(
  NAME
    auto_flip_decide_target_test
  SRCS
    auto_flip_decide_target_test.cpp
  DEPS
    GTest::gtest_main
)

# vlm_master_test is a standalone manual test binary and has no registered
# CTest target yet.

include(cc_test)

cc_test(
  NAME
    spawn_worker_protocol_test
  SRCS
    spawn_worker_protocol_test.cpp
  DEPS
    GTest::gtest_main
)

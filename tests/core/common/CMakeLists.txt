include(cc_test)

cc_test(
  NAME
    common_test
  SRCS
    flash_comm1_context_test.cpp
    options_test.cpp
    rate_limiter_test.cpp
  DEPS
    :config
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
)

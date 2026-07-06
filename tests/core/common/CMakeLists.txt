include(cc_test)

cc_test(
  NAME
    common_test
  SRCS
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

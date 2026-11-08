# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    model_util_test
  CPU_ONLY
  SRCS
    model_path_utils_test.cpp
    model_type_test.cpp
    vocab_validation_test.cpp
  DEPS
    glog::glog
    GTest::gtest_main
)

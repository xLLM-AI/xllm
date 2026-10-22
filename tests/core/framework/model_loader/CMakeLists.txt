# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

add_subdirectory(weight)

cc_test(
  NAME
    hf_model_loader_profile_test
  SRCS
    hf_model_loader_profile_test.cpp
  DEPS
    :model_loader
    GTest::gtest_main
)

# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

add_subdirectory(weight)

include(cc_test)

cc_test(
  NAME
    dit_model_discovery_test
  CPU_ONLY
  SRCS
    dit_model_discovery_test.cpp
  DEPS
    :dit_model_discovery
    GTest::gtest_main
)

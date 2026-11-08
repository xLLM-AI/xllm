# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    tensor_proto_test
  CPU_ONLY
  SRCS
    tensor_proto_test.cpp
  DEPS
    :tensor_codec
    GTest::gtest_main
)

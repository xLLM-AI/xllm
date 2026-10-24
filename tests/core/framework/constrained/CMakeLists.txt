# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    json_object_grammar_test
  SRCS
    json_object_grammar_test.cpp
  DEPS
    :json_object_grammar
    :sampler
    GTest::gtest_main
)

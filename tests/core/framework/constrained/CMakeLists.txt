# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    json_object_grammar_test
  CPU_ONLY
  SRCS
    json_object_grammar_test.cpp
  DEPS
    :json_object_grammar
    :sampler
    GTest::gtest_main
)

cc_test(
  NAME
    json_object_constraints_test
  SRCS
    json_object_constraints_test.cpp
  DEPS
    :json_object_grammar
    GTest::gtest_main
)

cc_test(
  NAME
    json_object_output_binding_test
  CPU_ONLY
  SRCS
    json_object_output_binding_test.cpp
  DEPS
    :json_object_grammar
    GTest::gtest_main
)

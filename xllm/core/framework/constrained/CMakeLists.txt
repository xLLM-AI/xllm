# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

cc_library(
  NAME
    json_object_grammar
  HDRS
    json_object_grammar.h
  SRCS
    json_object_grammar.cpp
  DEPS
    :tokenizer
    :common
    glog::glog
    torch)

cc_library(
  NAME
    json_object_constraint_batch
  HDRS
    json_object_constraint_batch.h
    json_object_output_binding.h
    json_object_output_error.h
  SRCS
    json_object_constraint_batch.cpp
  DEPS
    :json_object_grammar
    glog::glog
    torch)

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
    rec_constrained_decoding
  HDRS
    rec_constrained_decoding.h
  SRCS
    rec_constrained_decoding.cpp
  DEPS
    :state_dict
    :common
    util
    glog::glog
    torch
)

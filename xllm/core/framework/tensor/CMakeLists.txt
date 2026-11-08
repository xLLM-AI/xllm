# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

cc_library(
  NAME
    tensor_support
  HDRS
    linalg.h
    pybind_helper.h
    tensor_helper.h
    tensor_options.h
  SRCS
    linalg.cpp
    tensor_options.cpp
  DEPS
    :util
    torch
    glog::glog
    Boost::serialization
)

cc_library(
  NAME
    tensor_codec
  HDRS
    tensor_codec.h
  SRCS
    tensor_codec.cpp
  DEPS
    torch
    glog::glog
    :util
    proto::xllm_proto
)
